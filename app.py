import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from tensorflow.keras.models import load_model
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import os
import cv2

# === Load TFLite Optimized Classifier ===
interpreter = tf.lite.Interpreter(model_path="model/model_quantized.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Load CNN model & anomaly detection SVM ===
model = load_model("model/best_model.keras")
svm_model = joblib.load("model/one_class_svm.joblib")
scaler = joblib.load("model/scaler.joblib")

# === Class Labels & Disease Info ===
CLASS_NAMES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy Cassava Leaf"
]

DISEASE_INFO = {
    "Cassava Bacterial Blight (CBB)": "🦠 Caused by Xanthomonas bacteria. Angular spots, wilting, dieback.",
    "Cassava Brown Streak Disease (CBSD)": "🧬 Caused by CBS viruses. Brown streaks, root rot.",
    "Cassava Mosaic Disease (CMD)": "🧫 Geminiviruses cause mosaic leaf patterns, stunting.",
    "Healthy Cassava Leaf": "✅ Leaf appears healthy. No visible disease symptoms."
}

# === Enhanced Feature Extractor ===
class FeatureExtractor:
    def __init__(self, model, layer_index=8):
        self.model = model
        self.layer_index = layer_index
        
    def extract_features(self, x):
        """Extract features from intermediate layer"""
        @tf.function
        def get_intermediate_output(inputs):
            x = inputs
            for i, layer in enumerate(self.model.layers):
                x = layer(x)
                if i == self.layer_index:
                    return x
            return x
        
        features = get_intermediate_output(x)
        
        # Average pool if 4D tensor
        if len(features.shape) == 4:
            features = tf.reduce_mean(features, axis=[1, 2])
        
        return features.numpy()

# Initialize feature extractor
feature_extractor = FeatureExtractor(model, layer_index=8)

# === Multi-Modal Anomaly Detection ===
def detect_faces(image):
    """Detect human faces using OpenCV"""
    try:
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return len(faces) > 0, len(faces)
    except Exception as e:
        print(f"Face detection error: {e}")
        return False, 0

def analyze_image_properties(image):
    """Analyze basic image properties that might indicate non-leaf content"""
    try:
        img_array = np.array(image)
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Check for skin-like colors (hue range for human skin)
        skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
        skin_percentage = np.sum(skin_mask > 0) / (img_array.shape[0] * img_array.shape[1])
        
        # Check for green vegetation (leaves should have significant green)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_percentage = np.sum(green_mask > 0) / (img_array.shape[0] * img_array.shape[1])
        
        # Calculate color variance (leaves usually have more uniform colors)
        color_variance = np.var(img_array)
        
        # Edge density (leaves have specific edge patterns)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return {
            'skin_percentage': skin_percentage,
            'green_percentage': green_percentage,
            'color_variance': color_variance,
            'edge_density': edge_density
        }
    except Exception as e:
        print(f"Image analysis error: {e}")
        return None

def enhanced_anomaly_detection(image, features):
    """Multi-modal anomaly detection combining SVM, face detection, and image analysis"""
    
    # 1. Face Detection
    has_face, face_count = detect_faces(image)
    if has_face:
        return True, f"Anomaly image  detected ", -10.0
    
    # 2. Image Property Analysis
    props = analyze_image_properties(image)
    if props:
        # High skin percentage indicates human/animal content
        if props['skin_percentage'] > 0.15:  # 15% skin-like colors
            return True, "High skin-like color content detected", -8.0
        
        # Very low green percentage indicates non-vegetation
        if props['green_percentage'] < 0.05:  # Less than 5% green
            return True, "Insufficient vegetation color detected", -7.0
        
        # Very high color variance might indicate complex scenes
        if props['color_variance'] > 8000:
            return True, "High color complexity (non-leaf pattern)", -6.0
    
    # 3. SVM-based Feature Anomaly Detection
    try:
        scaled_features = scaler.transform(features)
        svm_score = svm_model.decision_function(scaled_features)[0]
        is_svm_outlier = svm_model.predict(scaled_features)[0] == -1.5
        
        # More strict SVM threshold
        if is_svm_outlier and svm_score < -0.5:
            return True, "Feature-based anomaly detected", svm_score
        
        return False, "Appears to be cassava leaf", svm_score
        
    except Exception as e:
        print(f"SVM anomaly detection error: {e}")
        return False, "SVM check failed, proceeding with prediction", 0.0

def confidence_threshold_check(output, threshold=0.6):
    """Check if the model is confident enough in its prediction"""
    max_confidence = np.max(output)
    
    # If confidence is too low, it might be an anomaly
    if max_confidence < threshold:
        return True, f"Low prediction confidence ({max_confidence:.3f})"
    
    return False, f"Good prediction confidence ({max_confidence:.3f})"

# === Image Preprocessing Helper ===
def preprocess_image(image: Image.Image):
    img_array = np.array(image.convert("RGB").resize((224, 224))) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

# === Extract Features for Anomaly Detection ===
def extract_feature(image: Image.Image):
    array = preprocess_image(image)
    features = feature_extractor.extract_features(array)
    return features

# === Convert Image to Base64 for UI ===
def image_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# === Full Prediction Pipeline ===
def predict_image(image: Image.Image):
    try:
        # Extract features for anomaly detection
        features = extract_feature(image)
        
        # Enhanced anomaly detection
        is_anomaly, anomaly_reason, anomaly_score = enhanced_anomaly_detection(image, features)
        
        # If anomaly detected, return warning
        if is_anomaly:
            return f"""
            <div style='color:#f87171; background:#1e293b; padding:16px; border-radius:12px; 
                        font-family:sans-serif; border-left: 4px solid #ef4444;'>
                <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                    <span style='font-size: 20px; margin-right: 8px;'>⚠️</span>
                    <strong>Non-Cassava Image Detected</strong>
                </div>
                <div style='margin-bottom: 8px; color: #fca5a5;'>
                    <strong>Reason:</strong> {anomaly_reason}
                </div>
                <div style='font-size: 12px; color: #94a3b8;'>
                    Anomaly Score: {anomaly_score:.4f}
                </div>
                <div style='margin-top: 10px; padding: 8px; background: #374151; border-radius: 6px;'>
                    <strong>💡 Tip:</strong> Please upload a clear image of cassava leaves for accurate disease detection.
                </div>
            </div>
            """

        # Proceed with TFLite prediction
        x = preprocess_image(image)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Check prediction confidence
        low_confidence, confidence_msg = confidence_threshold_check(output, threshold=0.7)
        
        if low_confidence:
            return f"""
            <div style='color:#f59e0b; background:#1e293b; padding:16px; border-radius:12px; 
                        font-family:sans-serif; border-left: 4px solid #f59e0b;'>
                <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                    <span style='font-size: 20px; margin-right: 8px;'>⚠️</span>
                    <strong>Low Confidence Prediction</strong>
                </div>
                <div style='margin-bottom: 8px; color: #fbbf24;'>
                    {confidence_msg}
                </div>
                <div style='margin-top: 10px; padding: 8px; background: #374151; border-radius: 6px;'>
                    <strong>💡 Tip:</strong> The model is not confident about this image. Please ensure it's a clear cassava leaf image.
                </div>
            </div>
            """

        pred_idx = int(np.argmax(output))
        confidence = float(output[0][pred_idx]) * 100
        class_name = CLASS_NAMES[pred_idx]
        description = DISEASE_INFO[class_name]

        # Success prediction
        html = f"""
        <div style="margin-bottom: 25px; padding: 20px; border-radius: 12px;
            background: linear-gradient(to right, #1e293b, #0f172a); color: #f1f5f9;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4); font-family: 'Segoe UI', sans-serif;
            display: flex; align-items: flex-start; border: 1px solid #22c55e;">
            <img src="{image_to_base64(image)}" style="width: 140px; height: auto;
                border-radius: 10px; border: 2px solid #22c55e; margin-right: 20px;">
            <div style="flex: 1;">
                <div style="font-size: 20px; font-weight: 600; margin-bottom: 10px; color: #22c55e;">
                    🍃 {class_name}
                </div>
                <div style="font-size: 16px; margin-bottom: 8px;">
                    📊 <span style="color: #facc15;">Confidence:</span> <span style="color: #34d399;">{confidence:.1f}%</span>
                </div>
                <div style="font-size: 14px; color: #cbd5e1; margin-bottom: 10px;">
                    {description}
                </div>
                <div style="font-size: 12px; color: #64748b; padding: 8px; background: #374151; border-radius: 6px;">
                    ✅ Image passed all anomaly checks • Score: {anomaly_score:.3f}
                </div>
            </div>
        </div>
        """
        return html
        
    except Exception as e:
        return f"""
        <div style='color:#f87171; background:#1e293b; padding:16px; border-radius:12px; 
                    font-family:sans-serif; border-left: 4px solid #ef4444;'>
            <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                <span style='font-size: 20px; margin-right: 8px;'>❌</span>
                <strong>Processing Error</strong>
            </div>
            <div style='color: #fca5a5;'>
                {str(e)}
            </div>
        </div>
        """

# === Predict Multiple Files ===
def predict_multiple(images):
    if not images:
        return gr.HTML("<div style='color:#f87171; padding:14px;'>No images uploaded.</div>")
    
    results = ""
    for i, img in enumerate(images):
        try:
            image = Image.open(img) if isinstance(img, str) else img
            results += f"<h3 style='color:#22c55e; margin-top: 20px;'>Image {i+1}</h3>"
            results += predict_image(image)
        except Exception as e:
            results += f"""
            <div style='color:#f87171; background:#1e293b; padding:14px; border-radius:10px; font-family:sans-serif'>
                ❌ <b>Error processing image {i+1}:</b> {str(e)}
            </div>
            """
    return gr.HTML(results)

# === Predict Webcam Input ===
def predict_webcam(image):
    if image is None:
        return gr.HTML("<div style='color:#f87171; padding:14px;'>No image captured.</div>")
    return gr.HTML(predict_image(image))

# === Build Gradio UI ===
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown("""
    # 🌿 Cassava Leaf Disease Detector
    
    **Features:**
    - ✅ Multi-modal anomaly detection (Anomaly detection + Color analysis + Feature analysis)
    - 🔍 Enhanced accuracy with confidence thresholds
    - 🚫 Automatic rejection of non-cassava images
    - 📊 Detailed prediction explanations
    """)
    
    with gr.Row():
        with gr.Column():
            upload_input = gr.File(
                label="📁 Upload Cassava Leaf Images", 
                file_types=["image"], 
                file_count="multiple"
            )
            webcam_input = gr.Image(
                label="📷 Capture from Webcam", 
                sources=["webcam"], 
                type="pil"
            )
            
            with gr.Row():
                predict_btn = gr.Button("🔍 Analyze Images", variant="primary")
                webcam_btn = gr.Button("📸 Analyze Webcam Image", variant="secondary")
                
        with gr.Column():
            result_area = gr.HTML(label="Analysis Results")

    predict_btn.click(fn=predict_multiple, inputs=[upload_input], outputs=[result_area])
    webcam_btn.click(fn=predict_webcam, inputs=[webcam_input], outputs=[result_area])

    gr.Markdown("---")
    gr.Markdown("""
    ### 💡 Tips for Best Results:
    - Use clear, well-lit images of cassava leaves
    - Avoid blurry or dark images
    - Ensure leaves fill most of the image frame
    - Remove any background objects or people
    """)
    
    gr.Markdown("<center style='color:#4ade80;'>🚀 Built with ❤️ by Jimoh Sodiq</center>")

# ✅ Launch App
if __name__ == "__main__":
    demo.launch(share=True)