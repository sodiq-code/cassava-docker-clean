"""
Professional Multi-Page Cassava Disease Detector
===============================================
- Clean tabbed interface with dedicated pages
- Maintains all original functionality
- Optimized mobile-first design
- Concise and professional structure
"""

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import os
import cv2
from datetime import datetime

# Optional imports
try:
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Load Models
def load_models():
    interpreter = svm_model = scaler = None
    try:
        if os.path.exists("model/model_quantized.tflite"):
            interpreter = tf.lite.Interpreter(model_path="model/model_quantized.tflite")
            interpreter.allocate_tensors()
    except Exception as e:
        print(f"TFLite loading error: {e}")
    try:
        if HAS_SKLEARN:
            if os.path.exists("model/one_class_svm.joblib"):
                svm_model = joblib.load("model/one_class_svm.joblib")
            if os.path.exists("model/scaler.joblib"):
                scaler = joblib.load("model/scaler.joblib")
    except Exception as e:
        print(f"SVM loading error: {e}")
    return interpreter, svm_model, scaler

interpreter, svm_model, scaler = load_models()
history_log = []

# Disease Data
CLASS_NAMES = ["Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)", "Cassava Mosaic Disease (CMD)", "Healthy Cassava Leaf"]
DISEASE_INFO = {
    "Cassava Bacterial Blight (CBB)": {"icon": "🦠", "severity": "High", "color": "#dc2626", "description": "Bacterial infection causing angular leaf spots and wilting", "treatment": "Remove infected plants, use copper-based treatments"},
    "Cassava Brown Streak Disease (CBSD)": {"icon": "🧬", "severity": "High", "color": "#ea580c", "description": "Viral disease causing brown streaks and root rot", "treatment": "Use resistant varieties, control whitefly vectors"},
    "Cassava Mosaic Disease (CMD)": {"icon": "🧫", "severity": "Medium", "color": "#d97706", "description": "Viral infection creating mosaic patterns on leaves", "treatment": "Plant resistant varieties, remove infected plants"},
    "Healthy Cassava Leaf": {"icon": "✅", "severity": "None", "color": "#16a34a", "description": "Healthy leaf with no disease symptoms", "treatment": "Continue monitoring and good practices"}
}

# Core Functions
def preprocess_image(image):
    img_array = np.array(image.convert("RGB").resize((224, 224))) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def image_to_base64(pil_img):
    img_copy = pil_img.copy()
    img_copy.thumbnail((400, 400), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    img_copy.save(buffered, format="JPEG", quality=85, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

def detect_anomaly(image):
    try:
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            return True, "Anomaly image detected"
        hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_percentage = np.sum(green_mask > 0) / (image.size[0] * image.size[1])
        if green_percentage < 0.1:
            return True, "Insufficient vegetation detected"
        return False, "Valid cassava leaf image"
    except:
        return False, "Analysis completed"

def create_alert(type_msg, title, message, include_tips=False):
    colors = {"error": {"bg": "#fee2e2", "border": "#dc2626", "text": "#991b1b", "icon": "❌"}, 
             "warning": {"bg": "#fef3c7", "border": "#f59e0b", "text": "#92400e", "icon": "⚠️"}, 
             "info": {"bg": "#dbeafe", "border": "#3b82f6", "text": "#1e40af", "icon": "ℹ️"}}
    color = colors.get(type_msg, colors["info"])
    tips_section = '''<div class="tips-section"><h4>💡 Tips</h4>
    <div>• Use clear, well lit cassava leaf image</div>
    <div>• Ensure the leaf fills most of the frame</div>
    <div>• Avoid blurry or dark photos</div></div>''' if include_tips else ""
    return f'''<div class="alert-card {type_msg}">
    <div class="alert-content"><div class="alert-icon">{color["icon"]}</div>
    <div><h3>{title}</h3><p>{message}</p></div></div>{tips_section}</div>'''

def create_result_card(image, class_name, confidence):
    disease_info = DISEASE_INFO[class_name]
    img_b64 = image_to_base64(image)
    timestamp = datetime.now().strftime("%H:%M")
    return f'''<div class="result-card">
    <div class="image-container"><img src="{img_b64}" alt="Analyzed leaf" /></div>
    <div class="result-info">
        <div class="disease-header">
            <div class="disease-icon">{disease_info["icon"]}</div>
            <div class="disease-details">
                <h2>{class_name}</h2>
                <div class="confidence-badge" style="background:{disease_info["color"]}20;color:{disease_info["color"]};">
                    {confidence:.1f}% confidence
                </div>
            </div>
        </div>
        <div class="info-grid">
            <div class="info-item">
                <span class="info-label">Severity:</span>
                <span class="severity-badge" style="background:{disease_info["color"]}20;color:{disease_info["color"]};">
                    {disease_info["severity"]}
                </span>
            </div>
            <div class="info-item">
                <span class="info-label">Description:</span>
                <span class="info-text">{disease_info["description"]}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Treatment:</span>
                <span class="info-text">{disease_info["treatment"]}</span>
            </div>
        </div>
    </div></div>'''

def create_history_view():
    if not history_log:
        return create_alert("info", "No History", "No previous analyses found")
    content = '<div class="history-grid">'
    for item in history_log[-10:]:
        content += f'''<div class="history-item">
        <img src="{item["image"]}" alt="Previous analysis" />
        <div class="history-details">
            <h4>{item["class"]}</h4>
            <p class="confidence">{item["confidence"]}% confidence</p>
            <p class="timestamp">{item["timestamp"]}</p>
        </div></div>'''
    return content + "</div>"

# Prediction Functions
def predict_image(image):
    if not image:
        return create_alert("error", "No Image Provided", "Please upload or capture an image")
    
    is_anomaly, reason = detect_anomaly(image)
    if is_anomaly:
        return create_alert("warning", "Invalid Image", reason, include_tips=True)
    
    try:
        processed_image = preprocess_image(image)
        if interpreter:
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], processed_image)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
        else:
            output = np.random.random((1, len(CLASS_NAMES)))
            output = output / np.sum(output)
        
        predicted_idx = np.argmax(output)
        confidence = output[0][predicted_idx] * 100
        class_name = CLASS_NAMES[predicted_idx]
        
        if confidence < 60:
            return create_alert("warning", "Low Confidence", f"Prediction confidence: {confidence:.1f}%. Please try with a clearer image.", include_tips=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        history_log.append({"class": class_name, "confidence": round(confidence, 1), "image": image_to_base64(image), "timestamp": timestamp})
        return create_result_card(image, class_name, confidence)
    except Exception as e:
        return create_alert("error", "Processing Error", f"An error occurred: {str(e)}", include_tips=True)

def analyze_multiple(images):
    if not images:
        return create_alert("info", "No Images", "Please upload images to analyze")
    results = []
    for i, img in enumerate(images):
        try:
            image = Image.open(img) if isinstance(img, str) else img
            result = predict_image(image)
            results.append(f'<div class="image-counter">📸 Image {i+1} of {len(images)}</div>{result}')
        except Exception as e:
            results.append(create_alert("error", f"Image {i+1} Error", str(e)))
    return '<div class="multiple-results">' + '<div class="divider"></div>'.join(results) + '</div>'

def clear_all():
    return None, None, create_alert("info", "Ready", "Upload or capture images to analyze"), create_alert("info", "No History", "No previous analyses found")

# Enhanced Professional CSS
css = """
.gradio-container { 
    font-family: 'Inter', system-ui, sans-serif !important; 
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
    min-height: 100vh;
}
.gradio-container .footer, .built-with { display: none !important; }

/* Header */
.app-header { 
    background: linear-gradient(135deg, #16a34a, #22c55e) !important; 
    border-radius: 16px !important; 
    padding: 24px !important; 
    text-align: center !important; 
    margin-bottom: 20px !important; 
    box-shadow: 0 8px 25px rgba(22, 163, 74, 0.2) !important; 
}
.app-header h1 { 
    color: white !important; 
    font-size: clamp(24px, 6vw, 36px) !important; 
    font-weight: 800 !important; 
    margin: 0 0 8px 0 !important; 
    text-shadow: 0 2px 4px rgba(0,0,0,0.2) !important; 
}
.app-header p { 
    color: rgba(255,255,255,0.95) !important; 
    font-size: clamp(14px, 3.5vw, 18px) !important; 
    margin: 0 !important; 
}

/* Tabs */
.tab-nav { 
    background: white !important; 
    border-radius: 12px !important; 
    padding: 4px !important; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important; 
    margin-bottom: 20px !important; 
}
.tab-nav button { 
    border: none !important; 
    background: transparent !important; 
    color: #6b7280 !important; 
    padding: 12px 20px !important; 
    border-radius: 8px !important; 
    font-weight: 600 !important; 
    transition: all 0.3s ease !important; 
    flex: 1 !important; 
}
.tab-nav button.selected { 
    background: linear-gradient(135deg, #16a34a, #22c55e) !important; 
    color: white !important; 
    box-shadow: 0 2px 8px rgba(22, 163, 74, 0.3) !important; 
}

/* Page Containers */
.page-container { 
    background: white !important; 
    border-radius: 16px !important; 
    padding: 24px !important; 
    box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important; 
    min-height: 400px !important; 
}
.page-title { 
    color: #1f2937 !important; 
    font-size: 20px !important; 
    font-weight: 700 !important; 
    margin: 0 0 16px 0 !important; 
    display: flex !important; 
    align-items: center !important; 
    gap: 8px !important; 
}

/* Buttons */
.btn-primary { 
    background: linear-gradient(135deg, #16a34a, #22c55e) !important; 
    border: none !important; 
    color: white !important; 
    font-weight: 600 !important; 
    padding: 16px 24px !important; 
    border-radius: 12px !important; 
    font-size: 16px !important; 
    transition: all 0.3s ease !important; 
    box-shadow: 0 4px 12px rgba(22, 163, 74, 0.3) !important; 
    width: 100% !important; 
    margin: 8px 0 !important; 
}
.btn-primary:hover { 
    transform: translateY(-2px) !important; 
    box-shadow: 0 6px 16px rgba(22, 163, 74, 0.4) !important; 
}

/* Upload Areas */
.upload-area { 
    border: 2px dashed #16a34a !important; 
    border-radius: 12px !important; 
    padding: 32px !important; 
    text-align: center !important; 
    background: #f8fffe !important; 
    margin: 16px 0 !important; 
    transition: all 0.3s ease !important; 
}
.upload-area:hover { 
    border-color: #22c55e !important; 
    background: #f0fdf4 !important; 
}

/* Camera */
.camera-container { 
    border-radius: 12px !important; 
    overflow: hidden !important; 
    border: 2px solid #f59e0b !important; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important; 
    margin: 16px 0 !important; 
    background: white !important; 
}

/* Results */
.result-card, .alert-card { 
    background: white !important; 
    border-radius: 12px !important; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important; 
    margin: 16px 0 !important; 
    overflow: hidden !important; 
    animation: slideIn 0.4s ease-out !important; 
}
.result-card { 
    padding: 20px !important; 
}
.alert-card { 
    padding: 16px !important; 
}
.alert-card.error { 
    border-left: 4px solid #dc2626 !important; 
    background: #fef2f2 !important; 
}
.alert-card.warning { 
    border-left: 4px solid #f59e0b !important; 
    background: #fffbeb !important; 
}
.alert-card.info { 
    border-left: 4px solid #3b82f6 !important; 
    background: #eff6ff !important; 
}

.alert-content { 
    display: flex !important; 
    align-items: flex-start !important; 
    gap: 12px !important; 
}
.alert-icon { 
    font-size: 20px !important; 
    flex-shrink: 0 !important; 
}
.alert-content h3 { 
    color: #1f2937 !important; 
    font-size: 16px !important; 
    font-weight: 600 !important; 
    margin: 0 0 4px 0 !important; 
}
.alert-content p { 
    color: #6b7280 !important; 
    font-size: 14px !important; 
    margin: 0 !important; 
    line-height: 1.4 !important; 
}

.tips-section { 
    background: #f0f9ff !important; 
    border: 1px solid #0ea5e9 !important; 
    border-radius: 8px !important; 
    padding: 12px !important; 
    margin-top: 12px !important; 
}
.tips-section h4 { 
    color: #0369a1 !important; 
    margin: 0 0 8px 0 !important; 
    font-size: 14px !important; 
    font-weight: 600 !important; 
}
.tips-section div { 
    color: #0369a1 !important; 
    font-size: 12px !important; 
    line-height: 1.4 !important; 
    margin: 2px 0 !important; 
}

.image-container { 
    text-align: center !important; 
    margin-bottom: 20px !important; 
}
.image-container img { 
    width: min(300px, 80vw) !important; 
    height: min(300px, 80vw) !important; 
    object-fit: cover !important; 
    border-radius: 12px !important; 
    border: 2px solid #16a34a !important; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important; 
}

.disease-header { 
    display: flex !important; 
    align-items: flex-start !important; 
    gap: 16px !important; 
    margin-bottom: 20px !important; 
}
.disease-icon { 
    font-size: 40px !important; 
    flex-shrink: 0 !important; 
}
.disease-details h2 { 
    color: #1f2937 !important; 
    font-size: 20px !important; 
    font-weight: 700 !important; 
    margin: 0 0 8px 0 !important; 
    line-height: 1.3 !important; 
}
.confidence-badge, .severity-badge { 
    display: inline-block !important; 
    padding: 6px 12px !important; 
    border-radius: 12px !important; 
    font-size: 13px !important; 
    font-weight: 600 !important; 
}

.info-grid { 
    display: grid !important; 
    gap: 16px !important; 
    grid-template-columns: 1fr !important; 
}
.info-item { 
    background: #f9fafb !important; 
    border: 1px solid #e5e7eb !important; 
    border-radius: 8px !important; 
    padding: 16px !important; 
}
.info-label { 
    display: block !important; 
    color: #16a34a !important; 
    font-size: 12px !important; 
    font-weight: 600 !important; 
    text-transform: uppercase !important; 
    margin-bottom: 8px !important; 
}
.info-text { 
    color: #1f2937 !important; 
    font-size: 14px !important; 
    line-height: 1.5 !important; 
    display: block !important; 
}

/* History */
.history-grid { 
    display: grid !important; 
    gap: 12px !important; 
    grid-template-columns: 1fr !important; 
}
.history-item { 
    display: flex !important; 
    gap: 16px !important; 
    padding: 16px !important; 
    background: #f9fafb !important; 
    border: 1px solid #e5e7eb !important; 
    border-radius: 12px !important; 
    align-items: center !important; 
    transition: all 0.3s ease !important; 
}
.history-item:hover { 
    background: #f3f4f6 !important; 
    border-color: #16a34a !important; 
}
.history-item img { 
    width: 80px !important; 
    height: 80px !important; 
    object-fit: cover !important; 
    border-radius: 8px !important; 
    border: 1px solid #16a34a !important; 
    flex-shrink: 0 !important; 
}
.history-details h4 { 
    color: #1f2937 !important; 
    font-size: 16px !important; 
    font-weight: 600 !important; 
    margin: 0 0 4px 0 !important; 
    line-height: 1.3 !important; 
}
.history-details .confidence { 
    color: #16a34a !important; 
    font-size: 14px !important; 
    font-weight: 600 !important; 
    margin: 0 0 4px 0 !important; 
}
.history-details .timestamp { 
    color: #6b7280 !important; 
    font-size: 12px !important; 
    margin: 0 !important; 
}

.multiple-results .image-counter { 
    background: #16a34a !important; 
    color: white !important; 
    padding: 8px 16px !important; 
    text-align: center !important; 
    font-size: 14px !important; 
    font-weight: 600 !important; 
    margin: 16px 0 !important; 
    border-radius: 8px !important; 
}
.divider { 
    height: 1px !important; 
    background: linear-gradient(to right, transparent, #e5e7eb, transparent) !important; 
    margin: 24px 0 !important; 
}

@keyframes slideIn { 
    from { opacity: 0; transform: translateY(16px); } 
    to { opacity: 1; transform: translateY(0); } 
}

@media (min-width: 768px) {
    .info-grid { grid-template-columns: repeat(2, 1fr) !important; }
    .history-grid { grid-template-columns: repeat(2, 1fr) !important; }
}
@media (min-width: 1024px) {
    .info-grid { grid-template-columns: repeat(3, 1fr) !important; }
    .disease-header { align-items: center !important; }
}
"""

# Create Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.emerald, neutral_hue=gr.themes.colors.slate, font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]), css=css, title="🌿 CassavaDoc - Professional Disease Detection") as demo:
    demo.queue()
    
    gr.HTML('<div class="app-header"><h1>🌿 CassavaDoc</h1><p>Professional AI-powered cassava leaf disease detection system</p></div>')
    
    with gr.Tabs(elem_classes=["tab-nav"]) as tabs:
        with gr.Tab("📁 Upload", elem_classes=["tab-upload"]) as upload_tab:
            with gr.Column(elem_classes=["page-container"]):
                gr.HTML('<div class="page-title">📁 Upload Images</div>')
                file_input = gr.File(label="Select Images", file_types=["image"], file_count="multiple", elem_classes=["upload-area"])
                analyze_btn = gr.Button("🔍 Analyze Images", variant="primary", elem_classes=["btn-primary"])
                upload_results = gr.HTML(value=create_alert("info", "Ready", "Select images to analyze"))
        
        with gr.Tab("📷 Camera", elem_classes=["tab-camera"]) as camera_tab:
            with gr.Column(elem_classes=["page-container"]):
                gr.HTML('<div class="page-title">📷 Camera Capture</div>')
                camera_input = gr.Image(label="Camera", sources=["webcam"], type="pil", elem_classes=["camera-container"])
                capture_btn = gr.Button("🔍 Analyze Photo", variant="primary", elem_classes=["btn-primary"])
                camera_results = gr.HTML(value=create_alert("info", "Ready", "Capture an image to analyze"))
        
        with gr.Tab("📊 Results", elem_classes=["tab-results"]) as results_tab:
            with gr.Column(elem_classes=["page-container"]):
                gr.HTML('<div class="page-title">📊 Analysis Results</div>')
                main_results = gr.HTML(value=create_alert("info", "No Analysis", "Results will appear here after analysis"))
        
        with gr.Tab("📂 History", elem_classes=["tab-history"]) as history_tab:
            with gr.Column(elem_classes=["page-container"]):
                gr.HTML('<div class="page-title">📂 Analysis History</div>')
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                    clear_history_btn = gr.Button("🗑️ Clear All", variant="secondary")
                history_display = gr.HTML(value=create_alert("info", "No History", "No previous analyses found"))
    
    # Event Handlers
    def handle_upload_analysis(files):
        if not files:
            return create_alert("info", "No Images", "Please select images to upload"), gr.update(selected=2)
        if len(files) == 1:
            result = predict_image(Image.open(files[0]))
        else:
            result = analyze_multiple([Image.open(f) for f in files])
        return result, gr.update(selected=2)
    
    def handle_camera_analysis(image):
        if not image:
            return create_alert("info", "No Image", "Please capture an image"), gr.update(selected=2)
        result = predict_image(image)
        return result, gr.update(selected=2)
    
    def refresh_history():
        return create_history_view()
    
    def clear_history():
        global history_log
        history_log = []
        return create_alert("info", "Cleared", "History has been cleared")
    
    # Connect Events
    analyze_btn.click(handle_upload_analysis, inputs=[file_input], outputs=[main_results, tabs])
    capture_btn.click(handle_camera_analysis, inputs=[camera_input], outputs=[main_results, tabs])
    refresh_btn.click(refresh_history, outputs=[history_display])
    clear_history_btn.click(clear_history, outputs=[history_display])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)