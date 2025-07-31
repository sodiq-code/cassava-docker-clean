"""
Optimized Cassava Disease Detector - Enhanced Features
======================================================
- Clickable history items that show analysis results
- Tips section on homepage
- Improved navigation and UI
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
import uuid

# Optional imports with fallbacks
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

def create_alert_mobile(type_msg, title, message, include_tips=False):
    colors = {"error": {"bg": "#fee2e2", "border": "#dc2626", "text": "#991b1b", "icon": "❌"}, 
              "warning": {"bg": "#fef3c7", "border": "#f59e0b", "text": "#92400e", "icon": "⚠️"}, 
              "info": {"bg": "#dbeafe", "border": "#3b82f6", "text": "#1e40af", "icon": "ℹ️"}}
    color = colors.get(type_msg, colors["info"])
    tips_section = '<div style="background:#f0f9ff;border:1px solid #0ea5e9;border-radius:8px;padding:12px;margin:8px 0;"><h4 style="color:#0369a1;margin:0 0 8px 0;font-size:14px;font-weight:600;">💡 Tips</h4><div style="display:flex;flex-direction:column;gap:4px;"><div style="color:#0369a1;font-size:12px;line-height:1.4;">• Use clear, well lit cassava leaf image</div><div style="color:#0369a1;font-size:12px;line-height:1.4;">• Ensure the leaf fills most of the frame</div><div style="color:#0369a1;font-size:12px;line-height:1.4;">• Avoid blurry or dark photos</div></div></div>' if include_tips else ""
    return f'<div class="mobile-results"><div class="alert-card" style="background:{color["bg"]};border:1px solid {color["border"]};border-radius:12px;padding:16px;margin:8px 0;"><div style="display:flex;align-items:flex-start;gap:12px;"><div style="font-size:20px;flex-shrink:0;">{color["icon"]}</div><div style="flex:1;min-width:0;"><h3 style="color:{color["text"]};margin:0 0 4px 0;font-size:16px;font-weight:600;">{title}</h3><p style="color:{color["text"]};margin:0;font-size:14px;opacity:0.9;line-height:1.4;">{message}</p></div></div></div>{tips_section}</div>'

def create_result_card_mobile(image, class_name, confidence):
    disease_info = DISEASE_INFO[class_name]
    img_b64 = image_to_base64(image)
    return f'''
    <div class="mobile-results">
        <div class="result-card">
            <div class="image-container">
                <img src="{img_b64}" alt="Analyzed leaf" />
            </div>
            <div class="result-info">
                <div class="disease-header">
                    <div class="disease-icon">{disease_info["icon"]}</div>
                    <div class="disease-details">
                        <h2>{class_name}</h2>
                        <div class="confidence-badge" style="background:{disease_info["color"]}20;color:{disease_info["color"]};">{confidence:.1f}% confidence</div>
                    </div>
                </div>
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">Severity:</span>
                        <span class="severity-badge" style="background:{disease_info["color"]}20;color:{disease_info["color"]};">{disease_info["severity"]}</span>
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
            </div>
        </div>
    </div>
    '''

def create_multiple_results_mobile(results_list):
    header = '<div class="mobile-results"><div class="results-header"><h3>📊 Analysis Results</h3><p>Multiple images analyzed</p></div>'
    content = ""
    for i, result in enumerate(results_list):
        if len(results_list) > 1:
            content += f'<div class="image-counter">📸 Image {i+1} of {len(results_list)}</div>'
        if "result-card" in result:
            start_pos = result.find('<div class="result-card">')
            end_pos = result.rfind('</div>') + 6
            if start_pos != -1 and end_pos != -1:
                content += result[start_pos:end_pos]
        else:
            content += result
        if i < len(results_list) - 1:
            content += '<div class="divider"></div>'
    return header + content + "</div>"

def create_history_mobile():
    if not history_log:
        return create_alert_mobile("info", "No History", "No previous analyses found")
    
    header = '<div class="mobile-results"><div class="results-header"><h3>📂 Analysis History</h3><p>Your recent diagnoses</p></div>'
    history_content = ""
    for i, item in enumerate(history_log[-10:]):
        history_content += f'''
        <div class="history-item" onclick="showHistoryItem('{item['id']}')">
            <img src="{item["image"]}" alt="Previous analysis" />
            <div class="history-details">
                <h4>{item["class"]}</h4>
                <p class="confidence">{item["confidence"]}% confidence</p>
                <p class="timestamp">{item["timestamp"]}</p>
            </div>
        </div>
        '''
    return header + history_content + "</div>"

def create_default_mobile():
    return '''
    <div class="mobile-results">
        <div class="results-header">
            <h3>📊 Analysis Results</h3>
            <p>Your diagnosis will appear here</p>
        </div>
        <div class="placeholder-content">
            <div class="placeholder-icon">🌿</div>
            <p>Upload or capture a cassava leaf image to get started</p>
        </div>
    </div>
    '''

def predict_image(image):
    if not image:
        return create_alert_mobile("error", "No Image Provided", "Please upload or capture an image")
    
    is_anomaly, reason = detect_anomaly(image)
    if is_anomaly:
        return create_alert_mobile("warning", "Invalid Image", reason, include_tips=True)
    
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
            return create_alert_mobile("warning", "Low Confidence", f"Prediction confidence: {confidence:.1f}%. Please try with a clearer image.", include_tips=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        result_html = create_result_card_mobile(image, class_name, confidence)
        
        # Store with unique ID for history reference
        history_log.append({
            "id": str(uuid.uuid4()),
            "class": class_name,
            "confidence": round(confidence, 1),
            "image": image_to_base64(image),
            "timestamp": timestamp,
            "result_html": result_html
        })
        return result_html
    except Exception as e:
        return create_alert_mobile("error", "Processing Error", f"An error occurred: {str(e)}", include_tips=True)

def analyze_uploaded_images(uploaded_files):
    if not uploaded_files:
        return create_alert_mobile("info", "No Images", "Please select images to upload")
    
    if len(uploaded_files) == 1:
        return predict_image(Image.open(uploaded_files[0]))
    else:
        results_list = []
        for img in uploaded_files:
            results_list.append(predict_image(Image.open(img)))
        return create_multiple_results_mobile(results_list)

def analyze_camera_image(webcam_image):
    if webcam_image is None:
        return create_alert_mobile("info", "No Image", "Please capture an image from camera")
    return predict_image(webcam_image)

def find_history_item(history_id):
    for item in history_log:
        if item["id"] == history_id:
            return item["result_html"]
    return create_alert_mobile("error", "Not Found", "History item not found")

# Enhanced Mobile-Optimized CSS
css = """
.gradio-container .footer, .gradio-container .built-with, footer, .gr-button-tool, .built-with-gradio, .gradio-container > .built-with, .share-button, .duplicate-button { display: none !important; }
@media all and (display-mode: standalone) { body { padding-top: env(safe-area-inset-top) !important; padding-bottom: env(safe-area-inset-bottom) !important; } }
body { height: 100vh; overflow: hidden; }
.gradio-container { height: 100vh; overflow-y: auto; -webkit-overflow-scrolling: touch; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important; min-height: 100vh; padding: 8px !important; }
.app-header { background: linear-gradient(135deg, #16a34a, #22c55e) !important; border-radius: 16px !important; padding: 20px 16px !important; text-align: center !important; margin-bottom: 16px !important; box-shadow: 0 8px 25px rgba(22, 163, 74, 0.2) !important; }
.app-header h1 { color: white !important; font-size: clamp(20px, 6vw, 32px) !important; font-weight: 800 !important; margin: 0 !important; text-shadow: 0 2px 4px rgba(0,0,0,0.2) !important; line-height: 1.2 !important; }
.app-header p { color: rgba(255,255,255,0.95) !important; font-size: clamp(12px, 3.5vw, 16px) !important; margin: 8px 0 0 0 !important; }
.home-buttons { display: flex; flex-direction: column; gap: 12px; padding: 0 16px; }
.tips-section { background: white; border-radius: 16px; padding: 16px; margin-top: 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.tips-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.tips-header h3 { margin: 0; font-size: 18px; color: #1f2937; }
.tips-content { display: flex; flex-direction: column; gap: 10px; }
.tip-item { display: flex; gap: 12px; }
.tip-icon { font-size: 20px; flex-shrink: 0; }
.tip-text { font-size: 14px; color: #4b5563; }
.btn-upload, .btn-camera, .btn-history { border: none !important; color: white !important; font-weight: 700 !important; padding: 18px 24px !important; border-radius: 16px !important; font-size: 16px !important; transition: all 0.3s ease !important; width: 100% !important; margin-bottom: 0 !important; display: flex !important; align-items: center !important; justify-content: center !important; gap: 10px !important; }
.btn-upload { background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important; box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important; }
.btn-upload:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5) !important; background: linear-gradient(135deg, #2563eb, #1e40af) !important; }
.btn-camera { background: linear-gradient(135deg, #f59e0b, #d97706) !important; box-shadow: 0 6px 20px rgba(245, 158, 11, 0.4) !important; }
.btn-camera:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(245, 158, 11, 0.5) !important; background: linear-gradient(135deg, #d97706, #b45309) !important; }
.btn-history { background: linear-gradient(135deg, #8b5cf6, #7c3aed) !important; box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important; }
.btn-history:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5) !important; background: linear-gradient(135deg, #7c3aed, #6d28d9) !important; }
.mobile-results { background: white !important; border-radius: 16px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important; overflow: hidden !important; margin: 8px 0 !important; }
.results-header { background: linear-gradient(135deg, #f8fafc, #e2e8f0) !important; padding: 16px !important; text-align: center !important; border-bottom: 1px solid #e5e7eb !important; }
.results-header h3 { color: #1f2937 !important; font-size: 18px !important; font-weight: 700 !important; margin: 0 0 4px 0 !important; }
.results-header p { color: #6b7280 !important; font-size: 14px !important; margin: 0 !important; }
.result-card { padding: 16px !important; animation: slideIn 0.4s ease-out !important; }
.image-container { text-align: center !important; margin-bottom: 16px !important; }
.image-container img { width: min(250px, 80vw) !important; height: min(250px, 80vw) !important; object-fit: cover !important; border-radius: 12px !important; border: 2px solid #16a34a !important; box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important; }
.disease-header { display: flex !important; align-items: flex-start !important; gap: 12px !important; margin-bottom: 16px !important; }
.disease-icon { font-size: 32px !important; flex-shrink: 0 !important; }
.disease-details { flex: 1 !important; min-width: 0 !important; }
.disease-details h2 { color: #1f2937 !important; font-size: 18px !important; font-weight: 700 !important; margin: 0 0 8px 0 !important; line-height: 1.3 !important; }
.confidence-badge, .severity-badge { display: inline-block !important; padding: 4px 10px !important; border-radius: 12px !important; font-size: 13px !important; font-weight: 600 !important; }
.info-grid { display: flex !important; flex-direction: column !important; gap: 12px !important; }
.info-item { background: #f9fafb !important; border: 1px solid #e5e7eb !important; border-radius: 8px !important; padding: 12px !important; }
.info-label { display: block !important; color: #16a34a !important; font-size: 12px !important; font-weight: 600 !important; text-transform: uppercase !important; margin-bottom: 6px !important; }
.info-text { color: #1f2937 !important; font-size: 14px !important; line-height: 1.4 !important; display: block !important; }
.history-item { display: flex !important; gap: 12px !important; padding: 12px 16px !important; border-bottom: 1px solid #e5e7eb !important; align-items: center !important; cursor: pointer; transition: all 0.2s; }
.history-item:hover { background-color: #f9fafb; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
.history-item:last-child { border-bottom: none !important; }
.history-item img { width: 60px !important; height: 60px !important; object-fit: cover !important; border-radius: 8px !important; border: 1px solid #16a34a !important; flex-shrink: 0 !important; }
.history-details { flex: 1 !important; min-width: 0 !important; }
.history-details h4 { color: #1f2937 !important; font-size: 14px !important; font-weight: 600 !important; margin: 0 0 4px 0 !important; line-height: 1.3 !important; }
.history-details .confidence { color: #16a34a !important; font-size: 12px !important; font-weight: 600 !important; margin: 0 0 2px 0 !important; }
.history-details .timestamp { color: #6b7280 !important; font-size: 11px !important; margin: 0 !important; }
.alert-card { margin: 16px !important; animation: slideIn 0.3s ease-out !important; }
.placeholder-content { padding: 40px 20px !important; text-align: center !important; color: #6b7280 !important; }
.placeholder-icon { font-size: 48px !important; margin-bottom: 12px !important; }
.placeholder-content p { font-size: 14px !important; margin: 0 !important; line-height: 1.4 !important; }
.image-counter { background: #16a34a !important; color: white !important; padding: 8px 16px !important; text-align: center !important; font-size: 14px !important; font-weight: 600 !important; margin: 8px 16px !important; border-radius: 8px !important; }
.divider { height: 1px !important; background: linear-gradient(to right, transparent, #e5e7eb, transparent) !important; margin: 16px !important; }
.btn-primary { background: linear-gradient(135deg, #16a34a, #22c55e) !important; border: none !important; color: white !important; font-weight: 600 !important; padding: 14px 20px !important; border-radius: 12px !important; font-size: 15px !important; transition: all 0.3s ease !important; box-shadow: 0 4px 12px rgba(22, 163, 74, 0.3) !important; width: 100% !important; margin-bottom: 8px !important; }
.btn-primary:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 16px rgba(22, 163, 74, 0.4) !important; background: linear-gradient(135deg, #15803d, #16a34a) !important; }
.btn-secondary { background: white !important; border: 1px solid #e5e7eb !important; color: #374151 !important; font-weight: 600 !important; padding: 10px 16px !important; border-radius: 10px !important; font-size: 14px !important; transition: all 0.3s ease !important; margin: 4px !important; }
.btn-secondary:hover { background: #f9fafb !important; border-color: #16a34a !important; color: #16a34a !important; }
.gr-file { background: white !important; border: 2px dashed #16a34a !important; border-radius: 12px !important; margin-bottom: 12px !important; }
.gr-image { background: white !important; border: 1px solid #e5e7eb !important; border-radius: 12px !important; margin-bottom: 12px !important; }
.native-camera { border-radius: 12px !important; overflow: hidden !important; border: 2px solid #f59e0b !important; box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important; margin-bottom: 12px !important; background: white !important; }
.native-camera img { width: 100% !important; height: auto !important; max-height: 400px !important; object-fit: cover !important; }
@keyframes slideIn { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
@media (min-width: 768px) { .gradio-container { padding: 16px !important; } .image-container img { width: 300px !important; height: 300px !important; } .info-grid { display: grid !important; grid-template-columns: 1fr 1fr !important; gap: 16px !important; } .btn-primary, .btn-upload, .btn-camera, .btn-history { width: auto !important; min-width: 200px !important; } .tips-content { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; } }
@media (min-width: 1024px) { .disease-header { align-items: center !important; } .disease-details h2 { font-size: 20px !important; } .info-grid { grid-template-columns: repeat(3, 1fr) !important; } }
@supports (-webkit-touch-callout: none) { .mobile-results { -webkit-transform: translateZ(0) !important; } }
"""

# Create Interface
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
    ),
    css=css,
    title="🌿 Cassava Disease Detector"
) as demo:
    demo.queue()
    
    # Add JavaScript for history item clicks
    gr.HTML('''
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="theme-color" content="#16a34a">
        <script>
            function showHistoryItem(itemId) {
                document.getElementById("selected_history_id").value = itemId;
                document.getElementById("select_history_btn").click();
            }
        </script>
    </head>
    <div class="app-header">
        <h1>🌿 CassavaDoc</h1>
        <p>AI-powered cassava leaf disease detection</p>
    </div>
    ''')
    
    # Hidden components for history selection
    selected_history_id = gr.Textbox(visible=False, elem_id="selected_history_id")
    select_history_btn = gr.Button(visible=False, elem_id="select_history_btn")
    
    # ===== PAGE COMPONENTS =====
    # Home Page
    with gr.Column(visible=True, elem_id="home_page") as home_page:
        gr.HTML('<h2 style="text-align:center;margin-bottom:24px;">Select Analysis Method</h2>')
        with gr.Column(elem_classes="home-buttons"):
            upload_btn = gr.Button("📁 Upload Image", elem_classes=["btn-upload"])
            camera_btn = gr.Button("📷 Camera Capture", elem_classes=["btn-camera"])
            history_btn = gr.Button("📂 View History", elem_classes=["btn-history"])
        
        # Tips Section
        with gr.Column(elem_classes="tips-section"):
            with gr.Row(elem_classes="tips-header"):
                gr.HTML('<div style="font-size:24px;">💡</div>')
                gr.HTML('<h3>Tips for Best Results</h3>')
            with gr.Column(elem_classes="tips-content"):
                with gr.Row():
                    gr.HTML('<div class="tip-item"><div class="tip-icon">🌿</div><div class="tip-text">Capture clear, well-lit images of cassava leaves</div></div>')
                    gr.HTML('<div class="tip-item"><div class="tip-icon">📱</div><div class="tip-text">Hold your phone steady when taking photos</div></div>')
                with gr.Row():
                    gr.HTML('<div class="tip-item"><div class="tip-icon">🔍</div><div class="tip-text">Make sure the leaf fills most of the frame</div></div>')
                    gr.HTML('<div class="tip-item"><div class="tip-icon">☀️</div><div class="tip-text">Use natural daylight for best accuracy</div></div>')
    
    # Upload Page
    with gr.Column(visible=False, elem_id="upload_page") as upload_page:
        back_btn_upload = gr.Button("⬅️ Back to Home", elem_classes=["btn-secondary"])
        file_input = gr.File(
            label="Upload Cassava Leaf Images",
            file_types=["image"],
            file_count="multiple",
            elem_classes=["mobile-file-input"]
        )
        analyze_upload_btn = gr.Button("🔍 Analyze Images", variant="primary", elem_classes=["btn-primary"])
    
    # Camera Page
    with gr.Column(visible=False, elem_id="camera_page") as camera_page:
        back_btn_camera = gr.Button("⬅️ Back to Home", elem_classes=["btn-secondary"])
        camera_input = gr.Image(
            label="Camera Capture",
            sources=["webcam"],
            type="pil",
            streaming=False,
            mirror_webcam=False,
            elem_classes=["native-camera"]
        )
        analyze_camera_btn = gr.Button("🔍 Analyze Image", variant="primary", elem_classes=["btn-primary"])
    
    # History Page
    with gr.Column(visible=False, elem_id="history_page") as history_page:
        back_btn_history = gr.Button("⬅️ Back to Home", elem_classes=["btn-secondary"])
        history_display = gr.HTML(create_history_mobile())
    
    # Results Page
    with gr.Column(visible=False, elem_id="results_page") as results_page:
        back_btn_results = gr.Button("⬅️ Back to Home", elem_classes=["btn-secondary"])
        results_display = gr.HTML(create_default_mobile())
    
    # ===== PAGE NAVIGATION =====
    # Home -> Upload
    upload_btn.click(
        lambda: [
            gr.Column(visible=False),  # Hide home
            gr.Column(visible=True),   # Show upload
            gr.Column(visible=False),  # Hide camera
            gr.Column(visible=False),  # Hide history
            gr.Column(visible=False)   # Hide results
        ],
        inputs=None,
        outputs=[home_page, upload_page, camera_page, history_page, results_page]
    )
    
    # Home -> Camera
    camera_btn.click(
        lambda: [
            gr.Column(visible=False),  # Hide home
            gr.Column(visible=False),  # Hide upload
            gr.Column(visible=True),   # Show camera
            gr.Column(visible=False),  # Hide history
            gr.Column(visible=False)   # Hide results
        ],
        inputs=None,
        outputs=[home_page, upload_page, camera_page, history_page, results_page]
    )
    
    # Home -> History
    history_btn.click(
        lambda: [
            gr.Column(visible=False),  # Hide home
            gr.Column(visible=False),  # Hide upload
            gr.Column(visible=False),  # Hide camera
            gr.Column(visible=True),   # Show history
            gr.Column(visible=False)   # Hide results
        ],
        inputs=None,
        outputs=[home_page, upload_page, camera_page, history_page, results_page]
    ).then(
        create_history_mobile, 
        inputs=None, 
        outputs=history_display
    )
    
    # Upload -> Results
    analyze_upload_btn.click(
        lambda: [
            gr.Column(visible=False),  # Hide home
            gr.Column(visible=False),  # Hide upload
            gr.Column(visible=False),  # Hide camera
            gr.Column(visible=False),  # Hide history
            gr.Column(visible=True)    # Show results
        ],
        inputs=None,
        outputs=[home_page, upload_page, camera_page, history_page, results_page]
    ).then(
        analyze_uploaded_images, 
        inputs=[file_input], 
        outputs=results_display
    )
    
    # Camera -> Results
    analyze_camera_btn.click(
        lambda: [
            gr.Column(visible=False),  # Hide home
            gr.Column(visible=False),  # Hide upload
            gr.Column(visible=False),  # Hide camera
            gr.Column(visible=False),  # Hide history
            gr.Column(visible=True)    # Show results
        ],
        inputs=None,
        outputs=[home_page, upload_page, camera_page, history_page, results_page]
    ).then(
        analyze_camera_image, 
        inputs=[camera_input], 
        outputs=results_display
    )
    
    # Back to Home from Upload
    back_btn_upload.click(
        lambda: [
            gr.Column(visible=True),   # Show home
            gr.Column(visible=False),  # Hide upload
            gr.Column(visible=False),  # Hide camera
            gr.Column(visible=False),  # Hide history
            gr.Column(visible=False)   # Hide results
        ],
        inputs=None,
        outputs=[home_page, upload_page, camera_page, history_page, results_page]
    )
    
    # Back to Home from Camera
    back_btn_camera.click(
        lambda: [
            gr.Column(visible=True),   # Show home
            gr.Column(visible=False),  # Hide upload
            gr.Column(visible=False),  # Hide camera
            gr.Column(visible=False),  # Hide history
            gr.Column(visible=False)   # Hide results
        ],
        inputs=None,
        outputs=[home_page, upload_page, camera_page, history_page, results_page]
    )
    
    # Back to Home from History
    back_btn_history.click(
        lambda: [
            gr.Column(visible=True),   # Show home
            gr.Column(visible=False),  # Hide upload
            gr.Column(visible=False),  # Hide camera
            gr.Column(visible=False),  # Hide history
            gr.Column(visible=False)   # Hide results
        ],
        inputs=None,
        outputs=[home_page, upload_page, camera_page, history_page, results_page]
    )
    
    # Back to Home from Results
    back_btn_results.click(
        lambda: [
            gr.Column(visible=True),   # Show home
            gr.Column(visible=False),  # Hide upload
            gr.Column(visible=False),  # Hide camera
            gr.Column(visible=False),  # Hide history
            gr.Column(visible=False)   # Hide results
        ],
        inputs=None,
        outputs=[home_page, upload_page, camera_page, history_page, results_page]
    )
    
    # History Item Selection
    select_history_btn.click(
        lambda history_id: [
            find_history_item(history_id),
            gr.Column(visible=False),  # Hide home
            gr.Column(visible=False),  # Hide upload
            gr.Column(visible=False),  # Hide camera
            gr.Column(visible=False),  # Hide history
            gr.Column(visible=True)    # Show results
        ],
        inputs=[selected_history_id],
        outputs=[results_display, home_page, upload_page, camera_page, history_page, results_page]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)