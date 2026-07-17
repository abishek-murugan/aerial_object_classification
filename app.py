import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
if not hasattr(cv2, 'imshow'):
    cv2.imshow = lambda *args, **kwargs: None
from ultralytics import YOLO
import os
import time
from src.config import MODELS_DIR, RESULTS_DIR, CLASSES, IMG_SIZE, logger
from download_models import download_models

# ==========================================
# Streamlit Page Configuration
# ==========================================
st.set_page_config(
    page_title="AeroEye AI - Aerial Object Intelligence",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Run Model Verification
# ==========================================
# Run check at startup
logger.info("Application starting, checking models...")
download_models()
logger.info("Model check complete.")

# ==========================================
# Custom Styling Injection (Aesthetics & WOW Factor)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom Gradient Title */
    .hero-title {
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 50%, #6BCB77 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.2rem;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #8E9AA6;
        font-size: 1.25rem;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }
    
    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(77, 150, 255, 0.3);
        background: rgba(255, 255, 255, 0.05);
        box-shadow: 0 12px 40px 0 rgba(77, 150, 255, 0.1);
    }
    
    /* Result Badge Styles */
    .badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .badge-drone {
        background: linear-gradient(135deg, rgba(77, 150, 255, 0.2) 0%, rgba(77, 150, 255, 0.4) 100%);
        border: 1px solid #4D96FF;
        color: #E6F0FF;
    }
    
    .badge-bird {
        background: linear-gradient(135deg, rgba(107, 203, 119, 0.2) 0%, rgba(107, 203, 119, 0.4) 100%);
        border: 1px solid #6BCB77;
        color: #EBF7ED;
    }
    
    /* Sidebar Styling Override */
    [data-testid="stSidebar"] {
        background-color: #0d0f13 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Monospace Text */
    .mono-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
    }
    
    /* Success, info boxes styling override */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Header
# ==========================================
st.markdown('<h1 class="hero-title">AeroEye AI 🦅</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Intelligence System for Aerial Classification and Object Detection</p>', unsafe_allow_html=True)

# ==========================================
# Models Loading Helpers
# ==========================================
@st.cache_resource
def load_classification_model(model_type):
    """
    Loads Keras classification model.
    """
    model_path = os.path.join(MODELS_DIR, f'{model_type}_model.keras')
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading {model_type} model: {e}")
            return None
    return None

@st.cache_resource
def load_yolo_model():
    """
    Loads trained YOLOv8 model, or falls back to pretrained yolov8n.
    """
    trained_yolo = os.path.join(RESULTS_DIR, 'yolov8_results', 'weights', 'best.pt')
    if os.path.exists(trained_yolo):
        try:
            return YOLO(trained_yolo), True
        except Exception as e:
            st.warning(f"Error loading trained YOLO model: {e}. Falling back to default YOLOv8.")
            return YOLO('yolov8n.pt'), False
    return YOLO('yolov8n.pt'), False

def preprocess_image(image):
    """
    Preprocess image for Keras model input.
    """
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    # Ensure 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:,:,:3]
        
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==========================================
# Tabs Organization
# ==========================================
tab_inference, tab_diagnostics, tab_insights = st.tabs([
    "🎯 AI Inference Center", 
    "📊 Model Performance & Metrics", 
    "🔬 System Diagnostics & Design"
])

# ==========================================
# TAB 1: AI Inference Center
# ==========================================
with tab_inference:
    st.markdown('<div class="glass-card"><h4>Configuration & Analysis Pipeline</h4>Choose the task mode and model architecture below, then upload an aerial image.</div>', unsafe_allow_html=True)
    
    col_cfg, col_upload = st.columns([1, 2])
    
    with col_cfg:
        st.subheader("Pipeline Settings")
        task_mode = st.radio("Task Mode", ["Classification (Binary)", "Object Detection (YOLOv8)"])
        
        if task_mode == "Classification (Binary)":
            model_choice = st.selectbox(
                "Classification Model Architecture", 
                ["Custom CNN Model", "Transfer Learning (MobileNetV2)"]
            )
            model_type = 'custom' if model_choice == "Custom CNN Model" else 'transfer'
            
            # Model details display
            if model_type == 'custom':
                st.info("💡 **Custom CNN**: 3 Convolutional blocks with batch normalization, max pooling, and dropout (0.5) optimized for quick classification.")
            else:
                st.info("💡 **MobileNetV2**: Leverage pre-trained features on ImageNet dataset, frozen base layers with custom dense head.")
        else:
            st.info("⚡ **YOLOv8 Nano**: Real-time object detection model designed to identify and localize both birds and drones in a single forward pass.")
            
    with col_upload:
        st.subheader("Image Input Source")
        uploaded_file = st.file_uploader("Upload Aerial Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Source Aerial Frame", use_container_width=True)
            
            run_btn = st.button("🚀 Execute AI Intelligence Pipeline", use_container_width=True)
            
            if run_btn:
                if task_mode == "Classification (Binary)":
                    with st.spinner(f"Running inference with {model_choice}..."):
                        model = load_classification_model(model_type)
                        
                        if model is not None:
                            # Preprocess and predict
                            processed_img = preprocess_image(image)
                            start_time = time.time()
                            prediction = model.predict(processed_img)[0][0]
                            latency = (time.time() - start_time) * 1000
                            
                            # Classes and metrics
                            label = "Drone" if prediction > 0.5 else "Bird"
                            confidence = prediction if prediction > 0.5 else 1.0 - prediction
                            
                            st.markdown("### 🔍 Pipeline Results")
                            
                            res_col1, res_col2, res_col3 = st.columns(3)
                            with res_col1:
                                badge_html = f'<div class="badge badge-drone">🤖 DRONE DETECTED</div>' if label == "Drone" else f'<div class="badge badge-bird">🦅 BIRD DETECTED</div>'
                                st.markdown("##### Prediction Category")
                                st.markdown(badge_html, unsafe_allow_html=True)
                                
                            with res_col2:
                                st.metric("Confidence Score", f"{confidence:.2%}")
                                
                            with res_col3:
                                st.metric("Inference Latency", f"{latency:.2f} ms")
                                
                            # Confidence Gauge
                            st.progress(float(confidence))
                            
                        else:
                            st.error(f"⚠️ Model file for '{model_choice}' is not found. Please run the training pipeline in Tab 2 or CLI first.")
                
                else:
                    with st.spinner("Executing YOLOv8 object detection..."):
                        model, is_trained = load_yolo_model()
                        
                        if is_trained:
                            st.sidebar.success("Loaded trained Custom YOLOv8 model.")
                        else:
                            st.sidebar.warning("Using pre-trained YOLOv8 model (trained on COCO).")
                            
                        # Run inference
                        start_time = time.time()
                        results = model(image)
                        latency = (time.time() - start_time) * 1000
                        
                        res_plotted = results[0].plot()
                        
                        # Save result image in a separate folder
                        os.makedirs("results", exist_ok=True)
                        result_path = f"results/detection_{int(time.time())}.jpg"
                        Image.fromarray(res_plotted[:, :, ::-1]).save(result_path)
                        
                        num_boxes = len(results[0].boxes)
                        
                        st.markdown("### 🔍 Detection Output Map")
                        st.image(res_plotted, caption=f"YOLOv8 Output (Saved to {result_path})", use_container_width=True)
                        
                        # Show latency and detections
                        det_col1, det_col2 = st.columns(2)
                        with det_col1:
                            st.metric("TF YOLOv8 Latency", f"{latency:.2f} ms")
                        with det_col2:
                            st.metric("Objects Located", f"{num_boxes}")

# ==========================================
# TAB 2: Model Performance & Metrics
# ==========================================
with tab_diagnostics:
    st.markdown('<div class="glass-card"><h4>Training Diagnostics & Performance Monitoring</h4>Visualize metrics, training loss/accuracy curves, and confusion matrices generated during training.</div>', unsafe_allow_html=True)
    
    st.subheader("Classification Diagnostics")
    
    diag_model = st.selectbox("Select Model for Diagnostics", ["Custom CNN Model", "Transfer Learning (MobileNetV2)"])
    model_key = 'custom' if diag_model == "Custom CNN Model" else 'transfer'
    
    col_curves, col_cm = st.columns(2)
    
    with col_curves:
        st.markdown("##### Training History Curves (Accuracy & Loss)")
        history_path = os.path.join(RESULTS_DIR, f'{model_key}_history.png')
        if os.path.exists(history_path):
            st.image(history_path, caption=f"{diag_model} - History", use_container_width=True)
        else:
            st.warning(f"No history plot found for {diag_model} at {history_path}. Please run train_classification first.")
            
    with col_cm:
        st.markdown("##### Test Set Confusion Matrix")
        cm_path = os.path.join(RESULTS_DIR, f'{model_key}_confusion_matrix.png')
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"{diag_model} - Confusion Matrix", use_container_width=True)
        else:
            st.warning(f"No confusion matrix plot found for {diag_model} at {cm_path}. Please run evaluate_classification first.")

    st.markdown("---")
    st.subheader("Object Detection Diagnostics (YOLOv8)")
    
    yolo_results_dir = os.path.join(RESULTS_DIR, 'yolov8_results')
    if os.path.exists(yolo_results_dir):
        yolo_cols = st.columns(2)
        
        # Check for results plot
        yolo_res_plot = os.path.join(yolo_results_dir, 'results.png')
        with yolo_cols[0]:
            st.markdown("##### YOLOv8 Training Trends")
            if os.path.exists(yolo_res_plot):
                st.image(yolo_res_plot, caption="YOLOv8 Loss & Metric Trends", use_container_width=True)
            else:
                st.info("YOLOv8 training results.png not found. Make sure training completed.")
                
        # Check for confusion matrix
        yolo_cm_plot = os.path.join(yolo_results_dir, 'confusion_matrix.png')
        with yolo_cols[1]:
            st.markdown("##### YOLOv8 Confusion Matrix")
            if os.path.exists(yolo_cm_plot):
                st.image(yolo_cm_plot, caption="YOLOv8 Confusion Matrix", use_container_width=True)
            else:
                st.info("YOLOv8 confusion_matrix.png not found.")
    else:
        st.info("YOLOv8 training directory not created. Complete the YOLOv8 training step first.")

# ==========================================
# TAB 3: System Diagnostics & Design
# ==========================================
with tab_insights:
    st.subheader("Model Comparison Matrix")
    st.markdown("""
    Below is a comparison of the three different architectures evaluated for our aerial object classification and detection task:
    """)
    
    # Custom styled comparison table
    st.markdown("""
    <div style="overflow-x:auto;">
        <table style="width:100%; border-collapse: collapse; margin: 10px 0; font-size: 0.95em; min-width: 400px; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
            <thead>
                <tr style="background-color: #1f232a; color: #ffffff; text-align: left; font-weight: bold; border-bottom: 2px solid #323946;">
                    <th style="padding: 12px 15px;">Model Architecture</th>
                    <th style="padding: 12px 15px;">Classifier Type</th>
                    <th style="padding: 12px 15px;">Ideal Use Case</th>
                    <th style="padding: 12px 15px;">Pros</th>
                    <th style="padding: 12px 15px;">Cons</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); background-color: rgba(255,255,255,0.01);">
                    <td style="padding: 12px 15px; font-weight: 600;">Custom CNN</td>
                    <td style="padding: 12px 15px; color: #6BCB77;">Binary Classifier</td>
                    <td style="padding: 12px 15px;">Embedded Systems / CPU Only</td>
                    <td style="padding: 12px 15px;">Ultra lightweight, fast inference, small storage footprint.</td>
                    <td style="padding: 12px 15px;">Lower accuracy compared to deep pre-trained networks.</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); background-color: rgba(255,255,255,0.03);">
                    <td style="padding: 12px 15px; font-weight: 600;">MobileNetV2 (Transfer)</td>
                    <td style="padding: 12px 15px; color: #6BCB77;">Binary Classifier</td>
                    <td style="padding: 12px 15px;">Mobile / Edge Deployments</td>
                    <td style="padding: 12px 15px;">High accuracy, leverages ImageNet features, fast transfer training.</td>
                    <td style="padding: 12px 15px;">Higher memory footprint than custom CNN.</td>
                </tr>
                <tr style="border-bottom: 2px solid #323946; background-color: rgba(255,255,255,0.01);">
                    <td style="padding: 12px 15px; font-weight: 600;">YOLOv8 Nano</td>
                    <td style="padding: 12px 15px; color: #4D96FF;">Object Detector</td>
                    <td style="padding: 12px 15px;">Real-Time Localization / Multi-target</td>
                    <td style="padding: 12px 15px;">Provides bounding box locations, handles multi-class, real-time speed.</td>
                    <td style="padding: 12px 15px;">Computationally heavier, requires annotation data.</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("System Architecture & Pipeline Flow")
    
    # Mermaid diagrams can visualize the pipeline
    st.markdown("""
    The flow diagram below displays the end-to-end processing pipeline for AeroEye AI:
    """)
    st.markdown("""
    ```text
    [Input Image/Frame] 
       ├──► Binary Classification (Custom CNN or MobileNetV2)
       └──► Object Detection (TF YOLOv8)
    ```
    """)
