import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import tempfile

# Set page config
st.set_page_config(
    page_title="😷 Mask Detector",
    page_icon="😷",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .mask {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .no-mask {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    # Custom deserialization config to handle TF version compatibility
    # This removes the 'groups' parameter that's not recognized in newer TF versions
    class CustomDepthwiseConv2D(layers.DepthwiseConv2D):
        def __init__(self, **kwargs):
            # Remove 'groups' parameter if present (not used in DepthwiseConv2D)
            kwargs.pop('groups', None)
            super().__init__(**kwargs)
    
    custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
    
    try:
        model = keras.models.load_model('keras_model.h5', 
                                       custom_objects=custom_objects,
                                       compile=False)
    except Exception as e:
        # Fallback: try loading with default settings
        st.warning(f"Using fallback loading method: {str(e)[:100]}")
        model = keras.models.load_model('keras_model.h5', compile=False)
    
    return model

# Load labels
@st.cache_data
def load_labels():
    with open('labels.txt', 'r') as f:
        labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    return labels

# Preprocess image for model
def preprocess_image(image):
    # Resize to 224x224 (Teachable Machine default)
    img = cv2.resize(image, (224, 224))
    # Normalize to [-1, 1]
    img = (img.astype(np.float32) / 127.5) - 1
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Main app
def main():
    st.title("😷 Face Mask Detection")
    st.markdown("### Real-time mask detection using your webcam")
    
    # Load model and labels
    try:
        model = load_model()
        labels = load_labels()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence required for prediction"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Stats")
        if 'predictions' not in st.session_state:
            st.session_state.predictions = {'Mask': 0, 'Non Mask': 0}
        
        st.metric("Mask Detected", st.session_state.predictions['Mask'])
        st.metric("No Mask Detected", st.session_state.predictions['Non Mask'])
        
        if st.button("Reset Stats"):
            st.session_state.predictions = {'Mask': 0, 'Non Mask': 0}
            st.rerun()
    
    # Main content
    tab1, tab2 = st.tabs(["📹 Webcam", "📤 Upload Image"])
    
    with tab1:
        st.markdown("**Click 'Start Webcam' to begin detection**")
        
        # Webcam capture
        run = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.empty()
        prediction_display = st.empty()
        
        if run:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not access webcam. Please check your camera permissions.")
            else:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to grab frame")
                        break
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Make prediction
                    preprocessed = preprocess_image(frame_rgb)
                    prediction = model.predict(preprocessed, verbose=0)
                    
                    # Get class and confidence
                    class_idx = np.argmax(prediction[0])
                    confidence = prediction[0][class_idx]
                    label = labels[class_idx]
                    
                    # Draw on frame
                    color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
                    cv2.putText(frame_rgb, f"{label}: {confidence:.2%}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, color, 2)
                    
                    # Display frame
                    FRAME_WINDOW.image(frame_rgb, channels="RGB")
                    
                    # Display prediction
                    if confidence >= confidence_threshold:
                        css_class = "mask" if label == "Mask" else "no-mask"
                        emoji = "✅" if label == "Mask" else "⚠️"
                        prediction_display.markdown(
                            f'<div class="prediction-box {css_class}">'
                            f'{emoji} {label} - {confidence:.1%} confidence'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        st.session_state.predictions[label] += 1
                    else:
                        prediction_display.markdown(
                            '<div class="prediction-box">'
                            '🤔 Low confidence - please adjust position'
                            '</div>',
                            unsafe_allow_html=True
                        )
                
                cap.release()
    
    with tab2:
        st.markdown("**Upload an image to detect mask**")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)
            
            # Make prediction
            preprocessed = preprocess_image(img_array)
            prediction = model.predict(preprocessed, verbose=0)
            
            # Get results
            class_idx = np.argmax(prediction[0])
            confidence = prediction[0][class_idx]
            label = labels[class_idx]
            
            with col2:
                st.markdown("**Prediction**")
                css_class = "mask" if label == "Mask" else "no-mask"
                emoji = "✅" if label == "Mask" else "⚠️"
                
                st.markdown(
                    f'<div class="prediction-box {css_class}">'
                    f'{emoji} {label}<br>{confidence:.1%} confidence'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Show confidence for both classes
                st.markdown("**Confidence Scores:**")
                for i, lbl in enumerate(labels):
                    st.progress(float(prediction[0][i]), text=f"{lbl}: {prediction[0][i]:.1%}")

if __name__ == "__main__":
    main()
