import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="😷 Mask Detector",
    page_icon="😷",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
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

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = keras.models.load_model("keras_model.h5", compile=False)
    return model

# ------------------ LOAD LABELS ------------------
@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]
    return labels

# ------------------ PREPROCESS IMAGE ------------------
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = (img.astype(np.float32) / 127.5) - 1
    img = np.expand_dims(img, axis=0)
    return img

# ------------------ MAIN APP ------------------
def main():
    st.title("😷 Face Mask Detection")
    st.markdown("Detect whether a person is wearing a mask.")

    try:
        model = load_model()
        labels = load_labels()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["📸 Camera", "📤 Upload Image"])

    # ================= CAMERA TAB =================
    with tab1:
        st.markdown("### Take a photo")

        camera_image = st.camera_input("Capture Image")

        if camera_image is not None:
            image = Image.open(camera_image)
            img_array = np.array(image)

            st.image(image, use_column_width=True)

            preprocessed = preprocess_image(img_array)
            prediction = model.predict(preprocessed, verbose=0)

            class_idx = np.argmax(prediction[0])
            confidence = prediction[0][class_idx]
            label = labels[class_idx]

            css_class = "mask" if "mask" in label.lower() else "no-mask"
            emoji = "✅" if css_class == "mask" else "⚠️"

            st.markdown(
                f'<div class="prediction-box {css_class}">'
                f'{emoji} {label}<br>{confidence:.1%} confidence'
                f'</div>',
                unsafe_allow_html=True
            )

    # ================= UPLOAD TAB =================
    with tab2:
        st.markdown("### Upload an image")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, use_column_width=True)

            preprocessed = preprocess_image(img_array)
            prediction = model.predict(preprocessed, verbose=0)

            class_idx = np.argmax(prediction[0])
            confidence = prediction[0][class_idx]
            label = labels[class_idx]

            with col2:
                css_class = "mask" if "mask" in label.lower() else "no-mask"
                emoji = "✅" if css_class == "mask" else "⚠️"

                st.markdown(
                    f'<div class="prediction-box {css_class}">'
                    f'{emoji} {label}<br>{confidence:.1%} confidence'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown("### Confidence Scores")
                for i, lbl in enumerate(labels):
                    st.progress(float(prediction[0][i]),
                                text=f"{lbl}: {prediction[0][i]:.1%}")

if __name__ == "__main__":
    main()
