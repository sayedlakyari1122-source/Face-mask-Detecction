# 😷 Face Mask Detection App

A real-time face mask detection application using Streamlit and TensorFlow (Teachable Machine model).

## Features

- 📹 **Real-time webcam detection** - See instant predictions as you move
- 📤 **Image upload** - Test with existing images
- 📊 **Live statistics** - Track mask vs no-mask detections
- ⚙️ **Adjustable confidence threshold** - Fine-tune sensitivity
- 🎨 **Clean, modern UI** - Easy to use interface

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Make sure your model files are in the same directory:**
   - `keras_model.h5`
   - `labels.txt`

3. **Run the app:**
```bash
streamlit run mask_detector_app.py
```

4. **Open your browser** at `http://localhost:8501`

## How to Use

### Webcam Mode
1. Click on the "📹 Webcam" tab
2. Check the "Start Webcam" box
3. Allow camera permissions when prompted
4. Position your face in frame
5. See real-time predictions!

### Upload Mode
1. Click on the "📤 Upload Image" tab
2. Upload a JPG/PNG image
3. View the prediction and confidence scores

## Troubleshooting

- **Camera not working?** Check browser permissions for camera access
- **Low confidence?** Adjust the confidence threshold in the sidebar
- **Model not loading?** Ensure `keras_model.h5` and `labels.txt` are in the same folder

## Tech Stack

- **Streamlit** - Web interface
- **TensorFlow/Keras** - Model inference
- **OpenCV** - Image processing
- **Teachable Machine** - Model training

---

Made with ❤️ using Teachable Machine and Streamlit
