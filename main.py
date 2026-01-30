import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import tf_keras
from tf_keras.layers import DepthwiseConv2D
import base64
from io import BytesIO
import time

st.set_page_config(page_title="Flower Classification", page_icon="🌸", layout="wide")


# Custom layer for model compatibility
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)


@st.cache_resource
def load_model():
    return tf_keras.models.load_model(
        "converted_keras/keras_model.h5",
        custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D},
        compile=False,
    )


@st.cache_data
def load_labels():
    labels = {}
    with open("converted_keras/labels.txt", "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1]
    return labels


def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224), Image.LANCZOS)
    arr = np.array(image, dtype=np.float32)
    arr = (arr / 127.5) - 1
    return np.expand_dims(arr, axis=0)


def predict_flower(image, model, labels):
    """Make prediction and return results"""
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)

    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    label = labels.get(predicted_class, "Unknown")

    all_results = sorted(
        [(labels[i], float(predictions[0][i])) for i in labels],
        key=lambda x: -x[1],
    )

    return label, confidence, all_results


def display_results(label, confidence, all_results):
    """Display prediction results with proper formatting"""
    emoji = "🌸" if confidence > 0.7 else "🤔" if confidence > 0.4 else "❓"

    if confidence > 0.7:
        st.success(f"## {emoji} {label}")
    elif confidence > 0.4:
        st.warning(f"## {emoji} {label}")
    else:
        st.error(f"## {emoji} {label}")

    st.metric("Confidence", f"{confidence * 100:.1f}%")

    st.markdown("**All Predictions:**")
    for lbl, prob in all_results:
        icon = "🟢" if prob > 0.7 else "🟠" if prob > 0.4 else "🔴"
        st.progress(prob, text=f"{icon} {lbl}: {prob * 100:.1f}%")


def create_realtime_camera():
    """Create HTML5 video component for real-time capture"""
    html_code = """
    <div style="text-align: center;">
        <video id="video" width="640" height="480" autoplay style="border: 3px solid #4CAF50; border-radius: 10px;"></video>
        <canvas id="canvas" width="224" height="224" style="display:none;"></canvas>
        <br><br>
        <button onclick="startCamera()" style="padding: 10px 20px; font-size: 16px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 5px;">
            📹 Start Camera
        </button>
        <button onclick="stopCamera()" style="padding: 10px 20px; font-size: 16px; background: #f44336; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 5px;">
            ⏹️ Stop Camera
        </button>
        <br>
        <p id="status" style="margin-top: 10px; font-weight: bold;"></p>
    </div>
    
    <script>
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let context = canvas.getContext('2d');
        let stream = null;
        let captureInterval = null;
        
        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: "environment"
                    } 
                });
                video.srcObject = stream;
                document.getElementById('status').textContent = '✅ Camera Active - Detecting flowers...';
                document.getElementById('status').style.color = '#4CAF50';
                
                // Capture frames every 1.5 seconds
                captureInterval = setInterval(captureFrame, 1500);
            } catch (err) {
                document.getElementById('status').textContent = '❌ Error: ' + err.message;
                document.getElementById('status').style.color = '#f44336';
            }
        }
        
        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                clearInterval(captureInterval);
                document.getElementById('status').textContent = '⏹️ Camera Stopped';
                document.getElementById('status').style.color = '#666';
            }
        }
        
        function captureFrame() {
            // Draw current video frame to canvas
            context.drawImage(video, 0, 0, 224, 224);
            
            // Convert canvas to base64 image
            let imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to Streamlit
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: imageData
            }, '*');
        }
        
        // Auto-start camera when loaded
        // startCamera();
    </script>
    """
    
    return components.html(html_code, height=650)


def display_results(label, confidence, all_results):
    """Display prediction results with proper formatting"""
    emoji = "🌸" if confidence > 0.7 else "🤔" if confidence > 0.4 else "❓"

    if confidence > 0.7:
        st.success(f"## {emoji} {label}")
    elif confidence > 0.4:
        st.warning(f"## {emoji} {label}")
    else:
        st.error(f"## {emoji} {label}")

    st.metric("Confidence", f"{confidence * 100:.1f}%")

    st.markdown("**All Predictions:**")
    for lbl, prob in all_results:
        icon = "🟢" if prob > 0.7 else "🟠" if prob > 0.4 else "🔴"
        st.progress(prob, text=f"{icon} {lbl}: {prob * 100:.1f}%")


def main():
    st.title("🌸 Flower Classification")
    st.caption("Take a photo or upload an image of a Tulip, Rose, or Sunflower!")

    # Load model and labels
    try:
        model = load_model()
        labels = load_labels()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Sidebar
    with st.sidebar:
        st.header("📖 Instructions")
        st.markdown(
            """
        **Real-Time Mode:**
        1. Click **Start Camera**
        2. Allow camera access
        3. Point at a flower
        4. See live predictions!
        
        **Snapshot Mode:**
        1. Take a photo or upload
        2. Get instant results
        """
        )

        st.markdown("---")
        st.header("🎨 Confidence Legend")
        st.markdown(
            """
        - 🟢 **Green** = High (>70%)
        - 🟠 **Orange** = Medium (40-70%)
        - 🔴 **Red** = Low (<40%)
        """
        )

        st.markdown("---")
        st.header("🌷 Supported Flowers")
        st.markdown(
            """
        - 🌷 Tulip
        - 🌹 Rose
        - 🌻 Sunflower
        """
        )

    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["🎥 Real-Time Detection", "📸 Camera Capture", "📁 Upload Image"])

    with tab1:
        st.markdown("### 🎥 Real-Time Flower Detection")
        st.info("Click 'Start Camera' to begin continuous flower detection!")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Real-time camera component
            image_data = create_realtime_camera()
        
        with col2:
            st.markdown("### 📊 Live Results")
            
            # Initialize session state for results
            if 'last_prediction' not in st.session_state:
                st.session_state.last_prediction = None
            
            # Process captured frame if available
            if image_data:
                try:
                    # Decode base64 image
                    image_data_clean = image_data.split(',')[1] if ',' in image_data else image_data
                    image_bytes = base64.b64decode(image_data_clean)
                    image = Image.open(BytesIO(image_bytes))
                    
                    # Make prediction
                    label, confidence, all_results = predict_flower(image, model, labels)
                    st.session_state.last_prediction = (label, confidence, all_results)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            # Display last prediction
            if st.session_state.last_prediction:
                label, confidence, all_results = st.session_state.last_prediction
                display_results(label, confidence, all_results)
            else:
                st.info("👆 Start the camera to see predictions!")
        
        st.markdown(
            """
        ---
        **Tips for best results:**
        - Ensure good lighting
        - Center the flower in frame
        - Hold camera steady
        - Get close to the flower
        - Allow 1-2 seconds between predictions
        """
        )

    with tab2:
        st.markdown("### 📸 Take a Photo")
        st.info("Click the button below to take a photo of a flower!")

        # Camera input - simpler and more reliable than webrtc
        camera_photo = st.camera_input("Take a picture")

        if camera_photo:
            # Read the image
            image = Image.open(camera_photo)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Captured Image", use_container_width=True)

            with col2:
                with st.spinner("Analyzing..."):
                    label, confidence, all_results = predict_flower(
                        image, model, labels
                    )

                display_results(label, confidence, all_results)

        st.markdown(
            """
        ---
        **Tips for best results:**
        - Ensure good lighting
        - Center the flower in frame
        - Hold camera steady
        - Get close to the flower
        """
        )

    with tab3:
        st.markdown("### 📁 Upload an Image")

        uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded:
            image = Image.open(uploaded)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                with st.spinner("Analyzing..."):
                    label, confidence, all_results = predict_flower(
                        image, model, labels
                    )

                display_results(label, confidence, all_results)


if __name__ == "__main__":
    main()
