import streamlit as st
import numpy as np
from PIL import Image
import tf_keras
from tf_keras.layers import DepthwiseConv2D

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
        1. Click **Take Photo** or **Browse files**
        2. Allow camera access (for camera)
        3. Point at a flower and capture
        4. See instant predictions!
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
    tab1, tab2 = st.tabs(["📸 Camera Capture", "📁 Upload Image"])

    with tab1:
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
                    label, confidence, all_results = predict_flower(image, model, labels)
                
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

    with tab2:
        st.markdown("### 📁 Upload an Image")

        uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded:
            image = Image.open(uploaded)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                with st.spinner("Analyzing..."):
                    label, confidence, all_results = predict_flower(image, model, labels)
                
                display_results(label, confidence, all_results)


if __name__ == "__main__":
    main()
