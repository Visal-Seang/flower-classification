import streamlit as st
import numpy as np
from PIL import Image

# Set page config FIRST
st.set_page_config(page_title="Flower Classification", page_icon="🌸", layout="wide")


# Lazy imports to avoid conflicts
@st.cache_resource
def get_model_and_labels():
    """Load TensorFlow and model in isolated way"""
    import tf_keras
    from tf_keras.layers import DepthwiseConv2D

    class CustomDepthwiseConv2D(DepthwiseConv2D):
        def __init__(self, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(**kwargs)

    model = tf_keras.models.load_model(
        "converted_keras/keras_model.h5",
        custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D},
        compile=False,
    )

    labels = {}
    with open("converted_keras/labels.txt", "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1]

    return model, labels


def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224), Image.LANCZOS)
    img_array = np.array(image, dtype=np.float32)
    img_array = (img_array / 127.5) - 1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(model, image, labels):
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])

    results = [
        (labels[i], float(predictions[0][i]))
        for i in range(len(predictions[0]))
        if i in labels
    ]
    results.sort(key=lambda x: x[1], reverse=True)

    return labels[predicted_class], confidence, results


def main():
    st.title("🌸 Real-Time Flower Classification")

    # Load model
    try:
        model, labels = get_model_and_labels()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Sidebar with instructions
    with st.sidebar:
        st.header("ℹ️ Instructions")
        st.markdown(
            """
        1. Click **Start Real-Time Detection**
        2. Allow camera access
        3. Point camera at a flower
        4. See instant predictions!
        
        **Supported Flowers:**
        - 🌷 Tulip
        - 🌹 Rose  
        - 🌻 Sunflower
        """
        )

        st.markdown("---")
        st.markdown("**Confidence Legend:**")
        st.markdown("- 🟢 High (>70%)")
        st.markdown("- 🟠 Medium (40-70%)")
        st.markdown("- 🔴 Low (<40%)")

    # Main content
    tab1, tab2 = st.tabs(["📹 Real-Time Camera", "📁 Upload Image"])

    with tab1:
        # Import webrtc only when needed (lazy import)
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            import av

            st.info("🎥 Click **START** below to begin real-time detection!")

            # Result placeholders
            col1, col2 = st.columns([2, 1])

            with col2:
                result_box = st.empty()
                confidence_box = st.empty()
                details_box = st.empty()

            # Frame counter for periodic inference
            if "frame_count" not in st.session_state:
                st.session_state.frame_count = 0
                st.session_state.last_result = None

            def video_callback(frame):
                img = frame.to_ndarray(format="bgr24")

                # Run inference every 10 frames to reduce load
                st.session_state.frame_count += 1

                if st.session_state.frame_count % 10 == 0:
                    try:
                        # Convert BGR to RGB
                        img_rgb = img[:, :, ::-1]
                        pil_image = Image.fromarray(img_rgb)

                        # Predict
                        label, conf, results = predict(model, pil_image, labels)
                        st.session_state.last_result = (label, conf, results)

                        # Draw on frame
                        color = (
                            (0, 255, 0)
                            if conf > 0.7
                            else (0, 165, 255) if conf > 0.4 else (0, 0, 255)
                        )

                        # Draw rectangle and text
                        img[10:60, 10:400] = color

                    except Exception:
                        pass

                return av.VideoFrame.from_ndarray(img, format="bgr24")

            with col1:
                ctx = webrtc_streamer(
                    key="flower-detect",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=video_callback,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    },
                )

            # Update results display
            if ctx.state.playing and st.session_state.last_result:
                label, conf, results = st.session_state.last_result

                with result_box:
                    if conf > 0.7:
                        st.success(f"### 🌸 {label}")
                    elif conf > 0.4:
                        st.warning(f"### 🤔 {label}")
                    else:
                        st.error(f"### ❓ {label}")

                with confidence_box:
                    st.metric("Confidence", f"{conf * 100:.1f}%")

                with details_box:
                    for lbl, prob in results:
                        prefix = "🟢" if prob > 0.7 else "🟠" if prob > 0.4 else "🔴"
                        st.progress(prob, text=f"{prefix} {lbl}: {prob*100:.1f}%")

        except ImportError:
            st.warning(
                "Real-time video not available. Using camera capture mode instead."
            )

            # Fallback to camera input
            camera_image = st.camera_input("📸 Take a photo")

            if camera_image:
                image = Image.open(camera_image)
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.image(image, use_container_width=True)

                with col2:
                    label, conf, results = predict(model, image, labels)

                    if conf > 0.7:
                        st.success(f"### 🌸 {label}")
                    elif conf > 0.4:
                        st.warning(f"### 🤔 {label}")
                    else:
                        st.error(f"### ❓ {label}")

                    st.metric("Confidence", f"{conf * 100:.1f}%")

                    for lbl, prob in results:
                        prefix = "🟢" if prob > 0.7 else "🟠" if prob > 0.4 else "🔴"
                        st.progress(prob, text=f"{prefix} {lbl}: {prob*100:.1f}%")

    with tab2:
        uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded:
            image = Image.open(uploaded)
            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                with st.spinner("Analyzing..."):
                    label, conf, results = predict(model, image, labels)

                if conf > 0.7:
                    st.success(f"### 🌸 {label}")
                elif conf > 0.4:
                    st.warning(f"### 🤔 {label}")
                else:
                    st.error(f"### ❓ {label}")

                st.metric("Confidence", f"{conf * 100:.1f}%")

                for lbl, prob in results:
                    prefix = "🟢" if prob > 0.7 else "🟠" if prob > 0.4 else "🔴"
                    st.progress(prob, text=f"{prefix} {lbl}: {prob*100:.1f}%")


if __name__ == "__main__":
    main()
