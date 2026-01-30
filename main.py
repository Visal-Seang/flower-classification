import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tf_keras
from tf_keras.layers import DepthwiseConv2D
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import threading
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


# Load model and labels globally
MODEL = None
LABELS = None


# Thread-safe result storage
class ResultStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.label = ""
        self.confidence = 0.0
        self.all_results = []
        self.last_update_time = 0

    def update(self, label, confidence, all_results):
        with self.lock:
            self.label = label
            self.confidence = confidence
            self.all_results = all_results
            self.last_update_time = time.time()

    def get(self):
        with self.lock:
            return {
                "label": self.label,
                "confidence": self.confidence,
                "all_results": self.all_results.copy(),
                "last_update_time": self.last_update_time,
            }


# Global result store
RESULT_STORE = ResultStore()


class FlowerDetector(VideoProcessorBase):
    def __init__(self):
        self.model = MODEL
        self.labels = LABELS

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")

        # Convert BGR to RGB for prediction
        img_rgb = img[:, :, ::-1].copy()
        pil_image = Image.fromarray(img_rgb)

        # Run prediction
        try:
            processed = preprocess_image(pil_image)
            predictions = self.model.predict(processed, verbose=0)

            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            label = self.labels.get(predicted_class, "Unknown")

            # Store results globally
            all_results = sorted(
                [(self.labels[i], float(predictions[0][i])) for i in self.labels],
                key=lambda x: -x[1],
            )

            RESULT_STORE.update(label, confidence, all_results)

            # Choose color based on confidence (BGR format for OpenCV)
            if confidence > 0.7:
                color = (0, 255, 0)  # Green
            elif confidence > 0.4:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red

            # Only draw border around frame (NO TEXT ON VIDEO)
            pil_frame = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_frame)

            # Draw colored border around frame
            border_width = 8
            w, h = pil_frame.size
            draw.rectangle([0, 0, w - 1, border_width], fill=color)
            draw.rectangle([0, h - border_width, w - 1, h - 1], fill=color)
            draw.rectangle([0, 0, border_width, h - 1], fill=color)
            draw.rectangle([w - border_width, 0, w - 1, h - 1], fill=color)

            img = np.array(pil_frame)

        except Exception as e:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    global MODEL, LABELS

    st.title("🌸 Real-Time Flower Classification")
    st.caption("Point your camera at a Tulip, Rose, or Sunflower!")

    # Load model and labels
    try:
        MODEL = load_model()
        LABELS = load_labels()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Sidebar
    with st.sidebar:
        st.header("📖 Instructions")
        st.markdown(
            """
        1. Click **START** below
        2. Allow camera access
        3. Point camera at a flower
        4. See real-time predictions!
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
    tab1, tab2 = st.tabs(["📹 Real-Time Detection", "📁 Upload Image"])

    with tab1:
        st.markdown("### 🎥 Live Camera Feed")
        st.info("Click **START** to begin real-time flower detection!")

        # Create two columns: camera on left, results on right
        col_camera, col_results = st.columns([1, 1])

        with col_camera:
            # WebRTC streamer for real-time video
            ctx = webrtc_streamer(
                key="flower-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=FlowerDetector,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 30},
                        "facingMode": "environment",
                    },
                    "audio": False,
                },
                async_processing=True,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
            )

        # Real-time results display - SEPARATE FROM CAMERA
        with col_results:
            st.markdown("### 📊 Detection Results")

            # Create placeholders for real-time updates
            status_placeholder = st.empty()
            result_text_placeholder = st.empty()
            confidence_placeholder = st.empty()
            progress_placeholder = st.empty()
            all_results_placeholder = st.empty()

            # Continuous update loop
            if ctx.state.playing:
                while ctx.state.playing:
                    results = RESULT_STORE.get()

                    # Check if we have recent results (within last 2 seconds)
                    if (
                        results["label"]
                        and (time.time() - results["last_update_time"]) < 2
                    ):
                        label = results["label"]
                        conf = results["confidence"]
                        all_results = results["all_results"]

                        # Status message
                        if conf > 0.7:
                            status_placeholder.success("✅ **Detection Active**")
                        elif conf > 0.4:
                            status_placeholder.warning("⚠️ **Detection Active**")
                        else:
                            status_placeholder.error("❓ **Low Confidence**")

                        # Large result text
                        result_text_placeholder.markdown(f"# {label}")

                        # Confidence percentage
                        confidence_placeholder.markdown(
                            f"### Confidence: {conf * 100:.1f}%"
                        )

                        # Progress bar
                        progress_placeholder.progress(conf)

                        # All predictions
                        with all_results_placeholder.container():
                            st.markdown("---")
                            st.markdown("**📋 All Predictions:**")
                            for lbl, prob in all_results:
                                icon = (
                                    "🟢" if prob > 0.7 else "🟠" if prob > 0.4 else "🔴"
                                )
                                st.write(f"{icon} **{lbl}**: {prob * 100:.1f}%")
                                st.progress(prob)
                    else:
                        # Show waiting state
                        status_placeholder.info("🔍 **Waiting for flower...**")
                        result_text_placeholder.markdown("### Point camera at a flower")
                        confidence_placeholder.empty()
                        progress_placeholder.empty()
                        all_results_placeholder.empty()

                    time.sleep(0.1)  # Small delay to prevent excessive updates
            else:
                status_placeholder.info("▶️ **Camera not started**")
                result_text_placeholder.markdown("### Click START to begin")
                confidence_placeholder.empty()
                progress_placeholder.empty()
                all_results_placeholder.empty()

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
                    processed = preprocess_image(image)
                    predictions = MODEL.predict(processed, verbose=0)

                    predicted_class = np.argmax(predictions[0])
                    confidence = float(predictions[0][predicted_class])
                    label = LABELS.get(predicted_class, "Unknown")

                    results = sorted(
                        [(LABELS[i], float(predictions[0][i])) for i in LABELS],
                        key=lambda x: -x[1],
                    )

                # Display results
                emoji = "🌸" if confidence > 0.7 else "🤔" if confidence > 0.4 else "❓"

                if confidence > 0.7:
                    st.success(f"## {emoji} {label}")
                elif confidence > 0.4:
                    st.warning(f"## {emoji} {label}")
                else:
                    st.error(f"## {emoji} {label}")

                st.metric("Confidence", f"{confidence * 100:.1f}%")

                st.markdown("**All Predictions:**")
                for lbl, prob in results:
                    icon = "🟢" if prob > 0.7 else "🟠" if prob > 0.4 else "🔴"
                    st.progress(prob, text=f"{icon} {lbl}: {prob * 100:.1f}%")


if __name__ == "__main__":
    main()
