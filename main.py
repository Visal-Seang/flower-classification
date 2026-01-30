"""
Flower Classification App - Improved Version
Real-time flower detection using webcam or uploaded images
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tf_keras
from tf_keras.layers import DepthwiseConv2D
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "converted_keras/keras_model.h5"
LABELS_PATH = "converted_keras/labels.txt"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLDS = {"high": 0.7, "medium": 0.4}


# ============================================================================
# CUSTOM STYLES
# ============================================================================


def inject_custom_css():
    """Inject custom CSS for better styling"""
    st.markdown(
        """
        <style>
        /* Main title styling */
        h1 {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            padding: 1rem 0;
        }
        
        /* Card-like containers */
        .prediction-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        /* Confidence bar styling */
        .confidence-bar {
            height: 8px;
            border-radius: 4px;
            background: #e0e0e0;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .confidence-fill {
            height: 100%;
            transition: width 0.3s ease;
        }
        
        /* Status messages */
        .status-message {
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-weight: 500;
        }
        
        .status-waiting {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .status-ready {
            background-color: #d1ecf1;
            color: #0c5460;
        }
        
        /* Reduce padding on mobile */
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem;
            }
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class PredictionResult:
    """Data class for prediction results"""

    label: str
    confidence: float
    all_predictions: List[Tuple[str, float]]

    def get_confidence_color(self) -> str:
        """Get color based on confidence level"""
        if self.confidence >= CONFIDENCE_THRESHOLDS["high"]:
            return "#4caf50"  # Green
        elif self.confidence >= CONFIDENCE_THRESHOLDS["medium"]:
            return "#ff9800"  # Orange
        return "#f44336"  # Red

    def get_confidence_text(self) -> str:
        """Get human-readable confidence text"""
        if self.confidence >= CONFIDENCE_THRESHOLDS["high"]:
            return "High Confidence"
        elif self.confidence >= CONFIDENCE_THRESHOLDS["medium"]:
            return "Medium Confidence"
        return "Low Confidence"


# ============================================================================
# CUSTOM KERAS LAYER
# ============================================================================


class CustomDepthwiseConv2D(DepthwiseConv2D):
    """Custom layer for model compatibility"""

    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)


# ============================================================================
# MODEL LOADING
# ============================================================================


@st.cache_resource
def load_model():
    """Load the Keras model with custom objects"""
    try:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        model = tf_keras.models.load_model(
            MODEL_PATH,
            custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D},
            compile=False,
        )
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please ensure the model file exists at the correct path.")
        return None


@st.cache_data
def load_labels() -> Dict[int, str]:
    """Load labels from file"""
    try:
        labels_path = Path(LABELS_PATH)
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")

        labels = {}
        with open(LABELS_PATH, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    labels[int(parts[0])] = parts[1]

        if not labels:
            raise ValueError("No labels found in labels file")

        return labels
    except Exception as e:
        st.error(f"❌ Error loading labels: {str(e)}")
        return {}


# ============================================================================
# IMAGE PROCESSING
# ============================================================================


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to model input size
    image = image.resize(IMG_SIZE, Image.LANCZOS)

    # Convert to array and normalize
    arr = np.array(image, dtype=np.float32)
    arr = (arr / 127.5) - 1.0

    return np.expand_dims(arr, axis=0)


def predict_image(
    image: Image.Image, model, labels: Dict[int, str]
) -> Optional[PredictionResult]:
    """Run prediction on an image"""
    try:
        # Preprocess
        processed = preprocess_image(image)

        # Predict
        predictions = model.predict(processed, verbose=0)

        # Get results
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        label = labels.get(predicted_class, "Unknown")

        # Sort all predictions
        all_predictions = sorted(
            [(labels[i], float(predictions[0][i])) for i in labels],
            key=lambda x: -x[1],
        )

        return PredictionResult(
            label=label, confidence=confidence, all_predictions=all_predictions
        )
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


# ============================================================================
# THREAD-SAFE RESULT STORAGE
# ============================================================================


class ResultStore:
    """Thread-safe storage for video stream results"""

    def __init__(self):
        self.lock = threading.Lock()
        self.result: Optional[PredictionResult] = None
        self.frame_count = 0

    def update(self, result: PredictionResult):
        """Update stored result"""
        with self.lock:
            self.result = result
            self.frame_count += 1

    def get(self) -> Tuple[Optional[PredictionResult], int]:
        """Get current result and frame count"""
        with self.lock:
            return self.result, self.frame_count


# Global result store
RESULT_STORE = ResultStore()


# ============================================================================
# VIDEO PROCESSOR
# ============================================================================


class FlowerDetector(VideoProcessorBase):
    """Video processor for real-time flower detection"""

    def __init__(self):
        self.model = st.session_state.get("model")
        self.labels = st.session_state.get("labels")
        self.frame_skip = 2  # Process every Nth frame for performance
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process video frame"""
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Skip frames for performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            # Convert BGR to RGB
            img_rgb = img[:, :, ::-1].copy()
            pil_image = Image.fromarray(img_rgb)

            # Run prediction
            result = predict_image(pil_image, self.model, self.labels)

            if result:
                # Update global store
                RESULT_STORE.update(result)

                # Draw on frame
                img = self._draw_prediction(img, result)

        except Exception as e:
            # Silently continue on errors to avoid breaking stream
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _draw_prediction(self, img: np.ndarray, result: PredictionResult) -> np.ndarray:
        """Draw prediction overlay on frame"""
        pil_frame = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_frame)

        # Get color based on confidence
        color_hex = result.get_confidence_color()
        color_rgb = tuple(int(color_hex[i : i + 2], 16) for i in (1, 3, 5))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # Convert to BGR

        # Draw background for text
        text = f"{result.label}: {result.confidence * 100:.1f}%"
        bbox = draw.textbbox((20, 20), text)
        draw.rectangle([10, 10, bbox[2] + 30, bbox[3] + 30], fill=color_bgr)

        # Draw text
        draw.text((20, 20), text, fill=(255, 255, 255))

        # Draw colored border
        border_width = 8
        w, h = pil_frame.size
        draw.rectangle([0, 0, w - 1, border_width], fill=color_bgr)
        draw.rectangle([0, h - border_width, w - 1, h - 1], fill=color_bgr)
        draw.rectangle([0, 0, border_width, h - 1], fill=color_bgr)
        draw.rectangle([w - border_width, 0, w - 1, h - 1], fill=color_bgr)

        return np.array(pil_frame)


# ============================================================================
# UI COMPONENTS
# ============================================================================


def display_prediction_card(result: PredictionResult, show_all: bool = True):
    """Display prediction results in a nice card format"""

    # Top prediction with large display
    color = result.get_confidence_color()
    confidence_text = result.get_confidence_text()

    st.markdown(
        f"""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, {color}22 0%, {color}11 100%); border-radius: 15px; margin: 1rem 0;'>
            <h2 style='color: {color}; margin: 0;'>{result.label}</h2>
            <p style='font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: {color};'>{result.confidence * 100:.1f}%</p>
            <p style='color: #666; margin: 0;'>{confidence_text}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Show all predictions if requested
    if show_all and len(result.all_predictions) > 1:
        st.markdown("### All Predictions")

        for label, conf in result.all_predictions:
            # Create a progress bar effect
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{label}**")
                st.progress(conf)
            with col2:
                st.markdown(
                    f"<p style='text-align: right; margin-top: 0.5rem;'>{conf * 100:.1f}%</p>",
                    unsafe_allow_html=True,
                )


def show_status_message(message: str, status_type: str = "waiting"):
    """Display a status message"""
    st.markdown(
        f"""
        <div class='status-message status-{status_type}'>
            {message}
        </div>
    """,
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """Main application"""

    # Inject custom CSS
    inject_custom_css()

    # Title
    st.markdown(
        "<h1 style='text-align: center;'>🌸 Flower Classification</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #666;'>Identify flowers in real-time or from uploaded images</p>",
        unsafe_allow_html=True,
    )

    # Load model and labels
    if "model" not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.model = load_model()
            st.session_state.labels = load_labels()

    model = st.session_state.model
    labels = st.session_state.labels

    # Check if model loaded successfully
    if model is None or not labels:
        st.error("⚠️ Unable to load model or labels. Please check your files.")
        return

    # Display number of classes
    st.info(f"ℹ️ Model loaded successfully with {len(labels)} flower classes")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📹 Real-Time Detection", "📁 Upload Image", "ℹ️ About"])

    # ========================================================================
    # TAB 1: REAL-TIME DETECTION
    # ========================================================================
    with tab1:
        st.markdown("### Real-Time Camera Detection")
        st.markdown("Point your camera at a flower to identify it in real-time.")

        # Center the camera feed
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # WebRTC streamer
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
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:global.stun.twilio.com:3478"]},
                        {
                            "urls": [
                                "turn:a.relay.metered.ca:80",
                                "turn:a.relay.metered.ca:80?transport=tcp",
                                "turn:a.relay.metered.ca:443",
                                "turn:a.relay.metered.ca:443?transport=tcp",
                            ],
                            "username": "e8dd65b92e4bd3d5c0b58d42",
                            "credential": "uWdxUCn0/ZULohEj",
                        },
                    ],
                    "iceTransportPolicy": "all",
                },
            )

        # Display results
        if ctx.state.playing:
            result, frame_count = RESULT_STORE.get()

            if result:
                display_prediction_card(result, show_all=True)

                # Show frame count for debugging
                with st.expander("📊 Detection Stats"):
                    st.metric("Frames Processed", frame_count)
                    st.metric("Current Confidence", f"{result.confidence * 100:.1f}%")
            else:
                show_status_message(
                    "🔍 Point your camera at a flower to start detecting...", "waiting"
                )
        else:
            show_status_message("▶️ Click START to begin detection", "ready")

    # ========================================================================
    # TAB 2: UPLOAD IMAGE
    # ========================================================================
    with tab2:
        st.markdown("### Upload an Image")
        st.markdown("Upload a photo of a flower for classification.")

        # Center the uploader
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear photo of a flower",
            )

            if uploaded_file:
                # Load and display image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                # Run prediction
                with st.spinner("🔍 Analyzing image..."):
                    result = predict_image(image, model, labels)

                if result:
                    display_prediction_card(result, show_all=True)

                    # Download button for results
                    results_text = f"Flower Classification Results\n\n"
                    results_text += f"Top Prediction: {result.label}\n"
                    results_text += f"Confidence: {result.confidence * 100:.1f}%\n\n"
                    results_text += "All Predictions:\n"
                    for label, conf in result.all_predictions:
                        results_text += f"  {label}: {conf * 100:.1f}%\n"

                    st.download_button(
                        label="📥 Download Results",
                        data=results_text,
                        file_name="flower_classification_results.txt",
                        mime="text/plain",
                    )

    # ========================================================================
    # TAB 3: ABOUT
    # ========================================================================
    with tab3:
        st.markdown("### About This Application")

        st.markdown(
            """
        This application uses a deep learning model to classify flowers in real-time or from uploaded images.
        
        #### Features:
        - 🎥 **Real-time detection** using your webcam
        - 📸 **Image upload** for analyzing photos
        - 🎯 **Confidence scoring** with color-coded results
        - 📊 **Multiple predictions** showing all possible classifications
        
        #### How to Use:
        1. **Real-Time**: Click START in the camera tab and point at a flower
        2. **Upload**: Choose an image file from your device
        3. View the classification results with confidence scores
        
        #### Tips for Best Results:
        - Ensure good lighting
        - Get close to the flower
        - Keep the flower centered in frame
        - Avoid blurry images
        
        #### Confidence Levels:
        - 🟢 **Green** (>70%): High confidence
        - 🟠 **Orange** (40-70%): Medium confidence
        - 🔴 **Red** (<40%): Low confidence
        """
        )

        # Model info
        st.markdown("### Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Type", "TensorFlow Keras")
            st.metric("Input Size", f"{IMG_SIZE[0]}x{IMG_SIZE[1]}")
        with col2:
            st.metric("Number of Classes", len(labels))
            st.metric("Model File", MODEL_PATH)


if __name__ == "__main__":
    main()
