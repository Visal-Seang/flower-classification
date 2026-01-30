import streamlit as st
import numpy as np
from PIL import Image
import tf_keras
from tf_keras.layers import DepthwiseConv2D
import av
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode


# Custom DepthwiseConv2D to handle 'groups' parameter issue
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove 'groups' parameter if present
        kwargs.pop("groups", None)
        super().__init__(**kwargs)


# Load the model and labels
@st.cache_resource
def load_model():
    model_path = "converted_keras/keras_model.h5"
    # Load with custom objects to handle compatibility issues
    model = tf_keras.models.load_model(
        model_path,
        custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D},
        compile=False,
    )
    return model


@st.cache_data
def load_labels():
    labels_path = "converted_keras/labels.txt"
    labels = {}
    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1]
    return labels


def preprocess_image(image):
    # Convert to RGB if needed (in case of RGBA or grayscale)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image to 224x224 (standard for Teachable Machine models)
    image = image.resize((224, 224), Image.LANCZOS)

    # Convert to array
    img_array = np.array(image, dtype=np.float32)

    # Normalize to [-1, 1] range (Teachable Machine uses this range)
    img_array = (img_array / 127.5) - 1

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_flower(model, image, labels):
    # Preprocess the image
    processed_img = preprocess_image(image)

    # Make prediction
    predictions = model.predict(processed_img)

    # Get the predicted class
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Get all predictions with confidence scores
    results = []
    for i, prob in enumerate(predictions[0]):
        if i in labels:
            results.append((labels[i], prob))

    results.sort(key=lambda x: x[1], reverse=True)

    return labels[predicted_class], confidence, results


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Flower Classification", page_icon="🌸", layout="centered"
    )

    st.title("🌸 Flower Classification App")
    st.write(
        "Upload an image or use webcam to identify if it's a **Tulip**, **Rose**, or **Sunflower**"
    )

    # Load model and labels
    try:
        model = load_model()
        labels = load_labels()
    except Exception as e:
        st.error(f"Error loading model or labels: {e}")
        return

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📷 Webcam", "📁 Upload Image"])

    with tab1:
        st.write("### Real-time Flower Detection")
        st.info(
            "🎥 Click START to begin real-time detection. Allow camera access when prompted."
        )

        # Video processor class for real-time detection
        class FlowerDetector(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")

                # Convert BGR to RGB for prediction
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)

                # Preprocess and predict
                processed_img = preprocess_image(pil_image)
                predictions = model.predict(processed_img, verbose=0)

                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class])
                predicted_label = labels.get(predicted_class, "Unknown")

                # Set color based on confidence
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green
                elif confidence > 0.4:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 0, 255)  # Red

                # Draw prediction text on frame
                cv2.putText(
                    img,
                    f"{predicted_label}: {confidence * 100:.1f}%",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    color,
                    3,
                )

                # Draw colored border
                img = cv2.copyMakeBorder(
                    img, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=color
                )

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        # WebRTC streamer for real-time video
        webrtc_streamer(
            key="flower-detector",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=FlowerDetector,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

        st.markdown(
            """
        **Legend:**
        - 🟢 Green border = High confidence (>70%)
        - 🟠 Orange border = Medium confidence (40-70%)
        - 🔴 Red border = Low confidence (<40%)
        """
        )

    with tab2:
        # File uploader - only accepts single image
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                # Make prediction
                with st.spinner("Analyzing..."):
                    predicted_label, confidence, all_results = predict_flower(
                        model, image, labels
                    )

                # Display results
                st.success(f"**Prediction: {predicted_label}**")
                st.metric("Confidence", f"{confidence * 100:.2f}%")

                # Show all predictions
                st.write("**All Predictions:**")
                for label, prob in all_results:
                    st.progress(float(prob), text=f"{label}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
