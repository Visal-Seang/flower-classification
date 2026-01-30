import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tf_keras
from tf_keras.layers import DepthwiseConv2D
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import queue
from typing import List
import time


# Custom DepthwiseConv2D to handle 'groups' parameter issue
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)


# Load the model and labels
@st.cache_resource
def load_model():
    model_path = "converted_keras/keras_model.h5"
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
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224), Image.LANCZOS)
    img_array = np.array(image, dtype=np.float32)
    img_array = (img_array / 127.5) - 1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_flower(model, image, labels):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    results = []
    for i, prob in enumerate(predictions[0]):
        if i in labels:
            results.append((labels[i], prob))
    results.sort(key=lambda x: x[1], reverse=True)
    return labels[predicted_class], confidence, results


# Global frame queue for real-time processing
frame_queue: "queue.Queue[av.VideoFrame]" = queue.Queue(maxsize=1)
result_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=1)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Callback that captures frames for processing"""
    img = frame.to_ndarray(format="bgr24")

    # Put frame in queue (non-blocking, replace old frame)
    try:
        frame_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        pass

    # Get latest result if available
    try:
        label, confidence, color = result_queue.get_nowait()

        # Draw on frame
        # Draw background rectangle
        img[5:55, 5:350] = color

        # Add text using PIL
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        text = f"{label}: {confidence * 100:.1f}%"
        draw.text((15, 15), text, fill=(255, 255, 255))
        img = np.array(pil_img)

    except queue.Empty:
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.set_page_config(
        page_title="Flower Classification", page_icon="🌸", layout="wide"
    )

    st.title("🌸 Real-Time Flower Classification")
    st.write(
        "Point your camera at a **Tulip**, **Rose**, or **Sunflower** for real-time detection!"
    )

    # Load model and labels
    try:
        model = load_model()
        labels = load_labels()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Camera Feed")

        # WebRTC streamer
        ctx = webrtc_streamer(
            key="flower-detection",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

    with col2:
        st.subheader("🔍 Detection Results")

        # Placeholders for results
        result_container = st.empty()
        confidence_container = st.empty()
        progress_container = st.empty()

        # Legend
        st.markdown(
            """
        ---
        **Confidence Legend:**
        - 🟢 High (>70%)
        - 🟠 Medium (40-70%)
        - 🔴 Low (<40%)
        """
        )

    # Real-time inference loop
    if ctx.state.playing:
        st.info("🎥 Camera is active! Processing frames...")

        while ctx.state.playing:
            try:
                # Get frame from queue
                frame = frame_queue.get(timeout=0.1)
                img = frame.to_ndarray(format="bgr24")

                # Convert BGR to RGB
                img_rgb = img[:, :, ::-1]
                pil_image = Image.fromarray(img_rgb)

                # Run prediction
                predicted_label, confidence, all_results = predict_flower(
                    model, pil_image, labels
                )

                # Determine color based on confidence
                if confidence > 0.7:
                    color = [0, 200, 0]  # Green
                    emoji = "🌸"
                elif confidence > 0.4:
                    color = [255, 165, 0]  # Orange
                    emoji = "🤔"
                else:
                    color = [255, 0, 0]  # Red
                    emoji = "❓"

                # Send result to video callback
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                result_queue.put_nowait((predicted_label, confidence, color))

                # Update UI
                with result_container:
                    if confidence > 0.7:
                        st.success(f"### {emoji} {predicted_label}")
                    elif confidence > 0.4:
                        st.warning(f"### {emoji} {predicted_label}")
                    else:
                        st.error(f"### {emoji} {predicted_label}")

                with confidence_container:
                    st.metric("Confidence", f"{confidence * 100:.1f}%")

                with progress_container:
                    for label, prob in all_results:
                        prefix = "🟢" if prob > 0.7 else "🟠" if prob > 0.4 else "🔴"
                        st.progress(
                            float(prob), text=f"{prefix} {label}: {prob * 100:.1f}%"
                        )

                time.sleep(0.1)  # Small delay to prevent overwhelming

            except queue.Empty:
                continue
            except Exception as e:
                st.error(f"Error: {e}")
                break


if __name__ == "__main__":
    main()
