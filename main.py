import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tf_keras
from tf_keras.layers import DepthwiseConv2D


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
    predictions = model.predict(processed_img, verbose=0)

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
    tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload Image"])

    with tab1:
        st.write("### 📸 Camera Detection")
        st.info(
            "Take a photo to classify the flower. For best results, ensure good lighting and center the flower in frame."
        )

        # Camera input
        camera_image = st.camera_input("Point your camera at a flower and take a photo")

        if camera_image is not None:
            # Open the captured image
            image = Image.open(camera_image)

            # Make prediction
            predicted_label, confidence, all_results = predict_flower(
                model, image, labels
            )

            # Create columns for layout
            col1, col2 = st.columns(2)

            with col1:
                # Add prediction overlay to image
                img_display = image.copy()
                draw = ImageDraw.Draw(img_display)

                # Set color based on confidence
                if confidence > 0.7:
                    color = (0, 200, 0)  # Green
                    emoji = "🌸"
                elif confidence > 0.4:
                    color = (255, 165, 0)  # Orange
                    emoji = "🤔"
                else:
                    color = (255, 0, 0)  # Red
                    emoji = "❓"

                # Draw result on image
                draw.rectangle([0, 0, img_display.width, 40], fill=color)
                draw.text(
                    (10, 10),
                    f"{predicted_label}: {confidence * 100:.1f}%",
                    fill=(255, 255, 255),
                )

                st.image(
                    img_display, caption="Captured Image", use_container_width=True
                )

            with col2:
                # Display results
                st.markdown(f"### {emoji} **{predicted_label}**")
                st.metric("Confidence", f"{confidence * 100:.1f}%")

                # Show all predictions with progress bars
                st.write("**All Predictions:**")
                for label, prob in all_results:
                    if prob > 0.7:
                        st.progress(float(prob), text=f"🟢 {label}: {prob * 100:.1f}%")
                    elif prob > 0.4:
                        st.progress(float(prob), text=f"🟠 {label}: {prob * 100:.1f}%")
                    else:
                        st.progress(float(prob), text=f"🔴 {label}: {prob * 100:.1f}%")

            # Add button to take another photo
            st.markdown("---")
            st.info("👆 Click the camera button above to take another photo!")

    with tab2:
        st.write("### 📁 Upload Image")

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

                # Set emoji based on confidence
                if confidence > 0.7:
                    emoji = "🌸"
                elif confidence > 0.4:
                    emoji = "🤔"
                else:
                    emoji = "❓"

                # Display results
                st.markdown(f"### {emoji} **{predicted_label}**")
                st.metric("Confidence", f"{confidence * 100:.2f}%")

                # Show all predictions
                st.write("**All Predictions:**")
                for label, prob in all_results:
                    if prob > 0.7:
                        st.progress(float(prob), text=f"🟢 {label}: {prob * 100:.1f}%")
                    elif prob > 0.4:
                        st.progress(float(prob), text=f"🟠 {label}: {prob * 100:.1f}%")
                    else:
                        st.progress(float(prob), text=f"🔴 {label}: {prob * 100:.1f}%")


if __name__ == "__main__":
    main()
