import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# ============================
# Page config (MUST be first)
# ============================
st.set_page_config(
    page_title="NeuraScan – Brain Tumor Detection",
    layout="centered"
)

# ============================
# LOGO — perfectly centered
# ============================
import base64

def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_image_base64("logo.png")

st.markdown(
    f"""
    <div style="display: flex; justify-content: center; margin-top: 10px;">
        <img src="data:image/png;base64,{logo_base64}" width="170">
    </div>
    <h1 style="text-align: center; margin-bottom: 4px;">NeuraScan</h1>
    <p style="text-align: center; margin-top: 0;">
        AI‑powered Brain Tumor Detection
    </p>
    """,
    unsafe_allow_html=True
)

# ============================
# Constants
# ============================
MODEL_PATH = "converted_keras/keras_model.h5"
CLASS_NAMES = ["No Brain Tumor", "Brain Tumor"]

# ============================
# Load model
# ============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model

model = load_model()

# ============================
# Find last convolutional layer (Grad‑CAM attempt)
# ============================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name
    return None

LAST_CONV_LAYER = find_last_conv_layer(model)
st.caption(f"Grad‑CAM layer detected: {LAST_CONV_LAYER}")

# ============================
# Preprocessing (Teachable Machine–style)
# ============================
def preprocess_image(pil_image):
    pil_image = pil_image.convert("RGB")

    w, h = pil_image.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    pil_image = pil_image.crop(
        (left, top, left + min_dim, top + min_dim)
    )

    pil_image = pil_image.resize((224, 224))
    img = np.array(pil_image).astype("float32") / 255.0
    original = img.copy()
    img = np.expand_dims(img, axis=0)

    return img, original

# ============================
# Grad‑CAM implementation (experimental)
# ============================
def generate_gradcam(img_array, original_img):
    if LAST_CONV_LAYER is None:
        raise RuntimeError("No convolutional layer found for Grad‑CAM.")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(LAST_CONV_LAYER).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_img = np.uint8(original_img * 255)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return overlay

# ============================
# UI
# ============================
st.markdown("---")
st.write("Upload an MRI image to predict whether a brain tumor is present.")

uploaded_file = st.file_uploader(
    "Upload MRI image",
    type=["jpg", "jpeg", "png"]
)

consent = st.checkbox(
    "I understand this tool is for educational purposes only "
    "and not a medical diagnosis."
)

submitted = st.button("Submit")

# ============================
# Prediction logic
# ============================
if submitted:
    if not uploaded_file:
        st.error("Please upload an image.")

    elif not consent:
        st.error("Please agree to the consent statement.")

    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", width=350)

        img_array, original_img = preprocess_image(image)
        predictions = model.predict(img_array)[0]

        st.markdown("### Raw prediction:")
        st.write(predictions)

        idx = int(np.argmax(predictions))
        label = CLASS_NAMES[idx]
        confidence = predictions[idx] * 100

        if idx == 1:
            st.success(f"✅ Result: {label}")
        else:
            st.info(f"ℹ️ Result: {label}")

        st.markdown(f"**Confidence:** {confidence:.2f}%")

        # ============================
        # Grad‑CAM attempt
        # ============================
        st.markdown("---")
        st.markdown("### 🔍 Grad‑CAM Visualization (Experimental)")

        try:
            heatmap = generate_gradcam(img_array, original_img)
            st.image(
                heatmap,
                caption="Grad‑CAM Heatmap Highlighting Influential Regions",
                width=350
            )
            st.caption("Grad‑CAM generated successfully.")

        except Exception as e:
            st.warning(
                "Grad‑CAM could not be generated for this model.\n\n"
                f"Reason: {str(e)}"
            )

# ============================
# Footer
# ============================
st.markdown("---")
st.caption(
    "NeuraScan is an educational AI demonstration. "
    "Not intended for clinical or diagnostic use."
)