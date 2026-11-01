
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Rice Disease Detection",
    page_icon="🌱"
)

MODEL_PATH = "final_rice_disease_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

class_names = [
    "Healthy",
    "bacterial_leaf_blight",
    "brown_spot",
    "leaf_blast",
    "leaf_scald",
    "narrow_brown_spot",
    "neck_blast",
    "rice_hispa",
    "sheath_blight",
    "tungro"
]

st.title("🌱 Rice Plant Disease Detection")
st.write("Upload a rice leaf image to predict its disease.")

uploaded_file = st.file_uploader(
    "Choose a rice leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Rice Leaf", use_container_width=True)

    if st.button("Predict Disease"):

        resized_image = image.resize((224, 224))

        image_array = np.array(resized_image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        predicted_index = np.argmax(prediction[0])
        predicted_disease = class_names[predicted_index]
        confidence = prediction[0][predicted_index] * 100
        st.subheader("Prediction Result")
        st.success(f"Predicted Disease: {predicted_disease}")
        st.write(f"Confidence: {confidence:.2f}%")
        if confidence < 60:
            st.warning("Low confidence. Please upload a clearer leaf image.")
        # Disease-specific recommendations
        disease_recommendations = {
            "Healthy": "Your rice leaf appears healthy. Continue regular crop monitoring.",
            "bacterial_leaf_blight": "Monitor yellowing and drying of leaf edges. Maintain proper water management.",
            "brown_spot": "Check soil fertility and crop nutrition.",
            "leaf_blast": "Monitor for spindle-shaped leaf lesions.",
            "leaf_scald": "Monitor leaf tips and margins for drying symptoms.",
            "narrow_brown_spot": "Check for narrow brown lesions and maintain balanced crop nutrition.",
            "neck_blast": "Inspect rice necks and panicles for symptoms.",
            "rice_hispa": "Inspect leaves for scraping damage and rice hispa insects.",
            "sheath_blight": "Monitor lesions on leaf sheaths and maintain proper plant spacing.",
            "tungro": "Monitor yellow-orange leaf discoloration and green leafhoppers."
        }

        st.subheader("🌾 Disease-specific Recommendations")
        st.info(disease_recommendations.get(
            predicted_disease,
            "Consult a local agricultural expert for guidance."
        ))

        st.info("This is an AI-assisted prediction, not a confirmed agricultural diagnosis.")
