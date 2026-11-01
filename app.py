
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

st.info("This is an AI-assisted prediction, not a confirmed agricultural diagnosis.")



# GPS Location Detection

from streamlit_geolocation import streamlit_geolocation
import requests

st.divider()
st.subheader("📍 Farmer Location Detection")

st.write("Click below and allow location access.")

location = streamlit_geolocation()

if location and location.get("latitude") is not None:

    latitude = location["latitude"]
    longitude = location["longitude"]

    st.success("Location detected successfully!")

    st.write(f"Latitude: {latitude}")
    st.write(f"Longitude: {longitude}")

    # Weather Information

    st.subheader("🌦️ Local Weather Information")

    if st.button("Get Weather Details"):

        url = "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain",
            "timezone": "auto"
        }

        try:
            response = requests.get(
                url,
                params=params,
                timeout=20
            )

            response.raise_for_status()

            weather = response.json()
            current = weather["current"]

            st.success("Weather data retrieved successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Temperature", f'{current["temperature_2m"]} °C')
                st.metric("Humidity", f'{current["relative_humidity_2m"]} %')

            with col2:
                st.metric("Rainfall", f'{current["precipitation"]} mm')
                st.metric("Rain", f'{current["rain"]} mm')

            st.caption("Weather data source: Open-Meteo")
            # Environmental Disease Risk Prediction

st.divider()
st.subheader("🌱 Environmental Disease Risk Prediction")

temperature = current["temperature_2m"]
humidity = current["relative_humidity_2m"]
rainfall = current["precipitation"]

risk_score = 0

# Humidity contribution
if humidity >= 85:
    risk_score += 40
elif humidity >= 70:
    risk_score += 25
elif humidity >= 60:
    risk_score += 10

# Temperature contribution
if 25 <= temperature <= 32:
    risk_score += 30
elif 20 <= temperature < 25 or 32 < temperature <= 35:
    risk_score += 15

# Rainfall contribution
if rainfall >= 5:
    risk_score += 30
elif rainfall > 0:
    risk_score += 15

risk_score = min(risk_score, 100)

if risk_score >= 70:
    risk_level = "High"
    st.error("⚠️ High environmental risk")
elif risk_score >= 40:
    risk_level = "Moderate"
    st.warning("⚠️ Moderate environmental risk")
else:
    risk_level = "Low"
    st.success("✅ Low environmental risk")

st.metric("Environmental Risk Score", f"{risk_score}/100")
st.write("Risk Level:", risk_level)

st.caption(
    "This is an initial weather-based indicator, "
    "not a confirmed disease probability."
)

        except Exception as e:
            st.error(f"Unable to retrieve weather: {e}")

else:
    st.info("Please allow location access to detect your location.")
