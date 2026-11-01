import requests
from streamlit_geolocation import streamlit_geolocation
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
       
        # Detailed disease information
        disease_details = {
            "Healthy": {
                "Symptoms": "No visible disease symptoms detected.",
                "Possible Causes": "Not applicable. Continue regular crop monitoring.",
                "Management Tips": "Maintain balanced nutrition, proper irrigation and regular field inspections.",
                "Expert Advice": "Continue monitoring the crop throughout the growing season."
            },

            "brown_spot": {
                "Symptoms": "Small circular or oval brown spots may appear on rice leaves.",
                "Possible Causes": "The disease is caused by a fungus. Poor soil fertility and environmental stress may increase its severity.",
                "Management Tips": "Maintain balanced soil nutrition, use healthy seeds and follow recommended field management practices.",
                "Expert Advice": "Consult an agricultural expert if symptoms spread or become severe."
            },

            "bacterial_leaf_blight": {
                "Symptoms": "Water-soaked lesions and yellowing may appear along leaf edges.",
                "Possible Causes": "Caused by the bacterium Xanthomonas oryzae pv. oryzae.",
                "Management Tips": "Use resistant varieties and follow recommended water and field management practices.",
                "Expert Advice": "Seek agricultural guidance if the disease spreads across the field."
            },

            "leaf_blast": {
                "Symptoms": "Spindle-shaped lesions with grey or whitish centres may appear on leaves.",
                "Possible Causes": "Caused by the fungus Magnaporthe oryzae.",
                "Management Tips": "Avoid excessive nitrogen application and use resistant varieties where available.",
                "Expert Advice": "Consult an agricultural expert for suitable disease management."
            },

            "leaf_scald": {
                "Symptoms": "Leaves may develop brown or straw-coloured areas starting near the tips.",
                "Possible Causes": "Caused by the fungus Microdochium oryzae.",
                "Management Tips": "Maintain balanced fertilizer application and follow good crop management.",
                "Expert Advice": "Get expert guidance if leaf damage increases."
            },

            "narrow_brown_spot": {
                "Symptoms": "Narrow, elongated brown lesions may appear on rice leaves.",
                "Possible Causes": "Caused by the fungus Cercospora janseana.",
                "Management Tips": "Maintain balanced crop nutrition and monitor affected plants.",
                "Expert Advice": "Consult an agricultural expert for confirmation and management."
            },

            "neck_blast": {
                "Symptoms": "Dark lesions may develop around the neck of the rice panicle.",
                "Possible Causes": "Caused by the fungus Magnaporthe oryzae.",
                "Management Tips": "Use resistant varieties and follow recommended crop protection practices.",
                "Expert Advice": "Seek prompt expert advice because panicle infection can affect grain production."
            },

            "rice_hispa": {
                "Symptoms": "Leaves may show white streaks and scraped areas caused by insect feeding.",
                "Possible Causes": "Feeding damage from rice hispa beetles and their larvae.",
                "Management Tips": "Monitor fields for beetles and damaged leaves. Follow locally recommended pest management.",
                "Expert Advice": "Consult an agricultural officer if insect damage increases."
            },

            "sheath_blight": {
                "Symptoms": "Oval or irregular greenish-grey lesions may develop on leaf sheaths.",
                "Possible Causes": "Caused by the fungus Rhizoctonia solani.",
                "Management Tips": "Avoid excessive nitrogen and maintain appropriate plant spacing.",
                "Expert Advice": "Seek expert guidance if lesions spread to upper plant parts."
            },

            "tungro": {
                "Symptoms": "Plants may show yellow or orange leaf discoloration and stunted growth.",
                "Possible Causes": "A viral disease transmitted by green leafhoppers.",
                "Management Tips": "Monitor leafhopper populations and follow locally recommended vector management.",
                "Expert Advice": "Consult an agricultural expert for confirmation and suitable management."
            }
        }

        st.subheader("🌾 Detailed Disease Information")

        details = disease_details.get(predicted_disease)

        if details:
            for heading, information in details.items():
                st.markdown(f"**{heading}:** {information}")
        else:
            st.info("Please consult an agricultural expert for further guidance.")

        st.info("This is an AI-assisted prediction, not a confirmed agricultural diagnosis.")
        # GPS Location Detection
        st.divider()
        st.subheader("📍 Farmer Location Detection")

        st.write("Click below and allow location access.")

        location = streamlit_geolocation()

        if location and location.get("latitude") is not None:
            latitude = location["latitude"]
            longitude = location["longitude"]

            st.success("Location detected successfully!")
            st.write("Latitude:", latitude)
            st.write("Longitude:", longitude)

            if st.button("Get Weather Details"):
                try:
                    weather_url = "https://api.open-meteo.com/v1/forecast"

                    params = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "current": "temperature_2m,relative_humidity_2m,precipitation,rain",
                        "timezone": "auto"
                    }

                    response = requests.get(
                        weather_url,
                        params=params,
                        timeout=20
                    )

                    response.raise_for_status()
                    weather = response.json()["current"]

                    temperature = weather["temperature_2m"]
                    humidity = weather["relative_humidity_2m"]
                    rainfall = weather["precipitation"]
                    rain = weather["rain"]

                    st.subheader("🌦️ Current Weather Details")
                    st.write("Temperature:", temperature, "°C")
                    st.write("Humidity:", humidity, "%")
                    st.write("Rainfall:", rainfall, "mm")
                    st.write("Rain:", rain, "mm")

                    # Initial environmental risk indicator
                    risk_score = 0

                    if humidity >= 85:
                        risk_score += 40
                    elif humidity >= 70:
                        risk_score += 25
                    elif humidity >= 60:
                        risk_score += 10

                    if 25 <= temperature <= 32:
                        risk_score += 30
                    elif 20 <= temperature < 25 or 32 < temperature <= 35:
                        risk_score += 15

                    if rainfall >= 5:
                        risk_score += 30
                    elif rainfall > 0:
                        risk_score += 15

                    risk_score = min(risk_score, 100)

                    st.subheader("🌱 Environmental Risk Indicator")
                    st.write("Risk Score:", risk_score, "/100")

                    if risk_score < 40:
                        st.success("Low Environmental Risk")
                    elif risk_score < 70:
                        st.warning("Moderate Environmental Risk")
                    else:
                        st.error("High Environmental Risk")

                    st.caption(
                        "This is an initial rule-based environmental indicator, "
                        "not a scientifically validated disease probability."
                    )

                except Exception as e:
                    st.error(f"Unable to retrieve weather: {e}")

        else:
            st.info("Please allow location access to detect your location.")


