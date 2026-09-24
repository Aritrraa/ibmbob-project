"""Streamlit frontend for the house price prediction API."""

import requests
import streamlit as st


st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Predictor")
st.caption("Estimate a property price using the trained machine-learning pipeline.")

with st.form("property_form"):
    area = st.number_input("Area", min_value=500, max_value=50000, value=7420, step=100)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=1, step=1)
    stories = st.number_input("Stories", min_value=1, max_value=10, value=2, step=1)
    parking = st.number_input("Parking spaces", min_value=0, max_value=10, value=1, step=1)
    col1, col2 = st.columns(2)
    with col1:
        mainroad = st.selectbox("Main road access", ["yes", "no"])
        guestroom = st.selectbox("Guest room", ["no", "yes"])
        basement = st.selectbox("Basement", ["no", "yes"])
        hotwaterheating = st.selectbox("Hot-water heating", ["no", "yes"])
    with col2:
        airconditioning = st.selectbox("Air conditioning", ["no", "yes"])
        prefarea = st.selectbox("Preferred area", ["no", "yes"])
        furnishingstatus = st.selectbox("Furnishing status", ["semi-furnished", "unfurnished", "furnished"])
    submitted = st.form_submit_button("Predict price")

if submitted:
    payload = {
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "stories": stories,
        "mainroad": mainroad, "guestroom": guestroom, "basement": basement,
        "hotwaterheating": hotwaterheating, "airconditioning": airconditioning,
        "parking": parking, "prefarea": prefarea, "furnishingstatus": furnishingstatus,
    }
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=payload, timeout=10)
        if response.ok:
            result = response.json()
            st.success(f"Estimated price: {result['predicted_price']:,.0f}")
        else:
            st.error(response.json().get("error", "The API returned an error."))
    except requests.RequestException:
        st.error("Could not reach the Flask API. Start it with: python app.py")
