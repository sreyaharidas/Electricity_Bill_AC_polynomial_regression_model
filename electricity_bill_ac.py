import streamlit as st
import joblib
import pandas as pd

# Load the saved Polynomial Regression model
model = joblib.load("polynomial_regression_model(1).pkl")

st.title("Electric Bill Prediction")
st.write("Predict the electricity bill based on AC units.")

# User input
ac_units = st.number_input(
    "Enter Number of AC Units",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

# Prediction button
if st.button("Predict"):
    new_data = pd.DataFrame({
        "AC_Units": [ac_units]
    })

    # Convert input to polynomial features
    new_data_poly = poly.transform(new_data)

    # Make prediction
    prediction = model.predict(new_data_poly)

    st.success(f"Predicted Electric Bill: ₹{prediction[0]:.2f}")

