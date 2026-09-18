import streamlit as st
import joblib
import pandas as pd

# Load the saved Polynomial Regression model
poly = joblib.load("polynomial_features.pkl")
model = joblib.load("polynomial_regression_model.pkl")
st.title("Electric Bill Prediction")
st.write("Predict the electricity bill based on AC units.")

# User input
ac_units = st.number_input(
    "Enter Number of AC Units",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)
fan_units = st.number_input(
    "Enter Number of FAN Units",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)
if ac_units<10 or ac_units>105:
    st.error("AC units must be between 10 and 105")
elif fan_units<20 or fan_units>115:
    st.error("Fan units must be between 20 and 115")
# Prediction button
else:
    new_data = pd.DataFrame({
        "AC_Units": [ac_units]
    })

    # Convert input to polynomial features
    new_data_poly = poly.transform(new_data)

    # Make prediction
    prediction = model.predict(new_data_poly)

    st.success(f"Predicted Electric Bill: ₹{prediction[0]:.2f}")

