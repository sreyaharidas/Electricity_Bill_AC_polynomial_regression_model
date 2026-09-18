import streamlit as st
import joblib
import pandas as pd

# Load the saved Polynomial Regression model
poly = joblib.load("polynomial_features.pkl")
model = joblib.load("polynomial_regression_model.pkl")

st.title("Electric Bill Prediction")
st.write("Predict the electricity bill based on AC units and Fan units.")

# User input
AC_Units = st.number_input(
    "Enter Number of AC Units",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

Fan_Units = st.number_input(
    "Enter Number of FAN Units",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

if AC_Units < 10 or AC_Units > 105:
    st.error("AC units must be between 10 and 105")

elif Fan_Units < 20 or Fan_Units > 115:
    st.error("Fan units must be between 20 and 115")

else:

    # Prediction button
    if st.button("Predict"):

        new_data = pd.DataFrame({
            "AC_Units": [AC_Units],
            "Fan_Units": [Fan_Units]
        })

        # Convert input to polynomial features
        new_data_poly = poly.transform(new_data)

        # Make prediction
        prediction = model.predict(new_data_poly)

        st.success(f"Predicted Electric Bill: ₹{prediction[0]:.2f}")
