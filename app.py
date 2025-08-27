import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the exported PyCaret model
model = load_model(r'wheat_classifier')

# App title
st.title("Wheat Type Classification")

# Input fields for features used in training
area = st.number_input("Area", min_value=0.0)
perimeter = st.number_input("Perimeter", min_value=0.0)
compactness = st.number_input("Compactness", min_value=0.0)
asymmetry_coeff = st.number_input("Asymmetry Coefficient", min_value=0.0)
groove = st.number_input("Groove", min_value=0.0)

# Predict button
if st.button("Predict"):
    # Prepare input dataframe
    input_data = pd.DataFrame({
        'Area': [area],
        'Perimeter': [perimeter],
        'Compactness': [compactness],
        'AsymmetryCoeff': [asymmetry_coeff],
        'Groove': [groove]
    })

    # Make prediction
    prediction = predict_model(model, data=input_data)

    # Debug: show the whole prediction dataframe (optional)
    st.write(prediction)

    # Get predicted class (label)
    st.write(prediction.columns)

