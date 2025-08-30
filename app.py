import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the trained model
model = load_model(r"wheat_classifier")  # adjust path if needed

st.title("🌾 Wheat Type Classification App")

# Mode selection
mode = st.radio("Choose input mode:", ["Enter all features", "Use Length & Width calculator"])

if mode == "Enter all features":
    # Full feature input
    area = st.number_input("Area", min_value=0.0)
    perimeter = st.number_input("Perimeter", min_value=0.0)
    compactness = st.number_input("Compactness", min_value=0.0)
    length = st.number_input("Length", min_value=0.0)
    width = st.number_input("Width", min_value=0.0)
    asymmetry = st.number_input("Asymmetry Coefficient", min_value=0.0)
    groove = st.number_input("Groove Length", min_value=0.0)

else:
    # Calculator mode
    length = st.number_input("Length", min_value=0.0)
    width = st.number_input("Width", min_value=0.0)

    # Compute area and perimeter automatically
    area = length * width
    perimeter = 2 * (length + width)

    st.write(f"📐 Calculated Area: **{area:.2f}**")
    st.write(f"📏 Calculated Perimeter: **{perimeter:.2f}**")

    # Disclaimer about calculation accuracy
    st.warning(
        "⚠️ Disclaimer: The area of a wheat seed depends on its shape. "
        "This calculator uses a simple Length × Width formula, which may not be accurate "
        "for irregular or curved seed shapes. Use the results as an approximation only."
    )

    compactness = st.number_input("Compactness", min_value=0.0)
    asymmetry = st.number_input("Asymmetry Coefficient", min_value=0.0)
    groove = st.number_input("Groove Length", min_value=0.0)

# When user clicks predict
if st.button("Predict Wheat Type"):
    input_df = pd.DataFrame([{
        "Area": area,
        "Perimeter": perimeter,
        "Compactness": compactness,
        "Length": length,
        "Width": width,
        "AsymmetryCoeff": asymmetry,
        "Groove": groove
    }])

    prediction = predict_model(model, data=input_df)
    predicted_class = prediction.loc[0, "Label"]
    predicted_score = prediction.loc[0, "Score"]

    st.success(f"Predicted Wheat Type: **{predicted_class}** (confidence: {predicted_score:.2f})")
