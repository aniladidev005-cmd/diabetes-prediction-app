
import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------------------------
# 2. Load the trained model and scaler
# --------------------------------------------------
@st.cache_resource
def load_files():
    with open("diabetes_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler


try:
    model, scaler = load_files()
except Exception as error:
    st.error(
        "Could not load the model or scaler. "
        "Check that diabetes_model.pkl and scaler.pkl "
        "are in the same folder as app.py."
    )
    st.stop()

# --------------------------------------------------
# 3. App title and description
# --------------------------------------------------
st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the health measurements below to obtain "
    "a machine-learning prediction."
)

st.info(
    "Educational project only. This tool cannot diagnose "
    "diabetes or replace advice from a qualified healthcare professional."
)

# --------------------------------------------------
# 4. User input form
# --------------------------------------------------
with st.form("diabetes_form"):

    st.subheader("Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

        glucose = st.number_input(
            "Glucose Level",
            min_value=0.0,
            max_value=500.0,
            value=120.0,
            step=1.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=300.0,
            value=70.0,
            step=1.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=150.0,
            value=20.0,
            step=1.0
        )

    with col2:
        insulin = st.number_input(
            "Insulin Level",
            min_value=0.0,
            max_value=1000.0,
            value=79.0,
            step=1.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.1
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01
        )

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=30,
            step=1
        )

    submitted = st.form_submit_button(
        "Predict Diabetes Risk",
        use_container_width=True
    )

# --------------------------------------------------
# 5. Process inputs and make prediction
# --------------------------------------------------
if submitted:

    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    try:
        # Match the feature names and order used during training.
        if hasattr(scaler, "feature_names_in_"):
            expected_features = list(scaler.feature_names_in_)
            input_data = input_data[expected_features]

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if int(prediction) == 1:
            st.warning(
                "The model predicts a positive diabetes classification."
            )
        else:
            st.success(
                "The model predicts a negative diabetes classification."
            )

        # Display the model's estimated class probability if supported.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_data)[0]
            positive_index = list(model.classes_).index(1)
            probability = probabilities[positive_index] * 100

            st.metric(
                "Model-estimated probability for the positive class",
                f"{probability:.2f}%"
            )

            st.caption(
                "This percentage is a model output, not a confirmed "
                "medical risk or diagnosis."
            )

        st.caption(
            "If you have concerns about diabetes, consult a healthcare "
            "professional for appropriate testing and interpretation."
        )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}. "
            "Check that the model, scaler, and input features match "
            "those used when training."
        )
