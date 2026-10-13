
import sys
from pathlib import Path

import streamlit as st

# Add the project root so we can import src.predict
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_churn


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    # page_icon="📊",
    layout="wide"
)

# Application heading
st.title("Customer Churn Prediction")
st.write(
    "Estimate the likelihood that a telecom customer "
    "will leave the service using a trained ML model."
)

st.info(
    "This application provides an estimate, not a guarantee "
    "of future customer behavior."
)

# Customer information form
with st.form("customer_form"):

    st.header("1. Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Partner", ["Yes", "No"])

    with col2:
        dependents = st.selectbox(
            "Dependents", ["Yes", "No"]
        )
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

    with col3:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    st.header("2. Internet Services")

    col1, col2, col3 = st.columns(3)

    with col1:
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    st.header("3. Contract and Billing")

    col1, col2 = st.columns(2)

    with col1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col2:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=840.0,
            step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    )


# Run prediction when the user submits the form
if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        result = predict_churn(customer)

        st.header("4. Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if result["prediction"] == "Churn":
                st.error("Prediction: Churn")
            else:
                st.success("Prediction: No Churn")

        with col2:
            probability = result["churn_probability"]
            st.metric(
                "Estimated Churn Probability",
                f"{probability:.1%}"
            )
            st.progress(probability)

        st.caption(
            "The probability is a model estimate based on learned "
            "patterns. It is not a guarantee and may not be "
            "well-calibrated to real-world churn rates."
        )

    except (ValueError, KeyError) as error:
        st.error(f"Unable to make prediction: {error}")
