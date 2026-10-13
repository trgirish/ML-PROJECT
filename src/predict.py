
from pathlib import Path

import joblib
import pandas as pd


# Locate the saved model relative to this Python file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

# Load the trained preprocessing + model pipeline
model = joblib.load(MODEL_PATH)

# Get the exact feature names used during training
EXPECTED_FEATURES = list(model.feature_names_in_)


def predict_churn(customer_data):
    """
    Predict customer churn from a dictionary of customer details.

    Returns:
        dict: Predicted class and estimated churn probability.
    """

    # Convert input dictionary to a one-row DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Check that every required feature is present
    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in customer_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Keep only the features used during training,
    # in the exact training order
    customer_df = customer_df[EXPECTED_FEATURES]

    # Reject missing input values
    if customer_df.isnull().any().any():
        raise ValueError(
            "Customer details cannot contain missing values."
        )

    # Make prediction
    prediction = int(model.predict(customer_df)[0])

    # Get probability of class 1 (Churn)
    churn_probability = float(
        model.predict_proba(customer_df)[0, 1]
    )

    # Return readable results
    return {
        "prediction": (
            "Churn" if prediction == 1 else "No Churn"
        ),
        "churn_probability": round(
            churn_probability, 4
        ),
        "churn_probability_percent": round(
            churn_probability * 100, 2
        )
    }
