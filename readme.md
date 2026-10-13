# Customer Churn Prediction

## Overview

Customer Churn Prediction is a machine learning project that estimates whether a telecommunications customer is likely to leave a service. It uses customer account details, subscribed services, contract information, and billing features to generate a churn prediction.

The project includes data preprocessing, exploratory data analysis, model training and evaluation, a reusable prediction module, and a Streamlit web application.

## Objectives

- Explore customer demographics, service subscriptions, and billing patterns.
- Prepare data for machine learning.
- Train and compare classification models.
- Evaluate churn detection using precision, recall, F1-score, and ROC-AUC.
- Save a trained model and use it to predict churn for new customer inputs.
- Provide a simple interface for testing predictions.

## Dataset

The project uses the IBM Telco Customer Churn dataset. It contains customer information, including tenure, contract type, internet services, payment method, and charges. The target variable is `Churn`, which indicates whether a customer left the service.

The dataset includes categorical and numerical features. Data preparation includes converting `TotalCharges` to numeric values, handling missing values, separating the target from the input features, and encoding categorical variables through a preprocessing pipeline.

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Perform exploratory data analysis.
3. Clean and prepare the data.
4. Split the data into training and testing sets.
5. Build preprocessing pipelines for numerical and categorical features.
6. Train and compare classification models.
7. Tune selected models using cross-validation.
8. Evaluate the final candidate on held-out test data.
9. Save the fitted model.
10. Use the model in a Streamlit application.

The candidate models include Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. The initial model-selection process prioritizes churn detection while also considering overall model performance.

## Model Evaluation

The tuned Logistic Regression candidate produced the following held-out test metrics in the recorded experiment:

| Metric | Result |
|---|---:|
| Accuracy | 72.57% |
| Precision | 49.01% |
| Recall | 79.68% |
| F1-score | 60.69% |
| ROC-AUC | 83.51% |

The cross-validation F1-score recorded during hyperparameter tuning was approximately 0.6333.

These results are specific to the recorded experiment and its evaluation setup. Update this section if the final model, test split, threshold, or evaluation results change.

Recall measures the proportion of actual churn cases identified by the model. Precision measures the proportion of predicted churn cases that actually churned. The appropriate trade-off depends on the business cost of missed churners versus unnecessary retention outreach.

## Run Locally

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install the required packages using `pip install -r requirements.txt`.
4. From the project root, start the app using `python -m streamlit run app/app.py`.
5. Open the local URL shown in the terminal.

The application loads the saved model from `models/churn_model.joblib`. Ensure that the model file exists and that the installed scikit-learn version is compatible with the version used to train it.

## Project Structure

- `data/` — raw and cleaned datasets
- `notebooks/` — exploratory analysis, model training, and evaluation
- `models/` — serialized trained model
- `src/` — prediction logic
- `app/` — Streamlit user interface
- `requirements.txt` — Python dependencies
- `PROJECT_REPORT.md` — detailed project report

## Limitations

- Results on this dataset may not generalize to every telecommunications provider.
- Model predictions indicate statistical associations, not causal explanations.
- Predicted probabilities may require calibration before being interpreted as real-world churn likelihoods.
- The classification threshold should be selected based on business objectives and retention costs.
- A time-based evaluation and monitoring for changes in customer behavior would strengthen a production implementation.

## Future Improvements

- Tune the classification threshold using validation data and business costs.
- Evaluate probability calibration.
- Test performance on more recent or external customer data.
- Add automated tests and input validation.
- Track model performance and data drift.
- Deploy the application to a hosting platform.

## Responsible Use

Predictions should support, not replace, business judgment. Customer outreach decisions should be evaluated for fairness, privacy, and practical effectiveness.

## Author

Add your name and relevant portfolio or GitHub links here.
