import streamlit as st
from pathlib import Path
import joblib

# -----------------------------
# Project paths
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "xgb_model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.pkl"
SHAP_DATA_PATH = ARTIFACTS_DIR / "shap_data.pkl"


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Interest Rate Prediction",
    page_icon="💰",
    layout="wide"
)


# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, preprocessor, feature_names

model, preprocessor, feature_names = load_artifacts()


# -----------------------------
# Home page
# -----------------------------
st.title("💰 Loan Interest Rate Prediction")

st.markdown(
    """
    ### XGBoost + SHAP Explainable AI

    This application predicts loan interest rates using borrower,
    loan, and credit-related characteristics.

    The model is built using **XGBoost Regression** and explained
    using **SHAP (SHapley Additive exPlanations)**.
    """
)

st.success("Model artifacts loaded successfully!")

st.write("Number of transformed features:", len(feature_names))

st.header("🔮 Loan Interest Rate Prediction")

st.write("Enter the borrower's loan and credit information.")

col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=500.0,
        value=10000.0,
        step=500.0
    )

    funded_amnt = st.number_input(
        "Funded Amount",
        min_value=500.0,
        value=10000.0,
        step=500.0
    )

    funded_amnt_inv = st.number_input(
        "Funded Amount Investor",
        min_value=500.0,
        value=10000.0,
        step=500.0
    )

    term = st.selectbox(
        "Loan Term",
        [" 36 months", " 60 months"]
    )

    sub_grade = st.selectbox(
        "Sub Grade",
        ["A1", "A2", "A3", "A4", "A5",
         "B1", "B2", "B3", "B4", "B5",
         "C1", "C2", "C3", "C4", "C5",
         "D1", "D2", "D3", "D4", "D5",
         "E1", "E2", "E3", "E4", "E5",
         "F1", "F2", "F3", "F4", "F5",
         "G1", "G2", "G3", "G4", "G5"])

with col2:
    annual_inc = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    dti = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        max_value=100.0,
        value=15.0
    )

    revol_util_num = st.number_input(
        "Revolving Utilization (%)",
        min_value=0.0,
        max_value=150.0,
        value=40.0
    )

    revol_bal = st.number_input(
        "Revolving Balance",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )

    emp_length_years = st.number_input(
        "Employment Length (Years)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    )

    st.subheader("👤 Borrower & Credit Information")

col1, col2, col3 = st.columns(3)

with col1:
    home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "MORTGAGE", "OWN", "OTHER", "NONE"]
    )

    verification_status = st.selectbox(
        "Verification Status",
        ["Not Verified", "Verified", "Source Verified"]
    )

    purpose = st.selectbox(
        "Loan Purpose",
        [
            "credit_card",
            "car",
            "small_business",
            "other",
            "wedding",
            "debt_consolidation",
            "home_improvement",
            "major_purchase",
            "medical",
            "moving",
            "vacation",
            "house",
            "renewable_energy",
            "educational"
        ]
    )

with col2:
    addr_state = st.selectbox(
        "State",
        [
            "CA", "NY", "FL", "TX", "NJ", "IL", "PA",
            "VA", "GA", "MA", "OH", "MD", "AZ", "WA",
            "CO", "NC", "CT", "MI", "MO", "MN", "NV",
            "OR", "AL", "WI", "SC", "LA", "KY", "OK",
            "KS", "UT", "AR", "DC", "RI", "NM", "WV",
            "HI", "NH", "DE", "MT", "WY", "AK", "MS",
            "TN", "IA", "NE", "ID", "ME", "VT", "SD",
            "IN"
        ]
    )

    delinq_2yrs = st.number_input(
        "Delinquencies (Last 2 Years)",
        min_value=0,
        value=0,
        step=1
    )

    inq_last_6mths = st.number_input(
        "Credit Inquiries (Last 6 Months)",
        min_value=0,
        value=0,
        step=1
    )

with col3:
    open_acc = st.number_input(
        "Open Credit Accounts",
        min_value=0,
        value=5,
        step=1
    )

    pub_rec = st.number_input(
        "Public Records",
        min_value=0,
        value=0,
        step=1
    )

    total_acc = st.number_input(
        "Total Credit Accounts",
        min_value=0,
        value=15,
        step=1
    )

    pub_rec_bankruptcies = st.number_input(
        "Public Record Bankruptcies",
        min_value=0,
        value=0,
        step=1
    )

    st.subheader("📅 Credit History & Loan Date")

col1, col2 = st.columns(2)

with col1:
    issue_year = st.number_input(
        "Loan Issue Year",
        min_value=2007,
        max_value=2026,
        value=2011,
        step=1
    )

    issue_month = st.number_input(
        "Loan Issue Month",
        min_value=1,
        max_value=12,
        value=12,
        step=1
    )

with col2:
    credit_history_years = st.number_input(
        "Credit History (Years)",
        min_value=0.0,
        max_value=60.0,
        value=10.0,
        step=0.5
    )

    st.divider()

predict_button = st.button(
    "🔮 Predict Interest Rate",
    type="primary"
)

if predict_button:

    input_data = {
        "loan_amnt": loan_amnt,
        "funded_amnt": funded_amnt,
        "funded_amnt_inv": funded_amnt_inv,
        "term": term,
        "sub_grade": sub_grade,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "addr_state": addr_state,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "total_acc": total_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "revol_util_num": revol_util_num,
        "emp_length_years": emp_length_years,
        "issue_year": issue_year,
        "issue_month": issue_month,
        "credit_history_years": credit_history_years
    }

    import pandas as pd

    input_df = pd.DataFrame([input_data])

    try:
        transformed_input = preprocessor.transform(input_df)

        prediction = model.predict(transformed_input)[0]

        st.success("Prediction generated successfully!")

        st.metric(
            "Predicted Interest Rate",
            f"{prediction:.2f}%"
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)