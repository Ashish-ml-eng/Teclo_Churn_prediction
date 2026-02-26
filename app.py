import streamlit as st
import pandas as pd
import joblib

# ── Load saved artifacts ──────────────────────────────────────────────────────
model           = joblib.load('trained_model.pkl')
t_encoder       = joblib.load('T_encoder.pkl')
b_encoder       = joblib.load('B_encoder.pkl')
s_scaler        = joblib.load('S_scaler.pkl')
selected_columns = joblib.load('selected_columns.pkl')

# These match exactly what was used during training in Telco_churn.ipynb
bin_num_col     = b_encoder.cols          
categorical_cols = t_encoder.cols         
num_cols        = s_scaler.feature_names_in_.tolist()  


st.set_page_config(page_title="📞 Telco Churn Prediction", layout="centered")
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Predict if a customer is likely to churn based on their subscription details.")

# ── Inputs ─────────────────────────────────────────────────────────────
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender          = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen  = st.selectbox('Senior Citizen', [0, 1])
    partner         = st.selectbox('Partner', ['Yes', 'No'])
    dependents      = st.selectbox('Dependents', ['Yes', 'No'])
    tenure          = st.number_input('Tenure (months)', min_value=0, max_value=72, value=12)
    phone_service   = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines  = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['Fiber optic', 'DSL', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])

with col2:
    online_backup       = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection   = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support        = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv        = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    streaming_movies    = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    contract            = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing   = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method      = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges     = st.number_input('Monthly Charges ($)', min_value=18.25, max_value=118.75, value=65.0)
    total_charges       = st.number_input('Total Charges ($)', min_value=18.8, max_value=8684.8, value=1000.0)

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(inputs: pd.DataFrame):
    
    inputs[categorical_cols] = t_encoder.transform(inputs[categorical_cols])

    bin_encoded_df = pd.DataFrame(
        b_encoder.transform(inputs[bin_num_col]),
        columns=b_encoder.get_feature_names_out(bin_num_col),
        index=inputs.index
    )

    scaled_df = pd.DataFrame(
        s_scaler.transform(inputs[num_cols]),
        columns=num_cols,
        index=inputs.index
    )

    # final dataframe
    X = pd.concat([
        inputs[categorical_cols],
        bin_encoded_df,
        scaled_df
    ], axis=1)

    X = X.reindex(columns=selected_columns)

    prediction = model.predict(X)

    if prediction[0] == 0:
        st.success("✅ This customer is **NOT likely to churn**.")
    else:
        st.error("⚠️ This customer is **likely to CHURN**. Consider a retention offer.")


if st.button("Predict Churn"):
    input_dict = {
        'gender':           gender,
        'SeniorCitizen':    senior_citizen,
        'Partner':          partner,
        'Dependents':       dependents,
        'tenure':           tenure,
        'PhoneService':     phone_service,
        'MultipleLines':    multiple_lines,
        'InternetService':  internet_service,
        'OnlineSecurity':   online_security,
        'OnlineBackup':     online_backup,
        'DeviceProtection': device_protection,
        'TechSupport':      tech_support,
        'StreamingTV':      streaming_tv,
        'StreamingMovies':  streaming_movies,
        'Contract':         contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod':    payment_method,
        'MonthlyCharges':   monthly_charges,
        'TotalCharges':     total_charges,
    }
    inputs = pd.DataFrame([input_dict])
    predict(inputs)