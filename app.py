import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("salary_model.pkl")
le_workclass = joblib.load("le_workclass.pkl")
le_education = joblib.load("le_education.pkl")
le_marital_status = joblib.load("le_marital-status.pkl")
le_occupation = joblib.load("le_occupation.pkl")
le_relationship = joblib.load("le_relationship.pkl")
le_race = joblib.load("le_race.pkl")
le_gender = joblib.load("le_gender.pkl")
le_native_country = joblib.load("le_native-country.pkl")

# Streamlit page setup
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.markdown("<h2 style='text-align: center;'>💼👨🏻‍💼 Employee Salary Prediction using AIML</h2><b>", unsafe_allow_html=True)
st.markdown("---")

with st.form("prediction_form"):
    st.subheader("📋 Enter Employee Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 90, 30)
        fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=1, value=100000)
        education = st.selectbox("Education", le_education.classes_)
        educational_num = st.slider("Education Number (educational-num)", 1, 16, 10)
        workclass = st.selectbox("Workclass", le_workclass.classes_)

    with col2:
        marital_status = st.selectbox("Marital Status", le_marital_status.classes_)
        occupation = st.selectbox("Occupation", le_occupation.classes_)
        relationship = st.selectbox("Relationship", le_relationship.classes_)
        race = st.selectbox("Race", le_race.classes_)
        gender = st.selectbox("Gender", le_gender.classes_)

    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", le_native_country.classes_)

    submitted = st.form_submit_button("🔍 Predict Salary")

if submitted:
    # Encode categorical features
    encoded_input = {
        "age": age,
        "workclass": le_workclass.transform([workclass])[0],
        "fnlwgt": fnlwgt,
        "education": le_education.transform([education])[0],
        "educational-num": educational_num,
        "marital-status": le_marital_status.transform([marital_status])[0],
        "occupation": le_occupation.transform([occupation])[0],
        "relationship": le_relationship.transform([relationship])[0],
        "race": le_race.transform([race])[0],
        "gender": le_gender.transform([gender])[0],
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": le_native_country.transform([native_country])[0],
    }

    input_df = pd.DataFrame([encoded_input])

    # Make prediction
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]

    st.markdown("---")
    st.markdown(f"### 🎯 Predicted Salary Range: `{'>50K' if prediction == 1 else '<=50K'}`")
    st.progress(min(int(prob * 100), 100), text=f"{prob*100:.2f}% confidence")
