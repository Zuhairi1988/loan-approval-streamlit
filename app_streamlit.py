# app_streamlit.py
import streamlit as st
import joblib
import numpy as np

# ==== load model & scaler (pastikan file ada di folder yang sama) ====
scaler = joblib.load("Scaler.pkl")
model = joblib.load("model.pkl")

st.title("Loan Approval Prediction")

st.markdown("""
Masukkan ciri-ciri pemohon. Contoh fitur: x1..x5.
(Ubah input control ikut nama ciri sebenar: Gender/Married/ApplicantIncome/Gender etc.)
""")

# contoh input — tukar ikut feature sebenar
x1 = st.selectbox("Credit History (1 = Good, 0 = Bad)", options=[1,0], index=0)
x2 = st.selectbox("Property Area (Urban=2, Semiurban=1, Rural=0)", options=[2,1,0], index=2)
x3 = st.selectbox("Married (1 = Married, 0 = Not Married)", options=[1,0], index=0)
x4 = st.number_input("Applicant Income (RM)", min_value=0, value=3000)
x5 = st.number_input("Loan Amount (RM)", min_value=0, value=100000)

# convert to numpy array + scale
input_array = np.array([[x1, x2, x3, x4, x5]], dtype=float)
scaled = scaler.transform(input_array)
pred = model.predict(scaled)[0]
pred_int = int(pred)

# show result
if st.button("Predict"):
    st.write("Prediction (raw):", pred)
    if pred_int == 1:
        st.success("Loan APPROVED (1)")
    else:
        st.error("Loan NOT APPROVED (0)")

# optional: show probability if model supports predict_proba
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(scaled)[0]
    st.write(f"Probability (class 0 / 1): {prob}")

