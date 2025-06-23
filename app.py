#import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,classification_report
import joblib
from tensorflow import keras
import streamlit as st

# Load the model
model = keras.models.load_model('Diabetic_model.h5')
# Load the scaler
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("Diabetes Prediction App")
st.markdown("enter the following details to predict diabetes")
# Input fields for user data
pregnancies = st.number_input("enter the no. of Pregnancies", min_value=0, max_value=10,value=1)
glucose = st.number_input("enter the Glucose level", min_value=0)
blood_pressure = st.number_input("enter the Blood Pressure", min_value=0)
skin_thickness = st.number_input("enter the Skin Thickness", min_value=0)
insulin = st.number_input("enter the amount of  Insulin in patient body", min_value=0)
bmi = st.number_input("enter the BMI", min_value=1)
diabetespedigreefunction = st.number_input("enter the Diabetes Pedigree Function", min_value=0)
age = st.number_input("enter the Age", min_value=0)

# Button to trigger prediction
if st.button("Predict Diabetes"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    result = "Not Diabetic" if prediction < 0.5 else "Diabetic"

    st.subheader("result of the prediction are ",result)