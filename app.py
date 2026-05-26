
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("titanic_model.pkl")

st.set_page_config(page_title="Titanic Survival Predictor")

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Gender", ["male", "female"])

age = st.slider("Age", 1, 100, 25)

sibsp = st.number_input("Siblings / Spouses", 0, 10, 0)

parch = st.number_input("Parents / Children", 0, 10, 0)

fare = st.number_input("Fare", 0.0, 600.0, 50.0)

embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# Encode
sex = 1 if sex == "male" else 0

embarked_map = {'C': 0, 'Q': 1, 'S': 2}
embarked = embarked_map[embarked]

# Predict
if st.button("Predict Survival"):

    data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did NOT Survive")
