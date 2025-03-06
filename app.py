import streamlit as st
import pandas as pd 
import sys 
from src.exception import CustomException
from src.pipeline.predict_pipeline import predictpipeline

# Tile of the App
st.title("Car Failure Prediction Model")

st.write(" Model is useful ...")

def get_user_input():
    fuel_consumption = st.number_input("Fuel consumption", min_value=0.0, max_value=10000.0, value=0.0)
    Temperature =  st.number_input("Temperature", min_value=0.0, max_value=1000.0, value=0.0)
    Usage = st.selectbox("Usage", options=["Medium", "Low", "High"], index=0)
    RPM = st.number_input("RPM", min_value=0.0, max_value=10000.0, value=0.0)
    Mileage = st.number_input("Mileage", min_value=0.0, max_value=100000000.0, value=0.0)
    Age_of_car = st.number_input("Age_of_car", min_value=0.0, max_value= 100.0, value=0.0)
    Color = st.selectbox("Color", options=["Black", "Blue", "Grey", "White", "Red"], index=0)
    Membership = st.selectbox("Membership", options=["Normal", "Premium"], index=0)


    data = {
        "Fuel consumption": fuel_consumption,
        "Temperature": Temperature,
        "Usage": Usage,
        "RPM": RPM,
        "Mileage" : Mileage,
        "Age_of_car": Age_of_car,
        "Model": "Model 5, 2018",
        "Color": Color,
        "Factory": "New York, U.S",
        "Membership": Membership

            }

    features = pd.DataFrame(data, index=[0])
    return features

user_input = get_user_input() #Get the user input data

st.header("User Input Features")
st.write(user_input)  #Display Input data

if st.button("Predict"):
    pipeline = predictpipeline()

    try:
        #Get prediction
        prediction = pipeline.predict(user_input)
        st.subheader("Prediction")
        st.write(prediction[0]) # Model Output
    except Exception as e:
        raise CustomException(e,sys)
