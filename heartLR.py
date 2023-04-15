import pandas as pd
import streamlit as st
import numpy as np
##from tensorflow import keras
import keras
from PIL import Image
import pickle

model=pickle.load(open(r'C:\Users\brill\OneDrive\Documents\DScourse\Heart prediction\logistic_regression_model.pkl', 'rb'))

df=pd.read_csv('heart_failure_clinical_records_dataset.csv')
st.set_page_config(page_title= 'Heart Risk prediction', layout= 'wide')

image =  Image.open(r'C:\Users\brill\OneDrive\Desktop\heart ML.jpg')
st.image(image, use_column_width = True)



age= st.number_input('age', min_value=0,max_value=90)
sex=st.number_input("enter 0 for female , 1 for male ", min_value=0, max_value=1)
anaemia=st.number_input("anaemia 0 for negative ,1 for positive" , min_value=0, max_value=1)
diabetes=st.number_input("diabetes 0 for negative, 1 for positve", min_value=0,max_value=1)
ejection_fraction=st.number_input("ejection_fraction", min_value=0, max_value=80 )
high_blood_pressure=st.number_input("high blood pressure 0 if your pressure is low, 1 if your pressure is high" , min_value=0,max_value=1)
##platelets=st.text_input('platelets')
smoking=st.number_input('smoking 0 if you do not smoke, 1 if you smoke', min_value=0,max_value=1)
serum_creatinine=st.number_input('serum creatinine enter levels between 0.5 and 9.0', min_value=0.5,max_value=9.0)
time=st.number_input("follow up periods days", min_value=0, max_value=350)
creatinine_phosphokinase=st.number_input("creatinine", min_value=0, max_value=11000)
platelets=st.number_input("platelets", min_value=0,max_value=11000)
serum_sodium=st.number_input("serum sodium", min_value=0,max_value=300)
inputs= [[age,time,creatinine_phosphokinase,platelets, serum_sodium, sex, anaemia,diabetes,ejection_fraction,high_blood_pressure,smoking, serum_creatinine]]


##make the prediction
if st.button("Predict"):
    result=model.predict(inputs)
    updated_res = result.flatten().astype(int)
    if updated_res > 0.5:
          st.write("you are at risk")
    else:
          st.write("you are not at risk")





##st.title("Heart Predictor")

##st.header("Predictor")

##st.subheader("heart predictor")
##st.success("Executed successfully")
##st.info("This is an information")
##st.warning("this is a warning")
##st.error("an error ocurred")
##st.checkbox("I agree")

##if(st.checkbox("I agree")):
    ##st.text("You agreed")

    ####slider 
    ##sal= st.slider("your salary", 10000, 50000)
    ##st.write("your salary is", sal)

##one code 

