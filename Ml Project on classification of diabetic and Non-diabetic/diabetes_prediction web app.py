# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 12:49:33 2022

@author: devan
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved Model
loaded_model = pickle.load(open('D:/ML Projects/Ml Project on classification of diabetic and Non-diabetic/trained_model.sav','rb'))

# Creating a func for Prediction

def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

    # standardization the input data
    #std_data = scaler.transform(input_data_reshape)
    #print(std_data)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)

    if(prediction==0):
        return 'The Person is non-Diabetic'
    else:
        return 'The Person is Diabetic'
    

def main():
    
    # Giving a title for the webpage
    
    st.title('Diabetes Prediction Web APP')
    
    # Getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnencies')
    Glucose = st.text_input('Blood Glucose Level')
    BloodPressure = st.text_input('BP Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin= st.text_input('Insulin Level')
    BMI= st.text_input('BMI Value')
    DiabetesPedigreeFunction= st.text_input('DiabetesPedigreeFunction Value')
    Age = st.text_input('Age Of the Person')
    
    
    # Code for Prediction 
    diagnosis = ''
    
    # Creating A Button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    