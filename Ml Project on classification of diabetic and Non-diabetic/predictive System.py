# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Loading the saved Model
loaded_model = pickle.load(open('D:/ML Projects/Ml Project on classification of diabetic and Non-diabetic/trained_model.sav','rb'))


input_data = (5,166,72,19,175,25.8,0.587,51)
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
    print("The Person is non-Diabetic")
else:
    print("The Person is Diabetic")