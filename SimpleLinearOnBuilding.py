#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 00:06:09 2020

@author: fahidkhan
"""

import numpy as np 
import matplotlib.pyplot as mtp

import pandas as pd

#reading the data 

data_set = pd.read_csv('building.csv')

x= data_set.iloc[:,1].values  
y= data_set.iloc[:,2].values  

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)


from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train.reshape(-1,1), y_train)  

y_pred= regressor.predict(x_test.reshape(-1,1))  
x_pred= regressor.predict(x_train.reshape(-1,1))

mtp.scatter(x_train, y_train, color="green")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Power vs Temp (Training Dataset)")  
mtp.xlabel("Power")  
mtp.ylabel("Temp")  
mtp.show() 


mtp.scatter(x_test, y_test, color="blue")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Power vs Temp (Test Dataset)")  
mtp.xlabel("Power")  
mtp.ylabel("Temp")  
mtp.show()    