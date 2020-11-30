#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 01:16:32 2020

@author: fahidkhan
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics

data_set = pd.read_csv("diabetes.csv")


X= data_set.iloc[:,:-1]

Y = data_set.iloc[:,-1]


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2,random_state =0 )

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mean_absolute_error=metrics.mean_absolute_error(y_test, y_pred) 
r2=metrics.r2_score(y_test, y_pred)


print(f'R2:  {round(r2,4)}')
print(f'MAE: {round(mean_absolute_error,4)}')

