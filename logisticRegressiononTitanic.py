#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:32:09 2020

@author: fahidkhan
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


df = pd.read_csv('titanic.csv',header=0,sep=',')

X = df.iloc[:, 0:3]

X_org = X
y = df.iloc[:, -1]

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['PClass', 'Gender'])], remainder='passthrough') # passthrough saves Age
X = np.array(ct.fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = LogisticRegression(random_state = 0)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


#predicting the survival chance of a girl and boy

y_girl = model.predict_proba(sc.transform([[0, 0, 0, 18]]))

y_boy=model.predict_proba(sc.transform([[0, 1, 1, 18]]))


print(f'girl: {round(y_girl[:,1][0]*100,2)}')
print(f'boy: {round(y_boy[:,1][0]*100,2)}')


cm = confusion_matrix(y_test, y_pred)

#The Confusiion metrics and accuracy)score 

print ('cm:')
print(cm)
print(f'accuracy_score: {accuracy_score(y_test, y_pred)}')
