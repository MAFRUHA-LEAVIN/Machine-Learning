#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:36:16 2020

@author: fahidkhan
"""

#importing dependency
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#uploading csv file

df = pd.read_csv('diabetes.csv')
df.tail()

df.head()

df.shape
#row column size
df.isna().sum()

df['Outcome'].value_counts()

df.info()

df2 = df

df = df[['Outcome', 'Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

df.head()

sns.pairplot(df.iloc[:,0:6], hue = 'Outcome')
df.iloc[:, 0:9].corr()

plt.figure(figsize=(12,12))
sns.heatmap(df.iloc[:, 0: 9].corr(), annot=True, fmt='.0%')

df.shape

X = df.iloc[:,1:9].values
Y = df.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

def models(X_train, Y_train):
  
  #LogisticRegression
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state = 0)
  log.fit(X_train, Y_train)

  #Decision Tree
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
  tree.fit(X_train, Y_train)

  #RandomForest Classifier
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)

  #estimating training accuracy
  print('[0]Logistic Regression Accuracy: ', log.score(X_train, Y_train))
  print('[1]Decision Tree Accuracy: ', tree.score(X_train, Y_train))
  print('[0]Random Forest Accuracy: ', forest.score(X_train, Y_train))

  return log, tree, forest

model = models(X_train, Y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
  print('model ', i)  
  print( classification_report(Y_test, model[i].predict(X_test)))
  print( accuracy_score(Y_test, model[i].predict(X_test)))
  print()
  
  
pred = model[2].predict(X_test)
print(pred)
print()
print(Y_test)


dif = 0
for i in range(len(pred)):
  if(pred[i] != Y_test[i]):
    dif = dif+1
    print(pred[i], Y_test[i])
print(dif)