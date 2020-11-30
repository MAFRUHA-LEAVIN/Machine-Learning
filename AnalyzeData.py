#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:36:33 2020

@author: fahidkhan
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns,numpy as np

path = "bank.csv"

df = pd.read_csv(path)
type(df)

#Explore and analyze data 

pd.set_option("display.max.columns",None)

df.head()

print(df.head())

#Showing data using pie chart for bank data 

labels = 'Good_Balance','Bad_balance'

sizes=[78,147]
colors=['orange','blue']


plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=150)

plt.axis('equal')
plt.show()


#Showing displot chart on random things from customer age 

sns.distplot(df['Age'])
plt.show()

#showing scatterplot on creditscore and balance 
sns.scatterplot(data=df, x="CreditScore",y="Balance")

#showing the trendline on the scatterplot of creditscore and balance 
sns.regplot(x="CreditScore",y="Balance",data=df,fit_reg=True)

#showing a boxplot graph on who is active member of the bank and has a credit card 
sns.boxplot(x="HasCrCard",y="IsActiveMember",data=df)

