from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('stroke.csv')
#print(df)

#checking for null values
print(df.isnull().sum(axis=0))

#drop NaN values
df = df.dropna()



#counts of residents
x = df['Residence_type'].value_counts()
x = x.reset_index()
print(x)

#filtering data and removing data with Unknow value types in dataframe
df = df[df['smoking_status'] != 'Unknown']
x = df['smoking_status'].value_counts()
x = x.reset_index()
print(x)
print(df.count())

#divinding into groups with stroke and nostroke as 0 and 1
print(df.groupby('stroke').size())

#glucose level based on stroke
df_avg_glucose_level= df.groupby(['stroke'],as_index=False).avg_glucose_level.mean()
print("Average glucose level based on stroke and no stroke")
print(df_avg_glucose_level)

#stroke count for males and females
df_test=df[df['stroke']==0].groupby(['gender','stroke']).size().reset_index(name='count')
print(df_test)

df_test2=df[df['stroke']==1].groupby(['gender','stroke']).size().reset_index(name='count')
print(df_test2)

#stroke for married and nit married
df_test=df[df['stroke']==0].groupby(['ever_married','stroke']).size().reset_index(name='count')
print(df_test)

df_test=df[df['stroke']==1].groupby(['ever_married','stroke']).size().reset_index(name='count')
print(df_test)

#stroke for heart_diseases
df_test=df[df['stroke']==0].groupby(['heart_disease','stroke']).size().reset_index(name='count')
print(df_test)

df_test=df[df['stroke']==1].groupby(['heart_disease','stroke']).size().reset_index(name='count')
print(df_test)

#stroke for hypertension
df_test=df[df['stroke']==0].groupby(['hypertension','stroke']).size().reset_index(name='count')
print(df_test)

df_test=df[df['stroke']==1].groupby(['hypertension','stroke']).size().reset_index(name='count')
print(df_test)

print('-----------------------')
#stroke for work_type
df_test=df[df['stroke']==0].groupby(['work_type','stroke']).size().reset_index(name='count')
print(df_test)

df_test=df[df['stroke']==1].groupby(['work_type','stroke']).size().reset_index(name='count')
print(df_test)


#bmi based on stroke
df_avg_bmi_level= df.groupby(['stroke'],as_index=False).bmi.mean()
print("Average bmi based on stroke and no stroke")
print(df_avg_bmi_level)

#age based on stroke
df_avg_age= df.groupby(['stroke'],as_index=False).age.mean()
print("Average age based on stroke and no stroke")
print(df_avg_age)

#stroke based on residency
df_test=df[df['stroke']==0].groupby(['Residence_type','stroke']).size().reset_index(name='count')
print(df_test)

df_test=df[df['stroke']==1].groupby(['Residence_type','stroke']).size().reset_index(name='count')
print(df_test)

#lets build machine learning model
X = df[['age', 'hypertension', 'heart_disease','avg_glucose_level']]
y = df['stroke']
print(X)
print(y)

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=40, max_iter=10000)
lr.fit(X_train, y_train)

# Coefficients of linear model (b_1,b_2,...,b_p): log(p/(1-p)) = b0+b_1x_1+b_2x_2+...+b_px_p
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# first example: a small fruit with mass 15g, color_score = 5.5, width 4.3 cm, height 5.5 cm
testFruit = pd.DataFrame([[101, 1, 1, 202.21]], columns=['age', 'hypertension', 'heart_disease','avg_glucose_level'])
fruit_prediction = lr.predict(testFruit)
print(fruit_prediction)