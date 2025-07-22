# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    CLASS = st.sidebar.selectbox('Titanic Passenger Class',('1st Class','2nd Class','3rd Class'))
    CLASS1 = 1 if CLASS == '1st Class' else 0 
    CLASS2 = 1 if CLASS == '2nd Class' else 0 
    CLASS3 = 1 if CLASS == '3rd Class' else 0 
    EMBARKED = st.sidebar.selectbox('Embarked',('Cherbourg','Queenstown','Southampton'))
    EMBARKEDC = 1 if EMBARKED == 'Cherbourg' else 0
    EMBARKEDQ = 1 if EMBARKED == 'Queenstown' else 0
    EMBARKEDS = 1 if EMBARKED == 'Southampton' else 0
    SEX = st.sidebar.selectbox('Titanic Passenger Sex',('Male','Female'))
    SEXM = 1 if SEX == 'Male' else 0
    SEXF = 1 if SEX == 'Female' else 0
    SIBSP = st.sidebar.number_input("No. of siblings and spouses of passenger on board")
    PARCH = st.sidebar.number_input("No. of parents and children of passenger on board")
    AGE = st.sidebar.number_input("Age of the passenger")
    FARE = st.sidebar.number_input("Fare")
    data = {'AGE':AGE,
            'SIBSP':SIBSP,
            'PARCH':PARCH,
            'FARE':FARE,
            'CLASS1':CLASS1,
            'CLASS2':CLASS2,
            'CLASS3':CLASS3,
            'EMBARKEDC':EMBARKEDC,
            'EMBARKEDQ':EMBARKEDQ,
            'EMBARKEDS':EMBARKEDS,
            'SEXF':SEXF,
            'SEXM':SEXM}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

Titanic_train = pd.read_csv("https://drive.google.com/file/d/1J_hVXcAMacpEsSRpLmpl1qM7eJ6L8r8M/view")
Titanic_train1 = Titanic_train.drop(columns=['PassengerId','Name','Ticket','Cabin'])
Titanic_train1['Embarked']=Titanic_train1['Embarked'].fillna('S')
Titanic_train1=Titanic_train1.dropna()
Titanic_train1.reset_index(inplace=True)
Titanic_train2_dummies= pd.get_dummies(Titanic_train1,columns=['Pclass','Embarked','Sex'],dtype='int')

X = Titanic_train2_dummies.iloc[:,2:]
X = X.rename({'Age':'AGE','SibSp':'SIBSP','Parch':'PARCH','Fare':'FARE','Pclass_1':'CLASS1','Pclass_2':'CLASS2','Pclass_3':'CLASS3','Embarked_C':'EMBARKEDC','Embarked_Q':'EMBARKEDQ','Embarked_S':'EMBARKEDS','Sex_female':'SEXF','Sex_male':'SEXM'},axis=1)
Y = Titanic_train2_dummies.iloc[:,[1]]
classifier = LogisticRegression()
classifier.fit(X,Y)

prediction = classifier.predict(df)
prediction_proba = classifier.predict_proba(df)

st.subheader('Predicted Result')
st.write('Survived' if prediction_proba[0][1] > 0.5 else 'Not Survived')

st.subheader('Prediction Probability')
st.write(prediction_proba)
