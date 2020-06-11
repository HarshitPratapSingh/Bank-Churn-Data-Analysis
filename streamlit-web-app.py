import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

model_path = "Models/"
model_set={"Default":"DT","Decision Tree":"DT","Logistic Regression":"LR","K-Nearest Neighbor":"KNN", "Support Vector Machine":"SVM", "Stochastic Gradient Descent": "SGD"}
@st.cache
def get_data():
    
    path = "dataset/churn.csv"
    return pd.read_csv(path)

def Country_fix(Inp_Country):
    if Inp_Country == "Germany":
        return 1,0,0
    if Inp_Country == "France":
        return 0,1,0
    else:
        return 0,0,1

def Gender_fix(gender_temp):
    if gender_temp == "Male":
        return 0,1
    else:
        return 1,0

def Bool_Fix(decision_temp):
    if decision_temp == "Yes":
        return 1
    else:
        return 0

Churn = get_data()
Churn = Churn.drop(["Exited","RowNumber", "CustomerId", "Surname"], axis=1)

list_cat = ['Geography', 'Gender']
Churn = pd.get_dummies(Churn, columns = list_cat, prefix = list_cat)



sc = StandardScaler()

train, test = train_test_split(Churn, test_size=0.2, random_state=1)
features = Churn.columns

train[features] = sc.fit_transform(train[features])


st.title("Bank Churn Project")

model_selection_name = st.selectbox("Select model you want to use", list(model_set.keys()))
st.write(model_selection_name," is the model through which we will predict")
model = joblib.load(model_path+model_set.get(model_selection_name)+"_model.pkl")

st.header("Enter the Values")
st.subheader("Enter the Params ands Values to predict any customer will leave or not : -")



crScore = st.slider("1. Credit Score :- ", min_value = 300, max_value = 800)
crScore
country = st.selectbox("2. Geography :- ", ["France", "Germany", "Spain"])
country
gender = st.radio("3. Gender",["Male", "Female"])
gender
age = st.slider("4. Age", min_value= 18, max_value = 95)
age
tenure = st.slider("5. Tenure", min_value= 0, max_value = 10)
tenure
bal = st.slider("6. Balance", min_value=0, max_value=260000)
bal
num_p = st.selectbox("7. Number of Products", [1,2,3,4])
num_p
crCard = st.radio("8. Have a Credit Card", ["Yes", "No"])
crCard
isActiveMem = st.radio("9. Are You an Active Member", ["Yes", "No"])
isActiveMem
Esalary = st.slider("10. Estimated Salary", min_value=0, max_value=200000)
Esalary


bt = st.button("Predict the Result")
if bt:
    
    G_Con, F_Con, S_Con = Country_fix(country)
    F_Gen, M_Gen = Gender_fix(gender)
    test_values = pd.DataFrame({"CreditScore": [crScore],
    "Geography_Spain": [S_Con],
    "Geography_Germany": [G_Con],
    "Geography_France": [F_Con],
    "Gender_Male":[M_Gen],
    "Gender_Female":[F_Gen],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[bal],
    "NumOfProducts":[num_p],
    "HasCrCard":[Bool_Fix(crCard)],
    "IsActiveMember":[Bool_Fix(isActiveMem)],
    "EstimatedSalary":[Esalary]})
    
    test_values[features] = sc.transform(test_values[features])
  
    pred = model.predict(test_values[features])
    if pred == 1:
        st.subheader("You could exit or retain your Bank Account")
    else:
        st.subheader("You couldn't exit or retain your Bank Account")
    test_values


st.header("Thank you for using my ML model I hope you liked it")
st.markdown(":smile: :smile: :smile: :smile: :smile:")
st.markdown("Made with :heart: By Harshit Pratap Singh \n")
st.markdown(" :email: 4harshitsingh@gmail.com :iphone: - 8840192004")