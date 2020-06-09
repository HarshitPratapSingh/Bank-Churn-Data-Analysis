import streamlit as st
import pandas as pd
import numpy as np


@st.cache
def get_data():
    url = "https://raw.githubusercontent.com/HarshitPratapSingh/Bank-Churn-Data-Analysis/master/dataset/churn.csv"
    return pd.read_csv(url)

df = get_data()


is_check = st.checkbox("Display Data")
if is_check:
    st.write(df)

countries = st.sidebar.multiselect("Enter Countries",df['Geography'].unique())
st.write("Your input countries", countries)

variables = st.sidebar.multiselect("Select the columns", df.columns)
st.write(variables)