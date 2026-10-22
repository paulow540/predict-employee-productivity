import streamlit as st
import pandas as pd
import numpy as np
import pickle


department = st.selectbox("Select Department: ", [ "sweing", "finishing"])
department = 1 if department == "finishing" else 0
date = st.date_input("Enter Date: ")
day	= date.day
team	= st.selectbox("Select Team: ", [1,2,3,4,5,6,7,8,9,10,11,12])
targeted_productivity	= st.number_input("Enter Targeted Productivity: ", min_value=0.0, step=0.1, max_value=1.0 )
smv	= st.number_input("Enter SMV: ", min_value=3.94, step=0.1, max_value=20.20 )
wip	= st.number_input("Enter WIP: ", min_value=7.0, step=0.1, max_value=23122.0 )
over_time	= st.number_input("Enter Over Time: ", min_value=0.0, step=1.0, max_value=25920.0 )
incentive = st.number_input("Enter Incentive: ", min_value=0.0, step=1.0, max_value=3600.0 )	
idle_time = st.number_input("Enter Idle Time: ", min_value=0.0, step=1.0, max_value=300.0 )
idle_men	= st.number_input("Enter Idle Men: ", min_value=0.0, step=1.0, max_value=45.0 )
no_of_style_change    = st.number_input("Enter No of Style Change: ", min_value=0.0, step=1.0, max_value=2.0 )	
no_of_workers	= st.number_input("Enter No of Workers: ", min_value=2.0, step=1.0, max_value=89.0 )
year	= date.year
Month	= date.month
dayweek	= date.weekday()


with open("xgboost_binary_productivity_model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)


X_new =np.array([department, day, team, targeted_productivity, smv, wip, over_time, incentive, idle_time, idle_men, no_of_style_change, no_of_workers, year, Month, dayweek] ).reshape(1, -1)
# Use it to predict new data
predictions = loaded_model.predict(X_new)

if st.button("Predict Productivity"):
    predicted_productivity = predictions[0]
    if predicted_productivity == 1:
        st.success("Predicted Productivity: High")
    else:
        st.warning("Predicted Productivity: Low")

# st.write(f"Predicted Productivity: {predictions[0]}")