import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ipl_data.csv")
    return df

df = load_data()

# Streamlit UI
st.title("IPL Score Predictor")
st.write("This app predicts the first innings score of an IPL match based on historical data.")

# Display Raw Data
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Selecting Features for Prediction
teams = df["batting_team"].unique()
venue = df["venue"].unique()

batting_team = st.selectbox("Select Batting Team", teams)
bowling_team = st.selectbox("Select Bowling Team", teams)
venue = st.selectbox("Select Venue", venue)
wickets = st.slider("Wickets Fallen", 0, 10, 3)
current_runs = st.number_input("Current Runs", min_value=0, max_value=250, value=50)
over_number = st.slider("Over Number", 1, 20, 10)

# Preprocessing
X = df[["wickets", "current_runs", "over_number"]]  # Features
y = df["total_score"]  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Prediction
if st.button("Predict Score"):
    input_data = np.array([[wickets, current_runs, over_number]])
    predicted_score = model.predict(input_data)[0]
    st.success(f"Predicted Total Score: {int(predicted_score)}")

# Display Model Performance
st.subheader("Model Performance")
y_pred = model.predict(X_test)
st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
