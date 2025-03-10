import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("ipl_data.csv")
        df.columns = df.columns.str.lower().str.strip()  # Normalize column names
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

df = load_data()

# Debug: Show available columns
st.write("### Debug: Columns in dataset:", df.columns.tolist())

# Check if dataset is empty
if df.empty:
    st.error("Dataset could not be loaded. Please check 'ipl_data.csv' file!")
    st.stop()

# Expected Columns
expected_columns = ["bat_team", "bowl_team", "venue", "wickets", "overs", "runs", "total"]
missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    st.error(f"⚠️ Missing columns in dataset: {missing_columns}")
    st.stop()

# Streamlit UI
st.title("🏏 IPL Score Predictor")
st.write("This app predicts the first innings score of an IPL match based on historical data.")

# Selecting Features for Prediction
teams = df["bat_team"].unique()
venues = df["venue"].unique()

batting_team = st.selectbox("Select Batting Team", teams)
bowling_team = st.selectbox("Select Bowling Team", teams)
venue = st.selectbox("Select Venue", venues)
wickets = st.slider("Wickets Fallen", 0, 10, 3)
current_runs = st.number_input("Current Runs", min_value=0, max_value=250, value=50)
overs = st.slider("Over Number", 1, 20, 10)
runs_last_5 = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=100, value=30)
wickets_last_5 = st.slider("Wickets in Last 5 Overs", 0, 5, 1)

# Prepare Data for Training
X = df[["wickets", "overs", "runs", "runs_last_5", "wickets_last_5"]]
y = df["total"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Make Prediction
if st.button("Predict Score"):
    input_data = np.array([[wickets, overs, current_runs, runs_last_5, wickets_last_5]])
    predicted_score = model.predict(input_data)[0]
    st.success(f"🎯 Predicted Total Score: {int(predicted_score)}")

# Model Performance
st.subheader("📊 Model Performance")
y_pred = model.predict(X_test)
st.write(f"📉 Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"📉 Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"📈 R² Score: {r2_score(y_test, y_pred):.2f}")
