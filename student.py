import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("📘 Student Exam Score Predictor")
st.write("Predict exam score using study hours, attendance, and previous score")

# Upload dataset
file = st.file_uploader("Upload Student Exam Dataset (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

    # Features & target
    X = df[["Study_Hours", "Attendance", "Previous_Score"]]
    y = df["Exam_Score"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction on test data
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    st.subheader("📈 Model Performance")
    st.write(f"Model Accuracy (R² Score): **{accuracy:.2f}**")

    # Visualization
    st.subheader("📊 Actual vs Predicted Scores")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    st.pyplot(fig)

    # User input for prediction
    st.subheader("🧠 Predict Student Exam Score")

    study_hours = st.number_input("Study Hours per Day", 0.0, 10.0, 2.0)
    attendance = st.number_input("Attendance (%)", 0.0, 100.0, 75.0)
    previous_score = st.number_input("Previous Exam Score", 0.0, 100.0, 60.0)

    if st.button("Predict Score"):
        input_data = [[study_hours, attendance, previous_score]]
        prediction = model.predict(input_data)
        st.success(f"🎯 Predicted Exam Score: **{prediction[0]:.2f}**")

else:
    st.info("👆 Upload a CSV file to train the prediction model")
