import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta


@st.cache_resource
def load_model_scaler():
    """Load the saved model and scaler"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        os.path.dirname(current_dir),
        "models",
        "energy_consumption",
        "random_forest_model.pkl",
    )
    scaler_path = os.path.join(
        os.path.dirname(current_dir),
        "models",
        "energy_consumption",
        "standard_scaler.pkl",
    )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# Load model and scaler using cached function
model, scaler = load_model_scaler()


def preprocess_input(data):
    """Process the user input data"""

    # Scale numerical features
    numerical_features = [
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_2",
        "Sub_metering_3",
        "hour",
    ]
    scaled_features = pd.DataFrame(
        scaler.transform(data[numerical_features]), columns=numerical_features
    )

    # Combine all features
    final_features = pd.concat([scaled_features, data[["is_weekend"]]], axis=1)
    final_features = final_features[
        [
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_2",
            "Sub_metering_3",
            "is_weekend",
            "hour",
        ]
    ]

    return final_features


def get_user_input():
    """Create input form for user data"""
    st.sidebar.header("Input Parameters")

    # Date and time input
    date = st.sidebar.date_input("Select Date", datetime.now())
    time = st.sidebar.time_input("Select Time", datetime.now())

    # Convert 'datetime' to datetime type and set as index
    datetime_input = datetime.combine(date, time)
    is_weekend = datetime_input.weekday() >= 5
    hour = datetime_input.hour

    # Other inputs
    reactive_power = st.sidebar.slider("Global Reactive Power (kW)", 0.0, 0.5, 0.1)
    voltage = st.sidebar.slider("Voltage (V)", 220.0, 250.0, 235.0)
    intensity = st.sidebar.slider("Global Intensity (A)", 0.0, 40.0, 20.0)
    sub_metering_2 = st.sidebar.slider("Sub Metering 2 (Wh)", 0.0, 20.0, 10.0)
    sub_metering_3 = st.sidebar.slider("Sub Metering 3 (Wh)", 0.0, 20.0, 10.0)

    # Create DataFrame
    data = {
        "Global_reactive_power": [reactive_power],
        "Voltage": [voltage],
        "Global_intensity": [intensity],
        "Sub_metering_2": [sub_metering_2],
        "Sub_metering_3": [sub_metering_3],
        "hour": [hour],
        "is_weekend": [is_weekend],
    }

    df = pd.DataFrame(data)
    return df


def make_prediction(user_input):
    """Make prediction using the model"""
    features = preprocess_input(user_input)
    prediction = model.predict(features)
    return prediction[0]


def plot_feature_importance():
    """Plot feature importance"""
    # Get feature importance from the model
    feature_importances = model.feature_importances_

    features = [
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_2",
        "Sub_metering_3",
        "is_weekend",
        "hour",
    ]

    # Sort features by importance in descending order
    indices = np.argsort(feature_importances)[::-1]

    # Create feature importance plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.arange(len(features)), feature_importances[indices], color="#4CAF50")
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels([features[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Importance")

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close()


def run_app():
    st.title("Household Energy Consumption Prediction")
    st.write(
        """
    This app predicts the household's Global Active Power consumption based on various input parameters.
    Adjust the sliders in the sidebar to see real-time predictions.
    """
    )

    # Get user input
    user_input = get_user_input()

    # Make prediction automatically when inputs change
    prediction = make_prediction(user_input)

    # Display prediction
    st.subheader("Prediction Results")
    st.markdown(
        f"""
        <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; text-align: center;">
            <h3 style="color: #0066cc;">Predicted Global Active Power</h3>
            <p style="font-size: 36px; font-weight: bold; color: #0066cc;">{prediction:.3f} kW</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show feature importance
    st.subheader("Feature Importance")
    plot_feature_importance()

    # Show input parameters
    st.subheader("Input Parameters")
    st.write(user_input)


if __name__ == "__main__":
    run_app()
