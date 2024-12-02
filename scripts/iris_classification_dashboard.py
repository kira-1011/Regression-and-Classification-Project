import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_iris

iris = load_iris()


def load_model_and_scaler():
    """Load the saved model and scaler"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        os.path.dirname(current_dir),
        "models",
        "iris_classification",
        "iris_classifier_model.pkl",
    )
    scaler_path = os.path.join(
        os.path.dirname(current_dir),
        "models",
        "iris_classification",
        "standard_scaler.pkl",
    )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


model, scaler = load_model_and_scaler()


def get_user_input():
    """Create input form for user data"""
    st.sidebar.header("Input Parameters")

    # Create sliders for each feature
    sepal_length = st.sidebar.slider(
        "Sepal Length (cm)",
        float(iris.data[:, 0].min()),
        float(iris.data[:, 0].max()),
        float(iris.data[:, 0].mean()),
    )

    sepal_width = st.sidebar.slider(
        "Sepal Width (cm)",
        float(iris.data[:, 1].min()),
        float(iris.data[:, 1].max()),
        float(iris.data[:, 1].mean()),
    )

    petal_length = st.sidebar.slider(
        "Petal Length (cm)",
        float(iris.data[:, 2].min()),
        float(iris.data[:, 2].max()),
        float(iris.data[:, 2].mean()),
    )

    petal_width = st.sidebar.slider(
        "Petal Width (cm)",
        float(iris.data[:, 3].min()),
        float(iris.data[:, 3].max()),
        float(iris.data[:, 3].mean()),
    )

    # Create DataFrame
    data = {
        "sepal length (cm)": [sepal_length],
        "sepal width (cm)": [sepal_width],
        "petal length (cm)": [petal_length],
        "petal width (cm)": [petal_width],
    }

    return pd.DataFrame(data)


def make_prediction(user_input):
    """Make prediction using the model"""
    # Scale the input
    scaled_features = scaler.transform(user_input)
    # Make prediction
    prediction = model.predict(scaled_features)
    # Get prediction probability
    prediction_proba = model.predict_proba(scaled_features)
    return prediction[0], prediction_proba[0]


def plot_feature_importance():
    """Plot feature importance"""
    # Get feature importance from the model
    feature_importances = model.feature_importances_
    features = iris.feature_names

    # Sort features by importance in descending order
    indices = np.argsort(feature_importances)[::-1]

    # Create feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(features)), feature_importances[indices], color="#4CAF50")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[i] for i in indices], rotation=45)
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest Feature Importance")

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close()


def plot_prediction_probabilities(probabilities):
    """Plot prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar plot
    bars = ax.bar(
        iris.target_names, probabilities, color=["#FF9999", "#66B2FF", "#99FF99"]
    )

    # Add percentage labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height*100:.1f}%",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities for Each Class")

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close()


def run_app():
    st.title("Iris Flower Classification")
    st.write(
        """
    This app predicts the Iris flower species based on the input measurements.
    Adjust the sliders in the sidebar to input flower measurements and get real-time predictions.
    """
    )

    # Get user input
    user_input = get_user_input()

    # Make prediction
    prediction, probabilities = make_prediction(user_input)

    # Display prediction
    st.subheader("Prediction Results")
    st.markdown(
        f"""
        <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; text-align: center;">
            <h3 style="color: #0066cc;">Predicted Species</h3>
            <p style="font-size: 36px; font-weight: bold; color: #0066cc;">{iris.target_names[prediction]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show prediction probabilities
    st.subheader("Prediction Probabilities")
    plot_prediction_probabilities(probabilities)

    # Show feature importance
    st.subheader("Feature Importance")
    plot_feature_importance()

    # Show input parameters
    st.subheader("Input Parameters")
    st.write(user_input)

    # Add a data visualization section
    st.subheader("Dataset Visualization")

    # Create a DataFrame with all iris data
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["target"] = iris.target

    # Allow user to select features for scatter plot
    st.write("### Scatter Plot")
    x_axis = st.selectbox("Select X-axis feature:", iris.feature_names)
    y_axis = st.selectbox("Select Y-axis feature:", iris.feature_names)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for target in range(3):
        mask = iris_df["target"] == target
        ax.scatter(
            iris_df[mask][x_axis],
            iris_df[mask][y_axis],
            label=iris.target_names[target],
        )

    # Add user input point to the scatter plot
    ax.scatter(
        user_input[x_axis],
        user_input[y_axis],
        color="red",
        marker="*",
        s=200,
        label="Your Input",
    )

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.legend()
    ax.set_title(f"{x_axis} vs {y_axis}")
    st.pyplot(fig)
    plt.close()


if __name__ == "__main__":
    run_app()
