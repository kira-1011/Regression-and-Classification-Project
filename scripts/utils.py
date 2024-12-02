import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

sns.set(style="whitegrid")


def cap_outliers(df, columns, lower_percentile, upper_percentile):
    lower_cap = df[columns].quantile(lower_percentile / 100)
    upper_cap = df[columns].quantile(upper_percentile / 100)
    for column in columns:
        df[column] = df[column].clip(lower_cap[column], upper_cap[column])


def plot_boxplots(df, columns, titles):
    plt.figure(figsize=(14, len(columns) * 3))
    for idx, feature in enumerate(columns):
        plt.subplot(len(columns), 2, idx + 1)
        sns.boxplot(df[feature])
        plt.title(f"{titles[idx]}")
    plt.tight_layout()
    plt.show()


def plot_boxplot(df, columns, title, size):
    plt.figure(figsize=size)
    df.boxplot(column=columns)
    plt.title(title)
    plt.show()


def plot_histplots(df, columns, titles):
    plt.figure(figsize=(14, len(columns) * 3))
    for idx, feature in enumerate(columns):
        plt.subplot(len(columns), 2, idx + 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f"{titles[idx]} | Skewness: {round(df[feature].skew(), 2)}")
    plt.tight_layout()
    plt.show()


def evaluate_model(y_true, y_pred, model_name):
    """Calculate and print evaluation metrics for a model"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    return rmse, mae, r2


def get_season(date):
    """
    Determine season based on month.
    Spring: March-May (3-5)
    Summer: June-August (6-8)
    Fall: September-November (9-11)
    Winter: December-February (12-2)
    """
    month = date.month
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:  # month in [12, 1, 2]
        return "Winter"
