import matplotlib.pyplot as plt
import seaborn as sns

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