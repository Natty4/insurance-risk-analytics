import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to set up the default plot style for consistency
def set_plot_style():
    """
    Sets the default style for visualizations.
    """
    sns.set(style="whitegrid", palette="muted")
    plt.rcParams["figure.figsize"] = (10, 6)

# Function to visualize missing values per column
def plot_missing_data(df):
    """
    Plot a bar chart of missing values per column.
    """
    missing_data = df.isnull().mean() * 100
    missing_data = missing_data[missing_data > 0]
    missing_data.sort_values(ascending=False).plot(kind='barh', color='coral')
    plt.title("Percentage of Missing Values per Column")
    plt.xlabel("Percentage")
    plt.ylabel("Columns")
    plt.show()

# Function to visualize the distribution of categorical variables
def plot_categorical_distribution(df, column, top_n=10):
    """
    Plots a bar chart for the distribution of a categorical variable.
    """
    top_categories = df[column].value_counts().head(top_n)
    top_categories.plot(kind='barh', color='skyblue')
    plt.title(f"Top {top_n} Most Frequent Categories in '{column}'")
    plt.xlabel("Frequency")
    plt.ylabel(column)
    plt.show()

# Function to visualize the distribution of numerical variables
def plot_numerical_distribution(df, column):
    """
    Plots the distribution of a numerical variable using a histogram.
    """
    df[column].dropna().plot(kind='hist', bins=30, color='seagreen', edgecolor='black')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# Function to visualize the relationship between two numerical variables
def plot_numerical_relationship(df, x_column, y_column):
    """
    Plots the relationship between two numerical variables.
    """
    sns.scatterplot(x=df[x_column], y=df[y_column], color='blue', alpha=0.6)
    plt.title(f"{x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

# Function to visualize the distribution of loss ratios
def plot_loss_ratio_distribution(df):
    """
    Plots the distribution of LossRatio.
    """
    df['LossRatio'].dropna().plot(kind='hist', bins=30, color='orange', edgecolor='black')
    plt.title("Distribution of Loss Ratio")
    plt.xlabel("Loss Ratio")
    plt.ylabel("Frequency")
    plt.show()

# Function to visualize the relationship between LossRatio and other variables
def plot_loss_ratio_vs_other(df, column):
    """
    Visualizes the relationship between LossRatio and another variable (e.g., 'TotalClaims', 'TotalPremium') 
    using a bar graph showing the mean LossRatio for each category.
    """
    # Group by the specified column and calculate the mean LossRatio for each category
    avg_loss_ratio = df.groupby(column)['LossRatio'].mean().reset_index()

    # Plotting the bar graph
    plt.figure(figsize=(12, 6))
    sns.barplot(x=column, y='LossRatio', data=avg_loss_ratio, palette='viridis')
    
    plt.title(f"Average Loss Ratio vs {column}")
    plt.xlabel(column)
    plt.ylabel("Average Loss Ratio")
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

# Function to visualize the correlation heatmap
def plot_correlation_heatmap(df, correlation_columns=None):
    """
    Plots a heatmap to show the correlation between numerical variables.
    """
    if correlation_columns is None:
        correlation_columns = df.select_dtypes(include=[np.number]).columns
    
    corr = df[correlation_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

# Function to visualize the count distribution of the 'WrittenOff' column
def plot_writtenoff_distribution(df):
    """
    Plots the count of values in the 'WrittenOff' column.
    """
    df['WrittenOff'].value_counts(dropna=False).plot(kind='bar', color='lightcoral')
    plt.title("Distribution of WrittenOff Status")
    plt.xlabel("WrittenOff")
    plt.ylabel("Count")
    plt.show()