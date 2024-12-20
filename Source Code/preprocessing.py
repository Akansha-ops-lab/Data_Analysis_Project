
# preprocessing.py: Scripts for cleaning and transforming data

import pandas as pd
import numpy as np

def load_data(file_path):
    """Load the raw data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the data by handling missing values and encoding categorical features."""
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean())
    df['Tenure'] = df['Tenure'].fillna(df['Tenure'].mean())
    df['Churn'] = df['Churn'].fillna(df['Churn'].mode()[0])
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def save_cleaned_data(df, file_path):
    """Save the cleaned data to a CSV file."""
    df.to_csv(file_path, index=False)
