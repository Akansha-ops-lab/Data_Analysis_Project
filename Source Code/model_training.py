
# model_training.py: Scripts for training machine learning models

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(cleaned_data_path):
    """Train a Random Forest model on the cleaned data."""
    # Load data
    data = pd.read_csv(cleaned_data_path)
    
    # Define features and target
    X = data[['Age', 'MonthlyCharges', 'Tenure']]
    y = data['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test
