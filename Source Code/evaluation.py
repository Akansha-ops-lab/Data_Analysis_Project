
# evaluation.py: Functions for evaluating model performance

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model's performance on test data."""
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Print results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("
Classification Report:")
    print(classification_report(y_test, y_pred))
