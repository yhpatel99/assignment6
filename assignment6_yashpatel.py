# Import Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Models
log_reg = LogisticRegression(max_iter=10000)
rand_forest = RandomForestClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)

# Train Models
log_reg.fit(X_train, y_train)
rand_forest.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Predict and Evaluate
models = {'Logistic Regression': log_reg, 'Random Forest': rand_forest, 'SVM': svm}
results = {}

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

# Print Results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print()

# Determine Best Model
best_model = max(results, key=lambda x: results[x]['F1 Score'])
print(f"Best Model: {best_model}")

# Explanation
explanation = """
Based on the evaluation metrics, the best performing model is {best_model}. 
The F1 Score is particularly important as it balances precision and recall, 
creating a single metric that looks for both false positives and false negatives. 
Other metrics such as accuracy, precision, recall, and ROC AUC also support the performance of this model.
"""
print(explanation.format(best_model=best_model))