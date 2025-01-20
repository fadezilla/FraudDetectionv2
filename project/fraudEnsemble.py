##SECOND ITERATION OF MODELS, COMBINGING MULTIPLE MODELS, RESULTS DID NOT IMPROVE


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

creditcard_df = pd.read_csv("creditcard.csv")

scaler = StandardScaler()
creditcard_df["NormalizedAmount"] = scaler.fit_transform(creditcard_df["Amount"].values.reshape(-1, 1))
creditcard_df["NormalizedTime"] = scaler.fit_transform(creditcard_df["Time"].values.reshape(-1, 1))

creditcard_df.drop(["Amount", "Time"], axis=1, inplace=True)

X = creditcard_df.drop("Class", axis=1)
y = creditcard_df["Class"]

important_features = [
    'V14', 'V10', 'V4', 'V17', 'V11', 'V12', 'V3', 'V16', 'V2', 'V7'
] 

X = X[important_features]

X["V14_V10"] = X["V14"] * X["V10"]
X["V14_V4"] = X["V14"] * X["V4"]
X["V10_V4"] = X["V10"] * X["V4"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote_enn = SMOTEENN(random_state=42)
X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train, y_train)

optimized_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)
optimized_model.fit(X_train_balanced, y_train_balanced)

y_pred_proba = optimized_model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred_adjusted = (y_pred_proba > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred_adjusted)
precision = precision_score(y_test, y_pred_adjusted)
recall = recall_score(y_test, y_pred_adjusted)
f1 = f1_score(y_test, y_pred_adjusted)

print(f"\nEvaluation with Optimized Features (threshold={threshold}):")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_adjusted))
