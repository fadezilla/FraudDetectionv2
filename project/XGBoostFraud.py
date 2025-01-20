import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

creditcard_df = pd.read_csv("creditcard.csv")

scaler = StandardScaler()
creditcard_df["NormalizedAmount"] = scaler.fit_transform(creditcard_df["Amount"].values.reshape(-1, 1))
creditcard_df["NormalizedTime"] = scaler.fit_transform(creditcard_df["Time"].values.reshape(-1, 1))

creditcard_df.drop(["Amount", "Time"], axis=1, inplace=True)

X = creditcard_df.drop("Class", axis=1)
y = creditcard_df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote_enn = SMOTEENN(random_state=42)
X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train, y_train)

balanced_class_counts = y_train_balanced.value_counts()
plt.bar(balanced_class_counts.index, balanced_class_counts.values, color=["blue", "orange"])
plt.xticks([0, 1], ["Non-Fraudulent", "Fraudulent"])
plt.ylabel("Number of Samples")
plt.title("Class Distribution After SMOTE-ENN")
plt.savefig("class_distribution.png") 

stacking_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('catboost', CatBoostClassifier(iterations=100, verbose=0)),
        ('xgboost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0))
    ],
    final_estimator=LogisticRegression(max_iter=500, class_weight='balanced'),
    cv=3,
    n_jobs=-1
)


stacking_model.fit(X_train_balanced, y_train_balanced)


y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]

threshold = 0.2
y_pred_adjusted = (y_pred_proba > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred_adjusted)
precision = precision_score(y_test, y_pred_adjusted)
recall = recall_score(y_test, y_pred_adjusted)
f1 = f1_score(y_test, y_pred_adjusted)

print(f"\nEvaluation with Adjusted Threshold (threshold={threshold}):")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_adjusted))
