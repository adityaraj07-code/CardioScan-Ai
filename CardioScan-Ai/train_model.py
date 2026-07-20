import os
import joblib
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("heart_disease_uci.csv")

# Binary target
df["target"] = (df["num"] > 0).astype(int)

df.drop("num", axis=1, inplace=True)

if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

# -----------------------------
# Missing Values
# -----------------------------
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# -----------------------------
# Encoding
# -----------------------------
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# Split
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# -----------------------------
# Models
# -----------------------------
log_model = LogisticRegression(max_iter=1000)
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)

log_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

log_acc = accuracy_score(y_test, log_model.predict(X_test))
tree_acc = accuracy_score(y_test, tree_model.predict(X_test))

print(f"Logistic Accuracy : {log_acc:.4f}")
print(f"Decision Tree Accuracy : {tree_acc:.4f}")

best_model = log_model if log_acc >= tree_acc else tree_model

# -----------------------------
# Save Everything
# -----------------------------
joblib.dump(best_model, f"{MODEL_DIR}/model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")
joblib.dump(num_imputer, f"{MODEL_DIR}/num_imputer.pkl")
joblib.dump(cat_imputer, f"{MODEL_DIR}/cat_imputer.pkl")
joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_names.pkl")

print("\nModel saved successfully!")
