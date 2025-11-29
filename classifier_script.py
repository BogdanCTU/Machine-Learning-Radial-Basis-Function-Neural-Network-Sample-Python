import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from evorbf import RbfClassifier
import joblib

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
csv_path = "INPUT.csv"
label_col = "CLASSIFICATION"
split_index = 16000  # Cutoff point: Train on first 16k, Test on rest

feature_cols = [
    "ENERGY", "PROTEIN", "CARBS", "TOTAL_FAT",
    "SATURATED_FAT", "FIBER", "SUGARS"
]

# ---------------------------------------------------------------------
# Load and clean dataset
# ---------------------------------------------------------------------
print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path, sep=';')

# Drop rows with missing values
df = df[feature_cols + [label_col]].dropna()

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------
# Encode labels (True/False -> 1/0)
encoder = LabelEncoder()
y_all = encoder.fit_transform(df[label_col])

# Print mapping so we know what 1 and 0 mean
print("-" * 30)
print("Label Mapping:")
for i, item in enumerate(encoder.classes_):
    print(f"  Class {i} = {item}")
print("-" * 30)

# Extract features matrix
X_all = df[feature_cols].values

# ---------------------------------------------------------------------
# SPLITTING: Manual Split at 16,000
# ---------------------------------------------------------------------
# Train set: 0 to 16,000
X_train = X_all[:split_index]
y_train = y_all[:split_index]

# Test set: 16,000 to End (approx 4,000)
X_test = X_all[split_index:]
y_test = y_all[split_index:]

print(f"Training Data Size: {len(X_train)} records")
print(f"Testing Data Size:  {len(X_test)} records")

# ---------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------
# IMPORTANT: Fit scaler ONLY on the training block to avoid leaking info
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------------------
# Train EvoRBF Classifier
# ---------------------------------------------------------------------
# Using the "tuned" settings for better accuracy on edge cases
model = RbfClassifier(
    size_hidden=80,  # Higher complexity
    center_finder="kmeans",
    sigmas=0.5,  # Sharper decision boundaries
    reg_lambda=0.01,
    seed=42
)

print("\nTraining model on first 16,000 records...")
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# Evaluation on the Last 4,000 Records
# ---------------------------------------------------------------------
print("\nEvaluating on the last 4,000 records...")
y_pred = model.predict(X_test)

# Calculate Accuracy Ratio
acc = accuracy_score(y_test, y_pred)
print("-" * 40)
print(f"RESULTS FOR LAST {len(y_test)} ITEMS:")
print(f"Correct Predictions Ratio: {acc:.2%}")  # Prints as percentage (e.g. 95.20%)
print(f"Correct Count: {np.sum(y_test == y_pred)} / {len(y_test)}")
print("-" * 40)

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in encoder.classes_]))


# ---------------------------------------------------------------------
# Specific Sample Check (The Edge Case)
# ---------------------------------------------------------------------
def classify_food(nutrient_dict):
    df_input = pd.DataFrame([nutrient_dict])[feature_cols]
    x_scaled = scaler.transform(df_input.values)
    pred_proba = model.predict_proba(x_scaled)
    # Get probability of Class 1
    prob_class1 = pred_proba[0][1]
    # Decision
    pred_index = 1 if prob_class1 >= 0.5 else 0
    pred_label = encoder.inverse_transform([pred_index])[0]
    return pred_label, prob_class1
