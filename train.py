import numpy as np
import pandas as pd
import pickle
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Load Datasets
data = pd.read_csv("Training.csv")
severity = pd.read_csv("Symptom_severity.csv")

# Clean severity dataset
severity.columns = [col.strip().lower() for col in severity.columns]
severity["symptom"] = severity["symptom"].str.strip().str.replace("_", " ")

# Additional supporting datasets
description = pd.read_csv("description.csv")
precautions = pd.read_csv("precautions_df.csv")
medications = pd.read_csv("medications.csv")
diets = pd.read_csv("diets.csv")
workout = pd.read_csv("workout_df.csv")


# Prepare Main Dataset
X = data.drop("prognosis", axis=1)
y = data["prognosis"]


# Build mappings (clean column names for consistency)
symptoms_dict = {symptom.strip().replace("_", " ").lower(): i for i, symptom in enumerate(X.columns)}
diseases_list = {i: d for i, d in enumerate(sorted(y.unique()))}


# Apply Symptom Severity Weights
severity_weights = {row["symptom"]: row["weight"] for _, row in severity.iterrows()}

def apply_severity_weights(df):
    weighted_df = df.copy()
    for col in weighted_df.columns:
        clean_col = col.strip().replace("_", " ").lower()
        if clean_col in severity_weights:
            weighted_df[col] = weighted_df[col] * severity_weights[clean_col]
    return weighted_df

X_weighted = apply_severity_weights(X)


# Split Dataset for Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size = 0.2, random_state = 42)


# Train or Load Model
model_path = "models/svc_weighted.pkl"
retrain_model = False  # set True if retraining is needed

if retrain_model or not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    svc = SVC(kernel="linear", probability=True)
    svc.fit(X_train, y_train)
    pickle.dump(svc, open(model_path, "wb"))
    print("✅ Model trained and saved.")
else:
    svc = pickle.load(open(model_path, "rb"))
    print("📂 Model loaded from file.")


# Evaluate Model
y_pred = svc.predict(X_test)

print("\n=== 🧮 Model Evaluation ===")
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Detailed classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Calculate overall precision, recall, and F1-score
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n=== 🔍 Overall Performance Metrics ===")
print(f"🎯 Precision (Weighted): {precision:.4f}")
print(f"📈 Recall (Weighted):    {recall:.4f}")
print(f"💥 F1-Score (Weighted):  {f1:.4f}")


# Helper Functions
def helper(dis):
    """Retrieve description, precautions, medications, diet, and workouts for a disease."""
    desc = " ".join(description[description["Disease"] == dis]["Description"])
    pre = precautions[precautions["Disease"] == dis][["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]].values.flatten().tolist()
    med = medications[medications["Disease"] == dis]["Medication"].tolist()
    die = diets[diets["Disease"] == dis]["Diet"].tolist()
    wrk = workout[workout["disease"] == dis]["workout"].tolist()
    return desc, pre, med, die, wrk


def get_predicted_value(patient_symptoms):
    """Convert symptom list to a severity-weighted input vector and predict disease."""
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        clean_item = item.strip().replace("_", " ").lower()
        if clean_item in symptoms_dict:
            input_vector[symptoms_dict[clean_item]] = 1

    # Apply severity weights
    for i, symptom in enumerate(symptoms_dict.keys()):
        if symptom in severity_weights:
            input_vector[i] *= severity_weights[symptom]

    input_vector = input_vector.reshape(1, -1)
    return svc.predict(input_vector)[0]