import pandas as pd
import numpy as np
from train import get_predicted_value, helper, symptoms_dict, svc


# Load Symptom Severity Data
severity = pd.read_csv("Symptom_severity.csv")

# Clean and normalize column names
severity.columns = [col.strip().lower() for col in severity.columns]
severity["symptom"] = severity["symptom"].str.strip().str.replace("_", " ")

# Create dictionary for symptom severity weights
severity_weights = {row["symptom"]: row["weight"] for _, row in severity.iterrows()}



# Weighted Prediction Function
def get_weighted_prediction(patient_symptoms):
    """
    Converts symptom list into a severity-weighted input vector and predicts disease.
    """
    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        clean_item = item.strip().replace("_", " ").lower()
        idx = None

        # Match user symptom to dataset
        if item in symptoms_dict:
            idx = symptoms_dict[item]
        else:
            # Try fuzzy match ignoring underscores/spaces
            for key in symptoms_dict.keys():
                if key.replace("_", " ").lower() == clean_item:
                    idx = symptoms_dict[key]
                    break

        # If no valid match, skip this symptom
        if idx is None:
            continue

        # Apply severity weight if available
        weight = severity_weights.get(clean_item, 1)
        input_vector[idx] = weight

    input_vector = input_vector.reshape(1, -1)
    return svc.predict(input_vector)[0]



# Main User Interaction
if __name__ == "__main__":
    print("=== 🧬 Disease Prediction System ===")
    print("Enter symptoms separated by commas (e.g., itching, skin rash, nodal skin eruptions):")

    user_input = input("Symptoms: ").strip()
    if not user_input:
        print("⚠️ No symptoms entered. Please try again.")
        exit()

    # Split input into a clean list of symptoms
    test_symptoms = [s.strip() for s in user_input.split(",") if s.strip()]

    # Predict disease
    disease = get_weighted_prediction(test_symptoms)

    # Get details
    desc, pre, med, die, wrk = helper(disease)

    # Display results
    print("\n=== 🧠 Diagnosis Result ===")
    print("🩺 Predicted Disease:", disease)
    print("\n📖 Description:", desc if desc else "No description available.")
    print("\n⚕️ Precautions:", ", ".join(pre) if pre else "No precautions available.")
    print("\n💊 Medications:", ", ".join(med) if med else "No medications listed.")
    print("\n🥗 Diet:", ", ".join(die) if die else "No diet recommendations.")
    print("\n🏋️ Workout:", ", ".join(wrk) if wrk else "No workout suggestions.")


