from flask import Flask, request, render_template
import pandas as pd, numpy as np, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ================== Train or load model ==================
model_path = "models/svc.pkl"
if not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    data = pd.read_csv("Training.csv")
    X, y = data.drop("prognosis", axis=1), data["prognosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svc = SVC(kernel="linear", probability=True)
    svc.fit(X_train, y_train)
    pickle.dump(svc, open(model_path, "wb"))
else:
    svc = pickle.load(open(model_path, "rb"))

# ================== Load datasets ==================
sym_des, precautions, workout = pd.read_csv("symtoms_df.csv"), pd.read_csv("precautions_df.csv"), pd.read_csv("workout_df.csv")
description, medications, diets = pd.read_csv("description.csv"), pd.read_csv("medications.csv"), pd.read_csv("diets.csv")

# ================== Helpers ==================
def helper(dis):
    desc = " ".join(description[description['Disease'] == dis]['Description'])
    pre = precautions[precautions['Disease']==dis][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']].values.flatten().tolist()
    med = medications[medications['Disease']==dis]['Medication'].tolist()
    die = diets[diets['Disease']==dis]['Diet'].tolist()
    wrk = workout[workout['disease']==dis]['workout'].tolist()
    return desc, pre, med, die, wrk

# Example symptom-disease mapping
symptoms_dict = {'itching':0,'skin_rash':1,'nodal_skin_eruptions':2}  # add full dict
diseases_list = {15:'Fungal infection', 4:'Allergy', 16:'GERD'}      # add full dict

def get_predicted_value(symptoms):
    vec = np.zeros(len(symptoms_dict))
    for s in symptoms:
        if s in symptoms_dict: vec[symptoms_dict[s]] = 1
    return diseases_list[svc.predict([vec])[0]]

# ================== Flask app ==================
app = Flask(__name__)

@app.route("/")
def index(): return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms','').strip().lower()
    if not symptoms or symptoms=="symptoms": 
        return render_template('index.html', message="⚠️ Please enter valid symptoms (comma-separated).")
    user_symptoms = [s.strip() for s in symptoms.split(",") if s.strip()]
    pred = get_predicted_value(user_symptoms)
    desc, pre, med, die, wrk = helper(pred)
    return render_template('index.html', predicted_disease=pred, dis_des=desc, my_precautions=pre, medications=med, my_diet=die, workout=wrk)

# Simple static pages
for page in ["about","contact","developer","blog"]:
    app.add_url_rule(f'/{page}', page, lambda p=page: render_template(f"{p}.html"))

if __name__=='__main__':
    app.run(debug=True)
