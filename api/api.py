from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import load
from flask_cors import CORS

# Charger le modèle
rf_model = load('rf_model.joblib')

# Charger les LabelEncoder
label_encoders = {}
for column in ['sexe', 'situation_familiale', 'type_de_vehicule', 'historique_accidents', 'usage_vehicule', 'couverture_souhaitee', 'offre_recommandee']:
    label_encoders[column] = load(f'{column}_encoder.joblib')

app = Flask(__name__)
CORS(app)  # Activez CORS pour toutes les routes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Transformer les données d'entrée
    df = pd.DataFrame(data, index=[0])
    df['sexe'] = label_encoders['sexe'].transform(df['sexe'])
    df['situation_familiale'] = label_encoders['situation_familiale'].transform(df['situation_familiale'])
    df['type_de_vehicule'] = label_encoders['type_de_vehicule'].transform(df['type_de_vehicule'])
    df['historique_accidents'] = label_encoders['historique_accidents'].transform(df['historique_accidents'])
    df['usage_vehicule'] = label_encoders['usage_vehicule'].transform(df['usage_vehicule'])
    df['couverture_souhaitee'] = label_encoders['couverture_souhaitee'].transform(df['couverture_souhaitee'])
    
    prediction = rf_model.predict(df)
    prediction_label = label_encoders['offre_recommandee'].inverse_transform(prediction)
    
    return jsonify({'prediction': prediction_label[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    'age': 30,
    'sexe': 'M',
    'situation_familiale': 'Célibataire',
    'enfants_a_charge': 0,
    'type_de_vehicule': 'Voiture',
    'experience_de_conduite': 5,
    'historique_accidents': 'Aucun',
    'usage_vehicule': 'Personnel',
    'couverture_souhaitee': 'Basique'
}

response = requests.post(url, json=data)
print(response.json())

