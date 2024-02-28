from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from flask_cors import CORS

# Charger le modèle (qui est un pipeline incluant le prétraitement et le modèle RandomForestClassifier)
rf_model = load('api/best_rf_model_pipeline.joblib')

app = Flask(__name__)
CORS(app)  # Activez CORS pour toutes les routes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Transformer les données d'entrée en DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Faire une prédiction avec le modèle
    # Le pipeline prendra soin de toutes les transformations nécessaires
    prediction = rf_model.predict(df)
    
    # Comme le modèle est maintenant dans un pipeline avec les transformations incluses,
    # il n'y a pas besoin de transformer la prédiction avec LabelEncoder
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
