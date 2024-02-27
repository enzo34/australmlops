import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the data
with open('data.json', encoding='utf-8') as file:
    data = json.load(file)

# Create a DataFrame
df = pd.DataFrame(data)

label_encoders = {}

for column in ['sexe', 'situation_familiale', 'type_de_vehicule', 'historique_accidents', 'usage_vehicule', 'couverture_souhaitee', 'offre_recommandee']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop('offre_recommandee', axis=1)
y = df['offre_recommandee']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

# Calculer la précision du modèle
accuracy = rf_model.score(X_test, y_test)
print(f'Précision du modèle sur les données de test : {accuracy:.2f}')

# Sauvegarder le modèle Random Forest
dump(rf_model, 'rf_model.joblib')

# Sauvegarder chaque LabelEncoder
for column, encoder in label_encoders.items():
    dump(encoder, f'{column}_encoder.joblib')



