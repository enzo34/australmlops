import pandas as pd
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
from sklearn.metrics import classification_report

# Load the data
with open('api/data.json', encoding='utf-8') as file:
    data = json.load(file)

# Create a DataFrame
df = pd.DataFrame(data)

# Nettoyage des données
medians = df.select_dtypes(include=['int64', 'float64']).median()
df.fillna(medians, inplace=True)
for column in df.select_dtypes(include=['object']).columns:
    df[column].fillna(df[column].mode()[0], inplace=True)
df = df[df['age'] < 100]  
df['age'] = df['age'].astype(int)

# Prétraitement des données
categorical_features = ['sexe', 'situation_familiale', 'type_de_vehicule', 'historique_accidents', 'usage_vehicule', 'couverture_souhaitee']
numerical_features = ['age', 'enfants_a_charge', 'experience_de_conduite']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(random_state=42))])

# Séparation des caractéristiques et de la cible
X = df.drop('offre_recommandee', axis=1)
y = df['offre_recommandee']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Validation croisée
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f'Scores de validation croisée : {cv_scores}')
print(f'Moyenne des scores de validation croisée : {cv_scores.mean():.2f}')

# Optimisation des hyperparamètres
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f'Meilleurs paramètres : {grid_search.best_params_}')
print(f'Meilleur score de validation croisée : {grid_search.best_score_:.2f}')

# Évaluation sur l'ensemble de test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Sauvegarde du meilleur modèle
dump(best_model, 'api/best_rf_model_pipeline.joblib')


