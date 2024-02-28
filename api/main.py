# Importing necessary libraries
import pandas as pd
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
from sklearn.metrics import classification_report

# Opening and loading the JSON data file
with open('data.json', encoding='utf-8') as file:
    data = json.load(file)

# Creating a DataFrame from the loaded JSON data
df = pd.DataFrame(data)

# Handling missing values by replacing them with the median of each numerical column
medians = df.select_dtypes(include=['int64', 'float64']).median()
df.fillna(medians, inplace=True)

# Handling missing values in categorical columns by replacing them with the mode (most frequent value)
for column in df.select_dtypes(include=['object']).columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Filtering out rows where the 'age' column is greater than or equal to 100
df = df[df['age'] < 100]

# Converting the 'age' column to integer type
df['age'] = df['age'].astype(int)

# Defining the categorical and numerical features
categorical_features = ['sexe', 'situation_familiale', 'type_de_vehicule', 'historique_accidents', 'usage_vehicule', 'couverture_souhaitee']
numerical_features = ['age', 'enfants_a_charge', 'experience_de_conduite']

# Creating transformers for categorical and numerical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

# Creating a preprocessor that applies the transformers to the respective features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Creating a pipeline that combines the preprocessor and a random forest classifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(random_state=42))])

# Splitting the data into training and testing sets
X = df.drop('offre_recommandee', axis=1)
y = df['offre_recommandee']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Performing cross-validation on the pipeline
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

# Printing the cross-validation scores
print(f'Scores de validation croisée : {cv_scores}')

# Printing the average cross-validation score
print(f'Moyenne des scores de validation croisée : {cv_scores.mean():.2f}')

# Defining the parameter grid for grid search
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30]
}

# Performing grid search to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Printing the best hyperparameters found by grid search
print(f'Meilleurs paramètres : {grid_search.best_params_}')

# Printing the best cross-validation score found by grid search
print(f'Meilleur score de validation croisée : {grid_search.best_score_:.2f}')

# Getting the best model from grid search
best_model = grid_search.best_estimator_

# Making predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Printing the classification report
print(classification_report(y_test, y_pred))

# Saving the best model as a joblib file
dump(best_model, 'best_rf_model_pipeline.joblib')
