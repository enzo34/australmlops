# Importing necessary libraries
from flask import Flask, request, jsonify  # Flask is a web framework, request is used to handle HTTP requests, jsonify is used to convert Python objects to JSON format
import pandas as pd  # pandas is a library for data manipulation and analysis
from joblib import load  # joblib is used to load pre-trained machine learning models
from flask_cors import CORS  # CORS is used to handle Cross-Origin Resource Sharing

# Loading the pre-trained random forest model
rf_model = load('best_rf_model_pipeline.joblib')

# Creating a Flask web application
app = Flask(__name__)
CORS(app)  # Enabling CORS for the Flask app

# Defining a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Getting the input data as JSON format
    data = request.get_json(force=True)
    
    # Converting the JSON data to a pandas DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Making predictions using the pre-trained random forest model
    prediction = rf_model.predict(df)
    
    # Returning the prediction as JSON format
    return jsonify({'prediction': prediction[0]})

# Running the Flask app if this file is executed directly
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
