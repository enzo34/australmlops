# Application Goal
This application is designed for insurance companies to help determine the most suitable offer for users based on their provided data. For example, consider a user input represented in JSON format:

```json
{
    "age": 35,
    "sexe": "M",
    "situation_familiale": "Divorcé(e)",
    "enfants_a_charge": 1,
    "type_de_vehicule": "Moto",
    "experience_de_conduite": 10,
    "historique_accidents": "Aucun",
    "usage_vehicule": "Les deux",
    "couverture_souhaitee": "Totale"
}
```

The application starts as a simple form, which is planned to be converted into Web Components for use across any application via a CDN. To initiate the application for testing, the following command can be run:

```bash
docker-compose up -d
```

Ensure Docker is installed on your machine. The application will be accessible at the address http://localhost.

# Machine Learning

The core functionality of this application is powered by a Flask API, integrated with a Machine Learning model utilizing a Random Forest Classifier. This choice is made due to the Random Forest's effectiveness in handling classification tasks, capable of processing the input data (such as age, sex, marital status, type of vehicle, driving experience, accident history, vehicle usage, and desired coverage) to predict the most suitable insurance offer.

The Random Forest Classifier works by constructing multiple decision trees during the training phase and outputs the class that is the mode of the classes (classification) of the individual trees. It is renowned for its accuracy and ability to deal with unbalanced data from real-world scenarios.

The application leverages a dataset stored in JSON format, initially filled with Dummy Data to mimic real user inputs closely. This approach ensures the application's design remains grounded in practicality, albeit the data does not represent real individuals.

## Code Explanation

Without delving into the specific code operations, the application functions through several steps:

1. **Data Preparation:** The user inputs are encoded into a numerical format suitable for machine learning models, transforming categorical data (like sex, marital status, etc.) into numerical values using `LabelEncoder`.

2. **Model Training:** The application splits the data into training and testing sets, using the training set to teach the Random Forest Classifier how to predict the most suitable offer based on the user's data.

3. **Evaluation:** The model's performance is assessed using accuracy metrics, determining how well it can predict the correct offers on a set of test data.

4. **Deployment:** Trained models and their respective encoders are saved for future use, allowing the Flask API to make predictions on new user data as it comes.

The application stands as a functional prototype requiring further testing with real data and enhancements to the base algorithm to refine its predictions. For those interested in the underpinnings of classification models, a recommended article provides in-depth insights into the mathematics of decision trees, random forests, and feature importance within machine learning frameworks.
