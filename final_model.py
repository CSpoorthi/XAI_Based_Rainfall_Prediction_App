import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from lime.lime_tabular import LimeTabularExplainer

# Load the dataset
dataset = pd.read_csv('cleaned_data.csv')

# Extract features and target variable
X = dataset[['Day','Month','Year','Location', 'MinTemp', 'MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
             'WindSpeed9am', 'WindSpeed3pm','Humidity9am','Humidity3pm','Cloud9am', 'Cloud3pm', 'AvgTemp', 'AvgPressure', 'RainToday']]
Y = dataset['RainTomorrow']

# Define feature names
feature_names = X.columns.tolist()

# Initialize LimeTabularExplainer with feature names
explainer = LimeTabularExplainer(X.values, feature_names=feature_names, discretize_continuous=True, mode='classification')

# Load models
rf_model = joblib.load('rf_classifier.pkl')
cb_model = joblib.load('catboost_model.pkl')

def predict_combined(input_data):
    input_data = input_data[X.columns].values  #Convert dataframe to Numpy array

    # Get predictions from both models
    rf_prediction = rf_model.predict(input_data)
    cb_prediction = cb_model.predict(input_data)

    # Combine predictions
    hybrid_pred_proba = (rf_prediction + cb_prediction) / 2
    
    # Convert prediction probabilities to binary predictions
    hybrid_preds = (hybrid_pred_proba > 0.5).astype(int)

    return hybrid_preds

def explain_prediction(input_data):
    # Convert non-numeric values to numeric
    input_df = input_data.apply(pd.to_numeric, errors='coerce')

    # Drop any rows with missing values
    input_df.dropna(inplace=True)

    # Explain the prediction using LimeTabularExplainer
    explanation = explainer.explain_instance(input_df.values[0], rf_model.predict_proba)

    # Get feature importances from Lime explanation
    importances_dict = dict(explanation.as_list())

    # Convert feature importances to human-readable format
    human_readable_explanations = []
    for feature, importance in importances_dict.items():
        feature_desc = f'"{feature}"'
        if feature.startswith('<='):
            feature_desc += f' with values less than or equal to {feature[2:]}'
        elif feature.startswith('>'):
            feature_desc += f' with values greater than {feature[1:]}'
        else:
            ranges = feature.split(' <= ')
            if len(ranges) == 2:
                feature_desc += f' with values between {ranges[0]} and {ranges[1]}'
        if importance > 0:
            human_readable_explanations.append(f"{feature_desc} seems to have a positive impact.")
        else:
            human_readable_explanations.append(f"{feature_desc} seems to have a negative impact.")

    return human_readable_explanations

def generate_feature_importance_plot(input_data):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data, columns=X.columns)

        # Ensure input data is numeric
        input_df = input_df.apply(pd.to_numeric, errors='coerce')

        # Drop rows with missing or non-numeric values
        input_df = input_df.dropna()

        # Compute feature importances using LimeTabularExplainer
        explanation = explainer.explain_instance(input_df.values[0], rf_model.predict_proba, num_features=len(X.columns))

        # Convert feature importances to DataFrame
        importances_dict = dict(explanation.as_list())
        importances_df = pd.DataFrame(importances_dict.items(), columns=['Feature', 'Importance'])

        # Sort feature importances
        importances_df = importances_df.sort_values(by='Importance', ascending=False)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importances_df, palette='viridis')
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        feature_importance_plot = base64.b64encode(img.getvalue()).decode()

        return feature_importance_plot

    except Exception as e:
        print(f"Error occurred while generating feature importance plot: {str(e)}")
        return None

    




