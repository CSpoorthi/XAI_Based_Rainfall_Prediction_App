import pandas as pd
import joblib

def predict_crop_item(X_crop_new):
    """
    Predicts crop items for new data using a trained RandomForest model.

    Args:
    X_crop_new (DataFrame): DataFrame containing new data features.
    model_path (str): Path to the trained RandomForest model file (default is 'rf_item_model.pkl').

    Returns:
    str: Predicted crop name.
    
    """
    model_path='rf_item_model.pkl'
    # Load the trained RandomForest model for crop item prediction
    rf_item_model = joblib.load(model_path)

    # Predict crop items using the loaded model
    predicted_items= rf_item_model.predict(X_crop_new)

    # Define a dictionary to map integer labels to crop names
    crop_names = {
        0: 'Maize',
        1: 'Potatoes',
        2: 'Rice',
        3: 'Sorghum',
        4: 'Soybeans',
        5: 'Sweet Potatoes',
        6: 'Wheat'
    }

    # Map the predicted index to the crop name
    predicted_crop_name = crop_names[predicted_items[0]]

    return predicted_crop_name

