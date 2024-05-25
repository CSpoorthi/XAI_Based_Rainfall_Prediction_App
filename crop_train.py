import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('crops.csv')

# Drop any rows with missing values
data.dropna(inplace=True)

# Splitting the data into features and target
X = data.drop("Item", axis=1)  # Features
y_item = data['Item']  # Crop item target

# Splitting the data into train and test sets
X_train, X_test, y_item_train, y_item_test = train_test_split(
    X, y_item, test_size=0.2, random_state=42
)

# Training RandomForest classifier for crop item prediction
print("Training RandomForest Classifier for Crop Item Prediction...")
rf_item_model = RandomForestClassifier()
rf_item_model.fit(X_train, y_item_train)

# Predicting crop item using the RandomForest model
print("Predicting Crop Item...")
rf_item_preds = rf_item_model.predict(X_test)

# Calculating accuracy for crop item prediction
item_accuracy_rf = accuracy_score(y_item_test, rf_item_preds)
print("Crop Item Accuracy for RandomForest Classifier:", item_accuracy_rf)

# Saving RandomForest model
print("Saving RandomForest model...")
joblib.dump(rf_item_model, 'rf_item_model.pkl')

print("RandomForest model saved successfully!")
