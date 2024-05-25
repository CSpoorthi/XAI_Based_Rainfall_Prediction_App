from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from final_model import predict_combined, explain_prediction, generate_feature_importance_plot
from crop_pred import predict_crop_item

app = Flask(__name__)

# Route to display the form
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/registration')
def registration():
    return render_template('registration.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/indata')
def indata():
    return render_template('indata.html')

# Route to handle form submission and display prediction
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs from the form
        inputs = {
            'Day': int(request.form['Day']),
            'Month': int(request.form['Month']),
            'Year': int(request.form['Year']),
            'Location': request.form['Location'],
            'MinTemp': float(request.form['MinTemp']),
            'MaxTemp': float(request.form['MaxTemp']),
            'Rainfall': float(request.form['Rainfall']),
            'Evaporation': float(request.form['Evaporation']),
            'Sunshine': float(request.form['Sunshine']),
            'WindGustDir': request.form['WindGustDir'],
            'WindGustSpeed': float(request.form['WindGustSpeed']),
            'WindDir9am': request.form['WindDir9am'],
            'WindDir3pm': request.form['WindDir3pm'],
            'WindSpeed9am': float(request.form['WindSpeed9am']),
            'WindSpeed3pm': float(request.form['WindSpeed3pm']),
            'Humidity9am': float(request.form['Humidity9am']),
            'Humidity3pm': float(request.form['Humidity3pm']),
            'Cloud9am': float(request.form['Cloud9am']),
            'Cloud3pm': float(request.form['Cloud3pm']),
            'AvgTemp': float(request.form['AvgTemp']),
            'AvgPressure': float(request.form['AvgPressure']),
            'RainToday': request.form['RainToday']
        }
        input_data = pd.DataFrame([inputs])
        hg_yield = float(request.form['hg/ha_yield'])
        # Make combined predictions using both models
        combined_prediction = predict_combined(input_data)

        # Explain the prediction using the same input data
        explanation = explain_prediction(input_data)
        # Generate feature importances plot
        feature_importance_plot = generate_feature_importance_plot(input_data)

        # Predict crop item using only the yield value and relevant features
        X_new_crop = pd.DataFrame([[inputs['Year'], hg_yield, inputs['AvgTemp'], inputs['Rainfall']]], columns=['Year', 'hg/ha_yield', 'AvgTemp', 'Rainfall'])
        predicted_crop_name = predict_crop_item(X_new_crop)

        return render_template('prediction.html', prediction=combined_prediction,explanation=explanation,feature_importance_plot=feature_importance_plot,predicted_crop=predicted_crop_name )



if __name__ == '__main__':
    app.run(debug=True)
