from utils import db_connect
from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, "random_forest_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

scaler_path = os.path.join(current_dir, "scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

label_encoders_path = os.path.join(current_dir, "label_encoders.pkl")
with open(label_encoders_path, "rb") as f:
    label_encoders = pickle.load(f)

def prepare_features(form_data, scaler, label_encoders):
    area = float(form_data['area'])
    bedrooms = int(form_data['bedrooms'])
    bathrooms = int(form_data['bathrooms'])
    stories = int(form_data['stories'])
    parking = int(form_data['parking'])
    mainroad = 1 if form_data['mainroad'] == 'yes' else 0
    guestroom = 1 if form_data['guestroom'] == 'yes' else 0
    basement = 1 if form_data['basement'] == 'yes' else 0
    hotwaterheating = 1 if form_data['hotwaterheating'] == 'yes' else 0
    airconditioning = 1 if form_data['airconditioning'] == 'yes' else 0
    prefarea = 1 if form_data['prefarea'] == 'yes' else 0
    furnishingstatus = label_encoders['furnishingstatus'].transform([form_data['furnishingstatus']])[0]
    price_category = label_encoders['price_category'].transform([form_data['price_category']])[0]

    features = np.array([[area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement,
                          hotwaterheating, airconditioning, prefarea, furnishingstatus, price_category]])

    return scaler.transform(features)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            features_scaled = prepare_features(request.form, scaler, label_encoders)
            predicted_price = model.predict(features_scaled)[0]
            return render_template('index.html', predicted_price=predicted_price)
        except Exception as e:
            return render_template('index.html', error="Ocurrió un error al procesar los datos. Por favor, verifica tus entradas.")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

engine = db_connect()
