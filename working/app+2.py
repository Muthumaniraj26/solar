# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
import xgboost as xgb
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_FILE_NAME = "solar_model.json"
model = xgb.XGBRegressor()
model.load_model(MODEL_FILE_NAME)
print("✅ Pre-trained model loaded successfully.")

def get_coordinates(city_name):
    """Geocodes a city name to latitude and longitude."""
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    response = requests.get(geo_url)
    response.raise_for_status()
    data = response.json()
    if not data.get('results'):
        raise ValueError(f"Could not find coordinates for '{city_name}'. Please check the spelling.")
    
    location = data['results'][0]
    return location['latitude'], location['longitude']

def get_weather_forecast(lat, lon):
    """Fetches weather forecast for a given location."""
    forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m,cloudcover,shortwave_radiation"
    response = requests.get(forecast_url)
    response.raise_for_status()
    return response.json()['hourly']

@app.route('/')
def index():
    """Renders the main dashboard page."""
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives a city, fetches data, and returns a prediction."""
    try:
        data = request.get_json()
        city = data['city']
        
        lat, lon = get_coordinates(city)
        forecast_data = get_weather_forecast(lat, lon)

        df_forecast = pd.DataFrame(forecast_data)
        df_forecast['timestamp'] = pd.to_datetime(df_forecast['time'])
        df_forecast = df_forecast.set_index('timestamp')
        
        df_forecast = df_forecast.rename(columns={
            'shortwave_radiation': 'solarradiation', 'temperature_2m': 'temp',
            'cloudcover': 'cloud_cover', 'relativehumidity_2m': 'humidity'
        })

        df_forecast['hour'] = df_forecast.index.hour
        df_forecast['day_of_year'] = df_forecast.index.dayofyear
        df_forecast['month'] = df_forecast.index.month
        
        required_features = ['solarradiation', 'temp', 'cloud_cover', 'humidity', 'hour', 'day_of_year', 'month']
        predictions = model.predict(df_forecast[required_features])
        df_forecast['predicted_kwh'] = predictions
        df_forecast.loc[df_forecast['predicted_kwh'] < 0, 'predicted_kwh'] = 0

        today_str = datetime.now().strftime('%Y-%m-%d')
        todays_predictions = df_forecast[df_forecast.index.strftime('%Y-%m-%d') == today_str]
        
        total_prediction = todays_predictions['predicted_kwh'].sum()
        
        output = {
            "total": f"{total_prediction:.2f}",
            "city": city.title(),
            # --- THIS IS THE CORRECTED LINE ---
            "labels": [t.strftime('%I %p').lstrip('0') for t in todays_predictions.index],
            "data": [round(p, 2) for p in todays_predictions['predicted_kwh']]
        }
        
        return jsonify(output)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)