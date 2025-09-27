import os
import requests
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from flask import Flask, render_template, jsonify

# --- CONFIGURATION (FOR GOOGLE API) ---
GOOGLE_API_KEY = "Apikey"
LATITUDE = 9.4716
LONGITUDE = 77.5569
SOLAR_DATA_CSV = "sample_inverter_data_large.csv"
HISTORICAL_WEATHER_CSV = "weather_data_corrected_2024.csv"

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
solar_model = None

# --- MACHINE LEARNING FUNCTIONS ---
def load_data_and_train_model():
    global solar_model
    print("--- Initializing Model ---")
    solar_df = pd.read_csv(SOLAR_DATA_CSV, parse_dates=['timestamp']).set_index('timestamp')
    weather_df = pd.read_csv(HISTORICAL_WEATHER_CSV, parse_dates=['datetime']).set_index('datetime')
    weather_df = weather_df.rename(columns={'temp': 'temperature', 'humidity': 'humidity', 'clouds': 'cloudCover', 'solarradiation': 'shortwaveRadiation'})
    df_combined = solar_df.join(weather_df, how='inner')
    df_combined.dropna(inplace=True)
    df_combined['hour'] = df_combined.index.hour
    df_combined['day_of_year'] = df_combined.index.dayofyear
    df_combined['month'] = df_combined.index.month
    features = ['shortwaveRadiation', 'temperature', 'cloudCover', 'humidity', 'hour', 'day_of_year', 'month']
    target = 'kwh'
    X, y = df_combined[features], df_combined[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, early_stopping_rounds=50, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"📈 Model Trained. MAE: {mae:.4f} kWh")
    solar_model = model
    print("✅ Model is ready.")

# --- FINAL CORRECTED PREDICTION FUNCTION (GOOGLE API) ---
def get_prediction_data():
    if not solar_model: return {"error": "Model not trained."}
    print("Fetching weather data from Google Weather API...")
    
    # Corrected GET request format
    base_params = f"location.latitude={LATITUDE}&location.longitude={LONGITUDE}"
    try:
        # Get Forecast
        forecast_url = f"https://weather.googleapis.com/v1/forecast:buildHourly?{base_params}&hourlyForecastParams=temperature&hourlyForecastParams=relativeHumidity&hourlyForecastParams=cloudCover&hourlyForecastParams=shortwaveRadiation&key={GOOGLE_API_KEY}"
        forecast_response = requests.get(forecast_url)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        # Get Current Conditions
        current_url = f"https://weather.googleapis.com/v1/currentConditions:lookup?{base_params}&key={GOOGLE_API_KEY}"
        current_response = requests.get(current_url)
        current_response.raise_for_status()
        current_data = current_response.json()
    except requests.exceptions.RequestException as e:
        error_info = e.response.json() if e.response else str(e)
        return {"error": f"Could not fetch weather data: {error_info}"}

    hourly_forecasts = forecast_data.get('hourlyForecasts', [])
    if not hourly_forecasts: return {"error": "No hourly forecast data in API response."}
    
    df_forecast = pd.DataFrame([dict(item, **{'timestamp': item['dateTime']}) for item in hourly_forecasts])
    df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])
    df_forecast = df_forecast.set_index('timestamp')
    df_forecast['hour'], df_forecast['day_of_year'], df_forecast['month'] = df_forecast.index.hour, df_forecast.index.dayofyear, df_forecast.index.month
    
    required_features = ['shortwaveRadiation', 'temperature', 'cloudCover', 'humidity', 'hour', 'day_of_year', 'month']
    df_forecast_features = df_forecast.reindex(columns=required_features).fillna(0)
    df_forecast['predicted_kwh'] = solar_model.predict(df_forecast_features).clip(0)
    
    todays_predictions = df_forecast[df_forecast.index.strftime('%Y-%m-%d') == datetime.now().strftime('%Y-%m-%d')]
    total_kwh = todays_predictions['predicted_kwh'].sum()
    hourly_json = [{'time': t.strftime('%I %p'), 'kwh': round(r['predicted_kwh'], 2)} for t, r in todays_predictions.iterrows()]

    if total_kwh < 5: day_type, suggestion = "Low Generation", "Defer high-power tasks."
    elif 5 <= total_kwh < 15: day_type, suggestion = "Medium Generation", "Good for some appliances."
    else: day_type, suggestion = "High Generation", "Maximize solar usage!"
    
    peak_hours = todays_predictions[todays_predictions['predicted_kwh'] >= todays_predictions['predicted_kwh'].max() * 0.7] if total_kwh > 0 else pd.DataFrame()
    peak_time_str = f"{peak_hours.index.min().strftime('%I %p')} to {peak_hours.index.max().strftime('%I %p')}" if not peak_hours.empty else "N/A"
    
    google_current = current_data.get('currentConditions', {})
    current_weather = {"temperature": google_current.get('temperature'), "weathercode": google_current.get('iconCode'), "windspeed": google_current.get('windSpeed')}

    return {"total_kwh": round(total_kwh, 2), "hourly_predictions": hourly_json, "current_weather": current_weather, "suggestions": {"day_type": day_type, "recommendation": suggestion, "peak_time": peak_time_str}}

@app.route('/')
def home(): return render_template('index.html')
@app.route('/api/prediction')
def api_prediction(): return jsonify(get_prediction_data())
if __name__ == "__main__":
    load_data_and_train_model()

    app.run(debug=True)
