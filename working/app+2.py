# main_app.py
import sys
import pandas as pd
import xgboost as xgb
import requests
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import pytz

# --- ☀️ CONFIGURATION ☀️ ---
# --- STEP 1: Set the capacity (in kW) of the solar panel system that
# --- generated your historical data file. For example: 3.5, 5.0, 10.0
HISTORICAL_SYSTEM_CAPACITY_KW = 5.0 

# --- File paths for your data ---
SOLAR_DATA_CSV = "sample_inverter_data_large.csv"
HISTORICAL_WEATHER_CSV = "weather_data_corrected_2024.csv"
MODEL_FILE_NAME = "solar_model.json"

# ==============================================================================
# === 🤖 1. MODEL TRAINING LOGIC 🤖 ============================================
# ==============================================================================

def create_normalized_model():
    """
    This function contains all the logic from the old 'model_trainer.py'.
    It creates and saves the correctly normalized model.
    """
    print("--- ⚙️ Starting Model Training Process ⚙️ ---")
    
    # --- Data Loading Functions ---
    def load_and_clean_solar_data(filepath):
        print(f"Loading and cleaning solar data from '{filepath}'...")
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        df_complete = df.reindex(full_date_range)
        df_complete['kwh'].fillna(0, inplace=True)
        return df_complete

    def load_historical_weather_data(filepath):
        print(f"Loading historical weather data from '{filepath}'...")
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['datetime'])
        df = df.set_index('timestamp')
        df = df[['temp', 'humidity', 'cloudcover', 'solarradiation']]
        df = df.rename(columns={'cloudcover': 'cloud_cover'})
        return df

    # --- Data Preparation Function (Crucial Normalization Step) ---
    def prepare_data(solar_df, weather_df):
        print("Combining and normalizing data...")
        df_combined = solar_df.join(weather_df, how='inner')
        df_combined.dropna(inplace=True)
        
        print(f"Normalizing kWh data using historical system capacity of {HISTORICAL_SYSTEM_CAPACITY_KW} kW...")
        df_combined['kwh_per_kw'] = df_combined['kwh'] / HISTORICAL_SYSTEM_CAPACITY_KW

        df_combined['hour'] = df_combined.index.hour
        df_combined['day_of_year'] = df_combined.index.dayofyear
        df_combined['month'] = df_combined.index.month
        
        features = ['solarradiation', 'temp', 'cloud_cover', 'humidity', 'hour', 'day_of_year', 'month']
        target = 'kwh_per_kw'
        
        X = df_combined[features]
        y = df_combined[target]
        print("✅ Data preparation complete.")
        return X, y

    # --- Model Training Function ---
    def train_and_save_model(X, y):
        print("Training the NORMALIZED prediction model...")
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, early_stopping_rounds=50, n_jobs=-1)
        model.fit(X, y, eval_set=[(X, y)], verbose=False)
        model.save_model(MODEL_FILE_NAME)
        print(f"✅ --- 🎉 Normalized model trained and saved as '{MODEL_FILE_NAME}'! 🎉 ---")

    # --- Run the training process ---
    solar_df = load_and_clean_solar_data(SOLAR_DATA_CSV)
    weather_df = load_historical_weather_data(HISTORICAL_WEATHER_CSV)
    
    if solar_df is not None and weather_df is not None:
        X, y = prepare_data(solar_df, weather_df)
        if X is not None:
            train_and_save_model(X, y)

# ==============================================================================
# === 🌍 2. FLASK WEB APPLICATION LOGIC 🌍 ======================================
# ==============================================================================

app = Flask(__name__)

# --- Load the model at startup ---
try:
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE_NAME)
    print(f"✅ Pre-trained normalized model '{MODEL_FILE_NAME}' loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not load model file '{MODEL_FILE_NAME}'.")
    print(f"❌ --- Please run 'python main_app.py train' first! ---")
    model = None

# --- Helper Functions for the Web App ---
def map_weather_code(code):
    if code == 0: return "Clear sky ☀️"
    if code == 1: return "Mainly clear 🌤️"
    if code == 2: return "Partly cloudy ⛅️"
    if code == 3: return "Overcast ☁️"
    if code in [45, 48]: return "Fog 🌫️"
    if code in [51, 53, 55]: return "Drizzle 🌦️"
    if code in [61, 63, 65]: return "Rain 🌧️"
    if code in [80, 81, 82]: return "Rain showers 🌧️"
    if code == 95: return "Thunderstorm ⛈️"
    return "Unknown"

def get_location_details(city_name):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    response = requests.get(geo_url)
    response.raise_for_status()
    data = response.json()
    if not data.get('results'):
        raise ValueError(f"Could not find coordinates for '{city_name}'.")
    location = data['results'][0]
    return location['latitude'], location['longitude'], location.get('timezone', 'UTC')

def get_weather_forecast(lat, lon, timezone):
    hourly_params = "temperature_2m,relativehumidity_2m,apparent_temperature,cloudcover,shortwave_radiation,weathercode,windspeed_10m,visibility"
    daily_params = "sunrise,sunset"
    forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly={hourly_params}&daily={daily_params}&timezone={timezone}"
    response = requests.get(forecast_url)
    response.raise_for_status()
    return response.json()

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first by running 'python main_app.py train'"}), 500

    try:
        data = request.get_json()
        city = data['city']
        capacity = float(data['capacity'])

        lat, lon, timezone = get_location_details(city)
        forecast_response = get_weather_forecast(lat, lon, timezone)
        
        hourly_data = forecast_response['hourly']
        daily_data = forecast_response['daily']

        df_forecast = pd.DataFrame(hourly_data)
        df_forecast['timestamp'] = pd.to_datetime(df_forecast['time'])
        df_forecast = df_forecast.set_index('timestamp')
        
        df_forecast = df_forecast.rename(columns={'shortwave_radiation': 'solarradiation', 'temperature_2m': 'temp','cloudcover': 'cloud_cover', 'relativehumidity_2m': 'humidity', 'apparent_temperature': 'feels_like', 'windspeed_10m': 'windspeed', 'weathercode': 'weather_code'})

        df_forecast['hour'] = df_forecast.index.hour
        df_forecast['day_of_year'] = df_forecast.index.dayofyear
        df_forecast['month'] = df_forecast.index.month
        
        required_features = ['solarradiation', 'temp', 'cloud_cover', 'humidity', 'hour', 'day_of_year', 'month']
        normalized_predictions = model.predict(df_forecast[required_features])
        df_forecast['predicted_kwh'] = normalized_predictions * capacity
        df_forecast.loc[df_forecast['predicted_kwh'] < 0, 'predicted_kwh'] = 0

        now_in_timezone = datetime.now(pytz.timezone(timezone))
        current_hour_naive = now_in_timezone.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        
        idx = df_forecast.index.get_indexer([current_hour_naive], method='nearest')[0]
        current_weather_data = df_forecast.iloc[idx]
        
        todays_predictions = df_forecast[df_forecast.index.date == now_in_timezone.date()]
        
        total_prediction = todays_predictions['predicted_kwh'].sum()
        peak_kwh = todays_predictions['predicted_kwh'].max()
        peak_hour_time = todays_predictions['predicted_kwh'].idxmax()
        productive_hours = todays_predictions[todays_predictions['predicted_kwh'] > 0]
        avg_kwh = productive_hours['predicted_kwh'].mean() if not productive_hours.empty else 0
        
        today_daily_index = pd.to_datetime(daily_data['time']).get_loc(now_in_timezone.strftime('%Y-%m-%d'))
        sunrise_iso = daily_data['sunrise'][today_daily_index]
        sunset_iso = daily_data['sunset'][today_daily_index]

        output = {
            "city": city.title(), "total": f"{total_prediction:.2f}",
            "labels": [t.strftime('%I %p').lstrip('0') for t in todays_predictions.index],
            "data": [round(p, 2) for p in todays_predictions['predicted_kwh']],
            "current_weather": {"temperature": f"{current_weather_data['temp']:.1f}", "feels_like": f"{current_weather_data['feels_like']:.1f}", "humidity": int(current_weather_data['humidity']), "windspeed": f"{current_weather_data['windspeed']:.1f}", "description": map_weather_code(current_weather_data['weather_code']), "visibility": f"{current_weather_data['visibility'] / 1000:.1f}"},
            "daily_info": {"sunrise": datetime.fromisoformat(sunrise_iso).strftime('%I:%M %p').lstrip('0'), "sunset": datetime.fromisoformat(sunset_iso).strftime('%I:%M %p').lstrip('0'), "avg_temp": f"{todays_predictions['temp'].mean():.1f}", "avg_cloud": f"{int(todays_predictions['cloud_cover'].mean())}", "avg_radiation": f"{int(productive_hours['solarradiation'].mean() if not productive_hours.empty else 0)}", "avg_humidity": f"{int(todays_predictions['humidity'].mean())}"},
            "stats": {"peak_hour": peak_hour_time.strftime('%I %p').lstrip('0'), "peak_kwh": f"{peak_kwh:.2f}", "avg_kwh": f"{avg_kwh:.2f}", "efficiency": f"{min(100, (peak_kwh / capacity) * 100):.0f}"}
        }
        return jsonify(output)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

# ==============================================================================
# === 🚀 3. SCRIPT RUNNER 🚀 ==================================================
# ==============================================================================

if __name__ == '__main__':
    # This allows you to run 'python main_app.py train' or 'python main_app.py run'
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            create_normalized_model()
        elif sys.argv[1] == 'run':
            app.run(debug=True)
        else:
            print("Invalid command. Use 'train' or 'run'.")
    else:
        print("Please specify a command: 'train' to create the model, or 'run' to start the web server.")

