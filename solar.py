import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from flask import Flask, render_template, jsonify
from datetime import datetime

# --- CONFIGURATION ---
SOLAR_DATA_CSV = "sample_inverter_data_large.csv"        # Your solar generation CSV
HISTORICAL_WEATHER_CSV = "weather_data.csv"             # Weather CSV (the one we created)

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
solar_model = None

# --- MACHINE LEARNING FUNCTIONS ---
def load_data_and_train_model():
    global solar_model
    print("--- Initializing Model ---")

    # Load CSVs
    solar_df = pd.read_csv(SOLAR_DATA_CSV, parse_dates=['timestamp']).set_index('timestamp')
    weather_df = pd.read_csv(HISTORICAL_WEATHER_CSV, parse_dates=['datetime']).set_index('datetime')

    # Rename columns to match ML model
    weather_df = weather_df.rename(columns={
        'temp': 'temperature',
        'humidity': 'humidity',
        'clouds': 'cloudCover',
        'solarradiation': 'shortwaveRadiation'
    })

    # Merge datasets on timestamp
    df_combined = solar_df.join(weather_df, how='inner')
    df_combined.dropna(inplace=True)

    # Add time features
    df_combined['hour'] = df_combined.index.hour
    df_combined['day_of_year'] = df_combined.index.dayofyear
    df_combined['month'] = df_combined.index.month

    # Features & target
    features = ['shortwaveRadiation', 'temperature', 'cloudCover', 'humidity', 'hour', 'day_of_year', 'month']
    target = 'kwh'

    X, y = df_combined[features], df_combined[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=1000, 
        learning_rate=0.05, 
        early_stopping_rounds=50, 
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"📈 Model Trained. MAE: {mae:.4f} kWh")
    solar_model = model
    print("✅ Model is ready.")

# --- PREDICTION FUNCTION USING CSV ---
def get_prediction_data():
    if not solar_model: 
        return {"error": "Model not trained."}

    # Load weather CSV
    weather_df = pd.read_csv(HISTORICAL_WEATHER_CSV, parse_dates=['datetime']).set_index('datetime')
    weather_df = weather_df.rename(columns={
        'temp': 'temperature',
        'humidity': 'humidity',
        'clouds': 'cloudCover',
        'solarradiation': 'shortwaveRadiation'
    })

    # Add time features
    weather_df['hour'] = weather_df.index.hour
    weather_df['day_of_year'] = weather_df.index.dayofyear
    weather_df['month'] = weather_df.index.month

    # Predict KWH
    required_features = ['shortwaveRadiation', 'temperature', 'cloudCover', 'humidity', 'hour', 'day_of_year', 'month']
    df_features = weather_df.reindex(columns=required_features).fillna(0)
    weather_df['predicted_kwh'] = solar_model.predict(df_features).clip(0)

    # Summarize
    total_kwh = weather_df['predicted_kwh'].sum()
    hourly_json = [{'time': t.strftime('%I %p'), 'kwh': round(r['predicted_kwh'], 2)} 
                   for t, r in weather_df.iterrows()]

    if total_kwh < 5:
        day_type, suggestion = "Low Generation", "Defer high-power tasks."
    elif 5 <= total_kwh < 15:
        day_type, suggestion = "Medium Generation", "Good for some appliances."
    else:
        day_type, suggestion = "High Generation", "Maximize solar usage!"

    # Peak hours
    peak_hours = weather_df[weather_df['predicted_kwh'] >= weather_df['predicted_kwh'].max() * 0.7]
    peak_time_str = f"{peak_hours.index.min().strftime('%I %p')} to {peak_hours.index.max().strftime('%I %p')}" \
                    if not peak_hours.empty else "N/A"

    return {
        "total_kwh": round(total_kwh, 2),
        "hourly_predictions": hourly_json,
        "suggestions": {
            "day_type": day_type,
            "recommendation": suggestion,
            "peak_time": peak_time_str
        }
    }

# --- FLASK ROUTES ---
@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/api/prediction')
def api_prediction(): 
    return jsonify(get_prediction_data())

# --- RUN APP ---
if __name__ == "__main__":
    load_data_and_train_model()
    app.run(debug=True)
