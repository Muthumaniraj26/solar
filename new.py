import os
import requests
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# --- CONFIGURATION -----------------------------------------------------------------
# Your location (Rajapalayam, Tamil Nadu, India)
LATITUDE = 9.4716
LONGITUDE = 77.5569

# File paths for your data (MAKE SURE THESE FILENAMES ARE CORRECT)
SOLAR_DATA_CSV = "sample_inverter_data_large.csv"
HISTORICAL_WEATHER_CSV = "weather_data_corrected_2024.csv"
# -----------------------------------------------------------------------------------

def load_and_clean_solar_data(filepath):
    """
    Loads solar data and fills in missing nighttime hours with 0 kWh.
    This is the key function to fix the nighttime prediction issue.
    """
    print(f"Loading and cleaning solar data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # --- FIX STARTS HERE ---
        # Create a full date range from the first to the last timestamp, with hourly frequency.
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        
        # Reindex the dataframe to include all hours. Missing hours (nighttime) will have NaN.
        df_complete = df.reindex(full_date_range)
        
        # Fill the NaN values in the 'kwh' column with 0.
        df_complete['kwh'].fillna(0, inplace=True)
        # --- FIX ENDS HERE ---

        print("✅ Solar data loaded and cleaned successfully (nighttime hours filled).")
        return df_complete
        
    except FileNotFoundError:
        print(f"❌ ERROR: Solar data file not found at '{filepath}'. Please create it.")
        return None

def load_historical_weather_data(filepath):
    """Loads historical weather data from a CSV file."""
    print(f"Loading historical weather data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['datetime'])
        df = df.set_index('timestamp')
        df = df[['temp', 'humidity', 'cloudcover', 'solarradiation']]
        df = df.rename(columns={'cloudcover': 'cloud_cover'})
        print("✅ Historical weather data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: Historical weather file not found at '{filepath}'.")
        return None
    except KeyError:
        print("❌ ERROR: The historical weather CSV has incorrect column names.")
        print("Please ensure it has 'datetime', 'temp', 'humidity', 'cloudcover', and 'solarradiation'.")
        return None

def prepare_data(solar_df, weather_df):
    """Merges solar and weather data and creates features for the model."""
    print("Combining data and creating features...")
    df_combined = solar_df.join(weather_df, how='inner')
    df_combined.dropna(inplace=True)

    if len(df_combined) == 0:
        print("❌ CRITICAL ERROR: After merging, there are 0 matching timestamps between your two files.")
        return None, None

    df_combined['hour'] = df_combined.index.hour
    df_combined['day_of_year'] = df_combined.index.dayofyear
    df_combined['month'] = df_combined.index.month

    features = ['solarradiation', 'temp', 'cloud_cover', 'humidity', 'hour', 'day_of_year', 'month']
    target = 'kwh'
    X = df_combined[features]
    y = df_combined[target]
    
    print("✅ Data preparation complete.")
    return X, y

def train_model(X, y):
    """Trains an XGBoost regression model."""
    print("Training the prediction model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=50,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"📈 Model Evaluation: Mean Absolute Error = {mae:.4f} kWh")
    print("✅ Model training complete.")
    return model

def get_todays_prediction(model):
    """Fetches today's weather forecast and predicts solar output."""
    print("\nFetching today's weather forecast...")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=temperature_2m,relativehumidity_2m,cloudcover,shortwave_radiation"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        forecast_data = response.json()['hourly']
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Could not fetch weather forecast. {e}")
        return

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
    predicted_kwh = model.predict(df_forecast[required_features])
    df_forecast['predicted_kwh'] = predicted_kwh
    df_forecast.loc[df_forecast['predicted_kwh'] < 0, 'predicted_kwh'] = 0

    today_str = datetime.now().strftime('%Y-%m-%d')
    todays_predictions = df_forecast[df_forecast.index.strftime('%Y-%m-%d') == today_str]
    total_prediction = todays_predictions['predicted_kwh'].sum()

    print("\n--- ☀️ Today's Solar Power Forecast ---")
    print(f"Total Predicted Generation for Today: {total_prediction:.2f} kWh")
    print("\nHourly Breakdown:")
    for time, row in todays_predictions.iterrows():
        # Only show hours with meaningful solar radiation or prediction
        if row['solarradiation'] > 0 or row['predicted_kwh'] > 0.05:
            print(f"  {time.strftime('%I:%M %p')}: {row['predicted_kwh']:.2f} kWh (☀️ {row['solarradiation']} W/m², ☁️ {row['cloud_cover']}%)")

if __name__ == "__main__":
    print("--- Solar Power Prediction Project ---")
    
    # Use the new cleaning function for solar data
    solar_df = load_and_clean_solar_data(SOLAR_DATA_CSV)
    weather_df = load_historical_weather_data(HISTORICAL_WEATHER_CSV)

    if solar_df is not None and weather_df is not None:
        X, y = prepare_data(solar_df, weather_df)
        if X is not None:
            solar_model = train_model(X, y)
            get_todays_prediction(solar_model)