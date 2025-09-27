# model_trainer.py
import pandas as pd
import xgboost as xgb

# --- CONFIGURATION ---
SOLAR_DATA_CSV = "sample_inverter_data_large.csv"
HISTORICAL_WEATHER_CSV = "weather_data_corrected_2024.csv"
MODEL_FILE_NAME = "solar_model_2.json"

# --- IMPORTANT! ---
# Set the capacity (in kW) of the solar panel system that
# generated your historical CSV file. For example: 3.5, 5.0, 10.0
HISTORICAL_SYSTEM_CAPACITY_KW = 5.0 

def load_and_clean_solar_data(filepath):
    """Loads solar data and fills in missing nighttime hours with 0 kWh."""
    print(f"Loading and cleaning solar data from '{filepath}'...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df_complete = df.reindex(full_date_range)
    df_complete['kwh'].fillna(0, inplace=True)
    return df_complete

def load_historical_weather_data(filepath):
    """Loads historical weather data from a CSV file."""
    print(f"Loading historical weather data from '{filepath}'...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['datetime'])
    df = df.set_index('timestamp')
    df = df[['temp', 'humidity', 'cloudcover', 'solarradiation']]
    df = df.rename(columns={'cloudcover': 'cloud_cover'})
    return df

def prepare_data(solar_df, weather_df):
    """Prepares data and creates a NORMALIZED target variable."""
    print("Combining and normalizing data...")
    df_combined = solar_df.join(weather_df, how='inner')
    df_combined.dropna(inplace=True)

    # --- This is the key step: Normalize the kWh by the system capacity ---
    df_combined['kwh_per_kw'] = df_combined['kwh'] / HISTORICAL_SYSTEM_CAPACITY_KW

    # Feature Engineering
    df_combined['hour'] = df_combined.index.hour
    df_combined['day_of_year'] = df_combined.index.dayofyear
    df_combined['month'] = df_combined.index.month
    
    features = ['solarradiation', 'temp', 'cloud_cover', 'humidity', 'hour', 'day_of_year', 'month']
    target = 'kwh_per_kw' # <-- The target is now the normalized value
    
    X = df_combined[features]
    y = df_combined[target]
    print("✅ Data preparation complete.")
    return X, y

def train_and_save_model(X, y):
    """Trains the model on normalized data and saves it to a file."""
    print("Training the NORMALIZED prediction model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=50,
        n_jobs=-1
    )
    # We train on ALL data for the final model
    model.fit(X, y, eval_set=[(X, y)], verbose=False)
    
    # Save the model
    model.save_model(MODEL_FILE_NAME)
    print(f"✅ Normalized model trained and saved successfully as '{MODEL_FILE_NAME}'")

if __name__ == "__main__":
    print("--- Starting Model Training ---")
    solar_df = load_and_clean_solar_data(SOLAR_DATA_CSV)
    weather_df = load_historical_weather_data(HISTORICAL_WEATHER_CSV)
    
    if solar_df is not None and weather_df is not None:
        X, y = prepare_data(solar_df, weather_df)
        if X is not None:
            train_and_save_model(X, y)