import requests
from flask import Flask, render_template, jsonify, request

# --- CONFIGURATION (GOOGLE API KEY) ---
GOOGLE_API_KEY = "GOOGLE API KEY"

app = Flask(__name__)

def get_location_from_google():
    """
    Uses Google Geolocation API to estimate the user's location (lat/lon)
    based on IP & WiFi data.
    """
    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_API_KEY}"
    # Google Geolocation API expects a POST (empty body works for IP-based)
    resp = requests.post(url, json={})
    resp.raise_for_status()
    data = resp.json()
    lat = data["location"]["lat"]
    lon = data["location"]["lng"]
    accuracy = data.get("accuracy", None)
    return lat, lon, accuracy

def reverse_geocode(lat, lon):
    """
    Turns latitude/longitude into a nice string using Google Geocoding API.
    """
    url = (
        f"https://maps.googleapis.com/maps/api/geocode/json"
        f"?latlng={lat},{lon}&key={GOOGLE_API_KEY}"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    if data["status"] == "OK" and len(data["results"]) > 0:
        # Use the first formatted address:
        return data["results"][0]["formatted_address"]
    return f"{lat:.3f}, {lon:.3f}"

@app.route("/")
def home():
    """Serves the main HTML page."""
    return render_template("index.html")

@app.route("/api/weather")
def get_weather():
    """Backend API: location via Google API, then weather via Open-Meteo."""
    try:
        # 1. Find user's location using Google Geolocation API
        lat, lon, accuracy = get_location_from_google()
        location_name = reverse_geocode(lat, lon)

        # 2. Call Open-Meteo API for weather
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,weather_code"
            "&current_weather=true&timezone=auto"
        )
        resp = requests.get(weather_url)
        resp.raise_for_status()
        data = resp.json()

        # 3. Send back JSON to frontend
        return jsonify({
            "location": location_name,
            "accuracy_m": accuracy,
            "current": data.get("current_weather"),
            "hourly": data.get("hourly")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Start the Flask dev server
    app.run(debug=True)


