from flask import Flask, request, jsonify
from eptr2 import EPTR2
from eptr2.composite.production import get_hourly_production_data
from meteostat import Stations
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

# Santral verisini parça parça çekmek için
def fetch_production_data_in_chunks(eptr, rt_pp_id, start_date, end_date, chunk_months=3):
    current_start = pd.to_datetime(start_date)
    final_end = pd.to_datetime(end_date)
    all_chunks = []

    while current_start < final_end:
        current_end = (current_start + pd.DateOffset(months=chunk_months)) - timedelta(days=1)
        if current_end > final_end:
            current_end = final_end
        try:
            df = get_hourly_production_data(
                eptr=eptr,
                start_date=current_start.strftime("%Y-%m-%d"),
                end_date=current_end.strftime("%Y-%m-%d"),
                rt_pp_id=int(rt_pp_id),
                verbose=False
            )
            all_chunks.append(df)
        except Exception as e:
            print(f"Hata: {e}")
        current_start = current_end + timedelta(days=1)

    return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()

@app.route("/", methods=["GET"])
def index():
    return "EPİAŞ Üretim Verisi API Aktif", 200

@app.route("/fetch-solar-data", methods=["POST"])
def fetch_solar_data():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        plant_name = data.get("plant", "").strip().upper()
        city_name = data.get("city", "").strip()

        if not username or not password or not plant_name or not city_name:
            return jsonify({"error": "Eksik bilgi"}), 400

        eptr = EPTR2(username=username, password=password)

        # Santral listesi
        today = datetime.now()
        start_date = (today - pd.DateOffset(months=9)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        plants = eptr.call("pp-list", start_date=start_date, end_date=end_date)
        matching = plants[plants["shortName"] == plant_name]

        if matching.empty:
            return jsonify({"error": "Santral bulunamadı"}), 404

        rt_pp_id = matching.iloc[0]["id"]

        # Üretim verisini çek
        df = fetch_production_data_in_chunks(eptr, rt_pp_id, start_date, end_date)
        if df.empty or "sun_rt" not in df.columns or "dt" not in df.columns:
            return jsonify({"error": "Veri eksik veya boş"}), 500

        df["dt"] = pd.to_datetime(df["dt"])
        df["hour"] = df["dt"].dt.hour
        filtered_df = df[(df["hour"] >= 3) & (df["hour"] <= 17)][["dt", "sun_rt"]]
        filtered_df = filtered_df.dropna()

        # Şehirden koordinat al
        geolocator = Nominatim(user_agent="epias-api")
        location = geolocator.geocode(city_name)
        if not location:
            return jsonify({"error": f"{city_name} için koordinat bulunamadı"}), 404

        lat, lon = location.latitude, location.longitude
        station = Stations().nearby(lat, lon).fetch(1)
        if station.empty:
            return jsonify({"error": "Hava durumu istasyonu bulunamadı"}), 404

        # JSON cevabı hazırla
        response = [
            {"datetime": row["dt"].strftime("%Y-%m-%d %H:%M:%S"), "sun_rt": round(row["sun_rt"], 2)}
            for _, row in filtered_df.iterrows()
        ]

        return jsonify({"data": response})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(debug=True)
