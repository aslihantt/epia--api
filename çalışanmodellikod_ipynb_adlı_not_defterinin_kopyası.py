from flask import Flask, request, jsonify
from eptr2 import EPTR2
from eptr2.composite.production import get_hourly_production_data
from meteostat import Stations, Hourly
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

app = Flask(__name__)

@app.route('/auth', methods=['POST'])
def auth():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({"auth": False, "error": "Eksik bilgi"}), 400

        eptr = EPTR2(username=username, password=password)
        _ = eptr.call('pp-list', start_date='2024-01-01', end_date='2024-01-02')

        return jsonify({"auth": True})

    except requests.exceptions.HTTPError:
        return jsonify({"auth": False})

    except Exception as e:
        return jsonify({"auth": False, "error": str(e)}), 500


@app.route('/predict-production', methods=['POST'])
def predict_production():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        plant_name = data.get("plant").strip().upper()
        city_name = data.get("city").strip()

        eptr = EPTR2(username=username, password=password)

        plants = eptr.call("pp-list", start_date="2024-04-01", end_date="2024-04-30")
        match = plants[plants["shortName"] == plant_name]
        if match.empty:
            return jsonify({"error": "Santral bulunamadı"}), 404
        rt_pp_id = match.iloc[0]["id"]

        def fetch_chunks(eptr, rt_pp_id, start_date, end_date):
            current_start = pd.to_datetime(start_date)
            final_end = pd.to_datetime(end_date)
            all_chunks = []
            while current_start < final_end:
                current_end = min(current_start + pd.DateOffset(months=3) - timedelta(days=1), final_end)
                try:
                    df = get_hourly_production_data(
                        eptr=eptr,
                        start_date=current_start.strftime("%Y-%m-%d"),
                        end_date=current_end.strftime("%Y-%m-%d"),
                        rt_pp_id=int(rt_pp_id),
                        verbose=False
                    )
                    all_chunks.append(df)
                except:
                    pass
                current_start = current_end + timedelta(days=1)
            return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()

        actual_df = fetch_chunks(eptr, rt_pp_id, "2024-04-01", "2024-12-31")
        if actual_df.empty:
            return jsonify({"error": "Üretim verisi alınamadı"}), 500

        actual_df["dt"] = pd.to_datetime(actual_df["dt"])
        actual_df["hour"] = actual_df["dt"].dt.hour
        solar_df = actual_df[(actual_df["hour"] >= 3) & (actual_df["hour"] <= 17)][["dt", "sun_rt"]].rename(columns={"dt": "date", "sun_rt": "gunes"})

        stations = Stations().nearby(38.0, 29.0)
        station = stations.fetch(1)
        if station.empty:
            return jsonify({"error": "Hava durumu istasyonu bulunamadı"}), 404
        station_id = station.index[0]

        weather = Hourly(station_id, datetime(2024, 4, 1), datetime(2024, 12, 31)).fetch()
        weather = weather[(weather.index.hour >= 3) & (weather.index.hour <= 17)]
        weather = weather.drop(columns=['snow', 'wdir', 'wpgt', 'tsun'], errors='ignore')
        weather = weather.reset_index().rename(columns={"time": "date"})

        # ✅ Timezone'ları kaldır
        solar_df["date"] = pd.to_datetime(solar_df["date"]).dt.tz_localize(None)
        weather["date"] = pd.to_datetime(weather["date"]).dt.tz_localize(None)

        solar_df["date"] = solar_df["date"].dt.floor("h")
        weather["date"] = weather["date"].dt.floor("h")

        all_dates = pd.date_range(solar_df["date"].min(), solar_df["date"].max(), freq="h")
        merged = pd.DataFrame({"date": all_dates})
        merged = merged.merge(solar_df, on="date", how="left").merge(weather, on="date", how="left").interpolate().fillna(0)
        merged["hour"] = merged["date"].dt.hour

        G_sc = 1367
        merged["day_of_year"] = merged["date"].dt.dayofyear
        merged["epsilon"] = 1 + 0.033 * np.cos((2 * np.pi * merged["day_of_year"]) / 365)
        merged["cos_theta"] = np.maximum(0, np.cos((np.pi / 12) * (merged["hour"] - 12)))
        merged["G0"] = merged["epsilon"] * G_sc * merged["cos_theta"]

        features = ["temp", "rhum", "pres", "prcp", "hour", "dwpt", "coco", "wspd", "G0"]
        X = merged[features]
        y = merged["gunes"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_scaled, y)

        start_future = datetime.now().replace(hour=3, minute=0, second=0, microsecond=0)
        end_future = start_future.replace(hour=17)
        future_weather = Hourly(station_id, start_future, end_future).fetch()
        future_weather = future_weather[(future_weather.index.hour >= 3) & (future_weather.index.hour <= 17)]
        future_weather = future_weather.drop(columns=['snow', 'wdir', 'wpgt', 'tsun'], errors='ignore')
        future_weather = future_weather.reset_index().rename(columns={"time": "date"})
        future_weather["date"] = pd.to_datetime(future_weather["date"]).dt.tz_localize(None)
        future_weather["hour"] = future_weather["date"].dt.hour
        future_weather["day_of_year"] = future_weather["date"].dt.dayofyear
        future_weather["epsilon"] = 1 + 0.033 * np.cos((2 * np.pi * future_weather["day_of_year"]) / 365)
        future_weather["cos_theta"] = np.maximum(0, np.cos((np.pi / 12) * (future_weather["hour"] - 12)))
        future_weather["G0"] = future_weather["epsilon"] * G_sc * future_weather["cos_theta"]

        X_future = scaler.transform(future_weather[features])
        future_weather["predicted_gunes"] = rf.predict(X_future)

        total = round(float(future_weather["predicted_gunes"].sum()), 2)
        hourly = list(map(float, future_weather["predicted_gunes"].round(2).tolist()))

        return jsonify({
            "total_prediction": total,
            "hourly_prediction": hourly
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run()
