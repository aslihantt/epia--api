from flask import Flask, request, jsonify
from eptr2 import EPTR2
from eptr2.composite.production import get_hourly_production_data
from meteostat import Stations, Hourly
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ✅ Kök URL'e yanıt verir
@app.route('/', methods=['GET'])
def index():
    return "Yonca Production Prediction API is live!", 200

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
        return jsonify({"auth": False, "error": "EPTR2 kimlik doğrulama hatası."})

    except Exception as e:
        return jsonify({"auth": False, "error": str(e)}), 500

@app.route('/predict-production', methods=['POST'])
def predict_production():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        plant_name_req = data.get("plant")
        city_name_req = data.get("city")

        if not username or not password or not plant_name_req or not city_name_req:
            return jsonify({"error": "Eksik bilgi: username, password, plant ve city gereklidir."}), 400

        plant_name = plant_name_req.strip().upper()
        city_name = city_name_req.strip()

        eptr = EPTR2(username=username, password=password)

        today = datetime.today()
        # Eğitim verisi için başlangıç ve bitiş tarihleri
        # Tahmin bugünün verileriyle yapılacağı için eğitim verisi dünden önceki 3 ay olmalı
        end_date_train_obj = today - timedelta(days=1)
        start_date_train_obj = end_date_train_obj - pd.DateOffset(months=3) # <--- DEĞİŞİKLİK BURADA

        start_date_train_str = start_date_train_obj.strftime("%Y-%m-%d")
        end_date_train_str = end_date_train_obj.strftime("%Y-%m-%d")

        plant_list_start_date = (today - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        plant_list_end_date = today.strftime("%Y-%m-%d")

        plants = eptr.call("pp-list", start_date=plant_list_start_date, end_date=plant_list_end_date)
        match = plants[plants["shortName"] == plant_name]
        if match.empty:
            return jsonify({"error": f"Santral bulunamadı: {plant_name}"}), 404
        rt_pp_id = match.iloc[0]["id"]

        def fetch_chunks(eptr_instance, rt_pp_id_val, start_date_val, end_date_val, max_retries=3):
            current_start = pd.to_datetime(start_date_val)
            final_end = pd.to_datetime(end_date_val)
            all_chunks_list = []
            # 3 aylık veri için tek bir chunk yeterli olabilir, ancak döngü genel bir çözüm sunar.
            # Eğer start_date ve end_date arasındaki fark 3 aydan az ise, döngü sadece bir kez çalışır.
            while current_start < final_end:
                # Tek seferde maksimum 3 ay çekebiliriz, bu nedenle current_end'i buna göre ayarlayalım
                # Eğer aralık zaten 3 aydan kısaysa, final_end kullanılır.
                max_chunk_end = current_start + pd.DateOffset(months=3) - timedelta(days=1)
                current_end = min(max_chunk_end, final_end)
                
                for attempt in range(max_retries):
                    try:
                        df_chunk = get_hourly_production_data(
                            eptr=eptr_instance,
                            start_date=current_start.strftime("%Y-%m-%d"),
                            end_date=current_end.strftime("%Y-%m-%d"),
                            rt_pp_id=int(rt_pp_id_val),
                            verbose=False
                        )
                        all_chunks_list.append(df_chunk)
                        break 
                    except Exception as ex_fetch:
                        if attempt + 1 == max_retries:
                            pass 
                current_start = current_end + timedelta(days=1)
            return pd.concat(all_chunks_list, ignore_index=True) if all_chunks_list else pd.DataFrame()

        actual_df = fetch_chunks(eptr, rt_pp_id, start_date_train_str, end_date_train_str)
        if actual_df.empty:
            return jsonify({"error": "Eğitim için üretim verisi alınamadı"}), 500
        
        if 'sun_rt' not in actual_df.columns:
             return jsonify({"error": "'sun_rt' kolonu üretim verilerinde bulunamadı. Mevcut kolonlar: " + ", ".join(actual_df.columns)}), 500

        actual_df["dt"] = pd.to_datetime(actual_df["dt"])
        actual_df["hour"] = actual_df["dt"].dt.hour
        solar_df = actual_df[(actual_df["hour"] >= 3) & (actual_df["hour"] <= 17)].copy()
        solar_df = solar_df[["dt", "sun_rt"]].rename(columns={"dt": "date", "sun_rt": "gunes"})

        geolocator = Nominatim(user_agent="yonca-api-client")
        location = geolocator.geocode(city_name, timeout=10)
        if not location:
            return jsonify({"error": f"{city_name} için koordinat bulunamadı"}), 404

        lat, lon = location.latitude, location.longitude
        stations = Stations().nearby(lat, lon)
        station = stations.fetch(1)
        if station.empty:
            return jsonify({"error": "Hava durumu istasyonu bulunamadı"}), 404
        station_id = station.index[0]

        weather_train = Hourly(station_id, pd.to_datetime(start_date_train_str), pd.to_datetime(end_date_train_str)).fetch()
        if weather_train.empty:
            return jsonify({"error": "Eğitim için hava durumu verisi alınamadı"}), 500
            
        weather_train = weather_train[(weather_train.index.hour >= 3) & (weather_train.index.hour <= 17)]
        weather_train = weather_train.drop(columns=['snow', 'wdir', 'wpgt', 'tsun'], errors='ignore')
        weather_train = weather_train.reset_index().rename(columns={"time": "date"})

        solar_df["date"] = pd.to_datetime(solar_df["date"]).dt.tz_localize(None)
        weather_train["date"] = pd.to_datetime(weather_train["date"]).dt.tz_localize(None)
        solar_df["date"] = solar_df["date"].dt.floor("h")
        weather_train["date"] = weather_train["date"].dt.floor("h")

        if solar_df.empty :
             return jsonify({"error": "Filtrelenmiş güneş üretim verisi (solar_df) boş. Eğitim yapılamaz."}), 500

        all_train_dates = pd.date_range(solar_df["date"].min(), solar_df["date"].max(), freq="h")
        merged_train = pd.DataFrame({"date": all_train_dates})
        merged_train = merged_train.merge(solar_df, on="date", how="left")
        merged_train = merged_train.merge(weather_train, on="date", how="left")
        
        if merged_train["gunes"].isnull().all():
             return jsonify({"error": "Birleştirme sonrası tüm 'gunes' değerleri NaN. Veri hizalamasını kontrol edin."}), 500

        merged_train = merged_train.interpolate(method='linear', limit_direction='both').fillna(0)
        merged_train["hour"] = merged_train["date"].dt.hour

        G_sc = 1367
        merged_train["day_of_year"] = merged_train["date"].dt.dayofyear
        merged_train["epsilon"] = 1 + 0.033 * np.cos((2 * np.pi * merged_train["day_of_year"]) / 365)
        merged_train["cos_theta"] = np.maximum(0, np.cos((np.pi / 12) * (merged_train["hour"] - 12)))
        merged_train["G0"] = merged_train["epsilon"] * G_sc * merged_train["cos_theta"]

        features = ["temp", "rhum", "pres", "prcp", "hour", "dwpt", "coco", "wspd", "G0"]
        for col in features:
            if col not in merged_train.columns:
                merged_train[col] = 0.0
        
        X_train = merged_train[features]
        y_train = merged_train["gunes"]

        if len(X_train) < 24 : 
            return jsonify({"error": f"Yetersiz eğitim verisi (satır sayısı: {len(X_train)}). Model eğitilemedi."}), 500

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)

        start_future_dt = datetime.now().replace(hour=3, minute=0, second=0, microsecond=0)
        end_future_dt = start_future_dt.replace(hour=17)

        future_weather_raw = Hourly(station_id, start_future_dt, end_future_dt).fetch()
        if future_weather_raw.empty:
            return jsonify({"error": f"Gelecek ({start_future_dt.date()}) için hava durumu verisi alınamadı."}), 500

        future_weather = future_weather_raw.copy()
        future_weather = future_weather[(future_weather.index.hour >= 3) & (future_weather.index.hour <= 17)]
        future_weather = future_weather.drop(columns=['snow', 'wdir', 'wpgt', 'tsun'], errors='ignore')
        future_weather = future_weather.reset_index().rename(columns={"time": "date"})
        future_weather["date"] = pd.to_datetime(future_weather["date"]).dt.tz_localize(None)
        
        expected_future_hours = pd.date_range(start_future_dt, end_future_dt, freq='h')
        future_df_template = pd.DataFrame({'date': expected_future_hours})
        future_weather = pd.merge(future_df_template, future_weather, on='date', how='left')

        future_weather["hour"] = future_weather["date"].dt.hour
        future_weather["day_of_year"] = future_weather["date"].dt.dayofyear
        future_weather["epsilon"] = 1 + 0.033 * np.cos((2 * np.pi * future_weather["day_of_year"]) / 365)
        future_weather["cos_theta"] = np.maximum(0, np.cos((np.pi / 12) * (future_weather["hour"] - 12)))
        future_weather["G0"] = future_weather["epsilon"] * G_sc * future_weather["cos_theta"]

        for col in features:
            if col not in future_weather.columns:
                future_weather[col] = 0.0
            elif col != 'hour' and future_weather[col].isnull().any():
                 future_weather[col] = future_weather[col].interpolate(method='linear', limit_direction='both')
        
        future_weather[features] = future_weather[features].fillna(0)

        if future_weather[features].isnull().any().any():
            pass

        X_future_scaled = scaler.transform(future_weather[features])
        predicted_values = rf.predict(X_future_scaled)
        
        predicted_values = np.maximum(0, predicted_values)
        future_weather["predicted_gunes"] = predicted_values

        total = round(float(future_weather["predicted_gunes"].sum()), 2)
        
        hourly_raw = future_weather["predicted_gunes"].round(2).tolist()
        hourly = [float(p) for p in hourly_raw]

        min_val = 0.0
        max_val = 0.0

        if hourly:
            min_val = min(hourly)
            max_val = max(hourly)

        return jsonify({
            "total_prediction": total,
            "hourly_prediction": hourly,
            "minimum": min_val,
            "maximum": max_val
        })

    except requests.exceptions.HTTPError as http_err:
        if hasattr(http_err, 'response') and http_err.response is not None and http_err.response.status_code == 401:
            return jsonify({"error": "EPTR2 kimlik doğrulama hatası. Kullanıcı adı veya şifre yanlış.", "auth_failed": True}), 401
        import traceback # Bu satır sadece bu except bloğu için gerekli
        return jsonify({"error": f"EPTR2 API Hatası: {str(http_err)}", "trace": traceback.format_exc()}), 500
        
    except Exception as e:
        import traceback # Bu satır sadece bu except bloğu için gerekli
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)