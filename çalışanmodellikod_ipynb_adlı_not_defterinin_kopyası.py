from flask import Flask, request, jsonify
from eptr2 import EPTR2
import pandas as pd

app = Flask(__name__)

@app.route('/get-merged-data', methods=['POST'])
def get_data():
    try:
        data = request.get_json()
        username = data['username']
        password = data['password']
        plant_name = data['plant']

        eptr = EPTR2(username=username, password=password)
        plants = eptr.call('pp-list', start_date="2024-04-01", end_date="2024-04-30")
        match = plants[plants["shortName"] == plant_name.upper()]
        if match.empty:
            return jsonify({"error": "Santral bulunamadı"}), 404

        return jsonify({"message": f"Santral bulundu: ID={match.iloc[0]['id']}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
