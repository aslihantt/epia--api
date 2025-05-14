from flask import Flask, request, jsonify
from eptr2 import EPTR2
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
        return jsonify({"auth": False})  # Giriş başarısız

    except Exception as e:
        return jsonify({"auth": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run()
