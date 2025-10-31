from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json
import re
from urllib.parse import urlparse
from scipy.sparse import hstack

# ===================================
# 1️⃣ Initialize Flask app and CORS
# ===================================
app = Flask(__name__)
CORS(app)  # allows frontend (React) to call this API

# ===================================
# 2️⃣ Load Trained Model & Preprocessors
# ===================================
print("🔄 Loading model and components...")
lgb_model = joblib.load("phishing_lightgbm_model_simple.joblib")
vectorizer = joblib.load("tfidf_vectorizer_simple.joblib")
scaler = joblib.load("scaler_simple.joblib")

with open("numeric_feature_names.json", "r") as f:
    numeric_feature_names = json.load(f)

PHISHING_THRESHOLD = 0.95
print("✅ Model and assets loaded successfully.")

# ===================================
# 3️⃣ Feature Extraction Function
# ===================================
def extract_features(url):
    """Extract handcrafted numerical features from a given URL."""
    url = str(url).strip()
    if not url:
        url = "http://invalid-url"
    if "://" not in url:
        url = "http://" + url

    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse("http://invalid-url")

    feats = {}
    feats["url_length"] = len(url)
    feats["num_dots"] = url.count(".")
    feats["num_hyphens"] = url.count("-")
    feats["num_digits"] = sum(c.isdigit() for c in url)
    feats["num_params"] = url.count("=") + url.count("?") + url.count("&")
    feats["has_https"] = 1 if url.startswith("https") else 0
    feats["has_ip"] = 1 if re.match(r"^(?:http[s]?://)?\d{1,3}(?:\.\d{1,3}){3}", url) else 0
    feats["num_subdirs"] = url.count("/")
    feats["has_at_symbol"] = 1 if "@" in url else 0
    feats["subdomain_count"] = max(0, parsed.netloc.count(".") - 1)
    feats["contains_login"] = 1 if "login" in url.lower() else 0
    feats["contains_verify"] = 1 if "verify" in url.lower() else 0
    feats["contains_secure"] = 1 if "secure" in url.lower() else 0
    return feats

# ===================================
# 4️⃣ Prediction Function
# ===================================
def predict_url(url):
    feats = extract_features(url)
    feats_df = pd.DataFrame([feats])

    # Align features to expected order
    for col in numeric_feature_names:
        if col not in feats_df.columns:
            feats_df[col] = 0
    feats_df = feats_df[numeric_feature_names]

    # Scale numeric and vectorize URL
    feats_scaled = scaler.transform(feats_df.fillna(0))
    tfidf_vec = vectorizer.transform([url]).tocsr()
    x = hstack([tfidf_vec, feats_scaled]).tocsr()

    prob = lgb_model.predict_proba(x)[0, 1]
    pred = 1 if prob >= PHISHING_THRESHOLD else 0

    return {
        "url": url,
        "prediction": "phishing" if pred == 1 else "legitimate",
        "probability": round(float(prob), 4),
        "confidence": f"{prob*100:.2f}%"
    }

# ===================================
# 5️⃣ Flask Routes
# ===================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Phishing URL Detection API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Please provide a 'url' field"}), 400
    url = data["url"]
    try:
        result = predict_url(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===================================
# 6️⃣ Run the App
# ===================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)