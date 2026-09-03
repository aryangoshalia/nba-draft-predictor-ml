"""Flask API for the NBA combine -> career outcome predictor."""

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

category_model = joblib.load("data/processed/category_model.pkl")
ppg_reg = joblib.load("data/processed/ppg_reg.pkl")
rpg_reg = joblib.load("data/processed/rpg_reg.pkl")
apg_reg = joblib.load("data/processed/apg_reg.pkl")
career_matrix = joblib.load("data/processed/career_matrix.pkl")
career_players = joblib.load("data/processed/career_players.pkl")
feature_meta = joblib.load("data/processed/feature_meta.pkl")

FEATURE_COLUMNS = feature_meta["feature_columns"]
MEDIANS = feature_meta["medians"]
CLASS_ORDER = feature_meta["class_order"]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}

    missing = []
    values = []
    for f in FEATURE_COLUMNS:
        raw = data.get(f)
        if raw is None or raw == "":
            values.append(MEDIANS[f])
            missing.append(f)
        else:
            values.append(float(raw))

    X = np.array(values).reshape(1, -1)

    probs = category_model.predict_proba(X)[0]
    category = category_model.classes_[int(np.argmax(probs))]
    probabilities = {cls: round(float(p), 3) for cls, p in zip(category_model.classes_, probs)}
    probabilities = {cls: probabilities[cls] for cls in CLASS_ORDER}

    ppg = float(ppg_reg.predict(X)[0])
    rpg = float(rpg_reg.predict(X)[0])
    apg = float(apg_reg.predict(X)[0])

    pred_vec = np.array([ppg, rpg, apg])
    diffs = np.linalg.norm(career_matrix - pred_vec, axis=1)
    comparison = career_players[int(np.argmin(diffs))]

    return jsonify({
        "category": category,
        "probabilities": probabilities,
        "predicted_ppg": round(ppg, 1),
        "predicted_rpg": round(rpg, 1),
        "predicted_apg": round(apg, 1),
        "comparison": comparison,
        "missing_fields_defaulted_to_median": missing,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "features": FEATURE_COLUMNS})


if __name__ == "__main__":
    # Port 5000 is claimed by macOS AirPlay Receiver by default, so this
    # uses 5001 instead to avoid a confusing silent conflict.
    app.run(debug=True, port=5001)
