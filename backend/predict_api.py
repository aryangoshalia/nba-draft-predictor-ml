"""Flask API for the NBA combine -> career outcome predictor."""

import os

import markdown
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import joblib

REPORT_MD_PATH = "report/REPORT.md"
REPORT_FIGURES_DIR = os.path.abspath("report/figures")

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
BOUNDS = feature_meta["bounds"]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}

    missing = []
    values = []
    field_errors = {}
    for f in FEATURE_COLUMNS:
        raw = data.get(f)
        if raw is None or raw == "":
            values.append(MEDIANS[f])
            missing.append(f)
            continue

        try:
            value = float(raw)
        except (TypeError, ValueError):
            field_errors[f] = f"\"{raw}\" is not a number."
            continue

        lo, hi = BOUNDS[f]
        if not (lo <= value <= hi):
            field_errors[f] = f"{value} is outside the plausible range ({lo}–{hi})."
            continue

        values.append(value)

    if field_errors:
        return jsonify({"error": "Invalid input.", "field_errors": field_errors}), 400

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


@app.route("/meta", methods=["GET"])
def meta():
    """Feature list + plausible-value bounds, so the frontend can build
    matching min/max input constraints instead of hardcoding a second copy
    that could drift from what the model was actually trained on."""
    return jsonify({"features": FEATURE_COLUMNS, "bounds": BOUNDS})


@app.route("/report/")
def report():
    with open(REPORT_MD_PATH) as f:
        source = f.read()
    body = markdown.markdown(source, extensions=["tables"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NBA Draft Predictor — Report</title>
<style>
  :root {{
    --bg: #f6f4f1; --surface: #ffffff; --border: #e6e1d8;
    --text: #22201c; --text-muted: #6e6759; --accent: #b2500f;
  }}
  body {{
    margin: 0; padding: 40px 20px 80px; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    line-height: 1.6;
  }}
  .report {{ max-width: 760px; margin: 0 auto; }}
  h1 {{ font-size: 28px; font-weight: 650; letter-spacing: -0.01em; margin-bottom: 6px; }}
  h2 {{ font-size: 19px; font-weight: 650; margin-top: 40px; border-top: 1px solid var(--border); padding-top: 28px; }}
  p {{ color: var(--text); }}
  strong {{ color: var(--text); }}
  a {{ color: var(--accent); }}
  code {{ background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; font-size: 0.9em; }}
  img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 10px; margin: 16px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14.5px; }}
  th, td {{ border: 1px solid var(--border); padding: 8px 12px; text-align: left; }}
  th {{ background: var(--surface); font-weight: 650; }}
  em {{ color: var(--text-muted); }}
</style>
</head>
<body>
<div class="report">{body}</div>
</body>
</html>"""


@app.route("/report/figures/<path:filename>")
def report_figures(filename):
    return send_from_directory(REPORT_FIGURES_DIR, filename)


if __name__ == "__main__":
    # Port 5000 is claimed by macOS AirPlay Receiver by default, so this
    # uses 5001 instead to avoid a confusing silent conflict.
    app.run(debug=True, port=5001)
