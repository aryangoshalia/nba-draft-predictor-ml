# NBA Draft Predictor

Does an NBA prospect's Combine performance (height, wingspan, vertical, agility, etc.)
predict how their career turns out? This project builds a small ML pipeline to answer
that, and a web app where you can plug in combine numbers and get a predicted outcome
tier (Bust / Role Player / Starter / Star).

**Short answer, see [`report/REPORT.md`](report/REPORT.md) for the full writeup: not
really.** A model trained only on combine measurables doesn't beat a naive baseline,
and individual measurables show essentially no correlation with career value.

## Project structure

```
data/raw/          Raw combine + career-average CSVs
data/processed/    Cleaned data + trained model files (generated, not hand-edited)
backend/           Data pipeline, model training, report generation, Flask API
backend/tests/     Unit tests for the data pipeline
report/            The analysis report and its charts
frontend/          React app: enter combine stats, get a prediction
```

## 1. Set up the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Build the data and train the models

Run these from the **project root** (the scripts read/write `data/...` relative paths):

```bash
python3 backend/preprocess_data.py    # cleans + matches combine <-> career data
python3 backend/model_training.py     # trains the classifier + regressors
python3 backend/generate_report.py    # writes report/REPORT.md and its charts
```

`preprocess_data.py` keeps only players who appear in **both** the combine data and
the career data — a combine prospect with no career record has no outcome to predict,
and a career player with no combine record has no measurables to predict from. See the
top of that file for how name-matching handles accents and shared names.

## 3. Run the API

```bash
python3 backend/predict_api.py
```

Starts a Flask server at `http://127.0.0.1:5001` (not 5000 — macOS's AirPlay Receiver
uses port 5000 by default, which causes confusing silent conflicts).

- `POST /predict` — body is a JSON object with the combine fields below; returns the
  predicted category, per-category probabilities, predicted PPG/RPG/APG, and the
  closest career comparison.
- `GET /health` — quick check that the model loaded.

Combine fields the API expects: `HGT`, `WGT`, `BMI`, `WNGSPN`, `STNDRCH`, `BAR`,
`STNDVERT`, `LANE`, `SPRINT`. Any field you omit is filled in with that feature's
median value from the training data.

## 4. Run the frontend

In a separate terminal:

```bash
cd frontend
npm install
npm start
```

Opens at `http://localhost:3000`. Fill in the combine measurables and submit to see
the predicted outcome tier, class probabilities, predicted stat line, and the closest
career comparison. It talks to the API at `http://127.0.0.1:5001` by default — set
`REACT_APP_API_URL` before `npm start` if you're running the API somewhere else.

## Re-running everything after a data change

If you edit the raw CSVs in `data/raw/`, redo steps 2 (preprocess → train → report) and
restart the API so it picks up the new model files.

## Running the tests

Tests cover the name-matching/disambiguation logic in `preprocess_data.py` — the
trickiest part of the pipeline (accent normalization, and telling apart different real
players who share a name). From the project root:

```bash
pip install -r backend/requirements-dev.txt
pytest
```

## License

[MIT](LICENSE)
