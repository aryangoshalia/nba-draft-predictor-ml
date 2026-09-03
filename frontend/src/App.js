import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:5001';

const FIELD_GROUPS = [
  {
    title: 'Body measurements',
    fields: [
      { name: 'HGT', label: 'Height', unit: 'in' },
      { name: 'WGT', label: 'Weight', unit: 'lbs' },
      { name: 'BMI', label: 'BMI', unit: '' },
      { name: 'WNGSPN', label: 'Wingspan', unit: 'in' },
      { name: 'STNDRCH', label: 'Standing reach', unit: 'in' },
      { name: 'BAR', label: 'Body adiposity ratio', unit: '%' },
    ],
  },
  {
    title: 'Athletic testing',
    fields: [
      { name: 'STNDVERT', label: 'Standing vertical', unit: 'in' },
      { name: 'LANE', label: 'Lane agility', unit: 'sec' },
      { name: 'SPRINT', label: '3/4 court sprint', unit: 'sec' },
    ],
  },
];

const CATEGORY_ORDER = ['Bust', 'Role Player', 'Starter', 'Star'];

const slugify = (label) => label.toLowerCase().replace(/\s+/g, '-');

function App() {
  const [form, setForm] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.post(`${API_URL}/predict`, form);
      setResult(res.data);
    } catch (err) {
      setResult({ error: 'Prediction failed. Is the backend running on port 5001?' });
    }
    setLoading(false);
  };

  return (
    <div className="page">
      <header className="site-header">
        <div className="site-header__inner">
          <p className="eyebrow">NBA Draft Analytics</p>
          <h1>Combine Outcome Predictor</h1>
          <p>
            Enter a prospect's combine measurables to see a predicted career
            outcome tier, based on career data from past combine participants.
          </p>
        </div>
      </header>

      <main className="layout">
        <section className="panel form-panel" aria-labelledby="form-heading">
          <h2 id="form-heading" className="panel-heading">Prospect measurables</h2>
          <p className="panel-description">All fields are required for a prediction.</p>

          <form className="form" onSubmit={handleSubmit}>
            {FIELD_GROUPS.map((group) => (
              <fieldset className="field-group" key={group.title}>
                <legend className="field-group__title">{group.title}</legend>
                <div className="field-grid">
                  {group.fields.map((field) => (
                    <div className="field" key={field.name}>
                      <label htmlFor={field.name}>{field.label}</label>
                      <div className="field-input">
                        <input
                          id={field.name}
                          type="number"
                          step="any"
                          name={field.name}
                          value={form[field.name] || ''}
                          onChange={handleChange}
                          required
                        />
                        {field.unit && <span className="field-input__unit">{field.unit}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </fieldset>
            ))}

            <div className="form-actions">
              <button type="submit" className="btn btn-primary" disabled={loading}>
                {loading && <span className="spinner" aria-hidden="true" />}
                {loading ? 'Predicting…' : 'Predict outcome'}
              </button>
              <span className="form-hint">Runs the trained model instantly</span>
            </div>
          </form>
        </section>

        <section className="panel results-panel" aria-live="polite">
          {!loading && !result && (
            <div className="empty-state">
              <div className="empty-state__icon" aria-hidden="true">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <path d="M12 3v18M3 12h18" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                </svg>
              </div>
              <p>Fill in the measurables and submit to see a predicted outcome.</p>
            </div>
          )}

          {loading && (
            <div>
              <div className="skeleton" style={{ height: 22, width: '55%', marginBottom: 18 }} />
              <div className="skeleton" style={{ height: 7, width: '100%', marginBottom: 9, borderRadius: 999 }} />
              <div className="skeleton" style={{ height: 7, width: '100%', marginBottom: 9, borderRadius: 999 }} />
              <div className="skeleton" style={{ height: 7, width: '100%', marginBottom: 9, borderRadius: 999 }} />
              <div className="skeleton" style={{ height: 7, width: '100%', marginBottom: 20, borderRadius: 999 }} />
              <div className="skeleton" style={{ height: 60, width: '100%' }} />
            </div>
          )}

          {!loading && result && result.error && (
            <div className="alert alert-error">{result.error}</div>
          )}

          {!loading && result && !result.error && (
            <div className="results">
              <div className="result-header">
                <span className="result-header__label">Predicted category</span>
                <span className={`badge badge--${slugify(result.category)}`}>{result.category}</span>
              </div>

              <div className="prob-list">
                {CATEGORY_ORDER.map((cat) => {
                  const pct = Math.round((result.probabilities[cat] || 0) * 100);
                  const slug = slugify(cat);
                  return (
                    <div className={`prob-row ${cat === result.category ? 'prob-row--active' : ''}`} key={cat}>
                      <span className="prob-row__label">{cat}</span>
                      <div className="prob-row__track">
                        <div className={`prob-row__fill prob-row__fill--${slug}`} style={{ width: `${pct}%` }} />
                      </div>
                      <span className="prob-row__value">{pct}%</span>
                    </div>
                  );
                })}
              </div>

              <hr className="divider" />

              <div className="stat-tiles">
                <div className="stat-tile">
                  <span className="stat-tile__value">{result.predicted_ppg}</span>
                  <span className="stat-tile__label">PPG</span>
                </div>
                <div className="stat-tile">
                  <span className="stat-tile__value">{result.predicted_rpg}</span>
                  <span className="stat-tile__label">RPG</span>
                </div>
                <div className="stat-tile">
                  <span className="stat-tile__value">{result.predicted_apg}</span>
                  <span className="stat-tile__label">APG</span>
                </div>
              </div>

              <div className="comparison">
                <span className="comparison__label">Closest career comp</span>
                <span className="comparison__value">{result.comparison}</span>
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="disclaimer">
        <div className="disclaimer__inner">
          <p>
            <strong>Reality check:</strong> our analysis (<code>report/REPORT.md</code>)
            found that combine measurables alone barely predict career outcome — treat
            this as a demo of the pipeline, not a scouting tool.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
