import { useEffect, useRef, useState } from 'react';
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

const EXAMPLES = [
  {
    category: 'Bust',
    player: 'Ben McLemore',
    year: 2013,
    confidence: 0.695,
    values: { HGT: 75.5, WGT: 189.2, BMI: 23.3, WNGSPN: 79.75, STNDRCH: 100.5, BAR: 105.6, STNDVERT: 32.5, LANE: 11.87, SPRINT: 3.27 },
  },
  {
    category: 'Role Player',
    player: 'Renaldo Balkman',
    year: 2006,
    confidence: 0.799,
    values: { HGT: 77.25, WGT: 206.0, BMI: 24.3, WNGSPN: 85.0, STNDRCH: 104.5, BAR: 110.0, STNDVERT: 30.5, LANE: 11.58, SPRINT: 3.22 },
  },
  {
    category: 'Starter',
    player: 'Ty Lawson',
    year: 2009,
    confidence: 0.539,
    values: { HGT: 71.25, WGT: 196.6, BMI: 27.2, WNGSPN: 72.75, STNDRCH: 94.5, BAR: 102.1, STNDVERT: 29.0, LANE: 10.98, SPRINT: 3.12 },
  },
  {
    category: 'Star',
    player: 'Nene',
    year: 2002,
    confidence: 0.373,
    values: { HGT: 81.25, WGT: 253.0, BMI: 26.9, WNGSPN: 88.5, STNDRCH: 109.0, BAR: 108.9, STNDVERT: 30.0, LANE: 10.73, SPRINT: 3.19 },
  },
];

const slugify = (label) => label.toLowerCase().replace(/\s+/g, '-');

function App() {
  const [form, setForm] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [bounds, setBounds] = useState({});
  const [fieldErrors, setFieldErrors] = useState({});
  const formTopRef = useRef(null);

  useEffect(() => {
    axios.get(`${API_URL}/meta`).then((res) => setBounds(res.data.bounds || {})).catch(() => {});
  }, []);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    if (fieldErrors[e.target.name]) {
      const next = { ...fieldErrors };
      delete next[e.target.name];
      setFieldErrors(next);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setFieldErrors({});
    try {
      const res = await axios.post(`${API_URL}/predict`, form);
      setResult(res.data);
    } catch (err) {
      if (err.response && err.response.status === 400 && err.response.data.field_errors) {
        setFieldErrors(err.response.data.field_errors);
        setResult({ error: 'Some values look physically implausible — see the highlighted fields.' });
      } else {
        setResult({ error: 'Prediction failed. Is the backend running on port 5001?' });
      }
    }
    setLoading(false);
  };

  const handleTryExample = (example) => {
    const filled = {};
    Object.entries(example.values).forEach(([key, val]) => {
      filled[key] = String(val);
    });
    setForm(filled);
    setResult(null);
    setFieldErrors({});
    formTopRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <div className="page">
      <header className="site-header">
        <div className="site-header__inner">
          <div className="site-header__row">
            <div>
              <p className="eyebrow">NBA Draft Analytics</p>
              <h1>Combine Outcome Predictor</h1>
            </div>
            <a
              className="btn btn-secondary"
              href={`${API_URL}/report/`}
              target="_blank"
              rel="noopener noreferrer"
            >
              Report
            </a>
          </div>
          <p>
            Enter a prospect's combine measurables to see a predicted career
            outcome tier, based on career data from past combine participants.
          </p>
        </div>
      </header>

      <main className="layout">
        <section className="panel form-panel" aria-labelledby="form-heading" ref={formTopRef}>
          <h2 id="form-heading" className="panel-heading">Prospect measurables</h2>
          <p className="panel-description">All fields are required for a prediction.</p>

          <form className="form" onSubmit={handleSubmit}>
            {FIELD_GROUPS.map((group) => (
              <fieldset className="field-group" key={group.title}>
                <legend className="field-group__title">{group.title}</legend>
                <div className="field-grid">
                  {group.fields.map((field) => {
                    const [min, max] = bounds[field.name] || [];
                    const error = fieldErrors[field.name];
                    return (
                      <div className="field" key={field.name}>
                        <label htmlFor={field.name}>{field.label}</label>
                        <div className={`field-input ${error ? 'field-input--error' : ''}`}>
                          <input
                            id={field.name}
                            type="number"
                            step="any"
                            min={min}
                            max={max}
                            name={field.name}
                            value={form[field.name] || ''}
                            onChange={handleChange}
                            aria-invalid={error ? 'true' : undefined}
                            required
                          />
                          {field.unit && <span className="field-input__unit">{field.unit}</span>}
                        </div>
                        {error && <p className="field-error">{error}</p>}
                      </div>
                    );
                  })}
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

      <section className="examples-wrap">
        <div className="panel examples-panel">
          <h2 className="panel-heading">See a real example of each outcome</h2>
          <p className="panel-description">
            Each card is one real prospect's actual combine numbers — the player the
            model is most confident about for that outcome, and one it happens to get
            right. Click a card to load those measurables into the form above.
          </p>
          <div className="example-grid">
            {EXAMPLES.map((ex) => (
              <button
                type="button"
                key={ex.category}
                className="example-card"
                onClick={() => handleTryExample(ex)}
              >
                <span className={`badge badge--${slugify(ex.category)}`}>{ex.category}</span>
                <span className="example-card__player">{ex.player} · {ex.year} combine</span>
                <span className="example-card__confidence">{Math.round(ex.confidence * 100)}% model confidence</span>
              </button>
            ))}
          </div>
        </div>
      </section>

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
