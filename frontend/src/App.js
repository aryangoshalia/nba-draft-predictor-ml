import React, { useState } from 'react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:5001';

const combineFields = [
  { label: 'Height (inches)', name: 'HGT' },
  { label: 'Weight (lbs)', name: 'WGT' },
  { label: 'BMI', name: 'BMI' },
  { label: 'Wingspan (inches)', name: 'WNGSPN' },
  { label: 'Standing Reach (inches)', name: 'STNDRCH' },
  { label: 'Body Adiposity Ratio (%)', name: 'BAR' },
  { label: 'Standing Vertical (inches)', name: 'STNDVERT' },
  { label: 'Lane Agility (seconds)', name: 'LANE' },
  { label: 'Sprint (seconds)', name: 'SPRINT' },
];

const CATEGORY_ORDER = ['Bust', 'Role Player', 'Starter', 'Star'];

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
    try {
      const res = await axios.post(`${API_URL}/predict`, form);
      setResult(res.data);
    } catch (err) {
      setResult({ error: 'Prediction failed. Is the backend running on port 5001?' });
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 520, margin: 'auto', padding: 20, fontFamily: 'sans-serif' }}>
      <h1>NBA Draft Predictor</h1>
      <p style={{ color: '#555' }}>
        Enter a prospect's combine measurables to see a predicted career outcome tier.
        Heads up: the project's <code>report/REPORT.md</code> found that combine
        measurables alone barely predict career outcome — treat this as a demo of the
        pipeline, not a scouting tool.
      </p>
      <form onSubmit={handleSubmit}>
        {combineFields.map((field) => (
          <div key={field.name} style={{ marginBottom: 12 }}>
            <label>
              {field.label}:<br />
              <input
                type="number"
                step="any"
                name={field.name}
                value={form[field.name] || ''}
                onChange={handleChange}
                required
                style={{ width: '100%', padding: 8, boxSizing: 'border-box' }}
              />
            </label>
          </div>
        ))}
        <button type="submit" disabled={loading} style={{ padding: '10px 20px', fontSize: 16 }}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>
      {result && !result.error && (
        <div style={{ marginTop: 24, padding: 16, border: '1px solid #ccc', borderRadius: 8 }}>
          <h2>Predicted category: {result.category}</h2>
          <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
            <tbody>
              {CATEGORY_ORDER.map((cat) => (
                <tr key={cat}>
                  <td style={{ padding: '4px 0' }}>{cat}</td>
                  <td style={{ padding: '4px 0' }}>
                    <div style={{ background: '#eee', borderRadius: 4, overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${(result.probabilities[cat] || 0) * 100}%`,
                          background: cat === result.category ? '#2980b9' : '#aaa',
                          padding: '2px 6px',
                          color: 'white',
                          fontSize: 12,
                          minWidth: 32,
                          textAlign: 'right',
                        }}
                      >
                        {Math.round((result.probabilities[cat] || 0) * 100)}%
                      </div>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p>Predicted PPG: {result.predicted_ppg}</p>
          <p>Predicted RPG: {result.predicted_rpg}</p>
          <p>Predicted APG: {result.predicted_apg}</p>
          <h3>Closest career comparison: {result.comparison}</h3>
        </div>
      )}
      {result && result.error && (
        <div style={{ marginTop: 24, color: 'red' }}>{result.error}</div>
      )}
    </div>
  );
}

export default App;
