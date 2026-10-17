"""
=============================================================================
 app.py  -  Credit Scoring Web Application (Flask)
=============================================================================
 Run with:
   python -X utf8 app.py
 Then open: http://localhost:5000
=============================================================================
"""

import os
import sys
import glob
import json
import base64
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template_string, request, jsonify

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
OUTPUT_DIR = "outputs"

# ── Load model at startup ─────────────────────────────────────────────────────
def find_best_model():
    pattern = os.path.join(OUTPUT_DIR, "best_model_*.pkl")
    files   = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

MODEL_PATH = find_best_model()
MODEL      = joblib.load(MODEL_PATH) if MODEL_PATH else None
MODEL_NAME = os.path.basename(MODEL_PATH).replace("best_model_","").replace(".pkl","").replace("_"," ").title() if MODEL_PATH else "No Model"

def risk_tier(prob):
    if prob < 0.25:   return ("Low Risk",       "🟢", "#10b981")
    elif prob < 0.50: return ("Medium Risk",     "🟡", "#f59e0b")
    elif prob < 0.75: return ("High Risk",       "🟠", "#f97316")
    else:             return ("Very High Risk",  "🔴", "#ef4444")

def img_to_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Credit Scoring Model — Dashboard</title>
<meta name="description" content="AI-powered credit risk assessment dashboard with real-time predictions">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --surface2:  #1f2937;
    --border:    #374151;
    --primary:   #6366f1;
    --primary2:  #818cf8;
    --accent:    #06b6d4;
    --success:   #10b981;
    --warning:   #f59e0b;
    --danger:    #ef4444;
    --text:      #f9fafb;
    --muted:     #9ca3af;
    --radius:    12px;
  }

  html { scroll-behavior: smooth; }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Background glow ── */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background:
      radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,102,241,.18) 0%, transparent 60%),
      radial-gradient(ellipse 60% 40% at 80% 110%, rgba(6,182,212,.12) 0%, transparent 60%);
    pointer-events: none; z-index: 0;
  }

  /* ── Navbar ── */
  nav {
    position: sticky; top: 0; z-index: 100;
    background: rgba(10,14,26,.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 2rem; height: 64px;
  }
  .nav-brand { display: flex; align-items: center; gap: .7rem; font-weight: 700; font-size: 1.1rem; }
  .nav-brand .dot {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: grid; place-items: center; font-size: 1rem;
  }
  .nav-links { display: flex; gap: 1.5rem; }
  .nav-links a {
    color: var(--muted); text-decoration: none; font-size: .9rem;
    transition: color .2s;
  }
  .nav-links a:hover { color: var(--text); }

  /* ── Main layout ── */
  main { position: relative; z-index: 1; max-width: 1280px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }

  /* ── Hero ── */
  .hero { text-align: center; padding: 3rem 0 2rem; }
  .hero-badge {
    display: inline-flex; align-items: center; gap: .4rem;
    background: rgba(99,102,241,.15); border: 1px solid rgba(99,102,241,.4);
    color: var(--primary2); border-radius: 999px;
    padding: .3rem .9rem; font-size: .8rem; font-weight: 600; margin-bottom: 1.2rem;
  }
  .hero h1 { font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 800; line-height: 1.1;
    background: linear-gradient(135deg, #fff 30%, var(--primary2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }
  .hero p { color: var(--muted); margin-top: .8rem; font-size: 1.05rem; }

  /* ── Stats bar ── */
  .stats-bar {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem; margin: 2rem 0;
  }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    position: relative; overflow: hidden;
    transition: transform .2s, border-color .2s;
  }
  .stat-card:hover { transform: translateY(-3px); border-color: var(--primary); }
  .stat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
  }
  .stat-card .label { font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  .stat-card .value { font-size: 1.8rem; font-weight: 800; margin-top: .3rem;
    background: linear-gradient(135deg, #fff, var(--primary2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }
  .stat-card .sub { font-size: .78rem; color: var(--muted); margin-top: .2rem; }

  /* ── Section titles ── */
  .section-title {
    font-size: 1.3rem; font-weight: 700; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: .5rem;
  }
  .section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
  }

  /* ── Charts grid ── */
  .charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 1.2rem; margin-bottom: 2rem;
  }
  .chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    transition: transform .2s, border-color .2s;
  }
  .chart-card:hover { transform: translateY(-3px); border-color: var(--primary); }
  .chart-card h3 { font-size: .9rem; color: var(--muted); margin-bottom: .8rem; font-weight: 500; }
  .chart-card img { width: 100%; border-radius: 8px; display: block; }

  /* ── Predict form ── */
  #predict { scroll-margin-top: 80px; }
  .predict-wrapper {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 2rem;
  }
  .predict-header {
    padding: 1.5rem 2rem;
    background: linear-gradient(135deg, rgba(99,102,241,.1), rgba(6,182,212,.05));
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: .7rem;
  }
  .predict-header h2 { font-size: 1.1rem; font-weight: 700; }
  .predict-header p  { font-size: .85rem; color: var(--muted); margin-top: .2rem; }
  .predict-body { padding: 2rem; }

  .form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 1.2rem;
  }
  .form-group { display: flex; flex-direction: column; gap: .4rem; }
  .form-group label { font-size: .8rem; color: var(--muted); font-weight: 500; }
  .form-group select,
  .form-group input {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 8px;
    padding: .6rem .9rem;
    font-size: .9rem;
    font-family: inherit;
    transition: border-color .2s, box-shadow .2s;
    appearance: none;
  }
  .form-group select:focus,
  .form-group input:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(99,102,241,.2);
  }

  .form-actions { margin-top: 1.5rem; display: flex; gap: 1rem; align-items: center; }
  .btn-primary {
    background: linear-gradient(135deg, var(--primary), #8b5cf6);
    color: #fff; border: none; border-radius: 8px;
    padding: .75rem 2rem; font-size: .95rem; font-weight: 600;
    cursor: pointer; transition: opacity .2s, transform .15s;
    font-family: inherit;
  }
  .btn-primary:hover { opacity: .9; transform: translateY(-1px); }
  .btn-secondary {
    background: var(--surface2); color: var(--muted);
    border: 1px solid var(--border); border-radius: 8px;
    padding: .75rem 1.5rem; font-size: .9rem;
    cursor: pointer; transition: color .2s, border-color .2s;
    font-family: inherit;
  }
  .btn-secondary:hover { color: var(--text); border-color: var(--primary); }

  /* ── Result card ── */
  #result-panel { margin-top: 1.5rem; }
  .result-card {
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;
    border: 1px solid;
    animation: fadeInUp .4s ease;
  }
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .result-card.good { background: rgba(16,185,129,.08); border-color: rgba(16,185,129,.4); }
  .result-card.bad  { background: rgba(239,68,68,.08);  border-color: rgba(239,68,68,.4); }
  .result-icon { font-size: 3rem; }
  .result-label { font-size: 1.8rem; font-weight: 800; }
  .result-tier  { font-size: .95rem; color: var(--muted); margin-top: .2rem; }
  .result-prob  {
    margin-left: auto;
    text-align: center;
    background: var(--surface2);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    min-width: 140px;
  }
  .result-prob .prob-value { font-size: 2.5rem; font-weight: 800; }
  .result-prob .prob-label { font-size: .78rem; color: var(--muted); margin-top: .2rem; }

  /* Progress bar */
  .risk-bar-wrap { width: 100%; flex-basis: 100%; }
  .risk-bar-label { display: flex; justify-content: space-between; margin-bottom: .4rem; font-size: .8rem; color: var(--muted); }
  .risk-bar-bg { background: var(--surface2); border-radius: 999px; height: 10px; overflow: hidden; }
  .risk-bar-fill { height: 100%; border-radius: 999px; transition: width 1s ease; }

  /* ── Quick demos ── */
  .demos { display: flex; gap: .8rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
  .demo-btn {
    background: var(--surface2); border: 1px solid var(--border);
    color: var(--muted); border-radius: 8px;
    padding: .45rem 1rem; font-size: .82rem; font-weight: 500;
    cursor: pointer; transition: all .2s; font-family: inherit;
  }
  .demo-btn:hover { color: var(--text); border-color: var(--primary); background: rgba(99,102,241,.08); }

  /* ── Table ── */
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: .88rem; }
  thead tr { border-bottom: 2px solid var(--border); }
  th { padding: .7rem 1rem; text-align: left; color: var(--muted); font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
  td { padding: .65rem 1rem; border-bottom: 1px solid rgba(55,65,81,.5); }
  tr:hover td { background: rgba(99,102,241,.04); }
  .badge {
    display: inline-block; padding: .2rem .65rem; border-radius: 999px;
    font-size: .75rem; font-weight: 700;
  }
  .badge-good    { background: rgba(16,185,129,.15); color: #10b981; }
  .badge-bad     { background: rgba(239, 68, 68,.15); color: #ef4444; }
  .badge-low     { background: rgba(16,185,129,.12); color: #10b981; }
  .badge-medium  { background: rgba(245,158,11,.12); color: #f59e0b; }
  .badge-high    { background: rgba(249,115,22,.12); color: #f97316; }
  .badge-vhigh   { background: rgba(239, 68, 68,.12); color: #ef4444; }

  /* ── Loader ── */
  .spinner {
    display: inline-block; width: 18px; height: 18px;
    border: 2px solid rgba(255,255,255,.3);
    border-top-color: #fff;
    border-radius: 50%; animation: spin .6s linear infinite;
    vertical-align: middle; margin-right: .4rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Footer ── */
  footer {
    text-align: center; padding: 2rem;
    color: var(--muted); font-size: .82rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
  }

  @media (max-width: 640px) {
    .form-grid { grid-template-columns: 1fr 1fr; }
    .charts-grid { grid-template-columns: 1fr; }
    .result-prob { margin-left: 0; }
  }
</style>
</head>
<body>

<!-- Navbar -->
<nav>
  <div class="nav-brand">
    <div class="dot">💳</div>
    <span>CreditScore<span style="color:var(--primary2)">AI</span></span>
  </div>
  <div class="nav-links">
    <a href="#dashboard">Dashboard</a>
    <a href="#charts">Charts</a>
    <a href="#predict">Predict</a>
  </div>
</nav>

<main>

  <!-- Hero -->
  <div class="hero">
    <div class="hero-badge">🤖 AI-Powered Credit Risk Assessment</div>
    <h1>Credit Scoring<br>Intelligence Dashboard</h1>
    <p>Real-time credit risk prediction using the UCI German Credit Dataset with ensemble ML models</p>
  </div>

  <!-- Stats -->
  <div id="dashboard" class="stats-bar" style="scroll-margin-top:80px">
    <div class="stat-card">
      <div class="label">Best Model</div>
      <div class="value" style="font-size:1.15rem;margin-top:.5rem">{{ model_name }}</div>
      <div class="sub">Highest ROC-AUC on test set</div>
    </div>
    <div class="stat-card">
      <div class="label">ROC-AUC Score</div>
      <div class="value">0.8096</div>
      <div class="sub">Test set performance</div>
    </div>
    <div class="stat-card">
      <div class="label">Accuracy</div>
      <div class="value">79.0%</div>
      <div class="sub">On 200 held-out samples</div>
    </div>
    <div class="stat-card">
      <div class="label">Training Set</div>
      <div class="value">800</div>
      <div class="sub">Samples (80/20 split)</div>
    </div>
    <div class="stat-card">
      <div class="label">Features</div>
      <div class="value">20</div>
      <div class="sub">7 numeric · 13 categorical</div>
    </div>
    <div class="stat-card">
      <div class="label">Models Trained</div>
      <div class="value">6</div>
      <div class="sub">LR, DT, RF, GB, XGB, LGBM</div>
    </div>
  </div>

  <!-- Model Leaderboard -->
  <div class="section-title">🏆 Model Leaderboard</div>
  <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem;margin-bottom:2rem;">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>Model</th><th>Accuracy</th>
            <th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th>
          </tr>
        </thead>
        <tbody>
          {% for r in leaderboard %}
          <tr>
            <td><strong>{{ r.rank }}</strong>{% if r.rank == 1 %} 🥇{% elif r.rank == 2 %} 🥈{% elif r.rank == 3 %} 🥉{% endif %}</td>
            <td><strong>{{ r.model }}</strong></td>
            <td>{{ "%.1f"|format(r.accuracy*100) }}%</td>
            <td>{{ "%.4f"|format(r.precision) }}</td>
            <td>{{ "%.4f"|format(r.recall) }}</td>
            <td>{{ "%.4f"|format(r.f1) }}</td>
            <td>
              <strong style="color:{% if r.auc >= 0.8 %}#10b981{% elif r.auc >= 0.75 %}#f59e0b{% else %}#9ca3af{% endif %}">
                {{ "%.4f"|format(r.auc) }}
              </strong>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Charts -->
  <div id="charts" class="section-title" style="scroll-margin-top:80px">📊 Model Visualisations</div>
  <div class="charts-grid">
    {% if roc_img %}
    <div class="chart-card" style="grid-column: span 2;">
      <h3>ROC Curves — All Models</h3>
      <img src="data:image/png;base64,{{ roc_img }}" alt="ROC Curves">
    </div>
    {% endif %}
    {% if eda_dist_img %}
    <div class="chart-card">
      <h3>Feature Distributions (EDA)</h3>
      <img src="data:image/png;base64,{{ eda_dist_img }}" alt="EDA Distributions">
    </div>
    {% endif %}
    {% if eda_corr_img %}
    <div class="chart-card">
      <h3>Feature Correlation Heatmap</h3>
      <img src="data:image/png;base64,{{ eda_corr_img }}" alt="Correlation Heatmap">
    </div>
    {% endif %}
    {% if shap_bar_img %}
    <div class="chart-card">
      <h3>SHAP Feature Importance (Global)</h3>
      <img src="data:image/png;base64,{{ shap_bar_img }}" alt="SHAP Bar">
    </div>
    {% endif %}
    {% if shap_bee_img %}
    <div class="chart-card">
      <h3>SHAP Beeswarm Plot</h3>
      <img src="data:image/png;base64,{{ shap_bee_img }}" alt="SHAP Beeswarm">
    </div>
    {% endif %}
    {% for name, img in cm_imgs %}
    <div class="chart-card">
      <h3>Confusion Matrix — {{ name }}</h3>
      <img src="data:image/png;base64,{{ img }}" alt="CM {{ name }}">
    </div>
    {% endfor %}
  </div>

  <!-- Predict -->
  <div id="predict" class="section-title">🎯 Live Credit Risk Prediction</div>
  <div class="predict-wrapper">
    <div class="predict-header">
      <div>
        <h2>Applicant Risk Assessment</h2>
        <p>Fill in the applicant details below and click Predict to get an instant credit risk score</p>
      </div>
    </div>
    <div class="predict-body">

      <!-- Quick demo buttons -->
      <div style="margin-bottom:1.2rem">
        <div style="font-size:.8rem;color:var(--muted);margin-bottom:.6rem;font-weight:500">QUICK DEMO PROFILES</div>
        <div class="demos">
          <button class="demo-btn" onclick="loadDemo('low')">🟢 Low Risk Applicant</button>
          <button class="demo-btn" onclick="loadDemo('medium')">🟡 Medium Risk Applicant</button>
          <button class="demo-btn" onclick="loadDemo('high')">🔴 High Risk Applicant</button>
        </div>
      </div>

      <form id="predict-form" onsubmit="submitPredict(event)">
        <div class="form-grid">
          <div class="form-group">
            <label for="checking_account_status">Checking Account Status</label>
            <select id="checking_account_status" name="checking_account_status">
              <option value="A11">A11 — &lt; 0 DM</option>
              <option value="A12">A12 — 0–200 DM</option>
              <option value="A13">A13 — ≥ 200 DM</option>
              <option value="A14" selected>A14 — No account</option>
            </select>
          </div>
          <div class="form-group">
            <label for="duration_months">Duration (months)</label>
            <input type="number" id="duration_months" name="duration_months" value="12" min="1" max="72">
          </div>
          <div class="form-group">
            <label for="credit_history">Credit History</label>
            <select id="credit_history" name="credit_history">
              <option value="A30">A30 — No credits / all paid</option>
              <option value="A31">A31 — All paid duly at bank</option>
              <option value="A32">A32 — Existing credits paid duly</option>
              <option value="A33">A33 — Delay in paying</option>
              <option value="A34" selected>A34 — Critical account</option>
            </select>
          </div>
          <div class="form-group">
            <label for="purpose">Purpose</label>
            <select id="purpose" name="purpose">
              <option value="A40">A40 — Car (new)</option>
              <option value="A41">A41 — Car (used)</option>
              <option value="A42" selected>A42 — Furniture/equipment</option>
              <option value="A43">A43 — Radio/TV</option>
              <option value="A44">A44 — Domestic appliances</option>
              <option value="A45">A45 — Repairs</option>
              <option value="A46">A46 — Education</option>
              <option value="A48">A48 — Retraining</option>
              <option value="A49">A49 — Business</option>
              <option value="A410">A410 — Others</option>
            </select>
          </div>
          <div class="form-group">
            <label for="credit_amount">Credit Amount (DM)</label>
            <input type="number" id="credit_amount" name="credit_amount" value="2000" min="100">
          </div>
          <div class="form-group">
            <label for="savings_account">Savings Account</label>
            <select id="savings_account" name="savings_account">
              <option value="A61">A61 — &lt; 100 DM</option>
              <option value="A62">A62 — 100–500 DM</option>
              <option value="A63">A63 — 500–1000 DM</option>
              <option value="A64">A64 — ≥ 1000 DM</option>
              <option value="A65" selected>A65 — Unknown / no savings</option>
            </select>
          </div>
          <div class="form-group">
            <label for="employment_since">Employment Since</label>
            <select id="employment_since" name="employment_since">
              <option value="A71">A71 — Unemployed</option>
              <option value="A72">A72 — &lt; 1 year</option>
              <option value="A73">A73 — 1–4 years</option>
              <option value="A74">A74 — 4–7 years</option>
              <option value="A75" selected>A75 — ≥ 7 years</option>
            </select>
          </div>
          <div class="form-group">
            <label for="installment_rate">Installment Rate (%)</label>
            <input type="number" id="installment_rate" name="installment_rate" value="2" min="1" max="4">
          </div>
          <div class="form-group">
            <label for="personal_status_sex">Personal Status / Sex</label>
            <select id="personal_status_sex" name="personal_status_sex">
              <option value="A91">A91 — Male: divorced/separated</option>
              <option value="A92">A92 — Female: div/sep/married</option>
              <option value="A93" selected>A93 — Male: single</option>
              <option value="A94">A94 — Male: married/widowed</option>
              <option value="A95">A95 — Female: single</option>
            </select>
          </div>
          <div class="form-group">
            <label for="other_debtors">Other Debtors/Guarantors</label>
            <select id="other_debtors" name="other_debtors">
              <option value="A101" selected>A101 — None</option>
              <option value="A102">A102 — Co-applicant</option>
              <option value="A103">A103 — Guarantor</option>
            </select>
          </div>
          <div class="form-group">
            <label for="residence_since">Residence Since (years)</label>
            <input type="number" id="residence_since" name="residence_since" value="4" min="1" max="4">
          </div>
          <div class="form-group">
            <label for="property">Property</label>
            <select id="property" name="property">
              <option value="A121" selected>A121 — Real estate</option>
              <option value="A122">A122 — Building society savings</option>
              <option value="A123">A123 — Car / other</option>
              <option value="A124">A124 — Unknown / no property</option>
            </select>
          </div>
          <div class="form-group">
            <label for="age">Age (years)</label>
            <input type="number" id="age" name="age" value="45" min="18" max="100">
          </div>
          <div class="form-group">
            <label for="other_installment_plans">Other Installment Plans</label>
            <select id="other_installment_plans" name="other_installment_plans">
              <option value="A141">A141 — Bank</option>
              <option value="A142">A142 — Stores</option>
              <option value="A143" selected>A143 — None</option>
            </select>
          </div>
          <div class="form-group">
            <label for="housing">Housing</label>
            <select id="housing" name="housing">
              <option value="A151">A151 — Rent</option>
              <option value="A152" selected>A152 — Own</option>
              <option value="A153">A153 — For free</option>
            </select>
          </div>
          <div class="form-group">
            <label for="existing_credits">Existing Credits (count)</label>
            <input type="number" id="existing_credits" name="existing_credits" value="1" min="1" max="4">
          </div>
          <div class="form-group">
            <label for="job">Job</label>
            <select id="job" name="job">
              <option value="A171">A171 — Unemployed / non-resident</option>
              <option value="A172">A172 — Unskilled — resident</option>
              <option value="A173" selected>A173 — Skilled employee</option>
              <option value="A174">A174 — Management / highly qualified</option>
            </select>
          </div>
          <div class="form-group">
            <label for="dependents">Number of Dependents</label>
            <input type="number" id="dependents" name="dependents" value="1" min="1" max="2">
          </div>
          <div class="form-group">
            <label for="telephone">Telephone</label>
            <select id="telephone" name="telephone">
              <option value="A191">A191 — None</option>
              <option value="A192" selected>A192 — Yes, registered</option>
            </select>
          </div>
          <div class="form-group">
            <label for="foreign_worker">Foreign Worker</label>
            <select id="foreign_worker" name="foreign_worker">
              <option value="A201" selected>A201 — Yes</option>
              <option value="A202">A202 — No</option>
            </select>
          </div>
        </div>

        <div class="form-actions">
          <button type="submit" class="btn-primary" id="submit-btn">
            ⚡ Predict Credit Risk
          </button>
          <button type="button" class="btn-secondary" onclick="resetForm()">Reset</button>
        </div>
      </form>

      <!-- Result -->
      <div id="result-panel"></div>

    </div>
  </div>

</main>

<footer>
  Built with Flask + scikit-learn · UCI German Credit Dataset ·
  Model: <strong>{{ model_name }}</strong> · ROC-AUC 0.8096
</footer>

<script>
const DEMOS = {
  low: {
    checking_account_status:"A14", duration_months:12, credit_history:"A34",
    purpose:"A42", credit_amount:2000, savings_account:"A65",
    employment_since:"A75", installment_rate:2, personal_status_sex:"A93",
    other_debtors:"A101", residence_since:4, property:"A121", age:45,
    other_installment_plans:"A143", housing:"A152", existing_credits:1,
    job:"A173", dependents:1, telephone:"A192", foreign_worker:"A201"
  },
  medium: {
    checking_account_status:"A11", duration_months:36, credit_history:"A32",
    purpose:"A40", credit_amount:8500, savings_account:"A61",
    employment_since:"A72", installment_rate:3, personal_status_sex:"A92",
    other_debtors:"A101", residence_since:2, property:"A123", age:28,
    other_installment_plans:"A141", housing:"A151", existing_credits:2,
    job:"A172", dependents:2, telephone:"A191", foreign_worker:"A201"
  },
  high: {
    checking_account_status:"A11", duration_months:60, credit_history:"A30",
    purpose:"A49", credit_amount:18000, savings_account:"A61",
    employment_since:"A71", installment_rate:4, personal_status_sex:"A93",
    other_debtors:"A102", residence_since:1, property:"A124", age:22,
    other_installment_plans:"A141", housing:"A151", existing_credits:3,
    job:"A171", dependents:2, telephone:"A191", foreign_worker:"A201"
  }
};

function loadDemo(type) {
  const d = DEMOS[type];
  for (const [k, v] of Object.entries(d)) {
    const el = document.getElementById(k);
    if (el) el.value = v;
  }
  document.getElementById('result-panel').innerHTML = '';
}

function resetForm() {
  document.getElementById('predict-form').reset();
  document.getElementById('result-panel').innerHTML = '';
}

async function submitPredict(e) {
  e.preventDefault();
  const btn = document.getElementById('submit-btn');
  btn.innerHTML = '<span class="spinner"></span>Predicting…';
  btn.disabled = true;

  const form   = document.getElementById('predict-form');
  const data   = Object.fromEntries(new FormData(form).entries());

  try {
    const res  = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data) });
    const json = await res.json();
    if (json.error) throw new Error(json.error);

    const prob    = json.probability;
    const label   = json.label;
    const tier    = json.tier;
    const pct     = Math.round(prob * 100);
    const isGood  = label === 'GOOD';

    const tierClass = tier.toLowerCase().replace(' risk','').replace(' ','');
    const badgeMap  = { low:'badge-low', medium:'badge-medium', high:'badge-high', 'very high':'badge-vhigh' };
    const tierKey   = tier.toLowerCase().replace(' risk','').trim();

    const barColor  = isGood ? '#10b981' : pct < 50 ? '#f59e0b' : pct < 75 ? '#f97316' : '#ef4444';

    document.getElementById('result-panel').innerHTML = `
      <div class="result-card ${isGood ? 'good' : 'bad'}">
        <div class="result-icon">${isGood ? '✅' : '❌'}</div>
        <div>
          <div class="result-label" style="color:${isGood ? '#10b981' : '#ef4444'}">${label} CREDIT</div>
          <div class="result-tier">${tier}</div>
        </div>
        <div class="result-prob">
          <div class="prob-value" style="color:${barColor}">${pct}%</div>
          <div class="prob-label">P(Bad Credit)</div>
        </div>
        <div class="risk-bar-wrap">
          <div class="risk-bar-label"><span>0% — Safe</span><span>100% — Risky</span></div>
          <div class="risk-bar-bg">
            <div class="risk-bar-fill" id="risk-fill" style="width:0%;background:${barColor}"></div>
          </div>
        </div>
      </div>`;

    setTimeout(() => {
      const fill = document.getElementById('risk-fill');
      if (fill) fill.style.width = pct + '%';
    }, 50);

  } catch (err) {
    document.getElementById('result-panel').innerHTML = `
      <div style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.4);border-radius:var(--radius);padding:1.2rem;color:#ef4444;margin-top:1rem;">
        ⚠️ ${err.message}
      </div>`;
  } finally {
    btn.innerHTML = '⚡ Predict Credit Risk';
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    leaderboard = [
        {"rank":1, "model":"Logistic Regression", "accuracy":.790, "precision":.7143, "recall":.5000, "f1":.5882, "auc":.8096},
        {"rank":2, "model":"Random Forest",        "accuracy":.770, "precision":.6944, "recall":.4167, "f1":.5208, "auc":.8038},
        {"rank":3, "model":"Gradient Boosting",    "accuracy":.775, "precision":.6471, "recall":.5500, "f1":.5946, "auc":.7952},
        {"rank":4, "model":"XGBoost",              "accuracy":.740, "precision":.6000, "recall":.4000, "f1":.4800, "auc":.7820},
        {"rank":5, "model":"LightGBM",             "accuracy":.730, "precision":.5682, "recall":.4167, "f1":.4808, "auc":.7640},
        {"rank":6, "model":"Decision Tree",        "accuracy":.670, "precision":.4516, "recall":.4667, "f1":.4590, "auc":.6675},
    ]

    def img(name):
        return img_to_b64(os.path.join(OUTPUT_DIR, name))

    # Confusion matrix images
    cm_names = ["Logistic Regression","Decision Tree","Random Forest","Gradient Boosting","XGBoost","LightGBM"]
    cm_imgs  = []
    for n in cm_names:
        fname = f"cm_{n.lower().replace(' ','_')}.png"
        b64   = img(fname)
        if b64:
            cm_imgs.append((n, b64))

    return render_template_string(HTML,
        model_name    = MODEL_NAME,
        leaderboard   = leaderboard,
        roc_img       = img("roc_curves.png"),
        eda_dist_img  = img("eda_distributions.png"),
        eda_corr_img  = img("eda_correlation.png"),
        shap_bar_img  = img("shap_bar.png"),
        shap_bee_img  = img("shap_beeswarm.png"),
        cm_imgs       = cm_imgs,
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    if MODEL is None:
        return jsonify({"error": "No trained model found. Please run credit_scoring_model.py first."}), 500

    data = request.get_json()
    try:
        # Cast numeric fields
        int_fields   = ["duration_months","credit_amount","installment_rate",
                         "residence_since","age","existing_credits","dependents"]
        for f in int_fields:
            if f in data:
                data[f] = int(data[f])

        df   = pd.DataFrame([data])
        pred = MODEL.predict(df)[0]
        prob = float(MODEL.predict_proba(df)[0][1])
        tier, _, _ = risk_tier(prob)

        return jsonify({
            "label":       "BAD" if pred == 1 else "GOOD",
            "probability": round(prob, 4),
            "tier":        tier,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  CreditScore AI — Web Dashboard")
    print("="*60)
    print(f"  Model loaded : {MODEL_NAME}")
    print(f"  Model path   : {MODEL_PATH}")
    print("  Running at   : http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
