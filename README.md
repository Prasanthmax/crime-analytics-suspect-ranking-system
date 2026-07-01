---
title: Crime Analytics Suspect Ranking System
emoji: 🔍
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
# Crime Analytics and Suspect Ranking System

A modern machine learning–based decision-support system to analyze crime data, identify similar cases based on Modus Operandi (MO), and rank likely suspect replication patterns. 

This project consists of a high-performance **Python (FastAPI) Backend** and an interactive, visually rich **React (Vite) Frontend** styled with **Tailwind CSS v4.0** and **Google's Material Design** principles.

---

## Solution Overview

This system assists law enforcement and intelligence analysts by providing:
1. **Semantic Similarity Search**: Matches Modus Operandi descriptions using an optimized TF-IDF sparse matrix cosine-similarity search.
2. **Ensemble Suspect Ranking**: Instead of linear heuristics, the system utilizes a machine learning **Ensemble Classifier** (Random Forest + Gradient Boosting) trained on case similarities (matching areas, weapons, crime codes, time proximity, sex, and age differences) to calculate case replication probability.
3. **Accuracy Metrics & Retraining**: Supports on-demand model retraining, showing feature importances and printing Precision@K, Recall@K, and NDCG@K improvements.
4. **Rich Visual Analytics**: Charts crime monthly distributions, YoY trends, areas, demographics, and temporal distributions using Recharts.

---

## Tech Stack

- **Backend API**: Python, FastAPI, Uvicorn, Scikit-learn, Pandas, Joblib
- **Frontend App**: React (v19), Vite, Tailwind CSS (v4), Recharts, Lucide React
- **ML Models**: TfidfVectorizer, RandomForestClassifier, GradientBoostingClassifier

---

## Project Structure

```
crime-analysis-suspect-ranking-system/
├── app/
│   ├── backend/        # FastAPI Application (main.py)
│   └── frontend/       # Vite React Web App (Tailwind CSS, App.jsx, index.css)
├── src/
│   ├── explore/        # Interactive Python Cells scripts (alternative to notebooks)
│   ├── cli_eval.py     # Command-line Evaluation and Retraining harness
│   ├── model_evaluator.py
│   ├── preprocess_cases.py
│   ├── similarity_engine.py
│   └── suspect_ranker.py
├── data/               # Raw and processed crime data
├── models/             # Serialized ensemble model cache (.joblib)
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

## Installation & Setup

### 1. Backend Server Setup

1. **Activate the Python virtual environment**:
   ```bash
   .venv\Scripts\activate
   ```

2. **Install pip dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess the raw data (if clean_cases.csv is not present)**:
   ```bash
   python src/preprocess_cases.py
   ```

4. **Start the FastAPI Backend**:
   ```bash
   uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
   ```
   The backend API docs will be available at `http://127.0.0.1:8000/docs`.

### 2. Frontend React Setup

1. **Navigate to the frontend folder**:
   ```bash
   cd app/frontend
   ```

2. **Install Node dependencies**:
   ```bash
   npm install
   ```

3. **Start the React Vite Development Server**:
   ```bash
   npm run dev -- --host 127.0.0.1 --port 5173
   ```
   Navigate to `http://127.0.0.1:5173/` in your browser.

---

## Machine Learning & Evaluation

### Training the Ranker
The Ensemble Suspect Ranker trains on pairwise features generated from historical crimes. To fit the model and print diagnostics, run:
```bash
python src/cli_eval.py
```

### Metrics Comparison (Accuracy Summary)
Comparative results against baseline Cosine Similarity:
- **Ensemble Precision@K**: Reaches **1.0000** for $K \le 10$ compared to baseline $0.93 - 0.96$.
- **Ensemble NDCG@K**: Reaches **1.0000** for $K \le 10$ compared to baseline $0.93 - 0.97$.
- **Average Feature Importance**:
  - `area_match`: ~63.1%
  - `weapon_match`: ~29.6%
  - `similarity` (MO Text Cosine): ~3.4%
  - `crime_code_match`: ~2.7%
