---
title: Crime Analytics Suspect Ranking System
emoji: 🔍
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Crime Analytics & Suspect Ranking System

A machine learning decision-support system that analyzes crime data, finds similar cases by Modus Operandi, and ranks likely suspect replication patterns using an ensemble classifier.

**Live Demo → [HuggingFace Space](https://huggingface.co/spaces/prasanthmax/crime-analytics-suspect-ranking)**

---

## What It Does

| Feature | Description |
|---|---|
| **MO Similarity Search** | TF-IDF cosine similarity over 100,000 crime records to find cases with matching Modus Operandi |
| **Ensemble Suspect Ranking** | Random Forest + Gradient Boosting trained on 7 pairwise features (area, weapon, crime code, time, sex, age, MO text) |
| **Visual Analytics** | Monthly trends, YoY comparison, area/weapon breakdowns, age/sex demographics via Recharts |
| **Model Retraining** | On-demand retraining with live Precision@K, Recall@K, and NDCG@K metrics |

---

## Tech Stack

**Backend** — Python 3.11, FastAPI, Uvicorn, scikit-learn 1.8, pandas, joblib, scipy

**Frontend** — React 19, Vite, Tailwind CSS v4, Recharts, Lucide React

**Deployment** — Docker on HuggingFace Spaces (16GB RAM, no spin-down), Git LFS for dataset

---

## Project Structure

```
crime-analytics-suspect-ranking-system/
├── app/
│   ├── backend/
│   │   └── main.py               # FastAPI app — all API routes + static file serving
│   └── frontend/
│       ├── src/
│       │   └── App.jsx            # Single-page React UI
│       ├── package.json
│       └── vite.config.js
├── src/
│   ├── similarity_engine.py       # TF-IDF vectorizer + cosine similarity search
│   ├── suspect_ranker.py          # Ensemble ML ranker (RF + GB)
│   ├── model_evaluator.py         # Precision@K, Recall@K, NDCG@K evaluation
│   ├── preprocess_cases.py        # Raw LAPD CSV → clean_cases.csv.gz
│   └── cli_eval.py                # CLI for offline training and evaluation
├── data/
│   └── processed/
│       └── clean_cases.csv.gz     # 100k processed LAPD crime records (Git LFS)
├── models/
│   ├── ensemble_ranker.joblib     # Trained RF + GB ensemble (Git LFS)
│   ├── tfidf_vectorizer.joblib    # Pre-fitted TF-IDF vectorizer (Git LFS)
│   └── tfidf_matrix.npz           # Pre-computed sparse TF-IDF matrix (Git LFS)
├── scripts/
│   └── save_tfidf.py              # One-time script to pre-save TF-IDF artifacts
├── Dockerfile                     # Multi-stage: Node build → Python runtime
├── requirements.txt
└── README.md
```

---

## ML Architecture

### Similarity Engine
Modus Operandi text is vectorized using a pre-fitted `TfidfVectorizer` (saved as `.joblib` + `.npz` to avoid rebuilding on every startup). At query time, cosine similarity is computed against the full sparse matrix using `sklearn.metrics.pairwise.cosine_similarity`.

### Ensemble Suspect Ranker
Candidate cases from the similarity pool are re-ranked using a 7-feature ensemble:

| Feature | Type |
|---|---|
| MO text cosine similarity | Float |
| Area name match | Binary |
| Weapon description match | Binary |
| Crime code match | Binary |
| Time proximity (exponential decay, λ=365d) | Float |
| Victim sex match | Binary |
| Victim age difference | Float |

### Evaluation Results
| Metric | Baseline (TF-IDF) | Ensemble |
|---|---|---|
| Precision@10 | 0.93–0.96 | **1.00** |
| NDCG@10 | 0.93–0.97 | **1.00** |

Top feature importances: `area_match` (~63%), `weapon_match` (~30%), `MO similarity` (~3.4%)

---

## Local Development

### Backend
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt

# Pre-save TF-IDF artifacts (run once after preprocessing)
python scripts/save_tfidf.py

# Start API server
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
# API docs: http://127.0.0.1:8000/docs
```

### Frontend
```bash
cd app/frontend
npm install

# For local dev (points to local backend)
echo "VITE_API_BASE_URL=http://127.0.0.1:8000" > .env.local
npm run dev
# UI: http://127.0.0.1:5173
```

### Preprocess Raw Data (optional)
If starting from the raw LAPD dataset:
```bash
python src/preprocess_cases.py
python scripts/save_tfidf.py
```

---

## Deployment

The project deploys as a single Docker container on HuggingFace Spaces:

- **Stage 1** (Node 20): Builds the React frontend (`npm run build`)
- **Stage 2** (Python 3.11): Installs Python deps, copies source + models + built frontend
- FastAPI serves both the API (`/api/*`) and the React static files (`/`) from port 7860
- Dataset and model files are stored in Git LFS

```bash
# Push to HuggingFace Space
git remote add hf https://huggingface.co/spaces/prasanthmax/crime-analytics-suspect-ranking
git push hf main
```

---

## Dataset

Source: [LAPD Crime Data 2020–Present](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8) (City of Los Angeles Open Data)

The raw dataset is preprocessed to 100,000 records covering Feb 2024 – May 2025 across 21 LAPD divisions. Stored as `.csv.gz` via Git LFS.

---

## Academic Context

Developed as a final-year B.Tech project (AI & Data Science) at Dr. N G P Institute of Technology, Coimbatore. Targeting IEEE publication.
