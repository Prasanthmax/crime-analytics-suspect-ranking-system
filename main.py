from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import urllib.request

from src.similarity_engine import SimilarityEngine
from src.suspect_ranker import SuspectRanker
from src.model_evaluator import ModelEvaluator, is_relevant

app = FastAPI(title="Crime Analytics & Suspect Ranking API")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed/clean_cases.csv.gz"))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/ensemble_ranker.joblib"))

# Load dataset and engines once on startup
df = None
similarity_engine = None
suspect_ranker = None

@app.on_event("startup")
def startup_event():
    global df, similarity_engine, suspect_ranker
    
    # Check if directory exists
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        print("[SERVER] Clean cases CSV not found locally. Checking for remote DATA_URL...")
        data_url = os.getenv("DATA_URL")
        if not data_url:
            raise FileNotFoundError(
                f"Dataset not found at {DATA_PATH} and 'DATA_URL' environment variable is not set. "
                "Please upload clean_cases.csv to a public cloud storage (Google Drive direct link, Dropbox, "
                "or GitHub Release) and configure the 'DATA_URL' environment variable on Render."
            )
        try:
            print(f"[SERVER] Downloading dataset from {data_url}...")
            urllib.request.urlretrieve(data_url, DATA_PATH)
            print("[SERVER] Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset from {data_url}: {e}")
            
    print("[SERVER] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"[SERVER] Dataset loaded: {len(df):,} rows.")
    
    # Initialize ML components
    similarity_engine = SimilarityEngine(df)
    suspect_ranker = SuspectRanker(df, model_path=MODEL_PATH)
    # Warm up / load model
    suspect_ranker.load_model()

@app.get("/api/metrics")
def get_metrics():
    global df
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
        
    date_min = df["datetime"].min()
    date_max = df["datetime"].max()
    span_days = int((date_max - date_min).days) if pd.notna(date_min) and pd.notna(date_max) else 0
    
    # Top 5 Areas by count
    by_area = df["area_name"].value_counts().head(5).reset_index()
    by_area.columns = ["area", "count"]
    
    # Top 5 Weapons by count
    by_weapon = df["weapon_desc"].fillna("UNKNOWN").value_counts().head(5).reset_index()
    by_weapon.columns = ["weapon", "count"]
    
    # Preview of recent ID-Theft cases
    id_theft_sample = (
        df[df["crm_cd_desc"] == "THEFT OF IDENTITY"]
        .sort_values("datetime", ascending=False)
        .head(10)
    )
    # Format dates to string
    id_theft_list = []
    for _, row in id_theft_sample.iterrows():
        id_theft_list.append({
            "dr_no": int(row["dr_no"]),
            "datetime": str(row["datetime"].date()) if pd.notna(row["datetime"]) else "N/A",
            "area_name": str(row.get("area_name", "N/A")),
            "weapon_desc": str(row.get("weapon_desc", "N/A")),
            "vict_age": int(row["vict_age"]) if pd.notna(row["vict_age"]) else "N/A",
            "vict_sex": str(row.get("vict_sex", "N/A")),
        })
        
    return {
        "summary": {
            "total_cases": len(df),
            "distinct_areas": int(df["area_name"].nunique()),
            "crime_types": int(df["crm_cd_desc"].nunique()),
            "time_span_days": span_days,
            "min_date": str(date_min.date()) if pd.notna(date_min) else None,
            "max_date": str(date_max.date()) if pd.notna(date_max) else None
        },
        "top_areas": by_area.to_dict(orient="records"),
        "top_weapons": by_weapon.to_dict(orient="records"),
        "recent_id_theft": id_theft_list
    }

@app.get("/api/cases/{case_id}")
def get_case(case_id: int):
    global df
    row = df[df["dr_no"] == case_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Case ID not found")
        
    case = row.iloc[0]
    return {
        "dr_no": int(case["dr_no"]),
        "datetime": str(case["datetime"].date()) if pd.notna(case["datetime"]) else "N/A",
        "area_name": str(case.get("area_name", "N/A")),
        "crm_cd": int(case.get("crm_cd", 0)),
        "crm_cd_desc": str(case.get("crm_cd_desc", "N/A")),
        "weapon_desc": str(case.get("weapon_desc", "N/A")),
        "vict_age": int(case["vict_age"]) if pd.notna(case["vict_age"]) else None,
        "vict_sex": str(case.get("vict_sex", "N/A")),
        "mo_text": str(case.get("mo_text", "N/A"))
    }

@app.get("/api/cases/search/lookup")
def search_cases(q: str = ""):
    global df
    if not q:
        return []
    # Find matching case IDs
    matches = df[df["dr_no"].astype(str).str.contains(q)].head(10)
    return [int(cid) for cid in matches["dr_no"].tolist()]

@app.get("/api/similarity")
def get_similar(
    case_id: int, 
    top_k: int = 10, 
    city: str = "All", 
    weapon: str = "All"
):
    global similarity_engine
    if similarity_engine is None:
        raise HTTPException(status_code=503, detail="Similarity engine not initialized")
        
    similar = similarity_engine.get_similar_cases(
        case_id,
        top_k=top_k,
        city=None if city == "All" else city,
        weapon=None if weapon == "All" else weapon
    )
    
    if similar.empty:
        return []
        
    results = []
    for _, row in similar.iterrows():
        results.append({
            "dr_no": int(row["dr_no"]),
            "datetime": str(row["datetime"].date()) if pd.notna(row["datetime"]) else "N/A",
            "area_name": str(row.get("area_name", "N/A")),
            "crm_cd_desc": str(row.get("crm_cd_desc", "N/A")),
            "weapon_desc": str(row.get("weapon_desc", "N/A")),
            "vict_age": int(row["vict_age"]) if pd.notna(row["vict_age"]) else None,
            "vict_sex": str(row.get("vict_sex", "N/A")),
            "similarity": float(row["similarity"]),
            "mo_text": str(row.get("mo_text", "N/A"))
        })
        
    return results

@app.get("/api/ranking")
def get_ranking(case_id: int, top_k: int = 10):
    global df, similarity_engine, suspect_ranker
    if similarity_engine is None or suspect_ranker is None:
        raise HTTPException(status_code=503, detail="ML components not initialized")
        
    base_row = df[df["dr_no"] == case_id]
    if base_row.empty:
        raise HTTPException(status_code=404, detail="Base Case ID not found")
    
    base_case = base_row.iloc[0]
    
    # Get similar pool (TF-IDF cosine similarity)
    pool = similarity_engine.get_similar_cases(case_id, top_k=max(30, top_k * 3))
    if pool.empty:
        return []
        
    # Rank candidates using ensemble SuspectRanker
    ranked = suspect_ranker.rank_suspects(case_id, pool, top_k=top_k)
    
    # Extract match features for display purposes
    features_df = suspect_ranker.extract_features(base_case, ranked)
    
    rankings = []
    for idx, row in ranked.iterrows():
        feats = features_df.loc[idx]
        rankings.append({
            "dr_no": int(row["dr_no"]),
            "datetime": str(row["datetime"].date()) if pd.notna(row["datetime"]) else "N/A",
            "area_name": str(row.get("area_name", "N/A")),
            "crm_cd_desc": str(row.get("crm_cd_desc", "N/A")),
            "weapon_desc": str(row.get("weapon_desc", "N/A")),
            "vict_age": int(row["vict_age"]) if pd.notna(row["vict_age"]) else None,
            "vict_sex": str(row.get("vict_sex", "N/A")),
            "similarity": float(row["similarity"]),
            "score": float(row["score"]),
            "mo_text": str(row.get("mo_text", "N/A")),
            # Breakdown matches for explanation
            "features": {
                "area_match": bool(feats["area_match"]),
                "weapon_match": bool(feats["weapon_match"]),
                "crime_code_match": bool(feats["crime_code_match"]),
                "time_decay": float(feats["time_decay"]),
                "sex_match": bool(feats["sex_match"]),
                "age_difference": float(feats["age_difference"])
            }
        })
        
    return rankings

@app.get("/api/analytics")
def get_analytics():
    global df
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
        
    # 1. Monthly trend (area chart)
    monthly = df.set_index("datetime").resample("ME").size().reset_index(name="count")
    monthly["month"] = monthly["datetime"].dt.strftime("%Y-%m")
    monthly_data = monthly[["month", "count"]].to_dict(orient="records")
    
    # 2. Top 10 Areas (bar chart)
    top_areas = df["area_name"].value_counts().head(10).reset_index()
    top_areas.columns = ["name", "count"]
    areas_data = top_areas.to_dict(orient="records")
    
    # 3. Top 10 Weapons (bar chart)
    top_weapons = df["weapon_desc"].fillna("UNKNOWN").value_counts().head(10).reset_index()
    top_weapons.columns = ["name", "count"]
    weapons_data = top_weapons.to_dict(orient="records")
    
    # 4. Age groups
    bins = [0, 18, 30, 45, 60, 80, 120]
    labels = ["0-17", "18-29", "30-44", "45-59", "60-79", "80+"]
    age_groups = pd.cut(df["vict_age"].dropna(), bins=bins, labels=labels, right=False)
    age_counts = age_groups.value_counts().sort_index().reset_index()
    age_counts.columns = ["name", "count"]
    age_data = age_counts.to_dict(orient="records")
    
    # 5. Sex breakdown
    sex_breakdown = df["vict_sex"].fillna("Unknown").value_counts().reset_index()
    sex_breakdown.columns = ["name", "count"]
    sex_data = sex_breakdown.to_dict(orient="records")
    
    # 6. Crimes by Day of Week
    dow_counts = df["datetime"].dt.day_name().value_counts()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_sorted = dow_counts.reindex(day_order, fill_value=0).reset_index()
    dow_sorted.columns = ["name", "count"]
    dow_data = dow_sorted.to_dict(orient="records")
    
    # 7. Crimes by Hour of Day (from datetime)
    hour_counts = df["datetime"].dt.hour.value_counts().sort_index()
    hour_df = hour_counts.reset_index()
    hour_df.columns = ["name", "count"]
    hour_data = hour_df.to_dict(orient="records")
    
    # 8. YoY (Year over Year) Monthly comparison
    df_valid = df.dropna(subset=["datetime"])
    df_yoy = df_valid.copy()
    df_yoy["year"] = df_yoy["datetime"].dt.year
    df_yoy["month_num"] = df_yoy["datetime"].dt.month
    yoy_group = df_yoy.groupby(["year", "month_num"]).size().reset_index(name="count")
    
    # Restructure for Recharts
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    yoy_records = []
    for m in range(1, 13):
        rec = {"month": month_names[m-1]}
        for year in yoy_group["year"].unique():
            val = yoy_group[(yoy_group["year"] == year) & (yoy_group["month_num"] == m)]["count"]
            rec[str(year)] = int(val.iloc[0]) if not val.empty else 0
        yoy_records.append(rec)
        
    return {
        "monthly_trend": monthly_data,
        "top_areas": areas_data,
        "top_weapons": weapons_data,
        "age_groups": age_data,
        "sex_breakdown": sex_data,
        "day_of_week": dow_data,
        "hour_of_day": hour_data,
        "yoy_trend": yoy_records
    }

@app.post("/api/model/train")
def train_model():
    global df, similarity_engine, suspect_ranker
    if suspect_ranker is None or similarity_engine is None:
        raise HTTPException(status_code=503, detail="ML components not initialized")
        
    try:
        model_dict = suspect_ranker.train_model(num_samples=1000)
        
        # Calculate feature importances
        features = model_dict["feature_names"]
        rf_importances = model_dict["rf"].feature_importances_.tolist()
        gb_importances = model_dict["gb"].feature_importances_.tolist()
        
        importances = []
        for feat, rf_val, gb_val in zip(features, rf_importances, gb_importances):
            importances.append({
                "feature": feat,
                "rf": rf_val,
                "gb": gb_val,
                "avg": (rf_val + gb_val) / 2.0
            })
            
        importances = sorted(importances, key=lambda x: x["avg"], reverse=True)
        
        # Compute evaluation metrics
        evaluator = ModelEvaluator(df, is_relevant)
        eval_cases = df.dropna(subset=["crm_cd", "area_name", "mo_text"])["dr_no"].sample(15, random_state=42).tolist()
        
        def suspect_ranker_predict(case_id, k):
            candidates = similarity_engine.get_similar_cases(case_id, top_k=max(20, k * 3))
            return suspect_ranker.rank_suspects(case_id, candidates, top_k=k)
            
        similarity_results = evaluator.evaluate_model(similarity_engine, eval_cases, k_values=[5, 10])
        ranker_results = evaluator.evaluate_model(suspect_ranker_predict, eval_cases, k_values=[5, 10])
        
        metrics = []
        for idx in range(len(similarity_results)):
            metrics.append({
                "k": int(similarity_results["k"].iloc[idx]),
                "base_precision": float(similarity_results["precision"].iloc[idx]),
                "ensemble_precision": float(ranker_results["precision"].iloc[idx]),
                "base_recall": float(similarity_results["recall"].iloc[idx]),
                "ensemble_recall": float(ranker_results["recall"].iloc[idx]),
                "base_ndcg": float(similarity_results["ndcg"].iloc[idx]),
                "ensemble_ndcg": float(ranker_results["ndcg"].iloc[idx]),
            })
            
        return {
            "success": True,
            "importances": importances,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve React frontend (must be LAST — after all API routes) ────────────────
import pathlib
from fastapi.staticfiles import StaticFiles
import pathlib
_frontend = pathlib.Path(__file__).parent.parent.parent / "frontend_dist"
print(f"[Frontend] Looking for frontend_dist at: {_frontend}")
print(f"[Frontend] Exists: {_frontend.is_dir()}")
if _frontend.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
else:
    print("[Frontend] frontend_dist directory NOT found — frontend will not be served")
_frontend = pathlib.Path(__file__).parent.parent.parent / "frontend_dist"
if _frontend.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
