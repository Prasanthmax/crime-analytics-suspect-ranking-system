import os
import sys
import pathlib
import urllib.request

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Path setup ────────────────────────────────────────────────────────────────
# Works for both local dev (app/backend/main.py) and Docker (/app/app/backend/main.py)
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.similarity_engine import SimilarityEngine
from src.suspect_ranker import SuspectRanker
from src.model_evaluator import ModelEvaluator, is_relevant

DATA_PATH  = ROOT / "data" / "processed" / "clean_cases.csv.gz"
MODEL_PATH = ROOT / "models" / "ensemble_ranker.joblib"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Crime Analytics & Suspect Ranking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals ───────────────────────────────────────────────────────────────────
df: pd.DataFrame | None = None
similarity_engine: SimilarityEngine | None = None
suspect_ranker: SuspectRanker | None = None


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    global df, similarity_engine, suspect_ranker

    # Download dataset if missing (fallback for Render/local without data)
    if not DATA_PATH.exists():
        data_url = os.getenv("DATA_URL")
        if not data_url:
            raise FileNotFoundError(
                f"Dataset not found at {DATA_PATH}. "
                "Set the DATA_URL environment variable to a direct download link."
            )
        print(f"[SERVER] Downloading dataset from {data_url}...")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(data_url, str(DATA_PATH))
        print("[SERVER] Download complete.")

    print("[SERVER] Loading dataset...")
    df = pd.read_csv(str(DATA_PATH))
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"[SERVER] Dataset loaded: {len(df):,} rows.")

    similarity_engine = SimilarityEngine(df)
    suspect_ranker = SuspectRanker(df, model_path=str(MODEL_PATH))
    suspect_ranker.load_model()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/metrics")
def get_metrics():
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    date_min = df["datetime"].min()
    date_max = df["datetime"].max()
    span_days = int((date_max - date_min).days) if pd.notna(date_min) and pd.notna(date_max) else 0

    by_area = df["area_name"].value_counts().head(5).reset_index()
    by_area.columns = ["area", "count"]

    by_weapon = df["weapon_desc"].fillna("UNKNOWN").value_counts().head(5).reset_index()
    by_weapon.columns = ["weapon", "count"]

    id_theft_sample = (
        df[df["crm_cd_desc"] == "THEFT OF IDENTITY"]
        .sort_values("datetime", ascending=False)
        .head(10)
    )
    id_theft_list = [
        {
            "dr_no": int(row["dr_no"]),
            "datetime": str(row["datetime"].date()) if pd.notna(row["datetime"]) else "N/A",
            "area_name": str(row.get("area_name", "N/A")),
            "weapon_desc": str(row.get("weapon_desc", "N/A")),
            "vict_age": int(row["vict_age"]) if pd.notna(row["vict_age"]) else "N/A",
            "vict_sex": str(row.get("vict_sex", "N/A")),
        }
        for _, row in id_theft_sample.iterrows()
    ]

    return {
        "summary": {
            "total_cases": len(df),
            "distinct_areas": int(df["area_name"].nunique()),
            "crime_types": int(df["crm_cd_desc"].nunique()),
            "time_span_days": span_days,
            "min_date": str(date_min.date()) if pd.notna(date_min) else None,
            "max_date": str(date_max.date()) if pd.notna(date_max) else None,
        },
        "top_areas": by_area.to_dict(orient="records"),
        "top_weapons": by_weapon.to_dict(orient="records"),
        "recent_id_theft": id_theft_list,
    }


@app.get("/api/cases/search/lookup")
def search_cases(q: str = ""):
    if not q:
        return []
    matches = df[df["dr_no"].astype(str).str.contains(q)].head(10)
    return [int(cid) for cid in matches["dr_no"].tolist()]


@app.get("/api/cases/{case_id}")
def get_case(case_id: int):
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
        "mo_text": str(case.get("mo_text", "N/A")),
    }


@app.get("/api/similarity")
def get_similar(case_id: int, top_k: int = 10, city: str = "All", weapon: str = "All"):
    if similarity_engine is None:
        raise HTTPException(status_code=503, detail="Similarity engine not initialized")

    similar = similarity_engine.get_similar_cases(
        case_id,
        top_k=top_k,
        city=None if city == "All" else city,
        weapon=None if weapon == "All" else weapon,
    )
    if similar.empty:
        return []

    return [
        {
            "dr_no": int(row["dr_no"]),
            "datetime": str(row["datetime"].date()) if pd.notna(row["datetime"]) else "N/A",
            "area_name": str(row.get("area_name", "N/A")),
            "crm_cd_desc": str(row.get("crm_cd_desc", "N/A")),
            "weapon_desc": str(row.get("weapon_desc", "N/A")),
            "vict_age": int(row["vict_age"]) if pd.notna(row["vict_age"]) else None,
            "vict_sex": str(row.get("vict_sex", "N/A")),
            "similarity": float(row["similarity"]),
            "mo_text": str(row.get("mo_text", "N/A")),
        }
        for _, row in similar.iterrows()
    ]


@app.get("/api/ranking")
def get_ranking(case_id: int, top_k: int = 10):
    if similarity_engine is None or suspect_ranker is None:
        raise HTTPException(status_code=503, detail="ML components not initialized")

    base_row = df[df["dr_no"] == case_id]
    if base_row.empty:
        raise HTTPException(status_code=404, detail="Base Case ID not found")
    base_case = base_row.iloc[0]

    pool = similarity_engine.get_similar_cases(case_id, top_k=max(30, top_k * 3))
    if pool.empty:
        return []

    ranked = suspect_ranker.rank_suspects(case_id, pool, top_k=top_k)
    features_df = suspect_ranker.extract_features(base_case, ranked)

    return [
        {
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
            "features": {
                "area_match": bool(features_df.loc[idx]["area_match"]),
                "weapon_match": bool(features_df.loc[idx]["weapon_match"]),
                "crime_code_match": bool(features_df.loc[idx]["crime_code_match"]),
                "time_decay": float(features_df.loc[idx]["time_decay"]),
                "sex_match": bool(features_df.loc[idx]["sex_match"]),
                "age_difference": float(features_df.loc[idx]["age_difference"]),
            },
        }
        for idx, row in ranked.iterrows()
    ]


@app.get("/api/analytics")
def get_analytics():
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    monthly = df.set_index("datetime").resample("ME").size().reset_index(name="count")
    monthly["month"] = monthly["datetime"].dt.strftime("%Y-%m")

    top_areas = df["area_name"].value_counts().head(10).reset_index()
    top_areas.columns = ["name", "count"]

    top_weapons = df["weapon_desc"].fillna("UNKNOWN").value_counts().head(10).reset_index()
    top_weapons.columns = ["name", "count"]

    bins   = [0, 18, 30, 45, 60, 80, 120]
    labels = ["0-17", "18-29", "30-44", "45-59", "60-79", "80+"]
    age_groups = pd.cut(df["vict_age"].dropna(), bins=bins, labels=labels, right=False)
    age_counts = age_groups.value_counts().sort_index().reset_index()
    age_counts.columns = ["name", "count"]

    sex_breakdown = df["vict_sex"].fillna("Unknown").value_counts().reset_index()
    sex_breakdown.columns = ["name", "count"]

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_counts = df["datetime"].dt.day_name().value_counts().reindex(day_order, fill_value=0).reset_index()
    dow_counts.columns = ["name", "count"]

    hour_counts = df["datetime"].dt.hour.value_counts().sort_index().reset_index()
    hour_counts.columns = ["name", "count"]

    df_yoy = df.dropna(subset=["datetime"]).copy()
    df_yoy["year"]      = df_yoy["datetime"].dt.year
    df_yoy["month_num"] = df_yoy["datetime"].dt.month
    yoy_group = df_yoy.groupby(["year", "month_num"]).size().reset_index(name="count")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    yoy_records = []
    for m in range(1, 13):
        rec = {"month": month_names[m - 1]}
        for year in yoy_group["year"].unique():
            val = yoy_group[(yoy_group["year"] == year) & (yoy_group["month_num"] == m)]["count"]
            rec[str(year)] = int(val.iloc[0]) if not val.empty else 0
        yoy_records.append(rec)

    return {
        "monthly_trend": monthly[["month", "count"]].to_dict(orient="records"),
        "top_areas":     top_areas.to_dict(orient="records"),
        "top_weapons":   top_weapons.to_dict(orient="records"),
        "age_groups":    age_counts.to_dict(orient="records"),
        "sex_breakdown": sex_breakdown.to_dict(orient="records"),
        "day_of_week":   dow_counts.to_dict(orient="records"),
        "hour_of_day":   hour_counts.to_dict(orient="records"),
        "yoy_trend":     yoy_records,
    }


@app.post("/api/model/train")
def train_model():
    if suspect_ranker is None or similarity_engine is None:
        raise HTTPException(status_code=503, detail="ML components not initialized")

    try:
        model_dict = suspect_ranker.train_model(num_samples=1000)

        features        = model_dict["feature_names"]
        rf_importances  = model_dict["rf"].feature_importances_.tolist()
        gb_importances  = model_dict["gb"].feature_importances_.tolist()
        importances = sorted(
            [
                {"feature": f, "rf": rf, "gb": gb, "avg": (rf + gb) / 2.0}
                for f, rf, gb in zip(features, rf_importances, gb_importances)
            ],
            key=lambda x: x["avg"],
            reverse=True,
        )

        evaluator  = ModelEvaluator(df, is_relevant)
        eval_cases = (
            df.dropna(subset=["crm_cd", "area_name", "mo_text"])["dr_no"]
            .sample(15, random_state=42)
            .tolist()
        )

        def ranker_predict(case_id, k):
            candidates = similarity_engine.get_similar_cases(case_id, top_k=max(20, k * 3))
            return suspect_ranker.rank_suspects(case_id, candidates, top_k=k)

        sim_results    = evaluator.evaluate_model(similarity_engine, eval_cases, k_values=[5, 10])
        ranker_results = evaluator.evaluate_model(ranker_predict,    eval_cases, k_values=[5, 10])

        metrics = [
            {
                "k":                  int(sim_results["k"].iloc[i]),
                "base_precision":     float(sim_results["precision"].iloc[i]),
                "ensemble_precision": float(ranker_results["precision"].iloc[i]),
                "base_recall":        float(sim_results["recall"].iloc[i]),
                "ensemble_recall":    float(ranker_results["recall"].iloc[i]),
                "base_ndcg":          float(sim_results["ndcg"].iloc[i]),
                "ensemble_ndcg":      float(ranker_results["ndcg"].iloc[i]),
            }
            for i in range(len(sim_results))
        ]

        return {"success": True, "importances": importances, "metrics": metrics}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve React frontend ──────────────────────────────────────────────────────
# Must be LAST — after all API routes.
# Uses FRONTEND_DIST env var (set in Dockerfile) with fallback to relative path.
_frontend = pathlib.Path(os.environ.get("FRONTEND_DIST", str(ROOT / "frontend_dist")))
print(f"[Frontend] Looking for frontend at: {_frontend} — exists: {_frontend.is_dir()}")
if _frontend.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
else:
    print("[Frontend] NOT FOUND — API-only mode (frontend not served)")