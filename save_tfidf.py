"""
Run this ONCE locally before pushing to HuggingFace.
It pre-builds the TF-IDF matrix so the server doesn't rebuild it on every startup.

Usage (from repo root):
    python scripts/save_tfidf.py
"""

import os
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH   = "data/processed/clean_cases.csv.gz"
MODELS_DIR  = "models"
VEC_PATH    = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
MAT_PATH    = os.path.join(MODELS_DIR, "tfidf_matrix.npz")

print(f"Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df["mo_text"] = df["mo_text"].fillna("").astype(str)
print(f"Loaded {len(df):,} rows.")

print("Building TF-IDF matrix (this may take a minute)...")
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["mo_text"])
print(f"Matrix shape: {tfidf_matrix.shape}, stored elements: {tfidf_matrix.nnz:,}")

os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(vectorizer, VEC_PATH)
sp.save_npz(MAT_PATH, tfidf_matrix)

vec_mb  = os.path.getsize(VEC_PATH)  / 1024 / 1024
mat_mb  = os.path.getsize(MAT_PATH)  / 1024 / 1024
print(f"\nSaved:")
print(f"  {VEC_PATH}  ({vec_mb:.1f} MB)")
print(f"  {MAT_PATH}  ({mat_mb:.1f} MB)")
print("\nNow commit both files and push to HuggingFace.")
