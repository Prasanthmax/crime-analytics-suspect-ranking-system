import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os


class SuspectRanker:
    """
    Ranks candidate cases as likely suspects for a base case.
    Uses an Ensemble Machine Learning method (Random Forest + Gradient Boosting)
    trained on case-to-case similarity features:
      - text similarity (cosine similarity of MO text)
      - area match
      - weapon match
      - crime code match
      - time proximity (exponential decay)
      - victim sex match
      - victim age difference
    """

    def __init__(self, df: pd.DataFrame, model_path: str = "models/ensemble_ranker.joblib"):
        self.df = df.copy()
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        self.model_path = model_path
        self.model = None

    def extract_features(self, base_case: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Extract pairwise features between base_case and candidates.
        """
        features = pd.DataFrame(index=candidates.index)

        # 1. Text Similarity (already in candidates as 'similarity')
        features["similarity"] = candidates["similarity"].astype(float).fillna(0.0)

        # 2. Area Match (Categorical Match)
        features["area_match"] = (candidates["area_name"] == base_case["area_name"]).astype(int)

        # 3. Weapon Match (Categorical Match)
        features["weapon_match"] = (candidates["weapon_desc"] == base_case["weapon_desc"]).astype(int)

        # 4. Crime Code Match
        features["crime_code_match"] = (candidates["crm_cd"] == base_case["crm_cd"]).astype(int)

        # 5. Time decay based on days difference
        base_time = pd.to_datetime(base_case["datetime"])
        candidate_times = pd.to_datetime(candidates["datetime"])
        days_diff = (candidate_times - base_time).dt.days.abs()
        features["time_decay"] = np.exp(-days_diff / 365.0).astype(float)

        # 6. Victim Sex Match
        features["sex_match"] = (candidates["vict_sex"] == base_case["vict_sex"]).astype(int)

        # 7. Victim Age Difference
        base_age = float(base_case["vict_age"]) if pd.notna(base_case["vict_age"]) else 40.0
        candidate_ages = candidates["vict_age"].fillna(40.0).astype(float)
        features["age_difference"] = (candidate_ages - base_age).abs()

        return features

    def train_model(self, num_samples: int = 500) -> dict:
        """
        Dynamically sample historical cases, search for similar candidate pairs,
        compute target labels using is_relevant, and train an ensemble model.
        Saves the resulting model to models/ensemble_ranker.joblib.
        """
        print("[ML] Starting training of Ensemble Suspect Ranker...")
        from src.similarity_engine import SimilarityEngine
        from src.model_evaluator import is_relevant

        # Filter valid records for base cases sample
        valid_df = self.df.dropna(subset=["crm_cd", "area_name", "mo_text"])
        sample_size = min(num_samples, len(valid_df))
        sample_cases = valid_df.sample(sample_size, random_state=42)

        engine = SimilarityEngine(self.df)

        X_list = []
        y_list = []

        print(f"[ML] Generating training pairs from {sample_size} cases...")
        for _, base_case in sample_cases.iterrows():
            # Get similarity pool of top 15 cases (excluding base case itself)
            pool = engine.get_similar_cases(base_case["dr_no"], top_k=15)
            if pool.empty:
                continue

            # Compute features
            feats = self.extract_features(base_case, pool)
            
            # Compute relevance targets
            labels = [1 if is_relevant(base_case, row) else 0 for _, row in pool.iterrows()]

            X_list.append(feats)
            y_list.extend(labels)

        if not X_list:
            raise ValueError("Failed to generate training data pairs. Check your datasets.")

        X = pd.concat(X_list, ignore_index=True)
        y = np.array(y_list)

        pos_count = int(np.sum(y))
        print(f"[ML] Training features shape: {X.shape}. Positive labels: {pos_count} ({pos_count/len(y)*100:.1f}%)")

        # Train Ensemble: Random Forest + Gradient Boosting
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

        rf.fit(X, y)
        gb.fit(X, y)

        model_dict = {
            "rf": rf,
            "gb": gb,
            "feature_names": X.columns.tolist()
        }

        # Save to disk
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model_dict, self.model_path)
        print(f"[ML] Ensemble model successfully cached at {self.model_path}")
        
        self.model = model_dict
        return model_dict

    def load_model(self):
        """
        Loads ensemble model from models/ensemble_ranker.joblib.
        If it doesn't exist, fits and saves it first.
        """
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"[ML] Successfully loaded cached ensemble model from {self.model_path}")
            except Exception as e:
                print(f"[ML] Warning: Failed to load model ({e}). Retraining...")
                self.train_model()
        else:
            self.train_model()

    def rank_suspects(
        self,
        base_case_id,
        candidates: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        if candidates.empty:
            return pd.DataFrame()

        base_row = self.df[self.df["dr_no"] == base_case_id]
        if base_row.empty:
            return pd.DataFrame()

        base_row = base_row.iloc[0]

        # Load models
        if self.model is None:
            self.load_model()

        # Extract features between base case and candidates
        X_eval = self.extract_features(base_row, candidates)

        # Keep columns matched with model training
        feature_names = self.model["feature_names"]
        X_eval = X_eval[feature_names]

        # Predict probability of being relevant
        prob_rf = self.model["rf"].predict_proba(X_eval)[:, 1]
        prob_gb = self.model["gb"].predict_proba(X_eval)[:, 1]

        # Ensemble average score
        scores = (prob_rf + prob_gb) / 2.0

        ranked = candidates.copy()
        ranked["score"] = scores
        ranked = ranked.sort_values("score", ascending=False)

        return ranked.head(top_k)

