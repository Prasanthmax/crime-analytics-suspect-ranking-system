import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:
    """
    Text-based similarity on mo_text with optional city/weapon filters.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        self.df["mo_text"] = self.df["mo_text"].fillna("").astype(str)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf = self.vectorizer.fit_transform(self.df["mo_text"])

    def _idx_for_case(self, case_id):
        idx_list = self.df.index[self.df["dr_no"] == case_id].tolist()
        return idx_list[0] if idx_list else None

    def get_similar_cases(
        self,
        case_id,
        top_k: int = 20,
        city: str | None = None,
        weapon: str | None = None,
    ) -> pd.DataFrame:
        base_idx = self._idx_for_case(case_id)
        if base_idx is None:
            return pd.DataFrame()

        # Compute similarities
        sims = cosine_similarity(self.tfidf[base_idx], self.tfidf)[0]

        # Optimize: Instead of copying the full 1M row DataFrame, find the top candidates first.
        # We take a slightly larger pool (e.g., top_k * 10 or at least 200) to account for city/weapon filtering.
        pool_size = max(200, top_k * 10)
        if pool_size >= len(sims):
            top_indices = np.arange(len(sims))
        else:
            # Get partition of top indices
            top_indices = np.argpartition(sims, -pool_size)[-pool_size:]
            # Sort them in descending order of similarity
            top_indices = top_indices[np.argsort(-sims[top_indices])]

        # Extract only the top candidates from df
        result = self.df.iloc[top_indices].copy()
        result["similarity"] = sims[top_indices]

        # remove the base case itself
        result = result[result["dr_no"] != case_id]

        if city and city != "All":
            result = result[result["area_name"] == city]

        if weapon and weapon != "All":
            result = result[result["weapon_desc"] == weapon]

        result = result.sort_values("similarity", ascending=False)
        return result.head(top_k)
