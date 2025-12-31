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

        sims = cosine_similarity(self.tfidf[base_idx], self.tfidf)[0]
        result = self.df.copy()
        result["similarity"] = sims

        # remove the base case itself
        result = result[result["dr_no"] != case_id]

        if city and city != "All":
            result = result[result["area_name"] == city]

        if weapon and weapon != "All":
            result = result[result["weapon_desc"] == weapon]

        result = result.sort_values("similarity", ascending=False)
        return result.head(top_k)
