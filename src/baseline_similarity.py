import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BaselineSimilarityEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.df["text"] = (
            self.df["crm_cd_desc"].astype(str).str.lower()
            + " "
            + self.df["weapon_desc"].astype(str).str.lower()
        )

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["text"])

    def get_similar_cases(self, dr_no, top_k=10):
        idx_list = self.df.index[self.df["dr_no"] == dr_no].tolist()
        if not idx_list:
            return pd.DataFrame()

        idx = idx_list[0]

        similarity_scores = cosine_similarity(
            self.tfidf_matrix.getrow(idx),
            self.tfidf_matrix
        ).flatten()

        sorted_idx = np.argsort(similarity_scores)[::-1]
        sorted_idx = sorted_idx[sorted_idx != idx]

        result = self.df.iloc[sorted_idx[:top_k]].copy()
        result["similarity_score"] = similarity_scores[sorted_idx[:top_k]]

        return result
