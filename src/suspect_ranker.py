import numpy as np
import pandas as pd


class SuspectRanker:
    """
    Ranks candidate cases as likely suspects for a base case.
    Combines:
      - text similarity (already in 'similarity' column)
      - area match
      - weapon match
      - time proximity
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])

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
        base_time = pd.to_datetime(base_row["datetime"])

        def _score(row):
            s_text = float(row.get("similarity", 0.0))

            s_area = 1.0 if row["area_name"] == base_row["area_name"] else 0.0
            s_weapon = 1.0 if row["weapon_desc"] == base_row["weapon_desc"] else 0.0

            dt = pd.to_datetime(row["datetime"])
            days = abs((dt - base_time).days)
            s_time = float(np.exp(-days / 365.0))  # decay over ~1 year

            return 0.5 * s_text + 0.2 * s_area + 0.1 * s_weapon + 0.2 * s_time

        ranked = candidates.copy()
        ranked["score"] = ranked.apply(_score, axis=1)
        ranked = ranked.sort_values("score", ascending=False)

        return ranked.head(top_k)
