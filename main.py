from src.preprocess_cases import preprocess_cases
from src.similarity_engine import SimilarityEngine
from src.suspect_ranker import SuspectRanker
import pandas as pd

def run():
    preprocess_cases(
        "./data/raw/la_crime.csv",
        "./data/processed/clean_cases.csv"
    )

    df = pd.read_csv("./data/processed/clean_cases.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])

    engine = SimilarityEngine(df)
    test_case = df["dr_no"].iloc[0]
    similar = engine.get_similar_cases(test_case, top_k=5)

    ranker = SuspectRanker(df)
    ranking = ranker.rank_suspects(test_case, similar, top_k=5)

    print("\nSimilar Cases:\n", similar)
    print("\nSuspect Ranking:\n", ranking)

if __name__ == "__main__":
    run()
