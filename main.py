from src.preprocess_cases import preprocess_cases
from src.similarity_engine import HybridSimilarityEngine
from src.suspect_ranker import SuspectRanker
import pandas as pd

def run():
    preprocess_cases(
        "./data/raw/crime_dataset_india.csv",
        "./data/processed/clean_cases.csv"
    )

    df = pd.read_csv("./data/processed/clean_cases.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])

    engine = HybridSimilarityEngine(df)
    test_case = df["case_id"].iloc[0]
    similar = engine.get_similar_cases(test_case, top_k=5)

    ranker = SuspectRanker(df)
    ranking = ranker.rank_suspects(test_case, top_k=5)

    print("\nSimilar Cases:\n", similar)
    print("\nSuspect Ranking:\n", ranking)

if __name__ == "__main__":
    run()
