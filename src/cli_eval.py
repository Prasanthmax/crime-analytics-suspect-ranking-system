import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.similarity_engine import SimilarityEngine
from src.suspect_ranker import SuspectRanker
from src.model_evaluator import ModelEvaluator, is_relevant

def run_evaluation():
    print("=" * 60)
    print("Crime Analytics & Suspect Ranking System - Model Evaluation")
    print("=" * 60)
    
    # 1. Load Data
    data_path = "./data/processed/clean_cases.csv.gz"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run preprocessing first.")
        return
        
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded. Total rows: {len(df):,}")
    
    # 2. Instantiate engines
    print("\nInitializing Similarity Engine...")
    similarity_engine = SimilarityEngine(df)
    
    print("Initializing Suspect Ranker...")
    ranker = SuspectRanker(df)
    
    # Force train the model to output training metrics
    print("\nFitting/Retraining the Ensemble Model...")
    model_dict = ranker.train_model(num_samples=1000) # train on 1000 samples
    
    # 3. Print feature importances
    print("\n" + "-" * 45)
    print("FEATURE IMPORTANCES (Ensemble Model)")
    print("-" * 45)
    features = model_dict["feature_names"]
    rf_importances = model_dict["rf"].feature_importances_
    gb_importances = model_dict["gb"].feature_importances_
    
    importance_df = pd.DataFrame({
        "Feature": features,
        "Random Forest": rf_importances,
        "Gradient Boosting": gb_importances,
        "Average": (rf_importances + gb_importances) / 2.0
    }).sort_values("Average", ascending=False)
    
    print(importance_df.to_string(index=False, formatters={
        "Random Forest": "{:.4f}".format,
        "Gradient Boosting": "{:.4f}".format,
        "Average": "{:.4f}".format
    }))
    
    # 4. Select evaluation cases
    print("\nSelecting sample cases for evaluation...")
    # Find cases that have valid crime codes and areas to get high-quality evaluations
    eval_cases = df.dropna(subset=["crm_cd", "area_name", "mo_text"])["dr_no"].sample(30, random_state=42).tolist()
    print(f"Selected {len(eval_cases)} sample cases for comparative evaluation.")
    
    # 5. Evaluate models
    evaluator = ModelEvaluator(df, is_relevant)
    
    print("\nEvaluating Baseline Similarity Search...")
    similarity_results = evaluator.evaluate_model(
        similarity_engine,
        eval_cases,
        k_values=[1, 3, 5, 10]
    )
    
    print("\nEvaluating Ensemble Suspect Ranker...")
    # SuspectRanker needs to be run on candidate pools from SimilarityEngine.
    # We define a callable wrapper:
    def suspect_ranker_predict(case_id, k):
        # We fetch max(20, k * 3) similar cases from the SimilarityEngine as the pool
        candidates = similarity_engine.get_similar_cases(case_id, top_k=max(20, k * 3))
        return ranker.rank_suspects(case_id, candidates, top_k=k)
        
    ranker_results = evaluator.evaluate_model(
        suspect_ranker_predict,
        eval_cases,
        k_values=[1, 3, 5, 10]
    )
    
    # 6. Compare results
    print("\n" + "=" * 60)
    print("COMPARATIVE EVALUATION SUMMARY")
    print("=" * 60)
    
    comparison = pd.DataFrame({
        "K": similarity_results["k"],
        "Base Precision": similarity_results["precision"],
        "Ensemble Precision": ranker_results["precision"],
        "Base Recall": similarity_results["recall"],
        "Ensemble Recall": ranker_results["recall"],
        "Base NDCG": similarity_results["ndcg"],
        "Ensemble NDCG": ranker_results["ndcg"]
    })
    
    print(comparison.to_string(index=False, formatters={
        "Base Precision": "{:.4f}".format,
        "Ensemble Precision": "{:.4f}".format,
        "Base Recall": "{:.4f}".format,
        "Ensemble Recall": "{:.4f}".format,
        "Base NDCG": "{:.4f}".format,
        "Ensemble NDCG": "{:.4f}".format
    }))
    
    print("\nModel evaluation complete!")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
