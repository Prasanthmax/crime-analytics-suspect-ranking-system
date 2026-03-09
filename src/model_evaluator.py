"""
Model Evaluation Utilities for Crime Analysis System
"""

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from typing import Callable, List, Tuple

class ModelEvaluator:
    """Comprehensive evaluation metrics for similarity models"""
    
    def __init__(self, df, relevance_func: Callable):
        """
        Args:
            df: DataFrame with cases
            relevance_func: Function to determine if two cases are relevant
        """
        self.df = df
        self.relevance_func = relevance_func
    
    def precision_at_k(self, engine, case_id, k=10):
        """Calculate Precision@K"""
        base = self.df[self.df["dr_no"] == case_id].iloc[0]
        retrieved = engine.get_similar_cases(case_id, top_k=k)
        
        if retrieved.empty:
            return 0.0
        
        relevant = sum(
            self.relevance_func(base, row)
            for _, row in retrieved.iterrows()
        )
        
        return relevant / k
    
    def recall_at_k(self, engine, case_id, k=10):
        """
        Calculate Recall@K using an efficient approximation.
        Instead of checking all cases (which is very slow), we check a sample
        and estimate based on retrieved results.
        """
        base = self.df[self.df["dr_no"] == case_id].iloc[0]
        retrieved = engine.get_similar_cases(case_id, top_k=k)
        
        if retrieved.empty:
            return 0.0
        
        # Count relevant cases in retrieved set
        relevant_retrieved = sum(
            self.relevance_func(base, row)
            for _, row in retrieved.iterrows()
        )
        
        # For efficiency, estimate total relevant cases based on:
        # 1. Same crime code (most important factor)
        # 2. Approximate by counting cases with same crime code
        same_crime_count = len(self.df[self.df["crm_cd"] == base["crm_cd"]])
        
        # Estimate: assume about 30% of same-crime cases are truly relevant
        estimated_relevant = max(same_crime_count * 0.3, relevant_retrieved)
        
        return relevant_retrieved / estimated_relevant if estimated_relevant > 0 else 0.0
    
    def f1_score_at_k(self, engine, case_id, k=10):
        """Calculate F1-Score@K"""
        precision = self.precision_at_k(engine, case_id, k)
        recall = self.recall_at_k(engine, case_id, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self, engine, case_id, k=10):
        """Calculate NDCG@K"""
        base = self.df[self.df["dr_no"] == case_id].iloc[0]
        retrieved = engine.get_similar_cases(case_id, top_k=k)
        
        if retrieved.empty:
            return 0.0
        
        relevance = [
            1 if self.relevance_func(base, row) else 0
            for _, row in retrieved.iterrows()
        ]
        
        dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def mean_average_precision(self, engine, case_ids, k=10):
        """Calculate MAP (Mean Average Precision)"""
        aps = []
        
        for case_id in case_ids:
            base = self.df[self.df["dr_no"] == case_id].iloc[0]
            retrieved = engine.get_similar_cases(case_id, top_k=k)
            
            if retrieved.empty:
                aps.append(0.0)
                continue
            
            num_relevant = 0
            sum_precisions = 0.0
            
            for i, (_, row) in enumerate(retrieved.iterrows(), 1):
                if self.relevance_func(base, row):
                    num_relevant += 1
                    sum_precisions += num_relevant / i
            
            ap = sum_precisions / num_relevant if num_relevant > 0 else 0.0
            aps.append(ap)
        
        return np.mean(aps)
    
    def evaluate_model(self, engine, case_ids, k_values=[1, 3, 5, 10, 15, 20]):
        """Comprehensive evaluation across multiple K values"""
        results = {
            'k': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'ndcg': []
        }
        
        for k in k_values:
            precisions = [self.precision_at_k(engine, cid, k) for cid in case_ids]
            recalls = [self.recall_at_k(engine, cid, k) for cid in case_ids]
            f1s = [self.f1_score_at_k(engine, cid, k) for cid in case_ids]
            ndcgs = [self.ndcg_at_k(engine, cid, k) for cid in case_ids]
            
            results['k'].append(k)
            results['precision'].append(np.mean(precisions))
            results['recall'].append(np.mean(recalls))
            results['f1'].append(np.mean(f1s))
            results['ndcg'].append(np.mean(ndcgs))
        
        return pd.DataFrame(results)


def is_relevant(base_case, candidate_case):
    """
    Determine if a candidate case is relevant to the base case.
    
    A case is considered relevant if it matches on at least 2 of these criteria:
    - Same crime code (crm_cd)
    - Same area
    - Same weapon type
    - Similar MO text
    """
    score = 0
    
    if pd.notna(base_case["crm_cd"]) and pd.notna(candidate_case["crm_cd"]):
        if base_case["crm_cd"] == candidate_case["crm_cd"]:
            score += 1
    
    if pd.notna(base_case["area_name"]) and pd.notna(candidate_case["area_name"]):
        if base_case["area_name"] == candidate_case["area_name"]:
            score += 1
    
    if pd.notna(base_case["weapon_desc"]) and pd.notna(candidate_case["weapon_desc"]):
        if base_case["weapon_desc"] == candidate_case["weapon_desc"]:
            score += 1
    
    if pd.notna(base_case["mo_text"]) and pd.notna(candidate_case["mo_text"]):
        if base_case["mo_text"] == candidate_case["mo_text"]:
            score += 1
    
    return score >= 2
