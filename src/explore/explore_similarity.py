# %% [markdown]
# # Interactive Crime Similarity Exploration
# This script serves as a version-control-friendly alternative to Jupyter Notebooks.
# You can execute the cells below by clicking "Run Cell" in VS Code (with the Python extension).

# %%
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.similarity_engine import SimilarityEngine
from src.suspect_ranker import SuspectRanker

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %% [markdown]
# ## 1. Load Cleaned Crime Dataset
# We load the dataset preprocessed from raw LA crime data.

# %%
data_path = "../../data/processed/clean_cases.csv.gz"
if not os.path.exists(data_path):
    # Try alternate path just in case
    data_path = "../data/processed/clean_cases.csv.gz"

df = pd.read_csv(data_path)
df["datetime"] = pd.to_datetime(df["datetime"])
print(f"Loaded {len(df):,} cases.")
print(df.head(3))

# %% [markdown]
# ## 2. Explore Core Distributions
# Let's visualize the distribution of crime categories.

# %%
top_crimes = df["crm_cd_desc"].value_counts().head(10)
sns.barplot(x=top_crimes.values, y=top_crimes.index, palette="viridis")
plt.title("Top 10 Crime Types in Los Angeles")
plt.xlabel("Number of Cases")
plt.ylabel("Crime Category")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Query Similar Cases (TF-IDF Similarity Engine)
# Initialize the engine and look up similar modus-operandi text matching for a case.

# %%
engine = SimilarityEngine(df)
test_case_id = df["dr_no"].iloc[0]
print(f"Querying similar cases for Case ID: {test_case_id}")

similar_cases = engine.get_similar_cases(test_case_id, top_k=10)
print(similar_cases[["dr_no", "area_name", "crm_cd_desc", "similarity"]])

# %% [markdown]
# ## 4. Run Suspect Ranking (Ensemble Model)
# Evaluate the ensemble scoring model. If no cached model exists on disk, it trains dynamically.

# %%
ranker = SuspectRanker(df, model_path="../../models/ensemble_ranker.joblib")
ranked_suspects = ranker.rank_suspects(test_case_id, similar_cases, top_k=5)

print("\nTop 5 Ranked Suspects:")
print(ranked_suspects[["dr_no", "area_name", "crm_cd_desc", "score"]])

# %% [markdown]
# ## 5. Visualize Similarity vs Ensemble Score

# %%
plt.figure()
sns.scatterplot(data=ranked_suspects, x="similarity", y="score", hue="area_name", s=100)
plt.title("Similarity Score vs. Ensemble Suspect Ranking Score")
plt.xlabel("Text Modus Operandi Similarity")
plt.ylabel("Ensemble Scoring Probability")
plt.legend(title="Area")
plt.show()
