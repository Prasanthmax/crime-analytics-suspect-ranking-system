# Crime Analytics and Suspect Ranking System

A machine learning–based system to analyze crime data, identify similar cases, and assist investigations through similarity ranking and analytics.

# Crime Analytics and Suspect Ranking System

A machine learning–based system to analyze crime data, identify similar cases, and assist investigations through similarity ranking and analytics.

## Problem Statement

Law enforcement agencies collect large volumes of crime data, but manually analyzing past cases to identify patterns and similarities is time-consuming and inefficient. Existing systems often focus on prediction accuracy rather than practical investigative support.

## Solution Overview

This project proposes a similarity-based crime analytics system that:
- Identifies similar crime cases using contextual features
- Ranks related cases to assist investigations
- Provides analytical insights through an interactive interface

The system focuses on interpretability and practical decision support rather than direct crime prediction.

## Key Features

- Crime case similarity analysis
- Suspect and case ranking based on multiple attributes
- Interactive analytics dashboard
- Efficient and interpretable machine learning approach
- Modular and scalable architecture

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization & UI:** Streamlit, Matplotlib
- **Storage:** CSV
- **Tools:** Jupyter Notebook, VS Code, Git

## Project Architecture

1. Data collection and preprocessing
2. Feature engineering for crime similarity
3. Similarity computation and ranking
4. Analytical visualization through UI

## Project Structure

crime-analytics-suspect-ranking-system/
├── app/            # Streamlit application
├── src/            # Core logic modules
├── data/           # Raw and processed datasets
├── notebooks/      # Experiments and evaluation
├── requirements.txt
└── README.md

## Installation

1. Clone the repository:
   git clone <repo-url>

2. Create and activate virtual environment:
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

## Running the Project

1. Preprocess the dataset:
   python src/preprocess_cases.py

2. Run the application:
   streamlit run app/app.py

## Evaluation and Results

The system was evaluated using ranking-based metrics such as Precision@K and NDCG, along with efficiency and consistency analysis. Results demonstrate improved similarity relevance and significantly reduced investigation time compared to manual analysis.

## Academic Context

This project was developed for academic and learning purposes and is not intended for real-world law enforcement deployment without further validation.

## Future Enhancements

- Integration with real-time databases
- Geospatial crime mapping
- Advanced learning-based ranking models
- Secure role-based access

## Author

Developed as part of an academic and learning project focused on crime analytics and machine learning–based decision support.
