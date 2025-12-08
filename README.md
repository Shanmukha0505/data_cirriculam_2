# data_cirriculam_2

# Hybrid Recommendation Engine for Personalized E-Commerce Product Suggestions

The project's objective is to create a hybrid recommendation engine that offers tailored e-commerce recommendations by combining collaborative filtering, content-based techniques, and session context. The project will make use of publicly available datasets from Stanford SNAP and the Amazon Product Advertising API in addition to data that has been scraped from test e-commerce websites. In order to better predict user preferences, the main goal is to develop a machine learning model that integrates various data sources and performs better than conventional recommenders. An interactive Streamlit application showcasing the recommender's capabilities will be the end result.

---

## Project Objectives

- Build a full end-to-end recommendation pipeline:
  - Data collection, cleaning, integration
  - Feature engineering (embeddings, TF-IDF, user–item matrices)
  - Baseline and hybrid models
  - Evaluation and interpretation
- Compare traditional recommenders against hybrid deep learning approaches.
- Provide an interactive demo that:
  - Simulates a user’s browsing history
  - Shows the top-N recommendations
  - Includes short natural-language explanations for each recommendation.

---

## Key Features

- **Multiple model families**
  - Baseline collaborative filtering: SVD, K-NN
  - Hybrid models: LightFM, Neural Collaborative Filtering 
- **Hybrid signal fusion**
  - Interaction data (clicks/ratings/implicit feedback)
  - Content features (TF-IDF from product titles/descriptions)
  - Session-based context (recently viewed items)
- **Offline evaluation**
  - Ranking metrics such as **NDCG@10** to compare models
  - Ablation studies to measure the impact of content features
- **Streamlit demo**
  - Input a mock browsing history (list of product IDs)
  - View top-10 personalized recommendations
  - See simple textual explanations for each suggestion

---

## Repository Structure

```text
.
├── week1.ipynb         # Environment setup, data download, basic exploration
├── week2.ipynb         # Data cleaning, integration, initial EDA
├── week3&4.ipynb       # Feature engineering, embeddings, TF-IDF
├── week5.ipynb         # Baseline models (SVD, K-NN) + initial hybrid model work
├── week6.ipynb         # LightFM and NCF model training & tuning
├── week7&8.ipynb       # Ablation studies, interpretation, final analysis, demo wiring
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
