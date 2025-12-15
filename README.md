
# Hybrid Recommendation Engine for Personalized E-Commerce

This repository documents my end-to-end project to build a sophisticated **hybrid recommendation engine**. My goal was to move beyond traditional recommenders by fusing collaborative, content-based, and user-behavior signals to deliver highly personalized and context-aware product suggestions.

The project covers the entire machine learning lifecycle, from acquiring and integrating multiple complex datasets to advanced feature engineering, model tuning, and evaluation. The final result is a powerful recommendation API.

---

## 🚀 Project Summary

I developed a multi-stage recommendation pipeline that successfully integrates user interaction data, product metadata, and behavioral features to predict user preferences. By systematically comparing baseline models against advanced hybrid techniques, I identified a **feature-based Gradient Boosting model** as the top performer.

After rigorous hyperparameter tuning, this model achieved an **RMSE of 0.5782**, marking a **58% improvement** in rating prediction accuracy over a traditional SVD collaborative filtering baseline. The project culminates in a `RecommendationEngine` class that simulates a production-ready API to showcase its real-world application.

### The Workflow at a Glance

1.  **Data Acquisition & Synthesis:** I started by fetching product data from the FakeStoreAPI for initial prototyping and downloaded a large-scale Amazon Electronics dataset from Stanford SNAP for the core model. To create a realistic environment, I also generated synthetic user interaction data, including views, clicks, and purchases.
2.  **Data Cleaning & Integration:** A significant effort went into cleaning the raw Amazon data. This involved parsing millions of gzipped JSON records, handling complex nested structures, cleaning HTML entities from text fields, and merging disparate review and metadata files into a cohesive dataset.
3.  **Advanced Feature Engineering:** To capture diverse signals beyond simple ratings, I engineered a rich feature set:
    *   **Temporal Features:** Weighted recent interactions more heavily using a time-decay function.
    *   **Implicit Ratings:** Engineered a composite score from explicit ratings, review helpfulness, and review length.
    *   **Behavioral Aggregates:** Calculated user- and product-level statistics (e.g., average ratings, review counts).
    *   **Content Embeddings:** Generated TF-IDF vectors from product titles and descriptions.
4.  **Modeling & Experimentation:** I built and evaluated a spectrum of models:
    *   **Baselines:** Popularity, Collaborative Filtering (SVD), and Content-Based (K-NN).
    *   **Hybrid Models:** A weighted ensemble, a feature-rich Gradient Boosting model, and a Neural Collaborative Filtering (NCF) deep learning architecture.
5.  **Hyperparameter Tuning & Optimization:** I used `GridSearchCV` to systematically fine-tune the best-performing model (Gradient Boosting) and optimized ensemble weights using numerical methods.
6.  **Comprehensive Evaluation:** I assessed models on both prediction accuracy (RMSE, MAE) and ranking quality (NDCG@10), and even simulated an A/B test to validate performance gains.
7.  **Deployment:** I encapsulated the final logic into a `RecommendationEngine` class to demonstrate personalized recommendations.

---

## 📊 Data & Feature Engineering

The engine's strength comes from its ability to synthesize information from multiple sources and my custom-engineered features.

### Data Sources

| Data Source | Type | Purpose |
| :--- | :--- | :--- |
| **Amazon SNAP Electronics** | User Reviews & Product Metadata | The primary source for training, containing over 1.6M reviews and 498k products, providing real-world data complexity. |
| **FakeStoreAPI** | Product Catalog & Synthetic Interactions | A smaller, clean dataset used for initial prototyping and to simulate diverse user behaviors like views, clicks, and purchases. |

### Feature Engineering Highlights

The raw data was transformed into a powerful feature set. This was the most critical step for enabling the hybrid models to outperform traditional methods.

| Feature Type | Description | Impact on the Model |
| :--- | :--- | :--- |
| **Implicit Rating** | A weighted score combining the `overall` rating, `helpful_ratio`, and normalized `review_length`. | This feature moves beyond simple star ratings to capture deeper signals of user engagement and review quality. |
| **Behavioral Aggregates**| User stats (e.g., average rating, total reviews) and Product stats (e.g., average rating, popularity). | These features capture long-term user tendencies and baseline item popularity, grounding the model's predictions. |
| **Temporal Features** | A `time_weight` feature gives more importance to recent interactions. | This ensures the recommender adapts to evolving user tastes and seasonal trends, making suggestions more timely. |
| **TF-IDF Embeddings**| Vector representations of product titles and descriptions. | This allows the model to understand product content, enabling it to recommend items based on textual similarity and handle cold-start scenarios for new products. |

---

## 🤖 Modeling & Results

I compared several models to find the most effective approach. The feature-based hybrid model demonstrated a clear advantage by effectively combining all engineered signals.

### Model Performance Comparison
The models were evaluated on their ability to predict user ratings on the held-out test set. The **Feature-Based Hybrid** model achieved the lowest error, significantly outperforming all other approaches.

| Model | RMSE (lower is better) | MAE (lower is better) |
| :--- | :---: | :---: |
| SVD (Baseline) | 1.3838 | 0.9030 |
| Weighted Ensemble Hybrid | 1.2832 | 0.9779 |
| **Feature-Based Hybrid (Tuned)** | **0.5782** | **0.3193** |
| Neural Collaborative Filtering | 1.3564 | 1.2370 |

The final tuned model represents a **57.96% improvement in RMSE** over the traditional SVD baseline, a substantial gain that validates the hybrid, feature-based strategy.

<!-- Placeholder for Model Performance Chart -->
**Model Performance Comparison (RMSE)**
`![Model Performance Chart](images/model_performance_chart.png)`

### Ranking & Business Metrics

Beyond rating prediction, I evaluated the models on their ability to rank relevant items for users.

| Metric | Description | Result (Weighted Ensemble) |
|:---|:---|:---|
| **NDCG@10** | Measures the quality of the top-10 recommendations, rewarding models that place highly relevant items at the top. | **0.0095** |

While the NDCG score is modest, it establishes a crucial baseline for future improvements in ranking-specific optimization. I also developed frameworks for measuring **catalog coverage** and **recommendation diversity**, which are vital for a production system.

---

## 🛠️ Getting Started: Setup and Installation


### 1. Clone the Repository
First, clone this repository to your local machine.
```bash
https://github.com/Shanmukha0505/data_cirriculam_2.git
```

### 2. Create and Activate a Virtual Environment
Using a virtual environment is highly recommended to manage dependencies and avoid conflicts.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required packages using the `requirements.txt` file. This file contains all the necessary libraries to run the notebooks and the final application.
```bash
pip install -r requirements.txt
```

