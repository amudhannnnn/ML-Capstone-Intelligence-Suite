# 23CSE301 Machine Learning Capstone Project

**Academic Year:** 2026–27 | **Course:** 23CSE301 Machine Learning  
**Evaluation Scope:** Review 1 & Review 2 (Full Regression, Full Classification, Full Clustering tracks)

---

## 1. Project Overview & Problem Statements

This repository contains an end-to-end Machine Learning pipeline across three distinct learning paradigms:

### Track 1: Regression — Ames Housing Price Prediction
* **Directory:** `REGRESSION/`
* **Dataset:** `housing.csv` (Ames Housing Dataset — 2,930 instances, 80 features)
* **Problem Statement:** Predict the final sale price (`SalePrice`) of residential homes in Ames, Iowa, using physical, quality, and location attributes.
* **Key Tasks:** Comprehensive EDA, missing value imputation, log-transformation of target, one-hot encoding, feature scaling, 10 regression algorithms, hyperparameter tuning, residual diagnostics, and 5-fold cross-validation.

### Track 2: Classification — Online Shoppers Purchasing Intention
* **Directory:** `CLASSIFICATION/`
* **Dataset:** `online_shoppers_intention.csv` (12,330 e-commerce sessions, 18 features)
* **Problem Statement:** Predict whether an online session will conclude in a completed transaction (`Revenue: True/False`).
* **Key Tasks:** Imbalance handling (84.5% non-revenue vs 15.5% revenue), stratified splitting, standard scaling, training 10 classification algorithms (Part A baseline + Part B ensemble/advanced), hyperparameter tuning, confusion matrices, ROC-AUC curves, and 5-fold stratified cross-validation.

### Track 3: Clustering — Online Retail Customer Segmentation
* **Directory:** `CLUSTERING/`
* **Dataset:** `Online Retail.xlsx` (541,909 transactions across 38 countries)
* **Problem Statement:** Segment customers into distinct behavioural cohorts using unsupervised learning to formulate targeted marketing retention strategies.
* **Key Tasks:** Data audit & cleaning (removing cancelled invoices, negative quantities, missing CustomerIDs), RFM (Recency, Frequency, Monetary value) feature engineering, log-transformation, standard scaling, K-Means (Elbow method & Silhouette analysis), Agglomerative Hierarchical Clustering (Dendrogram & linkage evaluation), 2D PCA & t-SNE projections, and cluster profiling.

---

## 2. Repository Directory Structure

```text
.
├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf   # Official capstone instructions & rubrics
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation & summary
│
├── REGRESSION/
│   ├── housing.csv                             # Raw dataset
│   ├── regression.ipynb                        # Review 1 regression notebook
│   └── regression_review_2.ipynb               # Review 2 consolidated regression notebook
│
├── CLASSIFICATION/
│   ├── online_shoppers_intention.csv           # Raw dataset
│   ├── classification.ipynb                    # Review 1 Part A notebook (5 baseline models)
│   └── classification_review_2.ipynb           # Review 2 complete notebook (All 10 models)
│
├── CLUSTERING/
│   ├── Online Retail.xlsx                      # Raw dataset
│   └── clustering.ipynb                        # Complete unsupervised clustering notebook
│
├── GUI/                                        # Capstone Interactive Web GUI
│   ├── app.py                                  # Streamlit web application
│   ├── export_models.py                        # Model training & export script
│   ├── requirements.txt                        # GUI specific dependencies
│   └── models/                                 # Serialized .pkl model pipelines
│       ├── regression_model.pkl
│       ├── classification_model.pkl
│       └── clustering_model.pkl
│
└── DATASETS/                                   # Supplemental / reference datasets
    └── Online+Retail (Clustering)/
        └── Online Retail.xlsx
```

---

## 3. Results Summary

### 3.1 Classification Track: 10-Algorithm Comparison

Evaluated on held-out test set ($N=2,466$, stratified 80:20 split) from `CLASSIFICATION/classification_review_2.ipynb`:

| # | Algorithm | Category | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 1 | Bagging Classifier (DT Base) | Part B | 0.9058 | 0.8995 | 0.9058 | 0.8980 | 0.9310 |
| 2 | Random Forest Classifier | Part B | 0.9037 | 0.8971 | 0.9037 | 0.8977 | 0.9282 |
| 3 | Gradient Boosting Classifier | Part B | 0.9005 | 0.8958 | 0.9005 | 0.8975 | 0.9330 |
| 4 | AdaBoost Classifier | Part B | 0.8976 | 0.8935 | 0.8976 | 0.8951 | 0.9235 |
| 5 | Decision Tree Classifier (depth=5) | Part A | 0.8959 | 0.8942 | 0.8959 | 0.8950 | 0.9197 |
| 6 | MLP Classifier (Neural Network) | Part B | 0.8935 | 0.8889 | 0.8935 | 0.8907 | 0.9170 |
| 7 | Support Vector Classifier (SVC, C=10, RBF) | Part A | 0.8923 | 0.8836 | 0.8923 | 0.8848 | 0.8600 |
| 8 | Logistic Regression (Baseline) | Part A | 0.8861 | 0.8764 | 0.8861 | 0.8698 | 0.8984 |
| 9 | K-Nearest Neighbors (k=5) | Part A | 0.8775 | 0.8641 | 0.8775 | 0.8648 | 0.7893 |
| 10 | Gaussian Naive Bayes | Part A | 0.8460 | 0.8534 | 0.8460 | 0.8493 | 0.8383 |

#### Hyperparameter Tuning & Cross-Validation:
* **Best Model After Hyperparameter Tuning:** **Gradient Boosting Classifier (Tuned)**
  * **Test Metrics:** Accuracy: **0.9074** | Precision: **0.9032** | Recall: **0.9074** | Weighted F1: **0.9046** | ROC-AUC: **0.9356**
  * **Best Parameters:** `learning_rate=0.05`, `max_depth=3`, `n_estimators=200`, `subsample=0.8` (GridSearchCV tuned)
* **Runner-Up Tuned Model:** **Random Forest Classifier (Tuned)**
  * **Test Metrics:** Accuracy: **0.9041** | Precision: **0.8978** | Recall: **0.9041** | Weighted F1: **0.8986** | ROC-AUC: **0.9281**
  * **Best Parameters:** `n_estimators=200`, `max_depth=15`, `min_samples_split=10`, `min_samples_leaf=4`
* **5-Fold Stratified Cross-Validation (Top-2 Tuned Models):**
  * **Tuned Gradient Boosting:** Mean CV Accuracy = $0.9029 \pm 0.0055$ | Mean CV F1 = $0.8989 \pm 0.0056$ | Mean CV ROC-AUC = $0.9291 \pm 0.0062$
  * **Tuned Random Forest:** Mean CV Accuracy = $0.9023 \pm 0.0039$ | Mean CV F1 = $0.8962 \pm 0.0045$ | Mean CV ROC-AUC = $0.9239 \pm 0.0036$

---

### 3.2 Regression Track: 10-Algorithm Comparison

Evaluated on held-out test split (Ames Housing, 80:20 split) from `REGRESSION/regression_review_2.ipynb`:

| # | Model | Parameters / Details | $R^2$ Score | RMSE ($) | MAE ($) | Ranking |
|---|---|---|---|---|---|---|
| 1 | **Gradient Boosting Regressor** | `learning_rate=0.05` | **0.9238** | **19,554.26** | **12,639.86** | **1** |
| 2 | **Random Forest Regressor** | `n_estimators=200` | **0.9125** | **20,959.23** | **13,514.19** | **2** |
| 3 | Lasso Regression | `alpha=100` ($L_1$ sparsity) | 0.9020 | 22,176.11 | 15,951.50 | 3 |
| 4 | Ridge Regression | `alpha=10` ($L_2$ regularization) | 0.8991 | 22,503.13 | 16,120.21 | 4 |
| 5 | Linear Regression | Baseline OLS | 0.8982 | 22,596.58 | 16,177.65 | 5 |
| 6 | KNN Regressor | `k=7` (Distance-weighted) | 0.8742 | 25,122.55 | 17,679.55 | 6 |
| 7 | Decision Tree Regressor | `max_depth=6` | 0.8351 | 28,766.48 | 20,131.02 | 7 |
| 8 | Polynomial Regression | Degree 2 (`PolynomialFeatures`) | 0.7932 | 32,209.09 | 17,791.71 | 8 |
| 9 | ElasticNet Regression | `alpha=100`, `l1_ratio=0.5` | 0.3157 | 58,596.14 | 43,408.01 | 9 |
| 10 | Support Vector Regressor (SVR) | `C=10`, RBF Kernel | -0.0441 | 72,379.94 | 51,271.39 | 10 |

#### Hyperparameter Tuning & Cross-Validation:
* **Best Model:** Gradient Boosting Regressor (GridSearchCV tuned: $R^2$ improved from 0.9174 to 0.9196 on search space; raw best single fit test $R^2 = 0.9238$, RMSE = \$19,554).
* **5-Fold Cross-Validation on Top-2 Models:**
  * **Gradient Boosting (Tuned):** Mean CV $R^2 = \mathbf{0.9279 \pm 0.0100}$ (Fold scores: `[0.9418, 0.9147, 0.9371, 0.9224, 0.9237]`)
  * **Random Forest (Tuned):** Mean CV $R^2 = \mathbf{0.9113 \pm 0.0130}$ (Fold scores: `[0.9279, 0.8918, 0.9167, 0.9015, 0.9189]`)

---

### 3.3 Clustering Track: K-Means vs Agglomerative Hierarchical

Evaluated on customer-level RFM features ($N=4,338$ unique active customers, $K=4$ clusters) from `CLUSTERING/clustering.ipynb`:

| Algorithm | Linkage / Init | Silhouette Score ($\uparrow$) | Davies-Bouldin Index ($\downarrow$) | Calinski-Harabasz Index ($\uparrow$) |
|---|---|---|---|---|
| **K-Means Clustering ($K=4$)** | `k-means++` (Elbow method: $k=4$) | **0.4901** | **0.7463** | **6,099.34** |
| Agglomerative Hierarchical ($K=4$) | Ward Linkage (Dendrogram cutoff) | 0.4627 | 0.7294 | 5,278.91 |

#### Hierarchical Linkage Comparison ($K=4$):
* **Ward:** Silhouette: **0.4627** | Davies-Bouldin: **0.7294** | Calinski-Harabasz: **5,278.91** *(Best overall balance; minimizes within-cluster variance)*
* **Average:** Silhouette: 0.4886 | Davies-Bouldin: 0.7157 | Calinski-Harabasz: 1,288.57
* **Complete:** Silhouette: 0.3983 | Davies-Bouldin: 0.8135 | Calinski-Harabasz: 1,455.52
* **Single:** Silhouette: 0.4537 | Davies-Bouldin: 0.3661 | Calinski-Harabasz: 6.09 *(Severe chaining effect)*

#### Customer Segment Profiles Identified (K-Means $K=4$):
1. **Cluster 0 — High-Value VIP Champions ($N=824$):** Low Recency (mean 16.5 days), High Frequency (14.3 purchases), High Monetary (£5,113.30).  
   *Strategy:* VIP tier status, early access to new lines, dedicated account concierge.
2. **Cluster 1 — Moderate Active / Potential Loyalists ($N=1,792$):** Moderate Recency (51.6 days), Moderate Frequency (2.2 purchases), Spend (£634.40).  
   *Strategy:* Milestone rewards, cross-sell/up-sell personalized bundles.
3. **Cluster 2 — Hibernating / Lapsed Customers ($N=1,063$):** High Recency (252.9 days), Low Frequency (1.5 purchases), Low Monetary (£428.50).  
   *Strategy:* Automated reactivation email campaigns with time-limited discount coupons.
4. **Cluster 3 — High-Frequency Loyal Shoppers ($N=659$):** Low Recency (33.8 days), High Frequency (6.3 purchases), High Monetary (£2,566.70).  
   *Strategy:* Loyalty tier points booster, referral incentive programs.

---

## 4. Environment Setup & How to Run

### Step 1: Clone Repository & Create Virtual Environment
```bash
# Clone the repository
git clone <repository-url>
cd "CASE STUDY"

# Create virtual environment (Python 3.10+ recommended)
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Jupyter Notebook
```bash
jupyter notebook
```

Navigate to any of the track notebooks:
* `CLASSIFICATION/classification_review_2.ipynb`
* `REGRESSION/regression_review_2.ipynb`
* `CLUSTERING/clustering.ipynb`

All notebooks are designed to execute top-to-bottom without manual configuration.

### Step 4: Launch Interactive Web GUI (Review 2 Bonus)
```bash
streamlit run GUI/app.py
```
Open your browser at `http://localhost:8501` to access the unified **ML Intelligence Suite**.

---

## 5. Academic Integrity & Generative AI Statement

In compliance with **Section 7.5 (Academic Integrity)** of the `23CSE301` Capstone Guidelines:
* **Generative AI Scaffolding:** Generative AI tools (Antigravity IDE / Claude / ChatGPT) were employed strictly for code scaffolding, boilerplate structuring, and markdown styling assistance.
* **Originality of Work:** All Exploratory Data Analysis (EDA), feature engineering pipelines, algorithmic parameter selections, diagnostic evaluations, visual interpretations, and business insight formulations were independently analyzed, authored, and verified by the team members.
* **External Citations:** Algorithms, transformations, and evaluation metrics utilize official implementations from Scikit-Learn, Pandas, NumPy, Scipy, and Streamlit.

