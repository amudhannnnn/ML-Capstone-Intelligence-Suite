"""
Model Export Script for Capstone Review 2 GUI
Serializes preprocessors, models, and reference metadata for Regression, Classification, and Clustering.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

os.makedirs('GUI/models', exist_ok=True)
RANDOM_STATE = 42

print("="*60)
print("1. Training & Exporting REGRESSION Pipeline (Ames Housing)...")
print("="*60)
df_reg = pd.read_csv('REGRESSION/housing.csv')

# Drop non-feature identifiers if present
drop_cols = [c for c in ['Order', 'PID'] if c in df_reg.columns]
df_reg = df_reg.drop(columns=drop_cols)

# Key user-friendly intuitive features explaining highest variance
num_features = [
    'Overall Qual', 'Gr Liv Area', 'Total Bsmt SF', '1st Flr SF',
    'Garage Cars', 'Garage Area', 'Year Built', 'Year Remod/Add',
    'Full Bath', 'Fireplaces', 'Lot Area'
]
cat_features = ['Neighborhood', 'Central Air', 'Bldg Type', 'House Style']

# Clean column names by replacing spaces if needed, but we keep raw column names
reg_features = num_features + cat_features
X_reg = df_reg[reg_features].copy()
y_reg = df_reg['SalePrice'].copy()

# Filter out extreme outliers (Gr Liv Area > 4000 with SalePrice < 300000 per Ames guidelines)
valid_idx = ~((X_reg['Gr Liv Area'] > 4000) & (y_reg < 300000))
X_reg = X_reg[valid_idx]
y_reg = y_reg[valid_idx]

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor_reg = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

reg_model = GradientBoostingRegressor(
    n_estimators=250,
    learning_rate=0.06,
    max_depth=4,
    random_state=RANDOM_STATE
)

reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_reg),
    ('regressor', reg_model)
])

reg_pipeline.fit(X_reg, y_reg)

# Store feature summaries for UI widgets
reg_meta = {
    'num_features': num_features,
    'cat_features': cat_features,
    'neighborhoods': sorted([str(x) for x in df_reg['Neighborhood'].dropna().unique()]),
    'bldg_types': sorted([str(x) for x in df_reg['Bldg Type'].dropna().unique()]),
    'house_styles': sorted([str(x) for x in df_reg['House Style'].dropna().unique()]),
    'medians': X_reg[num_features].median().to_dict(),
    'mins': X_reg[num_features].min().to_dict(),
    'maxs': X_reg[num_features].max().to_dict(),
    'avg_price': float(y_reg.mean()),
    'median_price': float(y_reg.median())
}

joblib.dump({'pipeline': reg_pipeline, 'meta': reg_meta}, 'GUI/models/regression_model.pkl')
print(f"Regression model saved successfully to GUI/models/regression_model.pkl")


print("\n" + "="*60)
print("2. Training & Exporting CLASSIFICATION Pipeline (Shopper Intention)...")
print("="*60)
df_cls = pd.read_csv('CLASSIFICATION/online_shoppers_intention.csv')

cls_num_cols = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]
cls_cat_cols = [
    'Month', 'OperatingSystems', 'Browser',
    'Region', 'TrafficType', 'VisitorType', 'Weekend'
]

X_cls = df_cls[cls_num_cols + cls_cat_cols].copy()
y_cls = df_cls['Revenue'].astype(int).copy()

preprocessor_cls = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), cls_num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cls_cat_cols)
])

cls_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=3,
    random_state=RANDOM_STATE
)

cls_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_cls),
    ('classifier', cls_model)
])

cls_pipeline.fit(X_cls, y_cls)

cls_meta = {
    'num_cols': cls_num_cols,
    'cat_cols': cls_cat_cols,
    'months': sorted([str(m) for m in df_cls['Month'].unique()]),
    'visitor_types': sorted([str(v) for v in df_cls['VisitorType'].unique()]),
    'medians': X_cls[cls_num_cols].median().to_dict(),
    'mins': X_cls[cls_num_cols].min().to_dict(),
    'maxs': X_cls[cls_num_cols].max().to_dict(),
    'conversion_rate': float(y_cls.mean())
}

joblib.dump({'pipeline': cls_pipeline, 'meta': cls_meta}, 'GUI/models/classification_model.pkl')
print("Classification model saved successfully to GUI/models/classification_model.pkl")


print("\n" + "="*60)
print("3. Training & Exporting CLUSTERING Pipeline (Online Retail RFM)...")
print("="*60)
excel_path = 'CLUSTERING/Online Retail.xlsx'
if not os.path.exists(excel_path):
    excel_path = 'DATASETS/Online+Retail (Clustering)/Online Retail.xlsx'

print(f"Reading {excel_path}...")
df_retail = pd.read_excel(excel_path)

# Cleaning
df_retail = df_retail.dropna(subset=['CustomerID'])
df_retail = df_retail[(df_retail['Quantity'] > 0) & (df_retail['UnitPrice'] > 0)]
df_retail['TotalSpend'] = df_retail['Quantity'] * df_retail['UnitPrice']
df_retail['InvoiceDate'] = pd.to_datetime(df_retail['InvoiceDate'])

snapshot_date = df_retail['InvoiceDate'].max() + pd.Timedelta(days=1)

# RFM derivation
rfm = df_retail.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalSpend': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalSpend': 'Monetary'
})

# Filter extreme anomalies
rfm = rfm[(rfm['Monetary'] > 0) & (rfm['Frequency'] > 0)]

# Log transform + Scale
rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=RANDOM_STATE)
labels = kmeans.fit_predict(rfm_scaled)
rfm['Cluster'] = labels

# Fit PCA for 2D projection
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_coords = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = pca_coords[:, 0]
rfm['PCA2'] = pca_coords[:, 1]

# Persona mapping
cluster_profiles = {
    0: {
        'name': 'Potential Loyalists',
        'desc': 'Recent customers with moderate frequency and growing spend.',
        'action': 'Offer loyalty programs, multi-buy bundles, and personalized product recommendations.',
        'color': '#3B82F6'  # Blue
    },
    1: {
        'name': 'Champions / High-Value VIPs',
        'desc': 'Top spenders with high purchase frequency and recent transactions.',
        'action': 'Provide VIP concierge service, exclusive previews, and reward anniversary perks.',
        'color': '#10B981'  # Green
    },
    2: {
        'name': 'At-Risk / Lapsed High Spenders',
        'desc': 'Historical big spenders who have not visited in a significant timeframe.',
        'action': 'Target with win-back email sequences, aggressive limited-time renewal discounts.',
        'color': '#F59E0B'  # Amber
    },
    3: {
        'name': 'Occasional / One-Time Buyers',
        'desc': 'Infrequent buyers with low monetary volume and long recency.',
        'action': 'Low-cost marketing: automated newsletter digests and mass seasonal promotion campaigns.',
        'color': '#EF4444'  # Red
    }
}

# Subsample 800 reference points for fast interactive 2D visualization in GUI
sample_points = rfm.sample(n=min(800, len(rfm)), random_state=RANDOM_STATE)

clst_data = {
    'scaler': scaler,
    'kmeans': kmeans,
    'pca': pca,
    'cluster_profiles': cluster_profiles,
    'sample_points': sample_points[['Recency', 'Frequency', 'Monetary', 'Cluster', 'PCA1', 'PCA2']].to_dict(orient='records'),
    'rfm_summary': rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).round(1).to_dict(orient='index')
}

joblib.dump(clst_data, 'GUI/models/clustering_model.pkl')
print("Clustering model and reference data saved successfully to GUI/models/clustering_model.pkl")
print("\n[SUCCESS] All 3 models and artifacts successfully exported!")
