# =========================================================
# E-COMMERCE RECOMMENDATION SYSTEM
# PART 1 : DATASET GENERATION + EDA
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# RANDOM SEED
# =========================================================

np.random.seed(42)

# =========================================================
# GENERATE SYNTHETIC DATASET
# =========================================================

records = 5000

categories = [
    'Smartphone',
    'Laptop',
    'Headphones',
    'Smartwatch',
    'Television',
    'Camera',
    'Gaming Console',
    'Tablet'
]

brands = [
    'Apple',
    'Samsung',
    'Sony',
    'Dell',
    'HP',
    'Lenovo',
    'Asus',
    'OnePlus'
]

products = []

for i in range(records):

    category = np.random.choice(categories)

    brand = np.random.choice(brands)

    price = np.random.randint(5000, 150000)

    rating = round(np.random.uniform(2.5, 5.0), 1)

    popularity = np.random.randint(1, 100)

    discount = np.random.randint(0, 70)

    description = (
        f"{brand} {category} "
        f"premium quality product "
        f"rating {rating}"
    )

    purchase_status = np.random.choice(
        [0, 1],
        p=[0.4, 0.6]
    )

    products.append([
        i + 1,
        f"{brand}_{category}_{i}",
        category,
        brand,
        price,
        rating,
        popularity,
        discount,
        description,
        purchase_status
    ])

# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(
    products,
    columns=[
        "Product_ID",
        "Product_Name",
        "Category",
        "Brand",
        "Price",
        "Rating",
        "Popularity",
        "Discount",
        "Description",
        "Purchase_Status"
    ]
)

# =========================================================
# SAVE CSV
# =========================================================

df.to_csv(
    "ecommerce_products.csv",
    index=False
)

print(
    "\nDataset Saved : ecommerce_products.csv"
)

# =========================================================
# FIRST 5 RECORDS
# =========================================================

print("\nFIRST 5 RECORDS")
print(df.head())

# =========================================================
# LAST 5 RECORDS
# =========================================================

print("\nLAST 5 RECORDS")
print(df.tail())

# =========================================================
# SHAPE
# =========================================================

print("\nDATASET SHAPE")
print(df.shape)

# =========================================================
# INFO
# =========================================================

print("\nDATASET INFO")
print(df.info())

# =========================================================
# DATA TYPES
# =========================================================

print("\nDATA TYPES")
print(df.dtypes)

# =========================================================
# MISSING VALUES
# =========================================================

print("\nMISSING VALUES")
print(df.isnull().sum())

# =========================================================
# DUPLICATES
# =========================================================

print("\nDUPLICATES")
print(df.duplicated().sum())

# =========================================================
# STATISTICAL SUMMARY
# =========================================================

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# =========================================================
# CATEGORY DISTRIBUTION
# =========================================================

print("\nCATEGORY COUNTS")
print(df['Category'].value_counts())

# =========================================================
# BRAND DISTRIBUTION
# =========================================================

print("\nBRAND COUNTS")
print(df['Brand'].value_counts())

# =========================================================
# PURCHASE DISTRIBUTION
# =========================================================

print("\nPURCHASE STATUS")
print(df['Purchase_Status'].value_counts())

# =========================================================
# HISTOGRAM : PRICE
# =========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df['Price'],
    kde=True,
    bins=30
)

plt.title("Price Distribution")

plt.show()

# =========================================================
# HISTOGRAM : RATING
# =========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df['Rating'],
    kde=True,
    bins=20
)

plt.title("Rating Distribution")

plt.show()

# =========================================================
# HISTOGRAM : POPULARITY
# =========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df['Popularity'],
    kde=True,
    bins=20
)

plt.title("Popularity Distribution")

plt.show()

# =========================================================
# HISTOGRAM : DISCOUNT
# =========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df['Discount'],
    kde=True,
    bins=20
)

plt.title("Discount Distribution")

plt.show()

# =========================================================
# CATEGORY COUNTPLOT
# =========================================================

plt.figure(figsize=(10,5))

sns.countplot(
    x='Category',
    data=df
)

plt.xticks(rotation=45)

plt.title("Category Distribution")

plt.show()

# =========================================================
# BRAND COUNTPLOT
# =========================================================

plt.figure(figsize=(10,5))

sns.countplot(
    x='Brand',
    data=df
)

plt.xticks(rotation=45)

plt.title("Brand Distribution")

plt.show()

# =========================================================
# PURCHASE STATUS COUNTPLOT
# =========================================================

plt.figure(figsize=(6,4))

sns.countplot(
    x='Purchase_Status',
    data=df
)

plt.title("Purchase Status")

plt.show()

# =========================================================
# BOXPLOT : PRICE
# =========================================================

plt.figure(figsize=(8,4))

sns.boxplot(
    x=df['Price']
)

plt.title("Price Outliers")

plt.show()

# =========================================================
# BOXPLOT : RATING
# =========================================================

plt.figure(figsize=(8,4))

sns.boxplot(
    x=df['Rating']
)

plt.title("Rating Outliers")

plt.show()

# =========================================================
# CORRELATION HEATMAP
# =========================================================

numeric_df = df[
    [
        'Price',
        'Rating',
        'Popularity',
        'Discount',
        'Purchase_Status'
    ]
]

plt.figure(figsize=(8,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# =========================================================
# PRICE VS RATING
# =========================================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x='Price',
    y='Rating',
    data=df
)

plt.title("Price vs Rating")

plt.show()

# =========================================================
# CATEGORY VS PRICE
# =========================================================

plt.figure(figsize=(10,5))

sns.boxplot(
    x='Category',
    y='Price',
    data=df
)

plt.xticks(rotation=45)

plt.title("Category vs Price")

plt.show()

# =========================================================
# BRAND VS PRICE
# =========================================================

plt.figure(figsize=(10,5))

sns.boxplot(
    x='Brand',
    y='Price',
    data=df
)

plt.xticks(rotation=45)

plt.title("Brand vs Price")

plt.show()

# =========================================================
# PURCHASE STATUS PIE CHART
# =========================================================

purchase_counts = (
    df['Purchase_Status']
    .value_counts()
)

plt.figure(figsize=(6,6))

plt.pie(
    purchase_counts,
    labels=[
        'Purchased',
        'Not Purchased'
    ],
    autopct='%1.1f%%'
)

plt.title("Purchase Status Distribution")

plt.show()

# =========================================================
# END OF PART 1
# =========================================================

print(
    "\nPART 1 COMPLETED SUCCESSFULLY"
)



# =========================================================
# PART 2 : CONTENT-BASED RECOMMENDATION SYSTEM
# USING TF-IDF + COSINE SIMILARITY
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle

# =========================================================
# CREATE COMBINED FEATURES
# =========================================================

print("\nCreating Recommendation Features...")

df["Combined_Features"] = (
    df["Category"].astype(str) + " " +
    df["Brand"].astype(str) + " " +
    df["Description"].astype(str)
)

# =========================================================
# TF-IDF VECTORIZATION
# =========================================================

print("\nApplying TF-IDF Vectorization...")

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    df["Combined_Features"]
)

print(
    "\nTF-IDF Matrix Shape:",
    tfidf_matrix.shape
)

# =========================================================
# COSINE SIMILARITY MATRIX
# =========================================================

print("\nCalculating Cosine Similarity...")

cosine_sim = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

print(
    "Cosine Similarity Matrix Shape:",
    cosine_sim.shape
)

# =========================================================
# PRODUCT INDEX MAPPING
# =========================================================

indices = pd.Series(
    df.index,
    index=df["Product_Name"]
).drop_duplicates()

print(
    "\nTotal Products Indexed:",
    len(indices)
)

# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend_products(product_name, top_n=5):

    if product_name not in indices:

        print(
            f"\nProduct '{product_name}' not found."
        )

        return None

    idx = indices[product_name]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[
        1:top_n+1
    ]

    product_indices = [
        i[0]
        for i in similarity_scores
    ]

    recommendations = df.iloc[
        product_indices
    ][[
        "Product_Name",
        "Category",
        "Brand",
        "Price",
        "Rating"
    ]]

    return recommendations

# =========================================================
# TEST RECOMMENDATION SYSTEM
# =========================================================

sample_product = df.iloc[0]["Product_Name"]

print("\n")
print("=" * 60)
print("SAMPLE RECOMMENDATION TEST")
print("=" * 60)

print(
    "\nSelected Product:",
    sample_product
)

recommendations = recommend_products(
    sample_product,
    top_n=5
)

print("\nTOP 5 RECOMMENDATIONS")
print(recommendations)

# =========================================================
# MULTIPLE PRODUCT TESTS
# =========================================================

print("\n")
print("=" * 60)
print("MULTIPLE PRODUCT TESTING")
print("=" * 60)

test_products = [
    df.iloc[10]["Product_Name"],
    df.iloc[50]["Product_Name"],
    df.iloc[100]["Product_Name"]
]

for product in test_products:

    print("\n")
    print("-" * 50)

    print(
        "Selected Product:",
        product
    )

    recs = recommend_products(
        product,
        top_n=3
    )

    print(recs)

# =========================================================
# CATEGORY-WISE SAMPLE RECOMMENDATION
# =========================================================

print("\n")
print("=" * 60)
print("CATEGORY SAMPLE ANALYSIS")
print("=" * 60)

for category in df["Category"].unique():

    sample = df[
        df["Category"] == category
    ].iloc[0]["Product_Name"]

    print(
        f"\nCategory: {category}"
    )

    print(
        f"Sample Product: {sample}"
    )

# =========================================================
# VISUALIZE PRODUCT COUNTS
# =========================================================

plt.figure(figsize=(10,5))

category_counts = (
    df["Category"]
    .value_counts()
)

sns.barplot(
    x=category_counts.index,
    y=category_counts.values
)

plt.xticks(rotation=45)

plt.title(
    "Products Per Category"
)

plt.ylabel(
    "Count"
)

plt.show()

# =========================================================
# VISUALIZE BRAND COUNTS
# =========================================================

plt.figure(figsize=(10,5))

brand_counts = (
    df["Brand"]
    .value_counts()
)

sns.barplot(
    x=brand_counts.index,
    y=brand_counts.values
)

plt.title(
    "Products Per Brand"
)

plt.ylabel(
    "Count"
)

plt.show()

# =========================================================
# SAVE TF-IDF MODEL
# =========================================================

with open(
    "tfidf_vectorizer.pkl",
    "wb"
) as f:

    pickle.dump(
        tfidf,
        f
    )

print(
    "\nTF-IDF Vectorizer Saved"
)

# =========================================================
# SAVE COSINE SIMILARITY
# =========================================================

with open(
    "cosine_similarity.pkl",
    "wb"
) as f:

    pickle.dump(
        cosine_sim,
        f
    )

print(
    "Cosine Similarity Saved"
)

# =========================================================
# SAVE PRODUCT DATA
# =========================================================

with open(
    "products_dataframe.pkl",
    "wb"
) as f:

    pickle.dump(
        df,
        f
    )

print(
    "Product Data Saved"
)

# =========================================================
# SAVE PRODUCT INDICES
# =========================================================

with open(
    "product_indices.pkl",
    "wb"
) as f:

    pickle.dump(
        indices,
        f
    )

print(
    "Product Indices Saved"
)

# =========================================================
# QUICK RECOMMENDATION API
# =========================================================

def get_recommendations(
        product_name,
        n=5):

    result = recommend_products(
        product_name,
        top_n=n
    )

    return result

# =========================================================
# FINAL TEST
# =========================================================

print("\n")
print("=" * 60)
print("RECOMMENDATION ENGINE READY")
print("=" * 60)

print(
    "\nFiles Generated:"
)

print(
    "1. ecommerce_products.csv"
)

print(
    "2. tfidf_vectorizer.pkl"
)

print(
    "3. cosine_similarity.pkl"
)

print(
    "4. products_dataframe.pkl"
)

print(
    "5. product_indices.pkl"
)

# =========================================================
# END OF PART 2
# =========================================================

print(
    "\nPART 2 COMPLETED SUCCESSFULLY"
)




# =========================================================
# PART 3 : MACHINE LEARNING MODELS
# Decision Tree + Random Forest + XGBoost
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PREPARE DATA FOR ML
# =========================================================

print("\nPreparing Data For ML Models...")

ml_df = df.copy()

# Encode categorical features

label_encoders = {}

categorical_cols = [
    "Category",
    "Brand"
]

for col in categorical_cols:

    le = LabelEncoder()

    ml_df[col] = le.fit_transform(
        ml_df[col]
    )

    label_encoders[col] = le

# =========================================================
# FEATURES & TARGET
# =========================================================

X = ml_df[
    [
        "Category",
        "Brand",
        "Price",
        "Rating",
        "Popularity",
        "Discount"
    ]
]

y = ml_df["Purchase_Status"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# =========================================================
# HELPER FUNCTION
# =========================================================

results = []

def evaluate_model(
        model_name,
        model,
        y_train_pred,
        y_test_pred):

    train_acc = accuracy_score(
        y_train,
        y_train_pred
    )

    test_acc = accuracy_score(
        y_test,
        y_test_pred
    )

    precision = precision_score(
        y_test,
        y_test_pred
    )

    recall = recall_score(
        y_test,
        y_test_pred
    )

    f1 = f1_score(
        y_test,
        y_test_pred
    )

    results.append([
        model_name,
        train_acc,
        test_acc,
        precision,
        recall,
        f1
    ])

    print("\n")
    print("="*60)
    print(model_name)
    print("="*60)

    print(
        "\nTrain Accuracy:",
        round(train_acc,4)
    )

    print(
        "Test Accuracy:",
        round(test_acc,4)
    )

    print(
        "\nClassification Report"
    )

    print(
        classification_report(
            y_test,
            y_test_pred
        )
    )

    cm = confusion_matrix(
        y_test,
        y_test_pred
    )

    plt.figure(figsize=(6,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

# =========================================================
# DECISION TREE - UNDERFITTING
# =========================================================

dt_under = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

dt_under.fit(
    X_train,
    y_train
)

under_train_pred = dt_under.predict(
    X_train
)

under_test_pred = dt_under.predict(
    X_test
)

evaluate_model(
    "DT Underfitting",
    dt_under,
    under_train_pred,
    under_test_pred
)

# Tree Visualization

plt.figure(figsize=(12,6))

plot_tree(
    dt_under,
    feature_names=X.columns,
    class_names=[
        "Not Purchased",
        "Purchased"
    ],
    filled=True
)

plt.title(
    "Decision Tree Underfitting"
)

plt.show()

# =========================================================
# DECISION TREE - OVERFITTING
# =========================================================

dt_over = DecisionTreeClassifier(
    random_state=42
)

dt_over.fit(
    X_train,
    y_train
)

over_train_pred = dt_over.predict(
    X_train
)

over_test_pred = dt_over.predict(
    X_test
)

evaluate_model(
    "DT Overfitting",
    dt_over,
    over_train_pred,
    over_test_pred
)

plt.figure(figsize=(20,10))

plot_tree(
    dt_over,
    feature_names=X.columns,
    filled=True
)

plt.title(
    "Decision Tree Overfitting"
)

plt.show()

# =========================================================
# DECISION TREE - BALANCED
# =========================================================

dt_balanced = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

dt_balanced.fit(
    X_train,
    y_train
)

balanced_train_pred = dt_balanced.predict(
    X_train
)

balanced_test_pred = dt_balanced.predict(
    X_test
)

evaluate_model(
    "DT Balanced",
    dt_balanced,
    balanced_train_pred,
    balanced_test_pred
)

plt.figure(figsize=(18,8))

plot_tree(
    dt_balanced,
    feature_names=X.columns,
    filled=True
)

plt.title(
    "Decision Tree Balanced"
)

plt.show()

# =========================================================
# RANDOM FOREST
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_train_pred = rf_model.predict(
    X_train
)

rf_test_pred = rf_model.predict(
    X_test
)

evaluate_model(
    "Random Forest",
    rf_model,
    rf_train_pred,
    rf_test_pred
)

# =========================================================
# RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

importance_rf = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        rf_model.feature_importances_

})

importance_rf = importance_rf.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(8,5))

sns.barplot(
    data=importance_rf,
    x="Importance",
    y="Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.show()

# =========================================================
# XGBOOST
# =========================================================

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_train_pred = xgb_model.predict(
    X_train
)

xgb_test_pred = xgb_model.predict(
    X_test
)

evaluate_model(
    "XGBoost",
    xgb_model,
    xgb_train_pred,
    xgb_test_pred
)

# =========================================================
# XGBOOST FEATURE IMPORTANCE
# =========================================================

importance_xgb = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        xgb_model.feature_importances_

})

importance_xgb = importance_xgb.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(8,5))

sns.barplot(
    data=importance_xgb,
    x="Importance",
    y="Feature"
)

plt.title(
    "XGBoost Feature Importance"
)

plt.show()

# =========================================================
# FINAL COMPARISON TABLE
# =========================================================

comparison_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Train Accuracy",
        "Test Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n")
print("="*70)
print("FINAL MODEL COMPARISON")
print("="*70)

print(comparison_df)

# =========================================================
# COMPARISON GRAPH
# =========================================================

plt.figure(figsize=(12,6))

comparison_df.set_index(
    "Model"
)[
    [
        "Train Accuracy",
        "Test Accuracy"
    ]
].plot(
    kind="bar"
)

plt.title(
    "Model Accuracy Comparison"
)

plt.ylabel(
    "Accuracy"
)

plt.xticks(rotation=15)

plt.show()

# =========================================================
# SAVE MODELS
# =========================================================

import pickle

with open(
    "dt_balanced_model.pkl",
    "wb"
) as f:

    pickle.dump(
        dt_balanced,
        f
    )

with open(
    "random_forest_model.pkl",
    "wb"
) as f:

    pickle.dump(
        rf_model,
        f
    )

with open(
    "xgboost_model.pkl",
    "wb"
) as f:

    pickle.dump(
        xgb_model,
        f
    )

print("\nModels Saved Successfully")

print("dt_balanced_model.pkl")
print("random_forest_model.pkl")
print("xgboost_model.pkl")

# =========================================================
# END OF PART 3
# =========================================================

print(
    "\nPART 3 COMPLETED SUCCESSFULLY"
)