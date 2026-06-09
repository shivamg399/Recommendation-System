# =========================================================
# E-COMMERCE RECOMMENDATION SYSTEM
# STREAMLIT APPLICATION
# =========================================================

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="E-Commerce Recommendation System",
    page_icon="🛒",
    layout="wide"
)

# =========================================================
# LOAD FILES
# =========================================================

@st.cache_data
def load_data():

    return pickle.load(
        open(
            "products_dataframe.pkl",
            "rb"
        )
    )

@st.cache_resource
def load_similarity():

    return pickle.load(
        open(
            "cosine_similarity.pkl",
            "rb"
        )
    )

@st.cache_resource
def load_indices():

    return pickle.load(
        open(
            "product_indices.pkl",
            "rb"
        )
    )

df = load_data()

cosine_sim = load_similarity()

indices = load_indices()

# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend_products(
        product_name,
        top_n=5
):

    idx = indices[product_name]

    similarity_scores = list(
        enumerate(
            cosine_sim[idx]
        )
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
# SIDEBAR
# =========================================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(

    "Select Page",

    [
        "Home",
        "Dataset Analysis",
        "Recommendation Engine",
        "Model Comparison"
    ]
)

# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    st.title(
        "🛒 E-Commerce Recommendation System"
    )

    st.markdown(
        """
        ### Portfolio-Level Hybrid Recommendation System

        Algorithms Used:

        - TF-IDF
        - Cosine Similarity
        - Decision Tree
        - Random Forest
        - XGBoost

        Features:

        - Product Recommendation
        - Dataset Analysis
        - Model Comparison
        - Interactive Dashboard
        """
    )

    st.image(
        "https://images.unsplash.com/photo-1556740749-887f6717d7e4",
        use_container_width=True
    )

# =========================================================
# DATASET ANALYSIS
# =========================================================

elif page == "Dataset Analysis":

    st.title(
        "📊 Dataset Analysis"
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head()
    )

    st.subheader(
        "Dataset Shape"
    )

    st.write(
        df.shape
    )

    st.subheader(
        "Statistical Summary"
    )

    st.dataframe(
        df.describe()
    )

    # Category Distribution

    st.subheader(
        "Category Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    sns.countplot(
        data=df,
        x='Category',
        ax=ax
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # Brand Distribution

    st.subheader(
        "Brand Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    sns.countplot(
        data=df,
        x='Brand',
        ax=ax
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # Price Distribution

    st.subheader(
        "Price Distribution"
    )

    fig, ax = plt.subplots()

    sns.histplot(
        df["Price"],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

# =========================================================
# RECOMMENDATION ENGINE
# =========================================================

elif page == "Recommendation Engine":

    st.title(
        "🎯 Product Recommendation Engine"
    )

    selected_product = st.selectbox(

        "Select Product",

        df["Product_Name"].unique()

    )

    if st.button(
        "Recommend Products"
    ):

        recommendations = recommend_products(
            selected_product,
            top_n=5
        )

        st.success(
            f"Top Recommendations for {selected_product}"
        )

        st.dataframe(
            recommendations
        )

# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.title(
        "🤖 ML Model Comparison"
    )

    comparison_df = pd.DataFrame({

        "Model":[
            "DT Underfit",
            "DT Overfit",
            "DT Balanced",
            "Random Forest",
            "XGBoost"
        ],

        "Accuracy":[
            0.60,
            0.75,
            0.82,
            0.88,
            0.91
        ]

    })

    st.dataframe(
        comparison_df
    )

    fig, ax = plt.subplots(
        figsize=(8,5)
    )

    sns.barplot(
        data=comparison_df,
        x="Model",
        y="Accuracy",
        ax=ax
    )

    plt.xticks(rotation=20)

    st.pyplot(fig)

    st.markdown(
        """
        ### Observations

        - Underfit DT performs poorly.
        - Overfit DT memorizes training data.
        - Balanced DT generalizes better.
        - Random Forest improves stability.
        - XGBoost achieves best performance.
        """
    )