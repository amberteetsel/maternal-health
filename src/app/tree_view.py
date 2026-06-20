### HELPER TO DISPLAY DECISION TREE CLASSIFICATION RESULTS

# Dependencies
import os
import pandas as pd
import numpy as np
import inspect
import streamlit as st

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
nb_res = os.path.join(BASE_DIR, "resources", "nbayes")
tree_res = os.path.join(BASE_DIR, "resources", "trees")

# ==========================================================
# DISPLAY FUNCTION
# ==========================================================
def render_dt(
        overview_text: str,
        # overview_cols: list,

        # placeholder space for rest of intro

        prep_text: str,
        cleaning_code_url: str,
        sample_data_url: str,
        sample_data: pd.DataFrame,
        train_test_text: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,


        ):
    
    # Overview
    st.subheader("Decision Tree Overview")
    with st.expander("Read About Decision Trees", expanded=True):
        st.markdown(overview_text)

    # Data Preparation
    st.subheader("Data Preparation")
    st.markdown(prep_text)

    st.markdown(f"🔗 **[Download Sample Dataset]({sample_data_url})**")
    st.markdown(f"🔍 **[View Cleaning Code]({cleaning_code_url})**")
    st.dataframe(sample_data, width='stretch')

    st.markdown(train_test_text)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Training Dataset")
        st.dataframe(train_data, width='stretch')
    with c2:
        st.markdown("Testing Dataset")
        st.dataframe(test_data, width='stretch')
    st.markdown("---")

# ==========================================================
# ACTUAL INPUTS
# ==========================================================

# Overview
overview_dt = """
    A Decision Tree (DT) is a non-parametric supervised learning algorithm used for both classification and regression.
    It structures complex data into a flowchart-like architecture of sequential, logical rules. Starting at a single "root"
    node, the tree evaluates features based on specific thresholds, splits data into "branching" paths that lead
    down to inernal nodes, and finally terminates at "leaf" nodes that represent final classs predictions.

    Decision Trees mimic human decision-making, and can handle both numeric and categorical data without requiring scaling.
    This makes them broadly useful for a variety of classification and regression tasks across a wide range industries.
    In a clinical setting, DTs help healthcare professionals diagnose patients by creating rule-based pathways (e.g. 
    *If Blood Pressure > 150 AND Age > 45, flag patient as High Risk*).

    It is generally possible to create an infinite number of unique Decision Trees from a single dataset. For any
    continuous numerical feature, the algorithm must choose an exact point to split on (e.g. 20 vs. 20.5 vs. 20.7, etc.)
    which would each yield a structurally distinct tree. Hyperparameters can also be adjusted to control how many times
    a tree splits and change information gain thresholds. In ensemble methods like Random Forest, each tree is built using
    a randomized subset of data samples and a randomized subset of features. There are millions of ways to randomly
    sample rows and columns, and each way will result in a unique tree.
"""
gini_descript = """
    Measures the probability that a randomly chosen element from a node would be incorrectly labeled if it were
    randomly labeled according to the class distribution in that node. With a binary target class, Gini Impurity
    ranges from 0 (perfectly pure) to 0.5 (equally split between two classes).
"""
entropy_descript = """
    Measures the degree of disorder, randomness, or uncertainty in a node. It ranges from 0 (perfectly pure) to 1.0
    (maximum uncertainty).
"""
info_gain_descript = """
    Metric used to decide the actual "goodness" of a split. It calculates the net reduction in impurity (using Gini or 
    Entropy) achieved by splitting a parent node into smaller child nodes. The DT algorithm tests every possible split
    and chooses the one that maximizes Information Gain.
"""
goodness_of_fit = [
    {
        'title': "Gini Impurity",
        'description': gini_descript
    },
    {
        'title': 'Entropy',
        'description': entropy_descript
    },
    {
        'title': 'Information Gain',
        'description': info_gain_descript
    }
]

# Data Preparation
cleaning_code_dt = "https://github.com/amberteetsel/maternal-health/blob/8c2f89b8fbb9202b46e08dc31e631522d8d9850a/src/cleaning/birth_preproccessing.py"
sample_data_url = "https://github.com/amberteetsel/maternal-health/blob/8c2f89b8fbb9202b46e08dc31e631522d8d9850a/data/clean/NCHS-Birth/birth_icu_processed.csv"
data_prep_dt = """
    Supervised learning models require **labeled** data. In this case, the label is `ICU_Admit` with values Yes or No.
    Preparing Birth data files for Decision Tree classification starts with paring down the availabe 103,811,757 entries
    to a manageable sample while mitigating class imbalance. Only about 0.4 percent of births from 2018 - 2024 involved
    maternal ICU admission, so the extraction pipeline was configured to retrive all rows where `ICU_Admit = Yes` and
    just 1 percent of rows where `ICU_Admit = No`. The resulting dataset has 303,565 entries with 15 percent representing
    cases where the mother was admitted to the ICU. After handling missing or unknown data, continuous features are
    binned into distinct categories. Feature columns are separated from the target column (`ICU_Admit`), then all
    features are encoded (text values are mapped to numbers).
"""
input_data_dt = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "NCHS-Birth", "birth_icu_processed.csv"))
train_test_dt = """
    After encoding, the data is split into training and testing sets. The model will be fitted using training data, then
    used to make predictions about the testing data. These predictions are compared against real values to evaluate
    model performance. The training and testing sets must be entirely disjoint because otherwise the model will simply
    memorize answers instead of learning underlying patterns. Training and testing sets for this data were split using
    stratification to ensure each receives a proportional slice of positive ICU cases.
"""
train_data = pd.read_csv(os.path.join(nb_res, "X_train_nb.csv"))
test_data = pd.read_csv(os.path.join(nb_res, "X_test_nb.csv"))

# Code

# Results

# Conclusion