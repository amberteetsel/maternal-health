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
        overview_cols: list,
        # placeholder space for rest of intro
        prep_text: str,
        cleaning_code_url: str,
        sample_data_url: str,
        sample_data: pd.DataFrame,
        train_test_text: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        model_code_url: str,
        tree_list: list,
        tree_pics: list,
        performance_stats: pd.DataFrame,
        conclusion_text: str
        ):
    
    # Helper Function: Render Goodness of Fit Measures
    def _render_gof(metrics_list=None):
        if not metrics_list:
            return
        
        n = len(metrics_list)
        cols = st.columns(n)

        for col, item in zip(cols, metrics_list):
            with col:
                label = item.get('title')
                words = item.get('description')

                st.markdown(f"##### {label}")
                st.write(words)

    # Helper Function: Render Three Trees
    def _render_trees(tree_list=None):
        if not tree_list:
            return
        
        n = len(tree_list)
        cols = st.columns(n)

        for col, item in zip(cols, tree_list):
            cm = item.get("cm")
            descript = item.get('text')
            with col:
                st.markdown(f"##### {item.get('title')}")
                st.write(f"Max Depth = {item.get('max_depth')}")
                st.image(cm, width='stretch')
                st.write(descript)


    # Overview
    st.subheader("Decision Tree Overview")
    with st.expander("Read About Decision Trees", expanded=True):
        st.markdown(overview_text)

        # goodness of fit metrics
        _render_gof(overview_cols)

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

    # Modeling & Results
    st.subheader("Model Results")
    st.markdown(f"👾 [View Code]({model_code_url})")

    st.table(performance_df_dt)

    _render_trees(tree_list)

    with st.expander("View Trees", expanded=True):
        for pic in tree_pics:
            st.image(pic, width='stretch')

    # Conclusion
    st.subheader("Conclusions")
    st.markdown(conclusion_text)

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
# !!!!! need an example of information gain/goodness of fit calculation

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
model_code_url_dt = "https://github.com/amberteetsel/maternal-health"

# Results
shallow_tree_image = os.path.join(tree_res, "shallow_tree.png")
medium_tree_image = os.path.join(tree_res, "medium_tree.png")
tree_images = [shallow_tree_image, medium_tree_image]

cm_shallow_interpret = """
    This tree is highly compact, containing 8 terminal leaf nodes. The root split starts on `diabetes`; if `diabetes` is
    absent it moves to evaluate `perineal_lac` and `antibiotics`.

    Because this tree is highly contrained, it uses broad, sweeping splits to partition the data. This results in
    a very high sensitivity, catching 85.29% of all ICU admissions. However, its crude logic rules function like an
    overly wide net, dragging overall accuracy down to 68.64% and resulting in a massive amount of False Positives (17,663).
"""
cm_medium_interpret = """
    This tree expands to 32 terminal leaf nodes. While the trunk is still heavily anchored by `diabetes` and `perineal_lac`,
    the branch layers begin introducting demographic and care factors like `binned_age` (maternal age) and `binned_visits`
    (prenatal care visits).

    At a depth of 5, the tree finds highly reliable, convervative rule pathways. It drastically eliminates false alarms by
    dropping False Positives from over 17,000 to just 5,966. This optimizes overall baseline accuracy to its peak at
    84.49%. However, this comes at a cost: the model has a significantly higher clinical blind spot, missing 3,449 true
    ICU cases.
"""
cm_deep_interpret = """
    This is a massive, complex, multi-branched tree. It contains hundreds of distinct leaf clusters, splitting deep into
    features like `eclampsia`, `anesthesia`, and `pay_source`. It handles the non-linear realities of maternal medicine
    better than any single-rule baseline.

    The deep tree yields the best overall mathematical framework for this task, achieving the highest global discriminative
    ability with ROC AUC of 0.8579. By diving 10 layers deep, it successfully maps complex, overlapping pathologies (e.g. 
    matching combinations of diabetes, eclampsia, and age). It recovers the clinical utility lost by the medium tree,
    driving True Positives back up to 7,068 and recall up to 75.51%. Overall accuracy remains close to 80%.
"""

three_trees = [
    {
        'title': "Shallow Tree",
        'max_depth': 3,
        'cm': os.path.join(tree_res, "dt_cm_shallow.png"),
        'text': cm_shallow_interpret
    },
    {
        'title': "Medium Tree",
        'max_depth': 5,
        'cm': os.path.join(tree_res, "dt_cm_medium.png"),
        'text': cm_medium_interpret
    },
    {
        'title': "Deep Tree",
        'max_depth': 10,
        'cm': os.path.join(tree_res, "dt_cm_deep.png"),
        'text': cm_deep_interpret
    }
]
performance_df_dt = pd.read_csv(os.path.join(tree_res, "dt_results.csv"))

# Conclusion
conclusion_text_dt = """
    * **Pathology Overlap**: The superior performance of the deepest tree supports previous findings that maternal morbidity
    is not the result of a few simple factors. Instead, maternal health outcomes are driven by compounding interactions
    of pre-existing risk factors and acute clinical realities (such as delivery complications like perineal lacerations).

    * **Strategic Sampling**: In public health datasets, severe or life-threatening outcomes are (thankfully!) very rare.
    In practice, this means that basic machine learning models will over-optimize for the healthy majority class and fail
    in real-world applications. The intentional use of stratified downsampling in this study was the single most impactful
    data engineering choise and enabled the models to actually learn distinctive profiles of the at-risk minority
    class.

    * **Measuring Success**: These results demonstrate that when human lives are at risk, **Recall** is the primary
    metric of success, not overall accuracy. Minimizing false negatives (missing that a mother
    needs ICU admission) is the priority, even if it results in false alarms (sending a healthy mother to the ICU).

    * **Limitations**: Unfortunately, NCHS birth records do not contain state-level information past the 1970s. It would be
    interesting and more relevant to the goal of this study to overlay state-level maternal health policies and examine
    how they can be used as features to predict maternal morbidity.
"""