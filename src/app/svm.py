import os
import streamlit as st
import pandas as pd

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
svm_rec = os.path.join(BASE_DIR, "resources", "svm")
if not os.path.exists(svm_rec):
    os.makedirs(svm_rec)

def render_svm(
        overview_text: str,
        overview_pipeline: dict,
        sample_data_url: str,       # main dataprep DF link
        data_reqs: str,
        sample_data: pd.DataFrame,  # main dataprep DF
        train_test_text: str,       # explanation of train test split
        train_data: pd.DataFrame,   # training DF
        test_data: pd.DataFrame,    # testing DF
        model_code_url: str
):

    # Overview
    st.subheader("SVM Overview")
    with st.expander("Read About SVM", expanded=True):
        st.write(overview_text)
        for title, item in overview_pipeline.items():
            st.markdown(f"##### {title}")  # use dict key as header
            st.write(item['text'])
            if item['fig']:
                st.image(item['fig'])
                if item['caption']:
                    st.caption(item['caption'])

    # Data Prep
    st.subheader("Data Preparation")
    st.markdown(f"🔗 **[Download Sample Dataset]({sample_data_url})**")
    st.markdown("##### Data Requirements")
    st.write(data_reqs)
    st.dataframe(sample_data, width='stretch')
    st.markdown("##### Training vs. Testing Split")
    st.write(train_test_text)
    ctrain, ctest = st.columns(2)
    with ctrain:
        st.markdown("**Training Data**")
        st.dataframe(train_data)
    with ctest:
        st.markdown("**Testing Data**")
        st.dataframe(test_data)

    st.markdown(train_test_text)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Training Dataset**")
        st.dataframe(train_data, width='stretch')
    with c2:
        st.markdown("**Testing Dataset**")
        st.dataframe(test_data, width='stretch')
    st.markdown("---")

    # Code/Results
    st.subheader("Model Results")
    st.markdown(f"👾 [View Code]({model_code_url})")

    # Conclusions
    st.subheader("Conclusions")

# ==============================================================================
# Overview
# ==============================================================================
overview_text_svm = """
    A **Support Vector Machine (SVM)** is a supervised machine
    learning algorithm primarily used for classification. The fundamental
    goal of an SVM is to find an optimal *hyperplane* that separates
    data points of one class from another. This "maximal margin hyperplane"
    optimally separates classes in order to generalize to new data and
    make accurate classification predictions.
"""
overview_text_svm_2 = """
    SVMs are known as linear separators because, in their simplest form,
    they find a straight line (in 2D), a flat plane (in 3D), or a linear
    hyperplane (in higher dimensions) to divide classes of data points.

    * SVM specifically finds the boundary that maximizes the distance
    between the boundary and the closest data points from each class
    * These closest points are called "support vectors" and help define
    the boundary; if a support vector changes, so does the boundary
"""
overview_text_svm_3 = """
    In higher dimensions, SVMs are much harder to interpret: humans
    struggle to visualize how the data can be linearly separable and what
    an optimal decision boundary will look like. Compounding the problem
    is that in practice, most data is not linearly separable. In these cases
    it is necessary to apply transformations to the data which map it
    from the original feature space into a higher dimensional feature space
    where classes are linearly separable. The decision boundary will then be
    a hyperplane in the higher-dimensional space.
"""
overview_text_svm_4 = """
    In real applications, data might have many features and require transformations involving many polynomial combinations of
    those features. This leads to extremely high and impractical computation costs. The **Kernel Trick** solves this problem:
    instead of manually transforming data into a massive higher-dimensional space, a kernel function computes the similarity or
    **dot product** between points as if they were in that higher-dimensional space.

    The dot product $\mathbf{x_i} \cdot \mathbf{x_j}$ between vectors $i$ and $j$ is key because the mathematics of SVMs rely on *only* the dot
    products between vectors, not the individual coordinates, allowing analysts to bypass the actual coordinate transformation.

    **Common Kernel Functions**

    * Polynomial Kernel

    $$K(\mathbf{x}, \mathbf{z}) = (\mathbf{x} \cdot \mathbf{z} + r)^d$$

    * Radial Basis Function (RBF) Kernel

    $$K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$$
"""
overview_pipeline_svm = {
    "Linear Separators and the Maximum Margin": {
        'text': overview_text_svm_2,
        'fig': None,
        'caption': None
    },
    "Dimensionality and Transformations": {
        'text': overview_text_svm_3,
        'fig': os.path.join(svm_rec, "svm_viz_1.png"),
        'caption': "This data becomes linearly separable after a quadratic transformation to 2-dimensions."
    },
    "The Kernel Trick": {
        'text': overview_text_svm_4,
        'fig': None,
        'caption': None
    }
}

# ==============================================================================
# Data Prep
# ==============================================================================
sample_data_url_svm = "https://github.com/amberteetsel/maternal-health/blob/697486b29943ab2450fa1d95ad0fc38b528246fa/resources/svm/svm_preprocessed.csv"
data_reqs_svm = """
    Supervised modeling requires **labeled** data, meaning every row must have a known target outcome (`Abortion_Restricted`).
    The model uses these labels to learn the patterns that differentiate classes. Without labels, the algorithm has no
    comparison point to evaluate its performance against. For SVM, features must also be standardized/scaled.

    The target variable `Abortion_Restricted` is a binary flag for whether a certain U.S. state restricts abortion or not.
    States with total bans or gestational bans up to 20 weeks are considered restrictive and indicated by `Abortion_Restriction` = 1.
"""
sample_data_svm = pd.read_csv(os.path.join(svm_rec, "svm_preprocessed.csv"))
train_test_text_svm = """
    To properly evaluate the model, the labeled data is partitioned into two sets:

    * **Training Set:** Used by SVM algorithm to calculate optimal hyperplane

    * **Testing Set:** Kept completely hidden from model during training. It acts as evaluation tool to measure how well
    the model generalizes to unseen data.

    The training and testing sets are disjoint (non-overlapping) because otherwise the model can simply 'memorize' the test
    data answers rather than learning general patterns. This would lead to overfitting and poor model performance on unseen data.
    The feature columnsn in sample training/testing sets below have been z-score normalized. 
"""
train_data_svm = pd.read_csv(os.path.join(svm_rec, "train_display.csv"))
test_data_svm = pd.read_csv(os.path.join(svm_rec, "test_display.csv"))



# ==============================================================================
# Code
# ==============================================================================
model_code_url_svm = os.path.join(BASE_DIR, "src", "models", "svm.py")



# ==============================================================================
# Results
# ==============================================================================




# ==============================================================================
# Conclusions
# ==============================================================================