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
        model_code_url: str,
        result_summary: str,
        result_details: str,
        result_df: pd.DataFrame,
        result_viz: str,
        confusion_matrix: dict,
        conclusion_text: str,
        conclusion_df: pd.DataFrame,
        final_takeaway: str

):
    
    # Helper function to custom style the results DataFrame
    def highlight_cell(df):
        # empty dataframe to hold CSS strings
        style_df = pd.DataFrame('', index=df.index, columns=df.columns)
        row_condition = df['Kernel'] == 'rbf'
        target_column = 'Accuracy (Cost = 1.0)'
        style_df.loc[row_condition, target_column] = 'background-color: #e1f3dc; color: #2c944c; font-weight: bold;'

        row_condition_2 = df['Kernel'] == 'linear'
        style_df.loc[row_condition_2, target_column] = 'color: black; font-weight: bold;'

        row_condition_3 = df['Kernel'] == 'poly'
        target_column_2 = 'Accuracy (Cost = 10.0)'
        style_df.loc[row_condition_3, target_column_2] = 'color: black; font-weight: bold;'

        return style_df

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
    st.dataframe(sample_data, width='stretch', hide_index=True)
    st.markdown("##### Training vs. Testing Split")
    st.markdown(train_test_text)
    ctrain, ctest = st.columns(2)
    with ctrain:
        st.markdown("**Training Data**")
        st.dataframe(train_data, hide_index=True)
    with ctest:
        st.markdown("**Testing Data**")
        st.dataframe(test_data, hide_index=True)

    st.markdown("---")

    # Code/Results
    st.subheader("Model Results")
    st.markdown(f"👾 [View Code]({model_code_url})")
    st.markdown(f"##### Kernel Performance & Cost Analysis")
    st.write(result_summary)
    styled_df = (result_df.style.apply(highlight_cell, axis=None)).format({
        "Accuracy (Cost = 0.1)": lambda x: f"{x*100:.2f}%",
        "Accuracy (Cost = 1.0)": lambda x: f"{x*100:.2f}%",
        "Accuracy (Cost = 10.0)": lambda x: f"{x*100:.2f}%"
    })
    st.dataframe(styled_df, hide_index=True)
    st.image(result_viz, width='content')
    st.write(result_details)
    cols = st.columns(3)

    for col, (kernel_name, item) in zip(cols, confusion_matrix.items()):
        with col:
            st.markdown(f"##### {kernel_name}")
            st.metric(
                label="Accuracy (Cost = 1.0)",
                value = f"{item['c1_acc']*100:.2f}%"
            )
            st.image(item['cm'])

    # Conclusions
    st.subheader("Conclusions")
    st.markdown(conclusion_text)
    st.dataframe(conclusion_df, hide_index=True)
    st.markdown(final_takeaway)


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

    The dot product ${x_i} * {x_j}$ between vectors $i$ and $j$ is key because the mathematics of SVMs rely on *only* the dot
    products between vectors, not the individual coordinates, allowing analysts to bypass the actual coordinate transformation.

    **Common Kernel Functions**

    * Polynomial Kernel

    $$K({x}, {z}) = ({x} * {z} + r)^d$$

    * Radial Basis Function (RBF) Kernel

    $$K({x}, {z}) = \exp(-\gamma \|{x} - {z}\|^2)$$
"""
overview_pipeline_svm = {
    "Linear Separators and the Maximum Margin": {
        'text': overview_text_svm_2,
        'fig': os.path.join(svm_rec, "svm_viz_hyperplane.png"),
        'caption': "Hyperplanes as binary classification decision boundaries."
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
    comparison point to evaluate its performance against. For SVM, features must also be standardized/scaled, and encoded if
    categorical. SVMs rely on calculating geometric distances and dot products between vectors, concepts which do not exist for
    unnumeric categorical data.

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
    The feature columns in sample training/testing sets below have been z-score normalized. 
"""
train_data_svm = pd.read_csv(os.path.join(svm_rec, "train_display.csv"))
test_data_svm = pd.read_csv(os.path.join(svm_rec, "test_display.csv"))

# ==============================================================================
# Code
# ==============================================================================
model_code_url_svm = "https://github.com/amberteetsel/maternal-health/blob/1e46a07d0df0bee1839984d5195185808c28472e/src/models/svm_model.py"

# ==============================================================================
# Results
# ==============================================================================
result_summary_svm = """
    Of the three kernels tested, the RBF Kernel had the best performance when classifying states by abortion policy
    based on maternal health metrics. Its peak accuracy is 100$ with cost of 10.0, and it achieved 94.74% accuracy with
    a more moderate cost of 1.0. The Linear kernel's accuracy peaks at 89.47% (cost of 1.0), and the Poly kernel's accuracy
    peaks at 94.74% (cost of 10.0).
"""
result_df_svm = pd.read_csv(os.path.join(svm_rec, 'accuracy_table_svm.csv'))
result_plot_svm = os.path.join(svm_rec, "svm_accuracy_comparison.png")
result_details_svm = """
    The kernel performance reveals a complex, non-linear data structure. The linear kernel peaks at cost $C$ = 1.0 with 89.47%
    accuracy but drops back down to 84.21% at $C$ = 10.0. This indicates that an overly aggressive, strict linear boundary
    forces overfitting: a rigid straight line cannot perfectly separate states into classes without error.

    The RBF kernel with $C$ of 10.0 was a flawless classifier with 100% accuracy for the test set. However, this metric
    must be interpreted with caution. In machine learning, perfect accuracy often signals overfitting, where a high cost parameter
    ($C$ = 10.0) forces the model to create a hypercomplex, rigid boundary tailored tightly to the training data. Despite this
    overfitting risk, the RBF kernel achieved an impressive 94.74% accuracy at a much lower cost ($C$ = 1.0). This proves that
    the underlying separation between restrictive and non-restrictive states is structurally real, robust, and not merely a
    result of an overfitted model. All models were ultimately evaluated at cost of 1.0 for comparison's sake.
"""

c1_linear_acc = result_df_svm.loc[result_df_svm.Kernel=='linear', 'Accuracy (Cost = 1.0)'].values[0]
c1_poly_acc = result_df_svm.loc[result_df_svm.Kernel=='poly', 'Accuracy (Cost = 1.0)'].values[0]
c1_rbf_acc = result_df_svm.loc[result_df_svm.Kernel=='rbf', 'Accuracy (Cost = 1.0)'].values[0]
confusion_matrices_svm = {
    'Linear': {
        'cm': os.path.join(svm_rec, "best_linear_cm.png"),
        'c1_acc': c1_linear_acc
        },
    'Poly': {
        'cm': os.path.join(svm_rec, "best_poly_cm.png"),
        'c1_acc': c1_poly_acc
        },
    'RBF': {
        'cm': os.path.join(svm_rec, "best_rbf_cm.png"),
        'c1_acc': c1_rbf_acc,
        }
}

# ==============================================================================
# Conclusions
# ==============================================================================
conclusion_text_svm = """
    ##### High Accuracy → Structural Polarization

    When a machine learning model achieves 95% test accuracy without overfitting, it means the boundary separating
    the classes is wide and clear. In the context of this study, it indicates that states with abortion restrictions
    and states without abortion restrictions have vastly different healthcare realities. The model is able to examine an 
    anonymous state's health statistics, such as `Maternity Care Desert`, `Inadequate Prenatal Care`, `Unintended Pregnancy`,
    and `Patients Per Doctor`, and then "know" its political/legal landscape with regards to abortion.

    ##### Feature Interpretation

    Examining the mean values of feature by class (Abortion_Restricted = 0/1) proves that a state's abortion policy is
    related to distinct, measurable disparities in maternal and infant health outcomes. The table below illustrates this;
    with one exception, **states with legal restrictions on abortion have worse health metrics across the board**. States that
    restrict abortion have more maternity care deserts, higher infant and maternal mortality, and fewer people receiving
    adequate prental, preventative, and postpartum care.
    
    Note that features have been engineered such that a higher relative value always means a worse outcome.
"""
feature_mean_df = pd.read_csv(os.path.join(svm_rec, "feature_means.csv"))
feature_mean_df.index.name == "Health Metric"
final_takeaway_svm = """
    Whether abortion restrictions directly degrade the healthcare system or whether the political landscape that enacts
    restrictions simultaneously underfunds public health, the mathematical conclusion is concrete:

    **A state's legal environment with regard to abortion is inseparable from the actual health outcomes for real mothers.**
"""