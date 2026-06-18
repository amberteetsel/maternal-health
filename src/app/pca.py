#### HELPER FUNCTION TO DISPLAY PRINCIPAL COMPONENT ANALYSIS RESULTS

# Dependencies
import os
import pandas as pd
import numpy as np
import inspect
import streamlit as st
import plotly.graph_objects as go

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
pca_res = os.path.join(BASE_DIR, "resources", "pca")

# Function
def render_pca(overview: str,
               prep_text: str,
               cleaning_code: str,
               df_before: pd.DataFrame,
               df_after: pd.DataFrame,
               data_download_url: str,
               pipeline_all: list,
            #    pipeline_opt: list
               ):
    # ----------------------------------------------------
    # internal helper function for pipeline/results rendering
    # ----------------------------------------------------
    def _render_pipeline_asset(item: dict, step_index: int, unique_key_prefix: str):
        """Processes and handles titles, figures, maps, dataframes, captions, and interpretations."""
        if item.get("title"):
            # numbered index
            st.markdown(f"**{step_index + 1}. {item['title']}**")
            
        fig = item.get("fig")
        
        # check for dataframes first and exit early if matched
        if isinstance(fig, pd.DataFrame):
            st.dataframe(fig, width='stretch')
            # render annotations if present for this dataframe step
            if item.get("caption"):
                st.caption(item["caption"])
            if item.get("interpretation"):
                st.markdown(f"🔬 **Interpretation:** {item['interpretation']}")
            st.markdown("<br>", unsafe_allow_html=True)
            return  # exit function immediately for dataframes

        # check for interactive html maps
        if isinstance(fig, str) and fig.endswith(".html"):
            if os.path.exists(fig):
                with open(fig, 'r', encoding='utf-8') as f:
                    html_data = f.read()
                height = item.get("html_height", 460)
                st.iframe(src = html_data, height=height)
            else:
                st.error(f"Missing map component: {fig}")
                
        # check for local image paths
        elif isinstance(fig, str):
            # st.image(fig, width='stretch')
            st.image(fig, width='content')
            
        # check for interactive plotly go instances
        elif isinstance(fig, go.Figure):
            st.plotly_chart(fig, width='stretch', key=f"{unique_key_prefix}_chart_{step_index}")
            
        # fallback for matplotlib / seaborn figures only
        else:
            st.pyplot(fig)
            
        # standard rendering path for captions/interpretations of non-dataframe assets
        if item.get("caption"):
            st.caption(item["caption"])
            
        if item.get("interpretation"):
            st.markdown(f"🔬 **Interpretation:** {item['interpretation']}")
            
        st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # OVERVIEW
    # ----------------------------------------------------
    st.subheader("Principal Component Analysis (PCA) Overview")
    st.markdown(overview)

    # ----------------------------------------------------
    # DATA PREPARATION
    # ----------------------------------------------------
    st.subheader("Data Preparation")
    st.markdown(prep_text)

    st.markdown(f"🔗 **[Download Sample Unlabeled Dataset]({data_download_url})**")
    st.markdown(f"🔍 **[View Cleaning Code]({cleaning_code})**")
    # constructing two equal columns for side-by-side data comparison
    prep_col1, prep_col2 = st.columns(2)
    with prep_col1:
        st.markdown("📋 **Original Contextually Grounded Dataset**")
        st.dataframe(df_before, width='stretch')
        
        # schema info rendered within a clean layout expander
        with st.expander("🛠️ View Original Feature Schema Details", expanded=False):
            schema_df1 = pd.DataFrame({
                "Data Type": df_before.dtypes.astype(str),
                "Non-Null Count": df_before.notnull().sum(),
                "Null Count": df_before.isnull().sum()
            })
            st.table(schema_df1)

    with prep_col2:
        st.markdown("🔢 **Mathematical Input Feature Space (Strictly Unlabeled & Scaled)**")
        st.dataframe(df_after, width='stretch')
        
        with st.expander("🛠️ View Modeled Vector Space Schema Details", expanded=False):
            schema_df2 = pd.DataFrame({
                "Data Type": df_after.dtypes.astype(str),
                "Non-Null Count": df_after.notnull().sum(),
                "Null Count": df_after.isnull().sum()
            })
            st.table(schema_df2)
            
    st.markdown("---")

    # ----------------------------------------------------
    # MODELING & RESULTS
    # ----------------------------------------------------
    st.subheader("Model Run & Results")

    # st.markdown("#### Method 1: All Available Features")
    st.markdown(f"👾 [View Code]()")
    for i, item in enumerate(pipeline_all):
        _render_pipeline_asset(item, step_index=i, unique_key_prefix="all")

    # st.markdown("#### Method 2: Optimal Features Only")
    # st.markdown(f"👾 [View Code]()")
    # for i, item in enumerate(pipeline_opt):
    #     _render_pipeline_asset(item, step_index=i, unique_key_prefix="opt")

    # ----------------------------------------------------
    # CONCLUSION
    # ----------------------------------------------------
    st.subheader("Conclusions")

# ==========================================================
# ACTUAL INPUTS
# ==========================================================

overview_pca = inspect.cleandoc("""
    ##### The Curse of Dimensionality

    The Curse of Dimensionality refers to various challenges and complications that arise when analyzing and organizing
    data in high-dimensional spaces. Dimensions refer to the features or attributes of data; in the context of this study,
    dimensions of the Health Rankings dataset include measures of poverty, unemployment, preventative care, maternal
    mortality, and more. As the number of features grows, the volume of the data space increases exponentially and
    available data observations become sparse. In high-dimensional spaces traditional analytical models break down because
    geometric distances (e.g. Euclidean distance) are less distinct, meaning the distance between two highly
    dissimilar data points and the distance between two highly similar points look mathematically identical. Humans
    cannot visualize beyond three dimensions (3D) so it's also difficult for analysts to conceive of and understand
    high-dimensional feature spaces.

    ##### Dimensionality Reduction

    Dimensionality reduction is the process of compressing high-dimensional feature spaces into a lower-dimensional
    subspace, typically 2D or 3D, while retaining as much structural variation as possible. Dimensionality reduction
    is crucial to address the problems caused by high dimensionality:

    * Data sparsity
    * Increased computation
    * Multicollinearity
    * Overfitting
    * Performance degradation
    * Visualization challenges

    Reducing dimensionality can eliminate multicollinearity and background noise that destabilizes machine learning
    algorithms. By compressing the data, it drastically improves computational efficiency while solving the issues of data
    sparsity and data visualization by turning a multi-variable matrix into a clear, interpretable landscape in two or
    three dimensions.

    ##### Principal Component Analysis

    Principal Component Analysis (PCA) is an unsupervised linear transformation technique for dimensionality reduction
    that projects data onto brand-new, uncorrelated axes known as *Principal Components*. Rather than selecting or
    deleting specific raw variables, PCA uses all original features to calculate a completely new coordinate system.

    * **Principal Component 1 (PC1):** The first geometric axis, built to point in the direction of the absolute maximum
    variance in the data.
    * **Principal Component 2 (PC2):** The second geometric axis, constructed to be completely orthogonal to PC1 and 
    capture the highest remaining variance. 

    In this way, the first few components generated should contain the vast majority of a dataset's information (variance),
    allowing analysts to discard additional components wtih minimal information loss.

    ##### Eigenvalues and Eigenvectors

    In practice, PCA is based on the decomposition of a dataset's *covariance matrix*, which tracks how each feature
    moves in relation to every other feature. PCA uses linear algebra to extract critical information from this matrix:
    **Eigenvectors** and **Eigenvalues**.

    * **Eigenvectors** (The Direction): An eigenvector is a non-zero vector whose direction remains entirely unchanged
    when a linear transformation is applied. In PCA, eigenvectors represent the directions of the new principal component
    axes and are weighted/combined to build the new coordinate space.
    * **Eigenvalues** (The Magnitude): An eigenvalue is a scalar value that corresponds to a specific eigenvector. It
    measures the absolute magnitude of variance captured along that specific principal component axis. The principal
    component with the largest eigenvalue represents the axis of maximum variance, or PC1.
""")

prep_pca = inspect.cleandoc("""
    PCA requires continuous numerical variables.
    To this end, data from *America's Health Rankings* was further processed to conform with PCA requirements. First,
    the DataFrame was melted into wide format.
    Categorical data such as `State` and `Year` were stripped away to leave only numeric features.
    Features were also reconstructed so that for all columns, a higher number indicated a "worse" outcome than a lower number.
    For example, "Adequate Prenatal Care" was converted to "Inadequate Prenatal Care" by flipping the percentage.
    Z-Score standardization was also used so that varying data magnitudes would not disproportionately dominate the
    analysis.
""")

pca_df_after = pd.read_csv(os.path.join(pca_res, "pca_input_data.csv"))

pca_model_features_all = list(pca_df_after.columns)
pca_model_intro_all = """
    To begin, Principal Component Analysis was applied to the full set of 17 features:
    `{pca_model_features_all}`
"""
pca_all_scree_interpret = """
    Upon executing PCA across all available public health metrics, Principal Component 1 (PC1) and Principal Component 2 (PC2)
    were revealed to explain a cumulative total of only 45.18 percent of the dataset's variance. In public health analytics, 
    this low dimensionality compression indicates a high degree of systemic complexity. Rather than being driven by a 
    singular, underlying socioeconomic factor, state-level healthcare landscapes are deeply fragmented; a state's capacity
    in clinical provider retention does not linearly predict its performance in maternal care or preventative wellness 
    visits. To capture the standard 70% threshold of total system information, the model must be expanded to include 5 or
    6 components.
"""
pca_all_loading_interpet = inspect.cleandoc("""
                                            
    **Most Influential Features, PC1:**
                                            
    * Positive: `Maternal Mortality`, `Infant Mortality`, `Patients Per Doctor`, `Poverty`
    * Negative: None

    Principal Component 1 (PC1) is an axis of Systemic Healthcare Deprivation and Vulnerability. PC1 has no meaningful
    negative loading coefficients, indicating this component captures a unified, compounding directional force. The
    positive pole is driven by a mix of structural resource strain (poverty, patients per doctor) and severe medical
    failures (maternal and infant mortality). This makes PC1 a useful index of compounding public health dangers. States
    with high scores on this axis are stuck in a system where economic deprivation and shortage of healthcare providers
    directly relate to elevated risk of maternal and infant death. Conversely, states with low scores on this axis tend
    to have well-resourced, protective health infrastructure and better outcomes.

    **Most Influential Features, PC2:**
                                            
    * Positive: `No Postpartum Visit`, `Inadequate Prenatal Care`, `No Preventative`
    * Negative: `Smoking During Pregnancy`, `Gender Pay Gap`

    Principal Component 2 (PC2) can be characterized as an axis of Preventative Care Underutilization vs. Socio-Behavioral
    Risk. The positive pole is heavily influenced by clinical non-attendance metrics (no postpartum or preventative visits,
    inadequate prenatal care), capturing environments where system barriers prevent patients from utilizing available care.
    Conversely, the negative pole reveals an inverse relationship with `Smoking During Pregnancy` and the `Gender Pay Gap`.
    This means states suffering from high behavioral risks (smoking) and economic disparities (wage gaps) often exhibit
    *paradoxically higher* rates of prenatal and postpartum utilization.

    Notably, the feature `Maternity Care Desert` exhibits a nearly identical loading magnitude across both primary axes
    but in opposite directions (+0.297 for PC1, -0.278 for PC2). This indicates that `Maternity Care Desert` acts as a 
    structural pivot: it shares equal amount of variance with both components, moving in direct alignment with system
    traits of PC1 while simultaneously moving inversely with dynamics captured by PC2.
""")

pca_all_assets = [
    {
        'title': "Cumulative Variance Analysis",
        'fig': os.path.join(pca_res, "pca_scree_all.png"),
        'caption': 'Shows the cumulative variance captured by $n$ principal components. In this context, variance is a proxy for information',
        'interpretation': pca_all_scree_interpret
    },
    {
        'title': "Loadings Plot",
        'fig': os.path.join(pca_res, "pca_loading_all.png"),
        'caption': "Shows correlations between raw features and a principal component. A loading coefficient close to +1.0 or -1.0 means that feature has a massive influence on direction of component. A loading coefficient close to 0.0 means the feature has almost no impact on that component.",
        'interpretation': pca_all_loading_interpet
    }
]


pca_opt_assets = []

