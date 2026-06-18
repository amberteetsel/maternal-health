#### HELPER FUNCTION TO DISPLAY PRINCIPAL COMPONENT ANALYSIS RESULTS

# Dependencies
import os
import pandas as pd
import numpy as np
import inspect
import streamlit as st

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
pca_res = os.path.join(BASE_DIR, "resources", "pca")

# Function
def render_pca(overview: str,
               prep_text: str,
               cleaning_code: str,
               df_before: pd.DataFrame,
               df_after: pd.DataFrame,
               data_download_url: str
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
            st.image(fig, width='stretch')
            
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
    # Overview
    # ----------------------------------------------------
    st.subheader("Principal Component Analysis (PCA) Overview")
    st.markdown(overview)

    # ----------------------------------------------------
    # Data preparation
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
    # Data preparation
    # ----------------------------------------------------
    st.subheader("Model Run & Results")

    # ----------------------------------------------------
    # Data preparation
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

pca_df_after = pd.read_csv(os.path.join(pca_res, "pca_input_data.csvs"))
