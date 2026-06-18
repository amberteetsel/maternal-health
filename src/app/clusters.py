# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# import os

# def render_clustering_view(
#     overview_text: str,
#     overview_images: list,
#     prep_text: str,
#     cleaning_code: str,
#     df_sample: pd.DataFrame,
#     data_download_url: str,
#     kmeans_pipeline: list,
#     hclust_pipeline: list,
#     conclusions_text: str
# ):
#     """
#     Renders a side-by-side clustering analysis view matching requirements (a)-(e).
    
#     Parameters:
#     -----------
#     overview_text : str -> Markdown explaining partitional vs hierarchical concepts.
#     overview_images : list -> List of dicts with keys 'fig' and optionally 'caption' for Section (a).
#     prep_text : str -> Explanation of numeric/unlabeled requirements.
#     df_sample : pd.DataFrame -> Scaled numeric-only preview dataset for Section (b).
#     data_download_url : str -> Target dataset source string/link.
#     kmeans_pipeline : list -> Visual pipeline dictionaries for the K-Means column.
#     hclust_pipeline : list -> Visual pipeline dictionaries for the Hierarchical column.
#     conclusions_text : str -> Final evaluation insights.
#     """
    
#     # ----------------------------------------------------
#     # INTERNAL HELPER FUNCTION FOR PIPELINE/RESULTS RENDERING
#     # ----------------------------------------------------
#     def _render_pipeline_asset(item: dict, step_index: int, unique_key_prefix: str):
#             """Processes and handles titles, figures, maps, dataframes, captions, and interpretations."""
#             if item.get("title"):
#                 st.markdown(f"**{step_index + 1}. {item['title']}**")
                
#             fig = item.get("fig")
            
#             # dataframes
#             if isinstance(fig, pd.DataFrame):
#                 st.dataframe(fig, width='stretch')
#                 # Render annotations if present for this DataFrame step
#                 if item.get("caption"):
#                     st.caption(item["caption"])
#                 if item.get("interpretation"):
#                     st.markdown(f"🔬 **Interpretation:** {item['interpretation']}")
#                 st.markdown("<br>", unsafe_allow_html=True)
#                 return

#             # html maps
#             if isinstance(fig, str) and fig.endswith(".html"):
#                 if os.path.exists(fig):
#                     with open(fig, 'r', encoding='utf-8') as f:
#                         html_data = f.read()
#                     height = item.get("html_height", 460)
#                     iframecsrc = omponents.v1.html(html_d=height, scrolling=True)
#                 else:
#                     st.error(f"Missing map component: {fig}")
                    
#             # local paths
#             elif isinstance(fig, str):
#                 st.image(fig, width='stretch')
                
#             # plotly go
#             elif isinstance(fig, go.Figure):
#                 st.plotly_chart(fig, width='stretch', key=f"{unique_key_prefix}_chart_{step_index}")
                
#             # fallback
#             else:
#                 st.pyplot(fig)
                
#             # Standard rendering path for captions/interpretations of non-DataFrame assets
#             if item.get("caption"):
#                 st.caption(item["caption"])
                
#             if item.get("interpretation"):
#                 st.markdown(f"🔬 **Interpretation:** {item['interpretation']}")
                
#             st.markdown("<br>", unsafe_allow_html=True)

#     # ----------------------------------------------------
#     # (a) OVERVIEW SECTION (With flexible multi-image container)
#     # ----------------------------------------------------
#     st.subheader("Clustering Overview & Discovery")
#     st.markdown(overview_text)
    
#     if overview_images:
#         # Automatically split the layout into columns depending on how many images are supplied
#         img_cols = st.columns(len(overview_images))
#         for idx, col in enumerate(img_cols):
#             with col:
#                 img_data = overview_images[idx]
#                 if isinstance(img_data["fig"], str):
#                     st.image(img_data["fig"], width='stretch')
#                 else:
#                     st.pyplot(img_data["fig"])
#                 if img_data.get("caption"):
#                     st.caption(img_data["caption"])
                    
#     st.markdown("---")
    
#     # ----------------------------------------------------
#     # (b) DATA PREPARATION
#     # ----------------------------------------------------
#     st.subheader("Data Preparation & Requirements")
#     st.markdown(prep_text)
    
#     st.markdown(f"🔗 **[Download Sample Unlabeled Dataset]({data_download_url})**")
#     st.markdown(f"🔍 **[View Cleaning Code]({cleaning_code})**")
#     st.markdown("Numeric Input Feature Vector Preview")
#     st.dataframe(df_sample, width='stretch')
    
#     st.markdown("---")
    
#     # ----------------------------------------------------
#     # (c & d) CODE & RESULTS COMPARED SIDE-BY-SIDE
#     # ----------------------------------------------------
#     st.subheader("Model Run & Results")
    
#     col_km, col_hc = st.columns(2)
    
#     # Left Column: K-Means Execution Pipeline
#     with col_km:
#         st.markdown("#### Method 1: Partitional K-Means")
#         for i, item in enumerate(kmeans_pipeline):
#             _render_pipeline_asset(item, step_index=i, unique_key_prefix="km")
            
#     # Right Column: Hierarchical Execution Pipeline
#     with col_hc:
#         st.markdown("#### Method 2: Cosine Hierarchical")
#         for i, item in enumerate(hclust_pipeline):
#             _render_pipeline_asset(item, step_index=i, unique_key_prefix="hc")
            
#     st.markdown("---")
    
#     # ----------------------------------------------------
#     # (e) CONCLUSIONS
#     # ----------------------------------------------------
#     st.subheader("Conclusions")
#     st.markdown(conclusions_text)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

def render_clustering_view(
    overview_text: str,
    overview_images: list,
    prep_text: str,
    cleaning_code: str,
    df_raw_sample: pd.DataFrame,    # raw dataset sample matrix (with labels)
    df_scaled_sample: pd.DataFrame, # pure numeric, scaled input matrix
    data_download_url: str,
    kmeans_pipeline: list,
    hclust_pipeline: list,
    conclusions_text: str
):
    """
    Renders a side-by-side clustering analysis view matching requirements (a)-(e).
    
    Parameters:
    -----------
    overview_text : str -> Markdown explaining partitional vs hierarchical concepts.
    overview_images : list -> List of dicts with keys 'fig' and optionally 'caption' for Section (a).
    prep_text : str -> Explanation of numeric/unlabeled requirements.
    cleaning_code : str -> Link or string pointing to data preprocessing/cleaning steps.
    df_raw_sample : pd.DataFrame -> Raw dataset matrix sample containing labels/strings.
    df_scaled_sample : pd.DataFrame -> Purely numeric, scaled input matrix used for modeling.
    data_download_url : str -> Target dataset source string/link.
    kmeans_pipeline : list -> Visual pipeline dictionaries for the K-Means column.
    hclust_pipeline : list -> Visual pipeline dictionaries for the Hierarchical column.
    conclusions_text : str -> Final evaluation insights.
    """
    
    # ----------------------------------------------------
    # internal helper function for pipeline/results rendering
    # ----------------------------------------------------
    def _render_pipeline_asset(item: dict, step_index: int, unique_key_prefix: str):
        """Processes and handles titles, figures, maps, dataframes, captions, and interpretations."""
        if item.get("title"):
            # removed numbered index prefixing to keep layout unnumbered
            st.markdown(f"**{item['title']}**")
            
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
    # (a) overview section (with flexible multi-image container)
    # ----------------------------------------------------
    st.subheader("Clustering Overview & Discovery")
    st.markdown(overview_text)
    
    if overview_images:
        # automatically split the layout into columns depending on how many images are supplied
        img_cols = st.columns(len(overview_images))
        for idx, col in enumerate(img_cols):
            with col:
                img_data = overview_images[idx]
                if isinstance(img_data["fig"], str):
                    st.image(img_data["fig"], width='stretch')
                else:
                    st.pyplot(img_data["fig"])
                if img_data.get("caption"):
                    st.caption(img_data["caption"])
                    
    st.markdown("---")
    
    # ----------------------------------------------------
    # (b) data preparation (side-by-side dataframes + schemas)
    # ----------------------------------------------------
    st.subheader("Data Preparation & Requirements")
    st.markdown(prep_text)
    
    st.markdown(f"🔗 **[Download Sample Unlabeled Dataset]({data_download_url})**")
    st.markdown(f"🔍 **[View Cleaning Code]({cleaning_code})**")
    
    # constructing two equal columns for side-by-side data comparison
    prep_col1, prep_col2 = st.columns(2)
    
    with prep_col1:
        st.markdown("📋 **Original Contextual Ground Dataset (With Metadata/Labels)**")
        st.dataframe(df_raw_sample, width='stretch')
        
        # schema info rendered within a clean layout expander
        with st.expander("🛠️ View Original Feature Schema Details", expanded=False):
            schema_df1 = pd.DataFrame({
                "Data Type": df_raw_sample.dtypes.astype(str),
                "Non-Null Count": df_raw_sample.notnull().sum(),
                "Null Count": df_raw_sample.isnull().sum()
            })
            st.table(schema_df1)

    with prep_col2:
        st.markdown("🔢 **Mathematical Input Feature Space (Strictly Unlabeled & Scaled)**")
        st.dataframe(df_scaled_sample, width='stretch')
        
        with st.expander("🛠️ View Modeled Vector Space Schema Details", expanded=False):
            schema_df2 = pd.DataFrame({
                "Data Type": df_scaled_sample.dtypes.astype(str),
                "Non-Null Count": df_scaled_sample.notnull().sum(),
                "Null Count": df_scaled_sample.isnull().sum()
            })
            st.table(schema_df2)
            
    st.markdown("---")
    
    # ----------------------------------------------------
    # (c & d) code & results compared side-by-side
    # ----------------------------------------------------
    st.subheader("Model Run & Results")
    
    col_km, col_hc = st.columns(2)
    
    # left column: k-means execution pipeline
    with col_km:
        st.markdown("#### Method 1: Partitional K-Means")
        for i, item in enumerate(kmeans_pipeline):
            _render_pipeline_asset(item, step_index=i, unique_key_prefix="km")
            
    # right column: hierarchical execution pipeline
    with col_hc:
        st.markdown("#### Method 2: Cosine Hierarchical")
        for i, item in enumerate(hclust_pipeline):
            _render_pipeline_asset(item, step_index=i, unique_key_prefix="hc")
            
    st.markdown("---")
    
    # ----------------------------------------------------
    # (e) conclusions
    # ----------------------------------------------------
    st.subheader("Conclusions")
    st.markdown(conclusions_text)