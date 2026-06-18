import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

def render_clustering_view(
    overview_text: str,
    overview_images: list,
    prep_text: str,
    cleaning_code: str,
    df_sample: pd.DataFrame,
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
    df_sample : pd.DataFrame -> Scaled numeric-only preview dataset for Section (b).
    data_download_url : str -> Target dataset source string/link.
    kmeans_pipeline : list -> Visual pipeline dictionaries for the K-Means column.
    hclust_pipeline : list -> Visual pipeline dictionaries for the Hierarchical column.
    conclusions_text : str -> Final evaluation insights.
    """
    
    # ----------------------------------------------------
    # INTERNAL HELPER FUNCTION FOR PIPELINE/RESULTS RENDERING
    # ----------------------------------------------------
    def _render_pipeline_asset(item: dict, step_index: int, unique_key_prefix: str):
            """Processes and handles titles, figures, maps, dataframes, captions, and interpretations."""
            if item.get("title"):
                st.markdown(f"**{step_index + 1}. {item['title']}**")
                
            fig = item.get("fig")
            
            # dataframes
            if isinstance(fig, pd.DataFrame):
                st.dataframe(fig, width='stretch')
                # Render annotations if present for this DataFrame step
                if item.get("caption"):
                    st.caption(item["caption"])
                if item.get("interpretation"):
                    st.markdown(f"🔬 **Interpretation:** {item['interpretation']}")
                st.markdown("<br>", unsafe_allow_html=True)
                return

            # html maps
            if isinstance(fig, str) and fig.endswith(".html"):
                if os.path.exists(fig):
                    with open(fig, 'r', encoding='utf-8') as f:
                        html_data = f.read()
                    height = item.get("html_height", 460)
                    st.components.v1.html(html_data, height=height, scrolling=True)
                else:
                    st.error(f"Missing map component: {fig}")
                    
            # local paths
            elif isinstance(fig, str):
                st.image(fig, width='stretch')
                
            # plotly go
            elif isinstance(fig, go.Figure):
                st.plotly_chart(fig, width='stretch', key=f"{unique_key_prefix}_chart_{step_index}")
                
            # fallback
            else:
                st.pyplot(fig)
                
            # Standard rendering path for captions/interpretations of non-DataFrame assets
            if item.get("caption"):
                st.caption(item["caption"])
                
            if item.get("interpretation"):
                st.markdown(f"🔬 **Interpretation:** {item['interpretation']}")
                
            st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # (a) OVERVIEW SECTION (With flexible multi-image container)
    # ----------------------------------------------------
    st.subheader("Clustering Overview & Discovery")
    st.markdown(overview_text)
    
    if overview_images:
        # Automatically split the layout into columns depending on how many images are supplied
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
    # (b) DATA PREPARATION
    # ----------------------------------------------------
    st.subheader("Data Preparation & Requirements")
    st.markdown(prep_text)
    
    st.markdown(f"🔗 **[Download Sample Unlabeled Dataset]({data_download_url})**")
    st.markdown(f"🔍 **[View Cleaning Code]({cleaning_code})**")
    st.markdown("Numeric Input Feature Vector Preview")
    st.dataframe(df_sample, width='stretch')
    
    st.markdown("---")
    
    # ----------------------------------------------------
    # (c & d) CODE & RESULTS COMPARED SIDE-BY-SIDE
    # ----------------------------------------------------
    st.subheader("Model Run & Results")
    
    col_km, col_hc = st.columns(2)
    
    # Left Column: K-Means Execution Pipeline
    with col_km:
        st.markdown("#### Method 1: Partitional K-Means")
        for i, item in enumerate(kmeans_pipeline):
            _render_pipeline_asset(item, step_index=i, unique_key_prefix="km")
            
    # Right Column: Hierarchical Execution Pipeline
    with col_hc:
        st.markdown("#### Method 2: Cosine Hierarchical")
        for i, item in enumerate(hclust_pipeline):
            _render_pipeline_asset(item, step_index=i, unique_key_prefix="hc")
            
    st.markdown("---")
    
    # ----------------------------------------------------
    # (e) CONCLUSIONS
    # ----------------------------------------------------
    st.subheader("Conclusions")
    st.markdown(conclusions_text)