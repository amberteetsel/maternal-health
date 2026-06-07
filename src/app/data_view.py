import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Helper Function to Render Metrics
def render_metrics(metrics_dict=None):
    if not metrics_dict:
        return
    
    n = len(metrics_dict)
    cols = st.columns(n)

    for col, (key, metric_data) in zip(cols, metrics_dict.items()):
        with col:
            label = metric_data['label']
            value = metric_data['value']
            type = metric_data['type']

            delta = metric_data.get('delta')
            delta_color = metric_data.get('delta_color')
            help_txt = metric_data.get('help')

            if type == 'percentage':
                formatted_val = f"{value}%"
            elif type == 'rate':
                formatted_val = float(value)
            elif isinstance(value, (float, int)):
                formatted_val = float(value)
            else:
                formatted_val = str(value)

            if help_txt is not None:
                if delta is not None and delta_color is not None:
                    st.metric(label=label, value=formatted_val, delta=delta, delta_color=delta_color, help=help_txt)
                elif delta is not None:
                    st.metric(label=label, value=formatted_val, delta=delta, help=help_txt)
                else:
                    st.metric(label=label, value=formatted_val, help=help_txt)
            else:
                if delta is not None and delta_color is not None:
                    st.metric(label=label, value=formatted_val, delta=delta, delta_color=delta_color)
                elif delta is not None:
                    st.metric(label=label, value=formatted_val, delta=delta)
                else:
                    st.metric(label=label, value=formatted_val)

# Function to Display Data Exploration Results
def data_source_section(
        title,
        source_name: str,
        source_link: str,
        api_collect: bool,
        collection_method: str,
        description,
        raw,
        clean,
        cleaning_steps: dict=None,
        cleaning_code: str=None,
        api_code: str=None,
        visuals: dict=None,
        data_link: str=None,
        metrics: dict=None
):
    
    # Header
    st.subheader(title)

    # Source/Collection Info
    col_meta1, col_meta2 = st.columns(2)

    ## Source/Collection details
    with col_meta1:
        st.write(f"**Data Source:** [{source_name}]({source_link})")
        if data_link:
            st.write(f"**Collection Method:** {collection_method} - [Raw Data]({data_link})")
        else:
            st.write(f"**Collection Method:** {collection_method}")
        if api_collect:
            st.code(api_code, language='python', line_numbers=True)
    
    ## Data description
    with col_meta2:
        st.write(f"**Description:**")

        if isinstance(description, dict):
            for sub_title, sub_desc in description.items():
                st.markdown(f"#### {sub_title}")
                st.write(sub_desc)

        else:
            st.write(description)

    st.markdown("---")

    # Raw vs. Clean Snapshots
    col_raw, col_clean = st.columns(2)

    ## Raw
    with col_raw:
        st.write("🔍 **Raw Snapshot**")

        # DataFrames
        if isinstance(raw, pd.DataFrame):
            st.dataframe(raw, height=380)
            st.caption("Raw data types and values.")
            with st.expander("View Raw Schema"):
                st.code(raw.dtypes)

        # JSON
        elif isinstance(raw, (dict, list)):
            st.json(raw, expanded=False)
            st.caption("Raw JSON schema.")
            with st.expander("View Payload Object Type"):
                st.code(f"Type: {type(raw).__name__}\nKeys/Elements: {len(raw)}")

        # .txt
        elif isinstance(raw, str):
            preview_lines = "\n".join(raw.splitlines()[:15])
            st.code(preview_lines, language='text')
            st.caption("First 15 lines of raw character-spaced mainframe text block.")
            with st.expander("View Metadata Diagnostics"):
                st.code(f"Total Characters: {len(raw)}\nEstimated Rows: {len(raw.splitlines())}")


    ## Clean
    with col_clean:
        st.write("✨ **Processed Snapshot**")

        # DataFrames
        if isinstance(clean, pd.DataFrame):
            df_size_mb = clean.memory_usage(deep=True).sum() / (1024**2)

            if df_size_mb > 150:
                st.dataframe(clean.head(10000), height=380)
                st.caption(
                    f"⚠️ **Truncated Preview:** The full dataset is too large ({df_size_mb:.1f} MB) "
                    "to render smoothly in the browser. Showing the first 10,000 rows."
                )
            else:
                # Render normally for small datasets
                st.dataframe(clean, height=380)
                st.caption("Post-cleaning")

            with st.expander("View Processed Schema"):
                st.code(clean.dtypes)

    ## Cleaning Steps
    if cleaning_steps:
        with st.expander("Data Cleaning Summary", expanded=False):
            for step, text in cleaning_steps.items():
                st.markdown(f"**{step}:** {text}\n")
            if cleaning_code:
                st.write(f"[View Full Cleaning Code]({cleaning_code})")

    ## EDA Section
    st.write("📊 **Exploratory Data Analysis (EDA)**")

    if metrics:
        render_metrics(metrics)

    if visuals:
        st.markdown("---")
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            if "visual_1" in visuals:
                st.markdown(f"💡 **{visuals['visual_1']['title']}**")
                # image file paths
                if isinstance(visuals['visual_1']['fig'], str):
                    st.image(visuals['visual_1']['fig'])
                # interactive plotly figs
                elif isinstance(visuals['visual_1']['fig'], go.Figure):
                    st.plotly_chart(visuals['visual_1']['fig'])
                # fallback option
                else:
                    st.pyplot(visuals['visual_1']['fig'])
                
                st.caption(visuals['visual_1']['caption'])

        with col_v2:
            if "visual_2" in visuals:
                st.markdown(f"💡 **{visuals['visual_2']['title']}**")
                # image file paths
                if isinstance(visuals['visual_2']['fig'], str):
                    st.image(visuals['visual_2']['fig'])
                # interactive plotly
                elif isinstance(visuals['visual_2']['fig'], go.Figure):
                    st.plotly_chart(visuals['visual_2']['fig'])
                # fallback
                else:
                    st.pyplot(visuals['visual_2']['fig'])
                
                st.caption(visuals['visual_2']['caption'])


