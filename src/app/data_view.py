import streamlit as st
import pandas as pd

# Function to Display Data Exploration Results
def data_source_section(
        title,
        source_name: str, source_link: str,
        api_collect: bool,
        collection_method: str,
        description: str,
        raw, clean,
        cleaning_steps: dict=None,
        api_code: str=None,
        visuals: dict=None
):
    
    # Header
    st.subheader(title)

    # Source/Collection Info
    col_meta1, col_meta2 = st.columns(2)

    ## Source/Collection details
    with col_meta1:
        st.write(f"**Data Source:** [{source_name}]({source_link})")
        st.write(f"**Collection Method:** {collection_method}")
        if api_collect:
            st.code(api_code, language='python', line_numbers=True)
    
    ## Data description
    with col_meta2:
        st.write(f"**Description:** {description}")

    # Raw vs. Clean Snapshots
    col_raw, col_clean = st.columns(2)

    ## Raw
    with col_raw:
        st.write("🔍 **Raw Snapshot**")

        # DataFrames
        if isinstance(raw, pd.DataFrame):
            st.dataframe(raw.head(10), use_container_width=True)
            st.caption("Raw data types and values.")
            with st.expander("View Raw Schema"):
                st.code(raw.dtypes)

        # JSON

        # .txt

    ## Clean
    with col_clean:
        st.write("✨ **Processed Snapshot**")

        # DataFrames
        if isinstance(clean, pd.DataFrame):
            st.dataframe(clean.head(10), use_container_width=True)
            st.caption("Post-cleaning")
            with st.expander("View Processed Schema"):
                st.code(clean.dtypes)

