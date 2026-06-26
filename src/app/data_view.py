import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stacked_maps import generate_stacked_us_maps
from policy_maps import create_ban_limit_map, create_protection_map

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
            metric_type = metric_data['type']

            delta = metric_data.get('delta')
            delta_color = metric_data.get('delta_color')
            help_txt = metric_data.get('help')

            if metric_type == 'percentage':
                formatted_val = f"{value:.1f}%"
            elif metric_type == 'rate':
                formatted_val = f"{value:.1f}"
            elif isinstance(value, (float, int)):
                formatted_val = f"{value:.2f}"
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
    st.markdown("---")
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


# ==========================================================
# ACTUAL INPUTS
# ==========================================================
# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
viz_path = os.path.join(BASE_DIR, "resources", "visuals_eda")

# ================= EMERGENCY ROOM (CDC-ER) =========================
er_v1 = os.path.join(viz_path, 'er_v1.png')
er_v2 = os.path.join(viz_path, 'er_v2.png')
title_er = "National Hospital Ambulatory Medical Care Survey (NHAMCS)"
source_name_er = "National Center for Health Statistics (NCHS)"
source_link_er = "https://www.cdc.gov/nchs/nhamcs/documentation/about-the-data-2018.html"
api_collect_er = False
collection_method_er = "Direct Download (Stata files)"
description_er = """
    The National Hospital Ambulatory Medical Care Survey (NHAMCS) collected data about medical services for patients 
    who were treated in hospital emergency and outpatient departments. 
    Healthcare professionals call these services ambulatory medical care.
    Ambulatory surgery centers provide same-day surgeries without admitting patients overnight.
    During years of interest (2018-2022), NHAMCS only collected data about ambulatory visits to emergency departments.
    Visits were filtered to focus only on pregnancy-related emergency room data.
"""
er_visuals={}
er_visuals['visual_1'] = {
    'title': "Top 5 Primary Reasons for Pregnancy ER Visits",
    'fig': er_v1,
    'caption': "Distribution of most common Reasons for Visit (RFV) parsed from pregnancy-related emergency records (2018-22)."
}
er_visuals['visual_2'] = {
    'title': 'Top 5 Primary Diagnoses for Pregnancy ER Visits',
    'fig': er_v2,
    'caption': "Distribution of most common diagnoses parsed from pregnancy-related emergency records (2018-22)."
}
cleaning_steps_er = {
    'Multi-Year Data Ingestion': 'Loaded several annual raw Stata datasets (.dta) spanning 2018 to 2022',
    'ICD-10-CM Standard Mapping': 'Loaded external code index file and parsed line-by-line to construct mapping of codes to their descriptions',
    'Metadata Extraction': "Bundled structured categorical label configurations for each annual file's metadata dictionaries",
    'Data Consolidation': "Compiled individual annual files to consolidated, clean flat CSV baseline file"
}

# ========== PREGNANCY/BIRTH/ABORTION (GUTTMACHER) ==================
preg_v1 = os.path.join(viz_path, "preg_v1.png")
preg_v2 = os.path.join(viz_path, "preg_v2.png")
title_preg = "Pregnancies, Births and Abortions in the United States: National and State Trends by Age"
source_name_preg = "Guttmacher Institute"
source_link_preg = "https://osf.io/kthnf/overview"
api_collect_preg = False
collection_method_preg = "Direct Download (.csv file)"
description_preg = """
    A data set of comprehensive historical statistics on the incidence of pregnancy, birth, abortion, and miscarriage for people 
    of all reproductive ages in the United States. National statistics cover the period from 1973 to 2020, the most 
    recent year for which comparable data are available; state-level statistics are for selected years from 1988 to 2020.
    Rate data is per 100,000 population.
"""
preg_visuals={}
preg_visuals['visual_1'] = {
    'title': "Historic Reproductive Outcome Shifts (1988-present)",
    'fig': preg_v1,
    'caption': """
        Comparing overall rates (per 1,000 women) of pregnancy, birth, abortion, and miscarriage.
        Note the declining birth rate beginning in 2007.
        """
}
preg_visuals['visual_2'] = {
    'title': "Mean Historic Abortion and Miscarriage Rates (1988-present)",
    'fig': preg_v2,
    'caption': """
        Comparing rates of abortion and miscarriage (per pregnancy) across maternal age cohorts.
        Incidence of miscarriage is fairly consistent across age groups, whereas abortion is more prevalent among very young mothers
        and mothers over 40 years of age.
        """
}
cleaning_steps_preg = {
    'Wide-to-Long Structural Transformation': 'Restructured historical metrics by transforming broad, wide-format table into a uniform long-format dataset',
    'Label Standardization': 'Standardize raw codes into uniform category names',
    'Normalization': 'Compute metric rates per 100,000 population'
}

# ====================== POLICY (LAWATLAS) ==========================
### Policy Data
policy_datasets = {
    "Post-Dobbs State Abortion Restrictions and Protections": 
        "This dataset provides a high-level overview of state abortion restrictions and protections "
        "enacted post-Dobbs, tracking key legal developments from June 1, 2022, through June 1, 2023.",
        
    "Restrictions on Public Funding of Abortion": 
        "This longitudinal dataset explores abortion regulations in all 50 U.S. states and the "
        "District of Columbia in effect from December 1, 2018 through November 1 2022, as well as "
        "case law and attorney general opinions that affect the enforceability of these laws.",
        
    "Statutory and Constitutional Right to Abortion": 
        "This dataset explores abortion protections in all 50 U.S. states and the District of "
        "Columbia in effect from December 1, 2018 through November 1, 2022, as well as case law "
        "and attorney general opinions that affect the enforceability of these laws."
}
title_pol = "Healthcare Policy Frameworks"
source_name_pol = "LawAtlas"
source_link_pol = "https://lawatlas.org/explore-topics?_search=Abortion"
api_collect_pol = False
collection_method_pol = "Direct Download (Excel files)"
description_pol = policy_datasets
cleaning_steps_pol = {
    'Boolean Feature Derivation': 'Transformed legislative parameters into indicator columns with binary logic tests',
    'Timeline Engineering': 'Imputed policy indicators for untracked time-periods based on closest available data'
}

# ================= HEALTH (HEALTH RANKINGS) ========================
title_health = "Health of Women and Children Report"
source_name_health = "America's Health Rankings"
source_link_health = "https://www.americashealthrankings.org/publications/reports/2025-health-of-women-and-children-report"
api_collect_health = True
collection_method_health = "GraphQL API"
description_health = """
    The annual Health of Women and Children Report provides a comprehensive look at the health of women of reproductive age
    and children nationwide and on a state-by-state basis. Data for this project is pulled from annual reports from 2018-2025;
    ultimately it contains (mostly) complete data from 2018-2023. 
"""
cleaning_steps_health = {
    'Feature Selection': 'Filtered metrics down to regional reproductive healthcare criteria',
    'De-Duplication & Aggregation': "Executed multi-index alignment and sorting functions based on unique groupings"
}
api_code_health = """
# Securely initialize API credentials
load_dotenv()
API_KEY = os.getenv("AHR_API_SUBSCRIPTION_KEY")

if not API_KEY:
    raise ValueError("Error: API key failed to load out of local environment.")

url = 'https://api.americashealthrankings.org/graphql'
headers = {
    'Content-Type': 'application/json',
    'X-Api-Key': API_KEY
}

# Specific metrics of interest from annual HWC Reports
target_measures = [
    "Concentrated Disadvantage", "Food Insecurity",
    "Gender Pay Gap", "Poverty", "Unemployment", "College Graduate", "Infant Child Care Affordability",
    "Voter Participation (Average)", "Adequate Prenatal Care", "Avoided Care Due to Cost",
    "Maternity Care Desert", "Uninsured Women", "Women's Health Providers", "Cervical Cancer Screening",
    "Postpartum Visit", "Well-Woman Visit", "Low-Risk Cesarean Delivery", "Maternity Practices Score",
    "Unintended Pregnancy", "Smoking During Pregnancy", "Postpartum Depression",
    "Maternal Mortality", "Mortality Rate", "Severe Maternal Morbidity", "WIC Coverage",
    "Infant Mortality", "Neonatal Mortality", "Low Birth Weight"
]

# Build query string
query = "
query GetReportDataByMeasures($measureNames: [String!]) {
  measures_A(where: { name: { in: $measureNames } }) {
    name
    source {
      name
    }
    data {
      state
      dateLabel
      value
    }
  }
}
"

variables = {
    "measureNames": target_measures
}

# Send API request
response = requests.post(
    url, 
    json={'query': query, 'variables': variables}, 
    headers=headers, 
    timeout=60
)
response.raise_for_status()
payload = response.json()
"""

# ===================== BIRTHS (NCHS-BIRTH) =========================
birth_v1 = os.path.join(viz_path, 'birth_v1.png')
birth_v2 = os.path.join(viz_path, 'birth_v2.png')
title_birth = "Birth Data Files"
source_name_birth = "National Center for Health Statistics (NCHS)"
source_link_birth = "https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm"
api_collect_birth = False
collection_method_birth = "Direct Download (.txt files)"
description_birth = """
    Natality statistics for births occurring within the United States.
"""
cleaning_steps_birth = {
    'Decompression': "Processed massive compressed source archive by dynamically pulling raw internal text data",
    'Coordinate Slicing': 'Used codebook to isolate relevant demographic and healthcare variables based on pre-defined column bitwise boundaries',
    'Categorical Value Labels': 'Mapped raw alphanumeric codes to human-readable strings'
}
birth_visuals={}
birth_visuals['visual_1'] = {
    'title': "Maternal Age Cohort Distribution (2024)",
    'fig': birth_v1,
    'caption': """
        Age group breakdown across all recorded births in 2024. 
    """
}
birth_visuals['visual_2'] = {
    'title': "Maternal ICU Admission Risk",
    'fig': birth_v2,
    'caption': """
        Incidence rate of intensive care admissions across maternal age groups. Risk is heightened for very young mothers
        (less than 15 years of age) and older mothers, with risk dramatically increasing after age 50.
    """
}


