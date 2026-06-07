# Website Code
import sys
print("--- LOG: STREAMLIT IS RUNNING ON PYTHON EXECUTABLE:", sys.executable)
print("--- LOG: LOOKING FOR PACKAGES IN PATHS:", sys.path)

# Dependencies - Packages
import streamlit as st
import pandas as pd
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
# import streamlit.iframe as components

# Dependencies - Helper Modules
from data_view import data_source_section
from stacked_maps import generate_stacked_us_maps
from policy_maps import create_ban_limit_map, create_protection_map

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Homepage
st.set_page_config(page_title="MaternalHealth", layout="wide")

st.title("Exploring Maternal Health Outcomes in Relation to Healthcare Policy")
st.markdown("---")

# Initialize Tabs
t1, t2, t3, t4 = st.tabs([
    "Introduction",
    "Conclusions",
    "Data Prep & EDA",          # add sub-tabs inside for each data source
    "Models"                    # add sub-tabs inside for each model type
])

# Infographic - Commonwealth Fund, Maternal Mortality Comparison (international)
infogram_embed_html = """
<div class="infogram-embed" data-id="80a84092-43ff-46f5-a159-5ec782d8f07b" data-type="interactive" data-title="Insights into the U.S. Maternal Mortality Crisis: An International Comparison: Exhibit 1"></div>
<script>!function(e,n,i,s){var d="InfogramEmbeds";var o=e.getElementsByTagName(n)[0];if(window[d]&&window[d].initialized)window[d].process&&window[d].process();else if(!e.getElementById(i)){var r=e.createElement(n);r.async=1,r.id=i,r.src=s,o.parentNode.insertBefore(r,o)}}(document,"script","infogram-async","https://e.infogram.com/js/dist/embed-loader-min.js");</script>
"""

###############################
##### TAB 1: INTRODUCTION #####
###############################

with t1:
    st.header("Introduction")

    st.subheader("Background & Policy Context")
    st.write("""
            In June 2022, the Supreme Court of the United States issued its controversial ruling in 
             Dobbs v. Jackson Women's Health Organization, a historic decision that overturned nearly half a century of 
             federal precedent established by Roe v. Wade (1973).
             The Dobbs ruling eliminated the national right to abortion, effectively reverting authority to 
             regulate, restrict, and outlaw the practice back to individual states.
             As a result, the United States has become a fractured patchwork of reproductive healthcare access where 
             an individual's legal rights and medical choices are dependent on geographic location.
             According to policy tracking by the Guttmacher Institute (2024), more than a dozen states immediately 
             enacted total or near-total bans on abortion. 
             Conversely, many states moved to enshrine abortion protections in state law or state constitutions. This divergence
              in policy has created severe clinical confusion and legal gridlock, with healthcare providers left to navigate 
             vague and sometimes contradictory language regarding exceptions for the life or health of the mother.

             The Dobbs decision occurred during a worsening maternal health crisis in the United States, further exacerbating the issue.
             Data from the Centers for Disease Control and Prevention (CDC, 2023) indicate that the United States has the highest maternal 
             mortality rate among developed nations. There are also severe disparities affecting maternal mortality 
             for Black and Indigenous populations.
             Longitudinal tracking by the Guttmacher Institute demonstrates that unintended pregnancy rates fluctuate closely 
             with access to comprehensive family planning resources including contraceptive affordability and abortion services. 
             By exploring how changing state-level abortion restrictions intersect with existing maternal health infrasstructure 
             and maternal health outcomes, researchers can begin to quantify the real ramifications of restrictive abortion policies.
        """)
    
    st.subheader("Research Significance")
    st.write("""
            Examining the direct correlation between restrictive healthcare policies and maternal health outcomes is critically 
             important because legislative interventions carry profound, life-altering consequences for pregnant individuals, 
             infants, and medical networks. Public health research consistently warns that the implementation of strict 
             abortion bans and narrow gestational limits can inadvertently increase maternal morbidity by delaying essential care 
             for obstetric complications, such as ectopic pregnancies, premature rupture of membranes, or incomplete miscarriages. 
             Furthermore, states enacting the most stringent restrictions often exhibit pre-existing systemic vulnerabilities, 
             such as high rates of uninsured residents, severe shortages of obstetricians and gynecologists, 
             and widespread "maternity care deserts." For example, the restriction of reproductive healthcare services 
             frequently leads to the closure of local clinics and rural labor units, compounding barriers to 
             routine prenatal care and leading to higher rates of low-birth-weight infants and preterm births. 
             The economic and psychological strains placed on individuals forced to carry unintended pregnancies to term, 
             or travel thousands of miles across state lines for care, introduce significant socioeconomic stressors that 
             undermine long-term household and community stability. 
             Additionally, the chilling effect on medical education and physician recruitment in states with
              severe criminal penalties for doctors threatens to destabilize the broader OB/GYN and pediatric healthcare
              workforce for decades to come. Quantitative data analysis provides an objective framework to move past polarized 
             political rhetoric and systematically evaluate the empirical impacts of these legal shifts on 
             tangible medical outcomes. By aligning longitudinal datasets tracking policy status—such as total bans,
              heartbeat bans, and protections—with public health records covering infant birth metrics and 
             emergency room utilization, this study establishes a clear, data-driven narrative.
        """)
    
    # Add figure(s) here, each should have 2 sentence caption/explanation
    left_spacer, center_column, right_spacer = st.columns([1, 2, 1])
    with center_column:
        # Slightly lowered the height to 600px to match the reduced width proportions
        components.html(infogram_embed_html, height=600, scrolling=False)
        st.caption("Figure 1.1: The Commonwealth Fund International Comparison")

    # Research Qs
    st.subheader("Research Questions")
    with st.container():
        st.markdown("""
                    1. Is there a statistically significant difference in maternal mortality/morbidity rates between states
                        with explicit constitutional/legal protections and states with abortion bans or severe restrictions?
                    2. Does the enactment of a heartbeat ban or total ban correlate with an increase in emergency room 
                        visits related to pregnancy complications?
                    3. What is the average lag between legal enactment of abortion restrictions and measurable changes
                        in statewide birth outcomes?
                    4. Do shifts in reproductive policy correspond with changes in prenatal care utilization or
                        timing of first prenatal visits?
                    5. Can distinct clusters of states/geographic regions be identified based on reproductive policies?
                    6. Do clusters reliably predict disparities in maternal health outcomes?
                    7. What policy features are linked to a decline in rural reproductive healthcare infrastructure?
                    8. How did the 1973 decision in *Roe v. Wade* impact reproductive infrastructure usage and maternal health outcomes?
                    9. How did the 2022 decision in *Dobbs v. Jackson* impact reproductive infrastructure usage and maternal health outcomes?
                    10. How did the 1992 decision in *Planned Parenthood v. Casey* impact abortion restrictions and maternal health outcomes?    
                """)
        

###############################
#### TAB 3: DATA PREP, EDA ####
###############################

@st.cache_data
def load_project_data():
    """
    Safely reads and caches all raw and cleaned project files in memory.
    """
    # Base folder directories
    raw_path = os.path.join(BASE_DIR, "data", "raw")
    clean_path = os.path.join(BASE_DIR, "data", "clean")
    
    # raw
    er_raw_df = pd.read_csv(os.path.join(raw_path, "CDC-ER", "er_raw.csv"))
    pregnancy_raw_df = pd.read_csv(os.path.join(raw_path, "Guttmacher", "NatStatePregnancy.csv"))
    policy_raw_df = pd.read_csv(os.path.join(raw_path, "LawAtlas", "policy_raw.csv"))
    
    with open(os.path.join(raw_path, "HealthRankings", "raw_api_snapshot.json"), 'r') as f:
        health_raw_obj = json.load(f)
        
    with open(os.path.join(raw_path, "NCHS-Birth", "births2024_raw.txt"), 'r') as f:
        birth_raw_str = f.read()

    # clean
    er_clean_df = pd.read_csv(os.path.join(clean_path, "CDC-ER", "er.csv"))
    pregnancy_clean_df = pd.read_csv(os.path.join(clean_path, "Guttmacher", "pregnancy.csv"))
    policy_clean_df = pd.read_csv(os.path.join(clean_path, "LawAtlas", "policy.csv"))
    health_clean_df = pd.read_csv(os.path.join(clean_path, "HealthRankings", "health.csv"))
    birth_clean_df = pd.read_csv(
        os.path.join(clean_path, "NCHS-Birth", "births2024.csv.zip"),
        low_memory=False
    )
    
    return (
        er_raw_df, pregnancy_raw_df, policy_raw_df, health_raw_obj, birth_raw_str,
        er_clean_df, pregnancy_clean_df, policy_clean_df, health_clean_df, birth_clean_df
    )

(
    er_raw, pregnancy_raw, policy_raw, health_raw, birth_raw,
    er_clean, pregnancy_clean, policy_clean, health_clean, birth_clean
) = load_project_data()

# ==============================================================================

# Visuals
viz_path = os.path.join(BASE_DIR, "resources", "visuals_eda")
er_v1 = os.path.join(viz_path, 'er_v1.png')
er_v2 = os.path.join(viz_path, 'er_v2.png')
birth_v1 = os.path.join(viz_path, 'birth_v1.png')
birth_v2 = os.path.join(viz_path, 'birth_v2.png')
preg_v1 = os.path.join(viz_path, "preg_v1.png")
preg_v2 = os.path.join(viz_path, "preg_v2.png")
## Health Rankings Interactive Maps
health_v1 = generate_stacked_us_maps(
    df=health_clean, 
    measure_name="Adequate Prenatal Care", 
    color_scale="blues", 
    title_text="Prenatal Care Quality Comparison (2018 vs. 2023)"
)
health_v2 = generate_stacked_us_maps(
    df=health_clean, 
    measure_name="Severe Maternal Morbidity", 
    color_scale="Reds", 
    title_text="Severe Maternal Morbidity Comparison (2018 vs. 2023)"
)
health_visuals = {
    "visual_1": {
        "title": "Maternity Care Desert Distribution",
        "fig": health_v1,  # Live Plotly Figure object
        "caption": """
            Choropleth comparison tracking the evolution of prenatal care classifications between 2018 and 2023.
            Values represent the percentage of live births in which the mother received prenatal care
            beginning in the first four months or pregnancy with the appropriate number of visits for the infant's gestational
            age.
        """
    },
    "visual_2": {
        "title": "Severe Maternal Morbidity Trends",
        "fig": health_v2,  # Live Plotly Figure object
        "caption": """
            Multi-year tracking showing shift intensities in severe clinical morbidity prevalence by state over 5 years.
            Values represent the number of significant life-threatening maternal complications during delivery
            per 10,000 delivery hospitalizations.
        """
    }
}
# Policy Interactive Maps
policy_visuals = {
    "visual_1": {
        "title": "Abortion Bans and Gestational Limits Comparison",
        "fig": create_ban_limit_map(policy_clean),
        "caption": """
            Red indicates total bans, orange indicates heartbeat bans, and purple indicates varying
            gestational limits for obtaining an abortion.
        """
    },
    "visual_2": {
        "title": "State Legal & Constitutional Protections",
        "fig": create_protection_map(policy_clean),
        "caption": "Dark blue reflects explicit constitutional safety; light blue indicates legislative protections."
    }
}

### Emergency Room Data
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
    'caption': "Distribution of most common Reasons for Visit (RFV) parsed from emergency records (2018-22)."
}
er_visuals['visual_2'] = {
    'title': 'Top 5 Primary Diagnoses for Pregnancy ER Visits',
    'fig': er_v2,
    'caption': "Distribution of most common diagnoses parsed from emergency records (2018-22)."
}
cleaning_steps_er = {
    'Multi-Year Data Ingestion': 'Loaded several annual raw Stata datasets (.dta) spanning 2018 to 2022',
    'ICD-10-CM Standard Mapping': 'Loaded external code index file and parsed line-by-line to construct mapping of codes to their descriptions',
    'Metadata Extraction': "Bundled structured categorical label configurations for each annual file's metadata dictionaries",
    'Data Consolidation': "Compiled individual annual files to consolidated, clean flat CSV baseline file"
}

### Pregnancy Data
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
    'title': "Mean Historic Abortion and Miscarriage Rates",
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
        "This dataset explores abortion regulations in all 50 U.S. states and the District of "
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

### Health Data
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

### Birth Data
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
    'caption': "Age group breakdown across all recorded births in 2024."
}
birth_visuals['visual_2'] = {
    'title': "Maternal ICU Admission Risk",
    'fig': birth_v2,
    'caption': "Incidence rate of intensive care admissions across maternal age groups."
}

with t3:
    st.header("Data Sources")

    t_cdc_er, t_preg, t_birth, t_health, t_pol  = st.tabs([
        "Emergency Room Visits",
        "Pregnancy, Births, and Abortions",
        "Birth Records",
        "Health Rankings",
        "Healthcare Policy"
    ])

    # Emergency Room Data
    with t_cdc_er:    
        data_source_section(
            title=title_er, 
            source_name=source_name_er, 
            source_link=source_link_er,
            api_collect=api_collect_er, 
            collection_method=collection_method_er,
            description=description_er,
            raw=er_raw, 
            clean=er_clean,
            cleaning_steps=cleaning_steps_er,
            cleaning_code = "https://github.com/amberteetsel/maternal-health/blob/597d1edc47ef13548676ec8e92e0f1ef33a95ab4/src/cleaning/er_cleaning.py",
            api_code=None,
            visuals=er_visuals,
            data_link="https://github.com/amberteetsel/maternal-health/blob/cebd0bc60d68f180778fcbd9e47e027b2fd5df7a/data/raw/CDC-ER/er_raw.csv"
        )

    # Pregnancy Data
    with t_preg:
        data_source_section(
            title=title_preg, source_name=source_name_preg, source_link=source_link_preg,
            api_collect=api_collect_preg, collection_method=collection_method_preg,
            description=description_preg,
            raw=pregnancy_raw, clean=pregnancy_clean,
            visuals=preg_visuals,
            cleaning_steps=cleaning_steps_preg,
            cleaning_code="https://github.com/amberteetsel/maternal-health/blob/597d1edc47ef13548676ec8e92e0f1ef33a95ab4/src/cleaning/pregnancy_cleaning.py",
            data_link = "https://github.com/amberteetsel/maternal-health/blob/cebd0bc60d68f180778fcbd9e47e027b2fd5df7a/data/raw/Guttmacher/NatStatePregnancy.csv"
        )

    # Policy Data
    with t_pol:
        data_source_section(
            title=title_pol, source_name=source_name_pol, source_link=source_link_pol,
            api_collect=api_collect_pol, collection_method=collection_method_pol,
            description=description_pol,
            raw=policy_raw, clean=policy_clean,
            visuals=policy_visuals,
            cleaning_steps=cleaning_steps_pol,
            cleaning_code="https://github.com/amberteetsel/maternal-health/blob/597d1edc47ef13548676ec8e92e0f1ef33a95ab4/src/cleaning/policy_cleaning.py",
            data_link="https://github.com/amberteetsel/maternal-health/blob/cebd0bc60d68f180778fcbd9e47e027b2fd5df7a/data/raw/LawAtlas/policy_raw.csv"
        )

    # Health Data
    with t_health:
        data_source_section(
            title=title_health, 
            source_name=source_name_health, 
            source_link=source_link_health,
            api_collect=api_collect_health, 
            collection_method=collection_method_health, # Fixed variable typo here too!
            description=description_health, 
            raw=health_raw, 
            clean=health_clean,
            cleaning_steps=cleaning_steps_health,
            cleaning_code="https://github.com/amberteetsel/maternal-health/blob/597d1edc47ef13548676ec8e92e0f1ef33a95ab4/src/cleaning/health_cleaning.py",
            api_code=api_code_health,
            visuals=health_visuals,
            data_link="https://github.com/amberteetsel/maternal-health/blob/cebd0bc60d68f180778fcbd9e47e027b2fd5df7a/data/raw/HealthRankings/raw_api_snapshot.json"
        )

    # Birth Data
    with t_birth:
        data_source_section(
            title=title_birth, source_name=source_name_birth, source_link=source_link_birth,
            api_collect=api_collect_birth, collection_method=collection_method_birth,
            description=description_birth,
            raw=birth_raw, clean=birth_clean,
            visuals=birth_visuals,
            cleaning_steps=cleaning_steps_birth,
            cleaning_code="https://github.com/amberteetsel/maternal-health/blob/597d1edc47ef13548676ec8e92e0f1ef33a95ab4/src/cleaning/birth_cleaning.py",
            data_link="https://github.com/amberteetsel/maternal-health/blob/cebd0bc60d68f180778fcbd9e47e027b2fd5df7a/data/raw/NCHS-Birth/births2024_raw.txt"
        )

###############################
######## TAB 4: MODELS ########
###############################

with t4:
    st.header("Modeling Results")

    t_cluster, t_arm, t_nb, t_dt, d_svm, t_regr, t_nn = st.tabs([
        "Clustering",
        "Association Rule Mining (ARM)",
        "Naive Bayes",
        "Decision Trees",
        "Support Vector Machines (SVM)",
        "Regression",
        "Neural Networks"
    ])