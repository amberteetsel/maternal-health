# Website Code

# Dependencies - Packages
import streamlit as st
import pandas as pd
import json
import os

# Dependencies - Helper Modules

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

###############################
##### TAB 1: INTRODUCTION #####
###############################

with t1:
    st.header("Introduction")

    st.subheader("Background")
    st.write("""
             At least 2 paragraphs (10 sentences each) explaining context for research topic...
        """)
    
    # Add figure(s) here, each should have 2 sentence caption/explanation

    st.subheader("Research Questions")
    with st.container():
        st.markdown("""
                    1. RQ1
                    2. RQ2
                    3. RQ3
                    4. 
                    5. 
                    6.
                    7.
                    8.
                    9.
                    10.    
                """)
        

###############################
#### TAB 3: DATA PREP, EDA ####
###############################
from data_view import data_source_section

# Raw Data
raw_data_path = os.path.join(BASE_DIR, "data", "raw")
er_raw = pd.read_csv(os.path.join(raw_data_path, "CDC-ER", "er_raw.csv"))
pregnancy_raw = pd.read_csv(os.path.join(raw_data_path, "Guttmacher", "NatStatePregnancy.csv"))
policy_raw = pd.read_csv(os.path.join(raw_data_path, "LawAtlas", "policy_raw.csv"))
with open(os.path.join(raw_data_path, "HealthRankings", "raw_api_snapshot.json"), 'r') as f:
    health_raw = json.load(f)
with open(os.path.join(raw_data_path, "NCHS-Birth", "births2024_raw.txt"), 'r') as f:
    birth_raw = f.read()

# Clean Data
clean_data_path = os.path.join(BASE_DIR, "data", "clean")
er_clean = pd.read_csv(os.path.join(clean_data_path, "CDC-ER", "er.csv"))
pregnancy_clean = pd.read_csv(os.path.join(clean_data_path, "Guttmacher", "pregnancy.csv"))
policy_clean = pd.read_csv(os.path.join(clean_data_path, "LawAtlas", "policy.csv"))
health_clean = pd.read_csv(os.path.join(clean_data_path, "HealthRankings", "health.csv"))
birth_clean = pd.read_csv(os.path.join(clean_data_path, "NCHS-Birth", "births2024.csv.zip"))

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

### Pregnancy Data

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
            title=title_er, source_name=source_name_er, source_link=source_link_er,
            api_collect=api_collect_er, collection_method=collection_method_er,
            description=description_er,
            raw=er_raw, clean=er_clean
        )

    # Policy Data
    with t_pol:
        data_source_section(
            title=title_pol, source_name=source_name_pol, source_link=source_link_pol,
            api_collect=api_collect_pol, collection_method=collection_method_pol,
            description=description_pol,
            raw=policy_raw, clean=policy_clean
        )

    # Health Data
    with t_health:
        data_source_section(
            title=title_health, source_name=source_name_health, source_link=source_link_health,
            api_collect=api_collect_health, collection_method=collection_method_er,
            description=description_health, api_code=api_code_health,
            raw=health_raw, clean=health_clean
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