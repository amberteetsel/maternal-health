# Website Code

# Dependencies - Packages
import streamlit as st
import pandas as pd
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


with t3:
    st.header("Data Sources")

    t_cdc_er, t_gutt, t_birth, t_health, t_law  = st.tabs([
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