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

with t3:
    st.header("Data Sources")

    t_cdc_mort, t_cdc_er, t_gutt, t_birth, t_law  = st.tabs([
        "Cause of Death",
        "Emergency Room Visits",
        "Pregnancy, Births, and Abortions",
        "Birth Records",
        "Healthcare Policy"
        ""
    ])


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