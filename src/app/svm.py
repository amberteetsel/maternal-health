import os
import streamlit as st
import pandas as pd

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
svm_dir = os.path.join(BASE_DIR, "resources", "svm")
if not os.path.exists(svm_dir):
    os.makedirs(svm_dir)

def render_svm(
        
        model_code_url: str
):

    # Overview
    st.subheader("SVM Overview")
    with st.expander("Read About SVM", expanded=True):
        st.write("Support Vector Machines are...")

    # Data Prep
    st.subheader("Data Preparation")

    # Code/Results
    st.subheader("Model Results")
    st.markdown(f"👾 [View Code]({model_code_url})")

    # Conclusions
    st.subheader("Conclusions")

# ==============================================================================
# Overview
# ==============================================================================




# ==============================================================================
# Data Prep
# ==============================================================================




# ==============================================================================
# Code
# ==============================================================================
model_code_url_svm = os.path.join(BASE_DIR, "src", "models", "svm.py")



# ==============================================================================
# Results
# ==============================================================================




# ==============================================================================
# Conclusions
# ==============================================================================