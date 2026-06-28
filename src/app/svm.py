import os
import streamlit as st
import pandas as pd

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
svm_rec = os.path.join(BASE_DIR, "resources", "svm")
if not os.path.exists(svm_rec):
    os.makedirs(svm_rec)

def render_svm(
        sample_data_url: str,       # main dataprep DF link
        sample_data: pd.DataFrame,  # main dataprep DF
        train_test_text: str,       # explanation of train test split
        train_data: pd.DataFrame,   # training DF
        test_data: pd.DataFrame,    # testing DF
        model_code_url: str
):

    # Overview
    st.subheader("SVM Overview")
    with st.expander("Read About SVM", expanded=True):
        st.write("Support Vector Machines are...")

    # Data Prep
    st.subheader("Data Preparation")

    st.markdown(f"🔗 **[Download Sample Dataset]({sample_data_url})**")
    st.dataframe(sample_data, width='stretch')

    st.markdown(train_test_text)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Training Dataset")
        st.dataframe(train_data, width='stretch')
    with c2:
        st.markdown("Testing Dataset")
        st.dataframe(test_data, width='stretch')
    st.markdown("---")

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
sample_data_url_svm = "https://github.com/amberteetsel/maternal-health" # REPLACE!!!
sample_data_svm = pd.read_csv(os.path.join(svm_rec, "svm_preprocessed.csv"))
train_test_text_svm = """
    Separate target column, scale features, disjoint because otherwise model can memorize data etc.
"""
train_data_svm = pd.read_csv(os.path.join(svm_rec, "train_display.csv"))
test_data_svm = pd.read_csv(os.path.join(svm_rec, "test_display.csv"))



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