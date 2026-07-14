import os
import streamlit as st
import pandas as pd

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
nn_rec = os.path.join(BASE_DIR, "resources", "neural_net")
if not os.path.exists(nn_rec):
    os.makedirs(nn_rec)

def render_nn(
        overview_text: str,
        sample_data_url: str,
        data_reqs: str,
        sample_data: pd.DataFrame,
        train_test_text: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        model_code_url: str,
        results_pipeline: list
):
    
    # Overview
    st.subheader('Neural Network Overview')
    with st.expander('Read About Neural Networks', expanded=True):
        st.write(overview_text)

    # Data Prep
    st.subheader("Data Preparation")
    st.markdown(f"🔗 **[Download Sample Dataset]({sample_data_url})**")
    st.markdown("##### Data Requirements")
    st.write(data_reqs)
    st.dataframe(sample_data, width='stretch', hide_index=True)
    st.markdown("##### Training vs. Testing Split")
    st.markdown(train_test_text)
    ctrain, ctest = st.columns(2)
    with ctrain:
        st.markdown("**Training Data**")
        st.dataframe(train_data, hide_index=True)
    with ctest:
        st.markdown("**Testing Data**")
        st.dataframe(test_data, hide_index=True)

    st.markdown("---")

    # Code/Results
    st.subheader("Model Results")
    st.markdown(f"👾 [View Code]({model_code_url})")
    
    for x in results_pipeline:
        st.markdown(f"##### {x.get('name')}")
        fig = x.get('fig')
        if isinstance(fig, pd.DataFrame):
            st.dataframe(fig)
            st.caption(x.get('caption'))
        elif isinstance(fig, str):
            st.image(fig, width='content')
            st.caption(x.get('caption'))
        else:
            pass
        
        st.markdown(x.get('text'))

    st.markdown("---")

    # Conclusion
    st.subheader('Conclusions')


# ==============================================================================
# Overview
# ==============================================================================
overview_text_nn = """
    A **Neural Network (NN)** is a computational model inspired by the biological structure of neurons in the human brain,
    designed to recognize complex, non-linear relationships within data. The network works by passing input data through
    layers of interconnected processing *nodes* (neurons). Each connection has an adjustable weight that 
    magnifies or dampens the signal passing through it. As data flows through the neural network, the nodes compute
    weighted sums, apply mathematical activation functions, and pass the results to the next layer until reaching the
    output layer. The output layer generates a final prediction. During training, the NN uses an optimization process
    known as backpropagation to 'learn'. It evaluates its prediction errors against the true values using a loss function,
    calculates updates, and systematically adjusts internal weights to improve its accuracy over successive training 'epochs'.
"""

# ==============================================================================
# Data Prep
# ==============================================================================
sample_data_url_nn = "https://github.com/amberteetsel/maternal-health/blob/6e9119dbf12c6714d364bcef9e25ecb8d01aef09/resources/neural_net/nn_preprocessed.csv"
data_reqs_nn = """
    In this instance, a neural network is being used to perform supervised learning (classification).
    Supervised modeling requires **labeled** data, meaning every row must have a known target outcome (`Abortion_Restricted`).
    The model uses these labels to learn the patterns that differentiate classes. Without labels, the algorithm has no
    comparison point to evaluate its performance against. NNs are highly sensitive to scale of numbers, so health metrics are also
    z-score standardized with `StandardScaler` to have a mean of zero and standard deviation of 1. Neural networks also require
    purely numeric data, so categorical feature `Region` was encoded using One-Hot Encoding to turn the four geographical regions
    into binary flags. If regions were encoded as ordinal numbers (e.g. Northeast = 1, Midwest = 2, South = 3, West = 4) then the
    model would think West is "four times greater" than Northeast. 

    The target variable `Abortion_Restricted` is a binary flag for whether a certain U.S. state restricts abortion or not.
    States with total bans or gestational bans up to 20 weeks are considered restrictive and indicated by `Abortion_Restriction` = 1.
"""
sample_data_nn = pd.read_csv(os.path.join(nn_rec, 'nn_preprocessed.csv'))
train_test_text_nn = """
    Because the dataset is relatively small (250 rows), a random split could result in all 'Restricted' states assigned to
    training and the non-Restricted states assigned to testing. Instead, stratified splitting was employed to maintain the
    same proportional mix of protected vs. restricted states in the training set ($80\\%$ of data) and the testing set
    ($20\\%$ of data).
"""
train_data_nn = pd.read_csv(os.path.join(nn_rec, 'nn_train_display.csv'))
test_data_nn = pd.read_csv(os.path.join(nn_rec, 'nn_test_display.csv'))

# ==============================================================================
# Code
# ==============================================================================
model_code_url_nn = "https://github.com/amberteetsel/maternal-health"   ## PLACEHOLDER - REPLACE!!!

# ==============================================================================
# Results
# ==============================================================================
report_text_nn = """
    place
"""
cm_text_nn = """
    confusion matrix jksdfghjkadfg
"""
results_nn = [
    {
        'name': 'Quantitative Model Performance',
        'fig': pd.read_csv(os.path.join(nn_rec, 'classification_report_nn.csv')),
        'text': report_text_nn,
        'caption': "Neural Network Classification Report"
    },

    {
        'name': 'Confusion Matrix Diagnostics',
        'fig': os.path.join(nn_rec, 'nn_confusion_matrix.png'),
        'text': cm_text_nn,
        'caption': "The confusion matrix for test evaluation shows extremely strong model performance."
    }
]

# ==============================================================================
# Conclusion
# ==============================================================================