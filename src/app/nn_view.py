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
):
    
    # Overview
    st.subheader('Neural Network Overview')
    with st.expander('Read About Neural Networks', expanded=True):
        st.write(overview_text)

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


# ==============================================================================
# Code
# ==============================================================================


# ==============================================================================
# Results
# ==============================================================================


# ==============================================================================
# Conclusion
# ==============================================================================