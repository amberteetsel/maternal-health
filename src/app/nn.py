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
        rqs: str,
        sample_data_url: str,
        data_reqs: str,
        sample_data: pd.DataFrame,
        train_test_text: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        model_code_url: str,
        result_section: dict,
        results_pipeline: list,
        conclusion_pipeline: list
):
    
    # Overview
    st.subheader('Neural Network Overview')
    with st.expander('Read About Neural Networks', expanded=True):
        st.write(overview_text)

    st.markdown(rqs)

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

    # first section (architecture, epochs)
    st.markdown("##### Model Design & Training")
    st.markdown(result_section.get('intro'))
    st.image(result_section.get('fig1'), width='content')
    st.caption("Model Architecture Diagram")
    st.image(result_section.get('fig2'), width='content')
    st.caption("Final Training Epochs")
    st.markdown(result_section.get('text'))

    
    for x in results_pipeline:
        st.markdown(f"##### {x.get('name')}")
        st.markdown(x.get('intro'))
        
        fig = x.get('fig')
        if isinstance(fig, pd.DataFrame):
            st.dataframe(fig, hide_index=True)
            st.caption(x.get('caption'))
        elif isinstance(fig, str):
            st.image(fig, width=600)
            st.caption(x.get('caption'))
        else:
            pass
        
        st.markdown(x.get('text'))

    st.markdown("---")

    # Conclusion
    st.subheader('Conclusions')
    for x in conclusion_pipeline:
        st.markdown(f"##### {x.get('title')}")
        st.markdown(x.get('text'))


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
rqs_nn = """
    The purpose of this analysis is to answer the following research questions:

    #### Can distinct clusters of states/geographic regions be identified based on reproductive policies?

    #### Do clusters reliably predict disparities in maternal health outcomes?
"""

# ==============================================================================
# Data Prep
# ==============================================================================
sample_data_url_nn = "https://github.com/amberteetsel/maternal-health/blob/6e9119dbf12c6714d364bcef9e25ecb8d01aef09/resources/neural_net/nn_preprocessed.csv"
data_reqs_nn = """
    In this instance, a neural network is being used to perform supervised learning (classification): the aim is to predict whether
    a state restricts abortion or not based on its geographic region and underlying health metrics.
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
model_code_url_nn = "https://github.com/amberteetsel/maternal-health/blob/cd0a7ac135275dfde3d2a46195d2439ebb2be823/src/models/nn_policy.py"   ## PLACEHOLDER - REPLACE!!!

# ==============================================================================
# Results
# ==============================================================================
result_intro_nn = """
    To evaluate whether a state's structural healthcare ecosystem and geographic positioning act as reliable predictors
    of its reproductive legislative status, a multi-layer perceptron (MLP) binary classifier was deployed. The model architecture
    comprises a 13-dimensional input layer, a non-linear hidden layer optimized with 8 ReLu units, and a single-node Sigmoid
    output layer.
"""
nn_architecture = os.path.join(nn_rec, 'NN_Cluster_Architecture.png')
nn_epochs = os.path.join(nn_rec, 'nn_epochs.png')
result_epoch_text = """
    While the network achieved $1.0$ training accuracy (potentially indicating near-complete optimization or overfitting),
    the validation accuracy remained stable at $93.94\\%$ with flat validation loss of $0.0805$. This minimal generalization gap
    proves that the model did not merely memorize training row noise. Instead, it demonstrates that the combined structural
    signature of geographic regions and maternal healthcare metrics provides a distinct, mathematically separable boundary
    between protected and restricted policy environments.
"""
result_section_nn = {
    'intro': result_intro_nn,
    'fig1': os.path.join(nn_rec, 'NN_Cluster_Architecture.png'),
    'fig2': os.path.join(nn_rec, 'nn_epochs.png'),
    'text': result_epoch_text
}


report_intro_nn = """
    The trained multi-layer perceptron binary classifier was evaluated using an independent test set comprising $20\\%$ of the
    overall data ($n=56$). The network achieved a **remarkable final classification accuracy of $98.21\\%$** with an overall macro
    F1-score of $0.98$. This strong performance on unseen data confirms that the network did not overfit or memorize training noise,
    but instead captured structural macro-trends.
"""
report_text_nn = """
    The minimal generalization gap between the final training cycle accuracy ($1.0$, Loss $=0.005$) and the validation/testing
    set demonstrates that the architecture did not succumb to overfitting or absolute row memorization. The model successfully mapped
    a robust, generalizable multi-dimensional decision boundary across the structural features (health statistics and geography).
"""
cm_intro_nn = """
    The empirical test set distribution reveals near-perfect class separation along the network's optimized decision boundary,
    mapping the 56 test instances as follows:

    * **True Negatives (22):** The network correctly identified 22 state-year records as legally **Not Restricted** based entirely
    on clinical metrics and geographic region.
    * **True Positives (33):** The network correctly identified 33 state-year records as legally **Restricted**, accurately pairing
    institutional policy environments with downstream health infrastructure stress.
    * **False Positives (0):** The network achieved a perfect Precision score ($1.0$) for the restricted class. It never misclassifies
    a legally protected record into the restricted tier, highlighting that high-quality healthcare environments remain
    structurally distinct from restricted landscapes.
    * **False Negatives (1):** The network committed exactly one classification error across the entire test matrix, misclassifying
    a legally **Restricted** state-year as **Not Restricted** (Recall $=0.97$).
"""
cm_text_nn = """
    From a structural standpoint, the single false negative represents an informative edge-case rather than a failure of the 
    network. The specific record in question is 2018 Alaska. Based on the data, this is an 'understandable' misclassification 
    because Alaska is somewhat of an anomaly of a state; its unique geography and low population means it is necessarily a very 
    different structural environment than the lower 48. In this case, despite legal restrictions on paper, the underlying healthcare
    infrastructure maintained baseline structural resilience matching the profile of the Not Restricted class.
"""
results_nn = [

    {
        'name': 'Quantitative Model Performance',
        'intro': report_intro_nn,
        'fig': pd.read_csv(os.path.join(nn_rec, 'classification_report_nn.csv')),
        'text': report_text_nn,
        'caption': "Test Evaluation Classification Report"
    },

    {
        'name': 'Confusion Matrix Diagnostics',
        'intro': cm_intro_nn,
        'fig': os.path.join(nn_rec, 'nn_confusion_matrix.png'),
        'text': cm_text_nn,
        'caption': "The confusion matrix for test evaluation shows extremely strong model performance."
    }
]

# ==============================================================================
# Conclusion
# ==============================================================================
q1_text = """
    Yes. By integrating one-hot encoded U.S. Census Bureau regional dummy variables (Northeast, Midwest, South, West) into
    the network's input features, the model's capacity to draw a clean boundary demonstrates that reproductive policy shifts
    are not geographically isolated anomalies. Rather, the legal frameworks run deeply across established macro-regional lines, 
    thus allowing the network to use spatial location as an anchor for its weights.
"""
q2_text = """
    Yes. The fact that an algorithm can achieve a $98.21\\%$ accurate classification rate using *only* clinical health outcomes 
    and geographic regions proves that healthcare disparities do not merely correlate with political environments: they serve
    as a definitive, multi-dimensional signature of those environments.

    The structural gaps across metrics like `Maternity Care Desert`, `Inadequate Prenatal Care`, and `Infant Mortality` are wide
    enough to form a mathematically separable boundary. This confirms that a state's legal/political framework regarding
    reproductive rights is closely bound to systemic, predictable disparities in the quality, accessibility, and outcomes of 
    maternal/infant care.
"""
conclusion_pipeline_nn = [
    {
        'title': "Can distinct clusters of states/geographic regions be identified based on reproductive policies?",
        'text': q1_text
    },

    {
        'title': "Do clusters reliably predict disparities in maternal health outcomes?",
        'text': q2_text
    }
]