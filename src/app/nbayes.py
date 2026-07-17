### HELPER TO DISPLAY NAIVE BAYES CLASSIFICATION RESULTS

# Dependencies
import os
import pandas as pd
import numpy as np
import inspect
import streamlit as st

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
nb_res = os.path.join(BASE_DIR, "resources", "nbayes")

# ==========================================================
# DISPLAY FUNCTION
# ==========================================================
def render_nb(overview_text: str,
              prep_text: str,
              cleaning_code_url: str,
              sample_data_url: str,
              sample_data: pd.DataFrame,
              train_test_text: str,
              train_data: pd.DataFrame,
              test_data: pd.DataFrame,
              model_code_url: str,
              results_intro: str,
              result_table: pd.DataFrame,
              confusion_matrix,
              results_text,
              conclusion_text
              ):
    
    # Overview
    st.subheader("Naive Bayes Classification Overview")
    with st.expander("Read About Naive Bayes", expanded=True):
        st.markdown(overview_text)
    # st.markdown("---")

    # Data Preparation
    st.subheader("Data Preparation")
    st.markdown(prep_text)

    st.markdown(f"🔗 **[Download Sample Dataset]({sample_data_url})**")
    st.markdown(f"🔍 **[View Cleaning Code]({cleaning_code_url})**")
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

    # Modeling & Results
    st.subheader("Model Results")
    st.markdown(f"👾 [View Code]({model_code_url})")

    st.markdown(results_intro)

    st.markdown("#### Performance Evaluation")
    st.table(result_table)

    st.markdown("#### Confusion Matrix")
    st.image(confusion_matrix)
    st.markdown(results_text)
    st.markdown("---")

    # Conclusion
    st.subheader("Conclusion")
    st.markdown(conclusion_text)


# Display Function for Severe Maternal Morbidity Analysis
def render_nb_er(
        overview_text: str,
        data_prep_text: str,
        data_raw_link: str,
        data_clean_link: str,
        data_raw: pd.DataFrame,
        data_clean: pd.DataFrame,
        model_code_url: str,
        result_df: pd.DataFrame,
        result_plots: list,
        result_text: list,
        conclusion_text: str,
):
    
    # overview
    st.subheader("Overview")
    st.markdown(overview_text)

    # data prep
    st.subheader("Data Preparation")
    st.markdown(data_prep_text)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Raw Dataset (First 100 Rows)**")
        st.markdown(f"🔗 **[Download Dataset]({data_raw_link})**")
        st.dataframe(data_raw, hide_index=True)

    with c2:
        st.markdown("**Encoded Input Data (First 100 Rows)**")
        st.markdown(f"🔗 **[Download Dataset]({data_clean_link})**")
        st.dataframe(data_clean, hide_index=True)

    st.markdown('---')

    # model results
    st.subheader("Model Results")
    st.markdown(f"👾 [View Code]({model_code_url})")
    st.write("When the trained Naive Bayes classifier projected risk probabilities across identical demographic pairs in contrasting regions, the following probabilistic profiles emerged:")
    st.dataframe(result_df, hide_index=True)
    st.markdown(result_text[0])
    st.markdown("##### Insight 2: Racial Disparities")
    st.image(result_plots[3], width=800)
    st.markdown(result_text[1])

    st.markdown("##### Insight 3: Age Paradox")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Teen (15-19) Risk Profiles**")
        st.image(result_plots[0], width='stretch')
    
    with c2:
        st.markdown(f"**Average Age (20-34) Risk Profiles**")
        st.image(result_plots[1], width='stretch')
    
    with c3:
        st.markdown(f"**Advanced Age (35-49) Risk Profiles**")
        st.image(result_plots[2], width='stretch')
    st.markdown(result_text[2])
            
    # conclusions
    st.subheader("Conclusions")
    st.markdown(conclusion_text)


# ==========================================================
# ACTUAL INPUTS - E.R. SEVERE MATERNAL MORBIDITY
# ==========================================================
overview_text_er = """
    The primary objective of this analysis is to answer the following research question:

    #### Given an emergency room patient's demographic profile, what is the conditional probability of them presenting with a severe pregnancy-related complication based on their geographic macro-region's reproductive legal status (Highly Restricted South vs. Highly Protected Northeast)?

    To that end, a Categorical Naive Bayes classifier was trained on 12,174 clean patient records extracted from the CDC's
    National Hospital Ambulatory Medical Care Survey (NHAMCS) Emergency Department data (2018 - 2022). The target label,
    Severe Maternal Morbidity (SMM) was flagged using a variety of ICD-10-CM codes for severe, acute obstetric crises including
    ectopic pregnancies, pre-eclampsia, hemorrhage, placenta previa, and sepsis.

    By evaluating the conditional probability $P(\\text{SMM} | \\text{Demographics, Region})$ the model isolates the baseline
    impact of geographic policy standards on acute maternal emergencies.
"""
data_prep_text_er = """
   Categorical Naive Bayes requires features to be categorical and encoded as discrete integer values. During preprocessing,
   the continuous feature `Age` was binned to discrete categories representing Teens (15-19), Average Maternal Age (20-34), and
   Advanced Maternal Age (35-49). Data was filted by `Sex` to only include female patients.
   Records were filtered by region to isolate the Northeast and the South. These regions were
   selected because the states within tend to be more consistently ideologically aligned in terms of abortion law than states
   in the West or Midwest. A flag for the target SMM was calculated by checking if any diagnostic codes for each entry
   belonged to a pre-defined list of codes signaling acute obstetric emergencies. This revealed a class imbalance aligned with
   typical clinical findings: the overall probability of a patient presenting to the emergency room with an SMM-flagged crisis is
   only $2.01\\%$, while the probability of presenting without such a crisis is $97.99\\%$. 

   Naive Bayes also assumes class-conditional independence, meaning the value of any single feature is independent of any other
   feature given the class label. The selected data from NHAMCS meets this requirement as Age, Race/Ethnicity, and Region are
   completely unrelated.
"""
data_raw_link_er = "https://github.com/amberteetsel/maternal-health/blob/f8e06291876ca7889c27ccb462986ad3f0618af3/resources/nbayes/er_data_raw.csv"
data_clean_link_er = "https://github.com/amberteetsel/maternal-health/blob/f8e06291876ca7889c27ccb462986ad3f0618af3/resources/nbayes/er_data_clean.csv"
data_raw_er = pd.read_csv(os.path.join(nb_res, "er_data_raw.csv"))
data_clean_er = pd.read_csv(os.path.join(nb_res, "er_data_clean.csv"))
model_code_url_er = "https://github.com/amberteetsel/maternal-health/blob/5cfd959c422583ec4fc1accd42e29e9e1deb3b5e/src/models/nb_er_smm.py"
df_scenarios_er = pd.read_csv(os.path.join(nb_res, "er_scenarios_prob.csv"))
result_plots_er = [os.path.join(nb_res, "smm_risk_teens_15_19.png"),
                     os.path.join(nb_res, "smm_risk_average_maternal_age_20_34.png"),
                     os.path.join(nb_res, "smm_risk_advanced_maternal_age_35_49.png"),
                     os.path.join(nb_res, "smm_racial_disparities.png")]
result_text_1 = """
    ##### Insight 1: Systemic Geographic Policy Shift

    The most striking and mathematically rigorous finding of this model is the **universal relative risk multiplier of $1.33x$ to
    $1.35x$ associated with the Southern region.** Because Naive Bayes calculates conditional probabilities independently based
    on individual feature likelihoods, this model reveals that simply seeking care in the South increases a patient's probability
    of presenting with SMM by roughly $33\\%$ to $35\\%$ regardless of race or age.

    This confirms that regional differences in healthcare policy and infrastructure serve as a systemic, environmental risk factor.
    Patients in highly restrictive legal and medical environments experience acute, emergency-room severe maternal morbidity crises
    at a substantially higher rate than patients in protected regions. This may be due to delayed prenatal interventions, fewer
    clinics, and narrower clinical criteria for handling early-pregnancy complications that all contribute to pushing standard
    pregnancy complications into medical emergencies.
"""
result_text_2 = """
    Even when holding geographic region constant, the model exposes several racial disparities in emergency-room SMM:

    * Within every age tier, Non-Hispanic White patients consistently display the lowest conditional probability of SMM
    * In contrast, Hispanic and Non-Hispanic Other groups exhibit the highest risk profiles
    * When systemic regional pressure (the South) is overlaid with racial disparities, a compounding risk profile is created. For
    example, an Average Maternal Age Non-Hispanic White patient in the Northeast has an SMM probability of $2.45\\%$ but a
    demographic peer in the South has double the risk at $4.95\\%$. 
"""
result_text_3 = """
    A close read of the conditional probability results reveals an apparent paradox: Advanced Maternal Age (35-49) has a lower
    absolute probability of SMM than the Average Maternal Age (20-34) and Teen (15-19) groups. For example, a Hispanic mother
    in the South has a $4.38\\%$ probability of SMM if she is 20-34, but only a $0.93\\%$ probability if she is over 35.

    While counterintuitive to typical ideas about biological aging, the finding makes sense when considering the realities of patients
    navigating modern healthcare systems. Older pregnant patients (35+) are automatically pre-classified as "high-risk" pregnancies
    by the medical establishment. This means they are more likely to have established, frequent prenatal check-ins and care that
    may reduce the risk of severe complications later on. They are more likely to have a dedicated obstetrician who can help them
    bypass the emergency department and report directly to specialist labor and delivery medical services.

    Conversely, younger age groups are more likely to experience unplanned pregnancies, to lack quality prenatal care, and to not have
    insurance. Without a dedicated obstetrician or insurance, pregnant patients will typically turn to the emergency room. Therefore
    the conditional probabilities generated by the model are only representative of people reporting to the E.R., not the general
    pregnant population at large, because the ER sees a disproportionately lower amount of older mothers.

"""
results_text_er = [result_text_1, result_text_2, result_text_3]
conclusion_text_er = """
    ##### Given an emergency room patient's demographic profile, what is the conditional probability of them presenting to the E.R. with a severe pregnancy-related complication based on their geographic macro-region's reproductive legal status (Highly Restricted South vs. Highly Protected Northeast)?

    The analysis concludes that **geographic region acts as a powerful independent predictor of severe maternal complications
    in American emergency departments.** The consistent $1.33x$ to $1.35x$ relative risk multiplier in the South represents clear
    mathematical proof that a highly restrictive healthcare environment significantly increases the likelihood that a maternal
    complications will escalate into a life-threatening situation requiring the emergency room.
"""

# ==========================================================
# ACTUAL INPUTS - ICU ADMISSION
# ==========================================================

# Overview
overview_nb = inspect.cleandoc("""
    Naive Bayes is a family of supervised machine learning algorithms based on *Bayes' Theorem*, a mathematical formula
    used to determine the conditional probability of an event occuring given prior knowledge of conditions related to
    that event. The algorithm is "naive" because it relies on a strict, often unrealistic assumption: it assumes all
    features are independent of one another given the target class. In a maternal health context, a Naive Bayes model
    assumes risk factors like maternal age, smoking during pregnancy, and gestational hypertension are completely unrelated
    to maternal mortality. Despite this oversimplification, Naive Bayes models are computationally efficient and scalable.

    ##### Multinomial Naive Bayes

    Multinomial Naive Bayes is designed for data that represents discrete counts or frequencies. It models the
    feature distribution using a multinomial distribution, which calculates how often a specific event or feature occurs
    within a given class. Multinomial Naive Bayes is commonly used for text classification, e.g. spam detection, document
    classification, and sentiment analysis.

    ##### Bernoulli Naive Bayes

    Bernoulli Naive Bayes is designed for binary data (e.g. True/False, 1/0). It models the feature distribution using
    a multivariate Bernoulli distribution, which is only concerned with whether a feature is present or absent, not its
    frequency. Bernoulli Naive Bayes is commonly used for spam detection, text classification, sentiment analysis and to
    determine whether a certain word is present in a document.

    ##### Categorical Naive Bayes

    Categorical Naive Bayes is designed for categorical features that are *encoded as distinct choices or levels*, where
    categories do not necessarily have a numerical order or magnitude. It isolates each feature and calculates the 
    probability of each individual category within that feature for each class. Categorical Naive Bayes is commonly
    used for medical diagnoses, customer churn analysis, weather forecasting, and product categorization.

    This study employs Categorical Naive Bayes to predict whether or not a pregnant person is admitted to the Intensive
    Care Unit (ICU) during labor and delivery. It is well suited for birth records from [the National Center for Health
    Statistics (NCHS)](https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm), which contain many useful categorical
    features such as maternal demographic information and binary features such as indicators for risk factors and 
    clinical procedures.
""")

# Data Prep
cleaning_code_nb = "https://github.com/amberteetsel/maternal-health/blob/8c2f89b8fbb9202b46e08dc31e631522d8d9850a/src/cleaning/birth_preproccessing.py"
sample_data_url = "https://github.com/amberteetsel/maternal-health/blob/8c2f89b8fbb9202b46e08dc31e631522d8d9850a/data/clean/NCHS-Birth/birth_icu_processed.csv"
data_prep_nb = """
    Supervised learning models require **labeled** data. In this case, the label is `ICU_Admit` with values Yes or No.
    Preparing Birth data files for Naive Bayes classification starts with paring down the availabe 103,811,757 entries
    to a manageable sample while mitigating class imbalance. Only about 0.4 percent of births from 2018 - 2024 involved
    maternal ICU admission, so the extraction pipeline was configured to retrive all rows where `ICU_Admit = Yes` and
    just 1 percent of rows where `ICU_Admit = No`. The resulting dataset has 303,565 entries with 15 percent representing
    cases where the mother was admitted to the ICU. After handling missing or unknown data, continuous features are
    binned into distinct categories. Feature columns are separated from the target column (`ICU_Admit`), then all
    features are encoded (text values are mapped to numbers).
"""
input_data_nb = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "NCHS-Birth", "birth_icu_processed.csv"))
train_test_nb = """
    After encoding, the data is split into training and testing sets. The model will be fitted using training data, then
    used to make predictions about the testing data. These predictions are compared against real values to evaluate
    model performance. The training and testing sets must be entirely disjoint because otherwise the model will simply
    memorize answers instead of learning underlying patterns. Training and testing sets for this data were split using
    stratification to ensure each receives a proportional slice of positive ICU cases.
"""
train_data = pd.read_csv(os.path.join(nb_res, "X_train_nb.csv"))
test_data = pd.read_csv(os.path.join(nb_res, "X_test_nb.csv"))

# Code
model_code_url = "https://github.com/amberteetsel/maternal-health/blob/e19201223962c7a42d3fd0e95a19dad191d28469/src/models/naive_bayes.py"

# Results
class_report_nb = pd.read_csv(os.path.join(nb_res, "nb_report.csv"))

results_intro_nb = """
    The Categorical Naive Bayes model was evaluated on a stratified test dataset consisting of 60,713 births, optimized
    to preserve the rare minority class of maternal ICU admissions.
"""

result_table_nb = pd.read_csv(os.path.join(nb_res, "nb_results_interpret.csv"))
cm_nb = os.path.join(nb_res, "nb_cm.png")
results_interpret_nb = """
    ##### High Accuraacy but Poor Recall

    While an overall accuracy of 89.0% and ROC AUC of 0.8360 seem to suggest a highly successful model, diving deeper to
    class-specific metrics reveals significant issues. The model achieves a recall of only 50%. In a real-world clinical
    setting, this means that out of 9,360 mothers who experienced critical life-threatening emergencies requiring ICU 
    admission, the **model would fail to flag 4,704 of them (False Negative)**. In a public health setting, a 50% failure
    rate for identifying high-risk patients is unacceptable.

    ##### Fundamental Incompatibility of Naive Bayes
     
    The model has such poor recall because of the underlying math in Naive Bayes algorithms that assumes conditional
    independence. Naive Bayes treats every feature as a completely independent variable, but in reality maternal medicine
    is driven by highly correlated, compounding risks. For example, preeclampsia and gestational diabetes often co-occur
    and escalate in severity as maternal age increases. Since Naive Bayes treats these and other features as parallel
    silos, it mathematically disregards increased risk from compounding pathologies and thus returns a high volume
    of false negatives. 
"""

# Conclusions
conclusion_nb = """
    Though the Categorical Naive Bayes model exhibited strong baseline discriminative capabilities (ROC AUC 0.8630), this
    was largely achieved through the intentional stratified sampling approach during data preparation.
    The independence assumption makes Naive Bayes models fundamentally ill-suited for maternal risk classification.
"""