import os
import pandas as pd
import streamlit as st

# Display Function
def render_questions(qa_pipeline: list):

    for i, x in enumerate(qa_pipeline):

        st.markdown(f"#### {i+1}. {x.get('q')}")
        st.markdown(x.get('a'))
        st.divider()

# Actual Inputs
rq1 = "Is there a statistically significant difference in maternal mortality/morbidity rates between states with explicit constitutional/legal protections and states with abortion bans or severe restrictions?"
rq2 = "How did historical national trends in pregnancy, birth, abortion, and miscarriage rates cluser across time?"
rq3 = "Do these temporal clusters align with major federal judicial milestones (*1973 Roe, 1992 Casey, 2022 Dobbs*)?"
rq4 = "Can distinct clusters of states/geographic regions be identified based on reproductive policies?"
rq5 = "Do clusters reliably predict disparities in maternal health outcomes?"
rq6 = "Can national clinical risk factors and complications be compressed into a unified Maternal Clinical Risk Profile?"
rq7 = "How does this profile vary across racial and age demographics nationally?"
rq8 = "Given an emergency room patient's demographic profile, what is the conditional probability of them presenting with a severe pregnancy-related complication based on their geographic macro-region's reproductive legal status (Highly Restricted South vs. Highly Protected Northeast)?"
rq9 = "How did the 1973 decision in *Roe v. Wade* impact reproductive infrastructure usage and maternal health outcomes?"
rq10 = "How did the 2022 decision in *Dobbs v. Jackson* impact reproductive infrastructure usage and maternal health outcomes?"

a1 = """
    * **Yes.** Supervised learning models (SVM, Neural Networks) successfully classified a state's abortion policy with up to **98% to 100% accuracy**
    using *only* the state's clinical outcomes.
    * States with legal restrictions on abortion have demonstrably worse health metrics across the board including higher maternal/infant
    mortality, fewer doctors, and more maternity care deserts.
"""

a2 = """
    * National trends clustered into **four distinct, strictly sequential chronological eras** rather than fluctuating randomly:

        * **Cluster 2 (1973-1976):** Immediate Post-*Roe* Era
        * **Cluster 3 (1977 - 1996):** The Stablization Era
        * **Cluster 1 (1997 - 2010):** The Contraceptive Era
        * **Cluster 0 (2011-2020):** The Targeted Regulation of Abortion Providers (TRAP) Era
"""

a3 = """
    * **Yes.** The chronological breaks correspond directly to legal shifts.
    * The 1973 *Roe* decision initiated Cluster 2. The 1992 *Casey* decision sits near the apex of Cluster 3, demonstrating that its
    "undue burden" framework took a few years to structurally manifest. The rapid accumulation of state TRAP laws in Cluster 0 represent
    a time of legislative attack on abortion rights that directly set the stage for the 2022 *Dobbs* ruling.
"""

a4 = """
    * **No, not directly.** The unsupervised clustering models (K-Means and Hierarchical) were trained strictly on **underlying health and
    infrastructure metrics**, specifically maternity care deserts, maternal mortality, unintended pregnancy, and doctor-to-patient ratios.
    * However, when these health-profile clusters were mapped geographically, they **strongly mirrored* state-level reproductive policies.
    * The quagmire of state-wide post-*Dobbs* anti-abortion legislation proved difficult to navigate due to disjoint timelines and
    state-specific legal nuances.
"""

a5 = """
    * **Yes, by definition.** Because the clusters were built directly from healthcare access and maternal health outcomes, they mathematically
    outline the starkest disparities in the nation. The cluster representing 'High Risk' and 'Poor Access' reflects a reality where
    patients face structurally higher maternal mortality rates, larger care deserts, and fewer active OB/GYNs—disparities that align
    almost perfectly with restrictive policy environments.
"""

a6 = """
    * **No.** Dimensionality reduction (PCA) was unable to successfully compress many variables into single features that explain most of
    overall variance, and decision trees showed that maternal morbidity is driven by compounding interactions of risk factors (e.g. diabetes) and
    acute clinical emergencies (e.g. ruptured uterus).
"""

a7 = """
    * While a single profile could not be identified, clear trends were observed across racial and age demographics.
    * **Race:** Non-Hispanic Black and Hispanic demographics show a higher baseline clinical risk profile compared to Non-Hispanic
    White demographics.
    * **Age:** Younger demographics (age 15-34) are disproportionately forced to seek care in emergency departments compared to mothers
    of advanced age (35+).
"""

a8 = """
    * Seeking care in the highly restricted South acts as a systemic environment risk factor, adding a **1.33x to 1.35x relative
    risk multiplier** across *all* demographic profiles.
    * For example, a White patient of average maternal age has a **2.45%** probability of presenting with Severe Maternal Morbidity (SMM) in
    the Northeast, but this risk doubles to **4.95%** for her demographic peer in the South.
"""

a9 = """
    * **To be determined.** Quantitative evaluation of this impact requires detailed patient-level longitudinal data from the 1960s and
    1970s. Unfortunately, the primary datasets used for this analysis had severe historical limitations.
    * State-level health metrics (America's Health Rankings) is only available for 2018 - 2023.
    * CDC data (birth records, emergency room data) does not contain the requisite level of granularity for 20th century records.
"""

a10 = """
    * **To be determined.** Reporting lags and geographical redactions of primary datasets prevented thorough analysis of this question.
    * The gold-standard dataset for tracking SMM (the CDC's NCHS birth data) typically has a 2-to-3-year release lag. Granular
    patient-level records mapping the post-*Dobbs* landscape are highly restricted and demand costly access fees.
    * Modern public-use NCHS files redact geographic identifiers like state of residence. Without state-level identifiers it is impossible
    to map health outcomes to state legal policies.
"""

q_and_a = [
    {'q': rq1, 'a': a1},
    {'q': rq2, 'a': a2},
    {'q': rq3, 'a': a3},
    {'q': rq4, 'a': a4},
    {'q': rq5, 'a': a5},
    {'q': rq6, 'a': a6},
    {'q': rq7, 'a': a7},
    {'q': rq8, 'a': a8},
    {'q': rq9, 'a': a9},
    {'q': rq10, 'a': a10}
]
