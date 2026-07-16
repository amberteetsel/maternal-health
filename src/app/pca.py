#### HELPER FUNCTION TO DISPLAY PRINCIPAL COMPONENT ANALYSIS RESULTS

# Dependencies
import os
import pandas as pd
import numpy as np
import inspect
import streamlit as st
import plotly.graph_objects as go

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
pca_res = os.path.join(BASE_DIR, "resources", "pca")

# Function
def render_pca(overview: list,
               prep_text: str,
               cleaning_code: str,
               df_before: pd.DataFrame,
               df_after: pd.DataFrame,
               data_download_url: str,
               pipeline_all: list,
               pca_code: str,
            #    pipeline_opt: list,
               conclusion: str
            
               ):
    # ----------------------------------------------------
    # internal helper function for pipeline/results rendering
    # ----------------------------------------------------
    def _render_pipeline_asset(item: dict, step_index: int, unique_key_prefix: str):
        """Processes and handles titles, figures, maps, dataframes, captions, and interpretations."""
        if item.get("title"):
            # numbered index
            st.markdown(f"**{step_index + 1}. {item['title']}**")
            
        fig = item.get("fig")
        
        # check for dataframes first and exit early if matched
        if isinstance(fig, pd.DataFrame):
            st.dataframe(fig, width='stretch')
            # render annotations if present for this dataframe step
            if item.get("caption"):
                st.caption(item["caption"])
            if item.get("interpretation"):
                st.markdown(f"🔬 **Interpretation:** {item['interpretation']}")
            st.markdown("<br>", unsafe_allow_html=True)
            return  # exit function immediately for dataframes

        # check for interactive html maps
        if isinstance(fig, str) and fig.endswith(".html"):
            if os.path.exists(fig):
                with open(fig, 'r', encoding='utf-8') as f:
                    html_data = f.read()
                height = item.get("html_height", 460)
                st.iframe(src = html_data, height=height)
            else:
                st.error(f"Missing map component: {fig}")
                
        # check for local image paths
        elif isinstance(fig, str):
            # st.image(fig, width='stretch')
            st.image(fig, width='content')
            
        # check for interactive plotly go instances
        elif isinstance(fig, go.Figure):
            st.plotly_chart(fig, width='stretch', key=f"{unique_key_prefix}_chart_{step_index}")
            
        # fallback for matplotlib / seaborn figures only
        else:
            st.pyplot(fig)
            
        # standard rendering path for captions/interpretations of non-dataframe assets
        if item.get("caption"):
            st.caption(item["caption"])
            
        if item.get("interpretation"):
            st.markdown(f"🔬 **Interpretation:** {item['interpretation']}")
            
        st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # internal helper function for overview rendering
    # ----------------------------------------------------
    def _render_overview_asset(item: dict):
        """Processes Overview Section, displays images next to text"""
        if item.get("title"):
            st.markdown(f"##### {item['title']}")

        try:
            fig = item.get("image")
        except:
            fig = None
        
        if fig:
            c1, c2 = st.columns(2)

            with c1:
                st.markdown(item['text'])
            with c2:
                st.image(fig, width=560, caption=item['caption'])
        
        else:
            st.markdown(item['text'])
            
        st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # OVERVIEW
    # ----------------------------------------------------
    st.subheader("Principal Component Analysis (PCA) Overview")
    with st.expander("Read About PCA", expanded=True):
        for item in overview:
            _render_overview_asset(item)

    # ----------------------------------------------------
    # DATA PREPARATION
    # ----------------------------------------------------
    st.subheader("Data Preparation")
    st.markdown(prep_text)

    st.markdown(f"🔗 **[Download Sample Unlabeled Dataset]({data_download_url})**")
    st.markdown(f"🔍 **[View Cleaning Code]({cleaning_code})**")
    # constructing two equal columns for side-by-side data comparison
    prep_col1, prep_col2 = st.columns(2)
    with prep_col1:
        st.markdown("📋 **Original Contextually Grounded Dataset**")
        st.dataframe(df_before, width='stretch')
        
        # schema info rendered within a clean layout expander
        with st.expander("🛠️ View Original Feature Schema Details", expanded=False):
            schema_df1 = pd.DataFrame({
                "Data Type": df_before.dtypes.astype(str),
                "Non-Null Count": df_before.notnull().sum(),
                "Null Count": df_before.isnull().sum()
            })
            st.table(schema_df1)

    with prep_col2:
        st.markdown("🔢 **Mathematical Input Feature Space (Strictly Unlabeled & Scaled)**")
        st.dataframe(df_after, width='stretch')
        
        with st.expander("🛠️ View Modeled Vector Space Schema Details", expanded=False):
            schema_df2 = pd.DataFrame({
                "Data Type": df_after.dtypes.astype(str),
                "Non-Null Count": df_after.notnull().sum(),
                "Null Count": df_after.isnull().sum()
            })
            st.table(schema_df2)
            
    st.markdown("---")

    # ----------------------------------------------------
    # MODELING & RESULTS
    # ----------------------------------------------------
    st.subheader("Model Run & Results")

    # st.markdown("#### Method 1: All Available Features")
    st.markdown(f"👾 [View Code]({pca_code})")
    for i, item in enumerate(pipeline_all):
        _render_pipeline_asset(item, step_index=i, unique_key_prefix="all")

    # st.markdown("#### Method 2: Optimal Features Only")
    # st.markdown(f"👾 [View Code]()")
    # for i, item in enumerate(pipeline_opt):
    #     _render_pipeline_asset(item, step_index=i, unique_key_prefix="opt")

    # ----------------------------------------------------
    # CONCLUSION
    # ----------------------------------------------------
    st.subheader("Conclusions")
    st.markdown(conclusion)

# Function to display PCA on Birth Records
def render_pca_birth(
        overview_text: str,
        overview_figs: list,
        overview_figs_text: list,
        cleaning_text: str,
        data_raw_link: str,
        data_clean_link: str,
        data_clean: pd.DataFrame,
        data_raw: pd.DataFrame,
        model_code_url: str,
        results_pipeline: list,
        conclusion_text: str
):
    
    # overview
    st.subheader("PCA Overview")
    st.markdown(overview_text)
    st.markdown("##### Exploratory Data Analysis")
    o1, o2 = st.columns(2)
    with o1:
        st.markdown(overview_figs_text[0])
        st.image(overview_figs[0], width='stretch')
        st.caption("Maternal morbidity and delivery complication features are far less common among U.S. mothers than risk factors like diabetes and hypertension.")
    
    with o2:
        st.markdown(overview_figs_text[1])
        st.image(overview_figs[1], width='stretch')
        st.caption("AIAN and NHOPI mothers comprise only 1% of the population in the dataset.")

    # data prep
    st.subheader("Data Preparation")
    st.markdown(cleaning_text)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Unscaled Dataset (First 100 Rows)**")
        st.markdown(f"🔗 **[Download Dataset]({data_raw_link})**")
        st.dataframe(data_raw, hide_index=True)

    with c2:
        st.markdown("**Scaled Input Data**")
        st.markdown(f"🔗 **[Download Dataset]({data_clean_link})**")
        st.dataframe(data_clean, hide_index=True)
    
    st.markdown("---")

    # modeling/results
    st.subheader("Model Results")
    st.markdown(f"👾 [View Code]({model_code_url})")

    for x in results_pipeline:
        st.markdown(f"##### {x.get('title')}")
        fig = x.get('fig')
        if isinstance(fig, pd.DataFrame):
            st.dataframe(fig, hide_index=True)
        elif isinstance(fig, str):
            if x.get('width'):
                st.image(fig, width=x.get('width'))
            else:
                st.image(fig, width='content')
        if x.get('caption'):
            st.caption(x.get('caption'))
        
        st.markdown(x.get('text'))
        
    st.markdown("---")

    # conclusions
    st.subheader("Conclusions")
    st.markdown(conclusion_text)

# ==========================================================
# ACTUAL INPUTS - BIRTH DATA
# ==========================================================
overview_text_birth = """
    The primary purpose of this analysis is to answer the following research questions using national birth records from 2018 to 2024:

    #### Can national clinical risk factors and complications be compressed into a unified Maternal Clinical Risk Profile?

    #### How does this profile vary across racial and age demographics nationally?

    By isolating demographic data (Race/Ethnicity, Age) before fitting the PCA model, these characteristics are prevented from
    mathematically biasing the clinical components (Risk Factors, Complications). This enables projecting the resulting
    patient-level clinical risk scores back onto demographic groups to expose systematic national health disparities.

    The following binary clinical data was collected for analysis:

    **Risk Factors Analyzed:**
    * Gestational Diabetes (`rf_gdb`)
    * Gestational Hypertension (`rf_ghyp`)
    * Hypertension Eclampsia (`rf_ehyp`)

    **Morbidity/Complications Analyzed:**
    * Maternal Transfusion (`mm_trans`)
    * Perineal Laceration (`mm_plac`)
    * Ruptured Uterus (`mm_rupt`)
    * Unplanned Hysterectomy (`mm_uhyst`)
    * Intensive Care Unit (ICU) Admission (`mm_icu`)

"""
feature_means_birth = os.path.join(pca_res, "birth_clinical_prevalence.png")
race_distr_birth = os.path.join(pca_res, "race_distribution.png")
overview_figs_birth = [feature_means_birth, race_distr_birth]
feature_means_text = """
    The chart below shows prevalence of each clinical feature within the dataset. In general, risk factors are far more common
    than severe complications. Of severe complications, perineal lacerations are the most common while a ruptured uterus is
    relatively rare (experienced by fewer than 1 woman out of 2,000). For risk factors, hypertension eclampsia is the rarest. 
"""
race_distr_text = """
    The chart below shows the distribution of maternal race and ethnicity within the dataset. Non-Hispanic White and Hispanic mothers comprise 3/4 of the data
    (Non-Hispanic White $50\\%$, Hispanic $25\\%$). Black and Asian mothers make up another $20\\%$ while American Indian/Alaska Native and Native Hawaiian/Pacific
    Islander mothers represent just $1\\%$ of the population.
"""
overview_fig_text_birth = [feature_means_text, race_distr_text]
data_prep_birth = """
    Data for this analysis comes from the [CDC's National Center for Health Statistics Birth records](https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm).
    After extracting a representative 5% sample of these records from 2018 to 2024, data was cleaned by mapping numeric codes for race
    and ethnicity to text descriptors and filling in missing binary clinical data as "No".

    A critical preparation step was scaling the clinical data. Because clinical complications occur at different baseline rates (for
    example, gestational diabetes is fairly common while a ruptured uterus is rare), unstandardized variables would cause the
    high-frequency conditions to dominate the principal component analysis. Standardizing features into z-scores with mean of 0.0 and
    standard deviation of 1.0 levels the playing field and allows the covariance of rarer, more serious complications to be detected.
"""
data_raw_link_birth = "https://github.com/amberteetsel/maternal-health/blob/22d25c792adeb876b699b9b450bfa27e1525bc3e/resources/pca/birth_data_raw.csv"
data_clean_link_birth = "https://github.com/amberteetsel/maternal-health/blob/22d25c792adeb876b699b9b450bfa27e1525bc3e/resources/pca/birth_data_processed.csv"
data_raw_birth = pd.read_csv(os.path.join(pca_res, "birth_data_raw.csv"))
data_clean_birth = pd.read_csv(os.path.join(pca_res, "birth_data_processed.csv"))

# Model Results
model_code_url_birth = "https://github.com/amberteetsel/maternal-health"    #### PLACEHOLDERS!!!!!!REPLACE!!!!!!!!!!!
exp_var_text = """
    The scree plot below demonstrates that when running PCA on 8 clinical risk and morbidity features, a single, unified profile
    of risk did not emerge. If that were the case, $PC_1$ and perhaps $PC_2$ together would explain a significant portion of overall
    variance. In this scenario $PC_1$ captures just $18.02\\%$ and $PC_2$ captures just $13.3\\%$. It would take at least seven 
    principal components to capture $90\\%$ of overall explained variance.
"""
loadings_text = """
    The loadings for each principal component reveal a clear, clinically logical split between $PC_1$ and $PC_2$. 

    **1. $PC_1$: Acute Trauma Index**
    
    The loadings for all maternal morbidity columns (excluding Perineal Laceration) are relatively high, ranging from $0.341$ for
    Ruptured Uterus to $0.575$ for an Unplanned Hysterectomy. On the other hand, loadings for all risk factor columns are close
    to zero (all under $0.06$). This means $PC_1$ is not a general measure of routine pregnancy complications, but rather a highly
    specific index of **severe, acutre maternal morbidity and delivery trauma.** A high $PC_1$ score indicates a patient who
    experienced a catastrophic delivery event involving a blood transfusion, ruptured uterus, unplanned hysterectomy, and intensive
    care. Recalling feature prevalence, it is intuitive that Perineal Laceration would not be a primary driver of $PC_1$ because
    it's a fairly common symptom and can be much less severe than the other delivery complications.

    **2. $PC_2$: Gestational Cardio-Metabolic Risk**

    $PC_2$ represents a completely separate pathway. It's mathematical structure is almost entirely orthogonal to $PC_1$.
    The weights for risk factors shifted dramatically, from near zero to at least
    0.68 for the more common gestational risks and to 0.22 for the less frequent eclampsia. In contrast, the acute delivery traumas
    drop to near-zero or even slightly negative loadings. This makes $PC_2$ a measure of gestational metabolic and cardiovascular
    dysfunction. A high $PC_2$ score indicates a patient managing chronic prenatal conditions (diabetes and high blood pressure) that
    develop during pregnancy without necessarily triggering a delivery emergency. Again, it makes intuitive sense that eclampsia has
    a lower relative weight for this component because it is a very severe, life-threatening condition more akin to the acute delivery
    traumas in nature than general risk factors like diabetes.

"""
rq2_text_birth = """
    Mapping each patient's individual $PC_1$ and $PC_2$ scores back onto their demographic information reveals stark, systemic
    national disparities across maternal ages and racial/ethnic groups.

    ##### Acute Delivery Trauma ($PC_1$)

    Plotting Mean $PC_1$ score against Maternal Age reveals two distinct trends:

    **1. Late-Life Age Risk**

    For the majority of the population (represented by the consolidated "Other" baseline and the lowest-risk group, Hispanic mothers),
    the risk of acute delivery trauma remains flat and near national average through early and mid-reproductive years. However,
    past age 35 (and accelerating exponentially past age 40) the $PC_1$ scores climb steeply. This provides clear mathematical and visual
    confirmation of heightened physical delivery risks associated with advanced maternal age.
    
    **2. Extreme Outliers**

    The visual highlights that acute trauma is not distributed evenly but concentrated within a very small portion of the data ($1\\%$), 
    recalling the distribution of maternal race and ethnicity). American Indian / Alaka Native (AIAN) mothers represent
    the highest-risk national outlier in acute delivery trauma, experiencing a volatile surge in $PC_1$ scores beginning as early
    as age 25 and peaking around age 43. Native Hawaiian / Pacific Islander (NHOPI) mothers also exhibit an elevated baseline risk
    that spikes much earlier than the national average. Hispanic mothers consistently track as the lowest relative risk group.

    ##### Gestational Cardio-Metabolic Risk ($PC_2$)

    The trajectory of $PC_2$ tells a different story.

    **1. Linear Aging Process**

    Unlike the acute trauma index, which remains flat before spiking later in life, cario-metabolic risk shows a steady, linear
    and universal upward march across the entire maternal lifespan. Regardless of race or ethnicity, older mothers are progressively
    more likely to experience gestational diabetes and hypertension.

    **2. Racial Divergence**

    Although the upward slope is universal, the vertical starting baselines are unequal. AIAN and Asian mothers carry the highest
    national burdens of gestational cardio-metabolic risk across virtually the entire age spectrum. Hispanic mothers consistently
    track at the bottom of the cardio-metabolic risk index.
"""
pca_pipeline_birth = [
    {
        'title': "Cumulative Explained Variance of Principal Components",
        'text': exp_var_text,
        'fig': os.path.join(pca_res, "birth_scree_plot.png"),
        'caption': "Principal Component 1 captures just 18% of overall variance, which is to be expected because the severe complications driving it are sparse among the data.",
        'width': 850
    },
    {
        'title': "Feature Loadings for $PC_1$ and $PC_2$",
        'text': loadings_text,
        'fig': pd.read_csv(os.path.join(pca_res, "birth_pc1_pc2.csv")),
        'caption': None,
        'width': None,
    },
    {
        'title': "Demographic Trends",
        'text': rq2_text_birth,
        'fig': os.path.join(pca_res, "birth_pca_race_plots.png"),
        'caption': "Comparing PC1 and PC2 scores for different racial/ethnic groups by maternal age.",
        'width': 'stretch'
    }
]

conclusions_text_birth = """
    #### Can national clinical risk factors and complications be compressed into a unified Maternal Clinical Risk Profile?

    No. Though PCA was applied effectively, the first two principal components ($PC_1, PC_2$) only explained $31.31\\%$ of
    overall variance in the dataset so they cannot be considered to comprise a "unified" risk profile. However, the exercise
    still provided valuable insight; the data naturally splits into two highly distinct, logical dimensions of maternal health
    risks and complications. These are Acute Delivery Trauma ($PC_1$) and Gestational Cardio-Metabolic Risk ($PC_2$).

    #### How does this profile vary across racial and age demographics nationally?

    The structural variance of each profile across demographic lines offers sociological and clinical insight. While the
    physical effects of aging lead to universal, steady increase in gestational cardio-metabolic risks ($PC_2$) for all pregnant
    people, the translation of those baseline risks into catastrophic, life-threatending delivery-room trauma ($PC_1$) is
    unequal.

    The sharp early-onset spikes in acute trauma ($PC_1$) observed for American Indian / Alaska Native (AIAN) and Native Hawaiian /
    Pacific Islander (NHOPI) mothers, even at younger ages, suggests that systemic healthcare disparities, structural prejudice,
    and unequal access to quality, timely prenatal and obstetric care can fail to resolve manageable cardio-metabolic risks 
    before they cascade into catastrophic delivery emergencies.
"""

# ==========================================================
# ACTUAL INPUTS - HEALTH RANKINGS
# ==========================================================
overview_pca_1 = inspect.cleandoc("""
    The Curse of Dimensionality refers to various challenges and complications that arise when analyzing and organizing
    data in high-dimensional spaces. Dimensions refer to the features or attributes of data; in the context of this study,
    dimensions of the Health Rankings dataset include measures of poverty, unemployment, preventative care, maternal
    mortality, and more. As the number of features grows, the volume of the data space increases exponentially and
    available data observations become sparse. In high-dimensional spaces traditional analytical models break down because
    geometric distances (e.g. Euclidean distance) are less distinct, meaning the distance between two highly
    dissimilar data points and the distance between two highly similar points look mathematically identical. Humans
    cannot visualize beyond three dimensions (3D) so it's also difficult for analysts to conceive of and understand
    high-dimensional feature spaces.
""")
overview_pca_1_image = os.path.join(pca_res, "dimensionality_curse.png")
overview_pca_2 = inspect.cleandoc("""
    Dimensionality reduction is the process of compressing high-dimensional feature spaces into a lower-dimensional
    subspace, typically 2D or 3D, while retaining as much structural variation as possible. Dimensionality reduction
    is crucial to address the problems caused by high dimensionality:

    * Data sparsity
    * Increased computation
    * Multicollinearity
    * Overfitting
    * Performance degradation
    * Visualization challenges

    Reducing dimensionality can eliminate multicollinearity and background noise that destabilizes machine learning
    algorithms. By compressing the data, it drastically improves computational efficiency while solving the issues of data
    sparsity and data visualization by turning a multi-variable matrix into a clear, interpretable landscape in two or
    three dimensions.
""")

overview_pca_3 = inspect.cleandoc("""
    Principal Component Analysis (PCA) is an unsupervised linear transformation technique for dimensionality reduction
    that projects data onto brand-new, uncorrelated axes known as *Principal Components*. Rather than selecting or
    deleting specific raw variables, PCA uses all original features to calculate a completely new coordinate system.

    * **Principal Component 1 (PC1):** The first geometric axis, built to point in the direction of the absolute maximum
    variance in the data.
    * **Principal Component 2 (PC2):** The second geometric axis, constructed to be completely orthogonal to PC1 and 
    capture the highest remaining variance. 

    In this way, the first few components generated should contain the vast majority of a dataset's information (variance),
    allowing analysts to discard additional components wtih minimal information loss.
""")
overview_pca_3_image = os.path.join(pca_res, "pca_projection.png")

overview_pca_4 = inspect.cleandoc("""
    In practice, PCA is based on the decomposition of a dataset's *covariance matrix*, which tracks how each feature
    moves in relation to every other feature. PCA uses linear algebra to extract critical information from this matrix:
    **Eigenvectors** and **Eigenvalues**.

    * **Eigenvectors** (The Direction): An eigenvector is a non-zero vector whose direction remains entirely unchanged
    when a linear transformation is applied. In PCA, eigenvectors represent the directions of the new principal component
    axes and are weighted/combined to build the new coordinate space.
    * **Eigenvalues** (The Magnitude): An eigenvalue is a scalar value that corresponds to a specific eigenvector. It
    measures the absolute magnitude of variance captured along that specific principal component axis. The principal
    component with the largest eigenvalue represents the axis of maximum variance, or PC1.
""")

overview_pca = [
    {
        'title': "The Curse of Dimensionality",
        'text': overview_pca_1,
        'image': overview_pca_1_image,
        'caption': "As the number of features increases, the classifier's performance also increases until it reaches the optimal dimensionality."
    },
    {
        'title': "Dimensionality Reduction",
        'text': overview_pca_2,
        'image': None,
        'caption': None
    },
    {
        'title': "Principal Component Analysis (PCA)",
        'text': overview_pca_3,
        'image': overview_pca_3_image,
        'caption': "PCA finds the optimal 2D plane to project 3D data while preserving maximum variance."
    },
    {
        'title': "Eigenvalues and Eigenvectors",
        'text': overview_pca_4,
        'image': None,
        'caption': None
    }
]

prep_pca = inspect.cleandoc("""
    PCA requires continuous numerical variables.
    To this end, data from *America's Health Rankings* was further processed to conform with PCA requirements. First,
    the DataFrame was melted into wide format.
    Categorical data such as `State` and `Year` were stripped away to leave only numeric features.
    Features were also reconstructed so that for all columns, a higher number indicated a "worse" outcome than a lower number.
    For example, "Adequate Prenatal Care" was converted to "Inadequate Prenatal Care" by flipping the percentage.
    Z-Score standardization was also used so that varying data magnitudes would not disproportionately dominate the
    analysis.
""")

pca_df_after = pd.read_csv(os.path.join(pca_res, "pca_input_data.csv"))

pca_model_features_all = list(pca_df_after.columns)
pca_model_intro_all = """
    To begin, Principal Component Analysis was applied to the full set of 17 features:
    `{pca_model_features_all}`
"""
pca_all_scree_interpret = """
    Upon executing PCA across all available public health metrics, Principal Component 1 (PC1) and Principal Component 2 (PC2)
    were revealed to explain a cumulative total of only $45.18\\%$ of the dataset's variance. In public health analytics, 
    this low dimensionality compression indicates a high degree of systemic complexity. Rather than being driven by a 
    singular, underlying socioeconomic factor, state-level healthcare landscapes are deeply fragmented; a state's capacity
    in clinical provider retention does not linearly predict its performance in maternal care or preventative wellness 
    visits. To capture the standard $70\\%$ threshold of total system information, the model must be expanded to include 5 or
    6 components.
"""
pca_all_loading_interpet = inspect.cleandoc("""
                                            
    **Most Influential Features, PC1:**
                                            
    * Positive: `Maternal Mortality`, `Infant Mortality`, `Patients Per Doctor`, `Poverty`
    * Negative: None

    Principal Component 1 (PC1) is an axis of Systemic Healthcare Deprivation and Vulnerability. PC1 has no meaningful
    negative loading coefficients, indicating this component captures a unified, compounding directional force. The
    positive pole is driven by a mix of structural resource strain (poverty, patients per doctor) and severe medical
    failures (maternal and infant mortality). This makes PC1 a useful index of compounding public health dangers. States
    with high scores on this axis are stuck in a system where economic deprivation and shortage of healthcare providers
    directly relate to elevated risk of maternal and infant death. Conversely, states with low scores on this axis tend
    to have well-resourced, protective health infrastructure and better outcomes.

    **Most Influential Features, PC2:**
                                            
    * Positive: `No Postpartum Visit`, `Inadequate Prenatal Care`, `No Preventative`
    * Negative: `Smoking During Pregnancy`, `Gender Pay Gap`

    Principal Component 2 (PC2) can be characterized as an axis of Preventative Care Underutilization vs. Socio-Behavioral
    Risk. The positive pole is heavily influenced by clinical non-attendance metrics (no postpartum or preventative visits,
    inadequate prenatal care), capturing environments where system barriers prevent patients from utilizing available care.
    Conversely, the negative pole reveals an inverse relationship with `Smoking During Pregnancy` and the `Gender Pay Gap`.
    This means states suffering from high behavioral risks (smoking) and economic disparities (wage gaps) often exhibit
    *paradoxically higher* rates of prenatal and postpartum utilization.

    Notably, the feature `Maternity Care Desert` exhibits a nearly identical loading magnitude across both primary axes
    but in opposite directions ($+0.297$ for PC1, $-0.278$ for PC2). This indicates that `Maternity Care Desert` acts as a 
    structural pivot: it shares equal amount of variance with both components, moving in direct alignment with system
    traits of PC1 while simultaneously moving inversely with dynamics captured by PC2.
""")

pca_all_assets = [
    {
        'title': "Cumulative Variance Analysis",
        'fig': os.path.join(pca_res, "pca_scree_all.png"),
        'caption': 'Shows the cumulative variance captured by $n$ principal components. In this context, variance is a proxy for information',
        'interpretation': pca_all_scree_interpret
    },
    {
        'title': "Loadings Plot",
        'fig': os.path.join(pca_res, "pca_loading_all.png"),
        'caption': "Shows correlations between raw features and a principal component. A loading coefficient close to +1.0 or -1.0 means that feature has a massive influence on direction of component. A loading coefficient close to 0.0 means the feature has almost no impact on that component.",
        'interpretation': pca_all_loading_interpet
    }
]

pca_conclusion = """
    The deployment of Principal Component Analysis on the state health rankings data yielded important structural insights
    to shape understanding and honestly diagnose US maternal healthcare deficiencies.

    ##### A Fragmented System

    In an ideal machine learning scenario, PCA is expected to generate a PC1 and PC2 that capture the majority of a
    dataset's information (70 - 90 percent). However, applying PCA to the full set of 17 features revealed that PC1 and
    PC2 accounted for a cumulative variance of only $45.18\\%$. From a data-reduction standpoint, forcing a flat 2D compression
    is not worth the substantial loss of information ($54.82\\%$). 

    ##### Takeaways

    Additional feature optimization or data manipulation may be needed to effectively reduce dimensionality.
"""


pca_opt_assets = []

