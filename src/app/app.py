import gc
# Website Code
import sys
print("--- LOG: STREAMLIT IS RUNNING ON PYTHON EXECUTABLE:", sys.executable)
print("--- LOG: LOOKING FOR PACKAGES IN PATHS:", sys.path)

# Dependencies - Packages
import streamlit as st
import pandas as pd
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import inspect

# Inject global CSS to color headers and subheaders
st.markdown(
    """
    <style>
    /* Targets st.subheader and ### markdown */
    h3 {
        color: #b13c6c !important;
    }
    /* Optional: Targets st.header and ## markdown */
    h2 {
        color: #8f3371 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dependencies - Helper Modules
from data_view import data_source_section
from stacked_maps import generate_stacked_us_maps
from policy_maps import create_ban_limit_map, create_protection_map
from memory import reduce_df_memory

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Homepage
st.set_page_config(page_title="MaternalHealth", layout="wide")

# Custom Styling
# Insert this near the top of src/app/app.py right after st.set_page_config()
st.markdown(
    """
    <style>
        /* Import premium web fonts */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');
        
        /* Apply font rules across titles, text, and data layouts */
        html, body, [data-testid="stMarkdownContainer"], p, h1, h2, h3, h4, h5, h6 {
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        /* Keep code metrics clean with monospacing if desired */
        code, pre {
            font-family: 'Fira Code', monospace !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Exploring Maternal Health Outcomes in Relation to Healthcare Policy")
st.markdown("---")

# Initialize Tabs
t1, t2, t3, t4, t5 = st.tabs([
    "Introduction",
    "Conclusions",
    "Data Prep & EDA",          # add sub-tabs inside for each data source
    "Models",                   # add sub-tabs inside for each model type
    "References"                # citations
])


##############################################################
# TAB 1: INTRODUCTION
##############################################################
import intro

with t1:
    st.header("Introduction")

    st.subheader("Background & Policy Context")
    st.write(intro.intro_p1)
    
    # Figure 1: Maternal Deaths, International Comparison
    left_spacer, center_column, right_spacer = st.columns([1, 2, 1])
    with center_column:
        # Slightly lowered the height to 600px to match the reduced width proportions
        st.iframe(src = intro.infogram_embed_html1, height=600)
        st.caption("""
                   Figure 1: The Commonwealth Fund International Comparison of Maternal Deaths.
                   The overall US maternal death rate (dark green) is 56 percent higher than that of Chile, the developed nation
                   with the second-highest incidence of maternal death, and over 300 percent higher than that of the United Kingdom.
                """)
    
    st.subheader("Research Significance")
    st.write(intro.intro_p2, unsafe_allow_html=True)
    
    # Figure 2: 
    left_spacer, center_column, right_spacer = st.columns([1, 2, 1])
    with center_column:
        # Slightly lowered the height to 600px to match the reduced width proportions
        st.iframe(src = intro.infogram_embed_html2, height=520)
        st.caption("""
                   Figure 2: The Commonwealth Fund National Comparison of Maternal Deaths by Abortion-Access Status.
                   There is a clear pattern of increased maternal death rates for abortion-restricted states.
                """)
        
    # Legal Evolution
    st.subheader("Legal Evolution")
    st.write(intro.intro_p3)

    # Real-World Consequences
    st.subheader("Real Women, Real Consequences")
    st.write(intro.intro_p4)

    # 5th Paragraph
    st.subheader("Research Foundations")
    st.write(intro.intro_p5)

    # Research Qs
    st.subheader("Research Questions")
    with st.container():
        st.markdown("""
                    1. Is there a statistically significant difference in maternal mortality/morbidity rates between states
                        with explicit constitutional/legal protections and states with abortion bans or severe restrictions?
                    2. Does the enactment of a heartbeat ban or total ban correlate with an increase in emergency room 
                        visits related to pregnancy complications?
                    3. What is the average lag between legal enactment of abortion restrictions and measurable changes
                        in statewide birth outcomes?
                    4. Do shifts in reproductive policy correspond with changes in prenatal care utilization or
                        timing of first prenatal visits?
                    5. Can distinct clusters of states/geographic regions be identified based on reproductive policies?
                    6. Do clusters reliably predict disparities in maternal health outcomes?
                    7. What policy features are linked to a decline in rural reproductive healthcare infrastructure?
                    8. How did the 1973 decision in *Roe v. Wade* impact reproductive infrastructure usage and maternal health outcomes?
                    9. How did the 1992 decision in *Planned Parenthood v. Casey* impact abortion restrictions and maternal health outcomes?
                    10. How did the 2022 decision in *Dobbs v. Jackson* impact reproductive infrastructure usage and maternal health outcomes?
                """)

##############################################################
# TAB 3: DATA SOURCES, EDA
##############################################################

@st.cache_data(ttl=3600)  # cache dataframes for 1 hour
def load_project_data():
    """
    Safely reads and caches all raw and cleaned project files in memory.
    """
    # Base folder directories
    raw_path = os.path.join(BASE_DIR, "data", "raw")
    clean_path = os.path.join(BASE_DIR, "data", "clean")
    
    # raw
    er_raw_df = pd.read_csv(os.path.join(raw_path, "CDC-ER", "er_raw.csv"))
    er_raw_df = reduce_df_memory(er_raw_df)

    pregnancy_raw_df = pd.read_csv(os.path.join(raw_path, "Guttmacher", "NatStatePregnancy.csv"))
    pregnancy_raw_df = reduce_df_memory(pregnancy_raw_df)

    policy_raw_df = pd.read_csv(os.path.join(raw_path, "LawAtlas", "policy_raw.csv"))
    policy_raw_df = reduce_df_memory(policy_raw_df)
    
    with open(os.path.join(raw_path, "HealthRankings", "raw_api_snapshot.json"), 'r') as f:
        health_raw_obj = json.load(f)
        
    with open(os.path.join(raw_path, "NCHS-Birth", "births2024_raw.txt"), 'r') as f:
        birth_raw_str = f.read()

    # clean
    er_clean_df = pd.read_csv(os.path.join(clean_path, "CDC-ER", "er.csv"))
    er_clean_df = reduce_df_memory(er_clean_df)

    pregnancy_clean_df = pd.read_csv(os.path.join(clean_path, "Guttmacher", "pregnancy.csv"))
    pregnancy_clean_df = reduce_df_memory(pregnancy_clean_df)
    
    policy_clean_df = pd.read_csv(os.path.join(clean_path, "LawAtlas", "policy.csv"))
    policy_clean_df = reduce_df_memory(policy_clean_df)

    health_clean_df = pd.read_csv(os.path.join(clean_path, "HealthRankings", "health.csv"))
    health_clean_df = reduce_df_memory(health_clean_df)

    birth_clean_df = pd.read_csv(
        os.path.join(clean_path, "NCHS-Birth", "births2024.csv.zip"),
        low_memory=False
    )
    
    return (
        er_raw_df, pregnancy_raw_df, policy_raw_df, health_raw_obj, birth_raw_str,
        er_clean_df, pregnancy_clean_df, policy_clean_df, health_clean_df, birth_clean_df
    )

(
    er_raw, pregnancy_raw, policy_raw, health_raw, birth_raw,
    er_clean, pregnancy_clean, policy_clean, health_clean, birth_clean
) = load_project_data()

# ==============================================================================
import data_view

# Visuals
viz_path = os.path.join(BASE_DIR, "resources", "visuals_eda")
## Health Rankings Interactive Maps
health_v1 = generate_stacked_us_maps(
    df=health_clean, 
    measure_name="Adequate Prenatal Care", 
    color_scale="blues", 
    title_text="Prenatal Care Quality Comparison (2018 vs. 2023)"
)
health_v2 = generate_stacked_us_maps(
    df=health_clean, 
    measure_name="Severe Maternal Morbidity", 
    color_scale="Reds", 
    title_text="Severe Maternal Morbidity Comparison (2018 vs. 2023)"
)
health_visuals = {
    "visual_1": {
        "title": "Adequate Prenatal Care Trends",
        "fig": health_v1,  # Live Plotly Figure object
        "caption": """
            Tracking the evolution of prenatal care classifications between 2018 and 2023.
            Values represent the percentage of live births in which the mother received prenatal care
            beginning in the first four months of pregnancy with the appropriate number of visits for the infant's gestational
            age.
        """
    },
    "visual_2": {
        "title": "Severe Maternal Morbidity Trends",
        "fig": health_v2,  # Live Plotly Figure object
        "caption": """
            Multi-year tracking showing shift intensities in severe clinical morbidity prevalence by state over 5 years.
            Values represent the number of significant life-threatening maternal complications during delivery
            per 10,000 delivery hospitalizations. As evidenced by the darker colors in 2023, rates of severe maternal morbidity
            *increased* relative to 2018. 
        """
    }
}
apc_2018 = health_clean.loc[(health_clean.Year==2018)&(health_clean.Measure=="Adequate Prenatal Care")].Value.mean()
apc_2023 = health_clean.loc[(health_clean.Year==2023)&(health_clean.Measure=="Adequate Prenatal Care")].Value.mean()
smm_2018 = health_clean.loc[(health_clean.Year==2018)&(health_clean.Measure=="Severe Maternal Morbidity")].Value.mean()
smm_2023 = health_clean.loc[(health_clean.Year==2023)&(health_clean.Measure=="Severe Maternal Morbidity")].Value.mean()
metrics_health = {
    '1': {
        'label': 'Adequate Prenatal Care (2018)',
        'value': round(apc_2018,1),
        'type': 'percentage'
    },
    '2': {
        'label': 'Adequate Prenatal Care (2023)',
        'value': round(apc_2023,1),
        'type': 'percentage',
        'delta': round(apc_2023-apc_2018, 1),
        'delta_color': 'red'
    },
    '3': { 
        'label': 'Severe Maternal Morbidity (2018)',
        'value': round(smm_2018, 1),
        'type': 'rate',
        'help': "per 10,000 deliveries"
    },
    '4': {
        'label': 'Severe Maternal Morbidity (2023)',
        'value': round(smm_2023, 1),
        'type': 'rate',
        'delta': round(smm_2023-smm_2018, 1),
        'delta_color': 'red',
        'help': "per 10,000 deliveries"
    }
}
# Policy Interactive Maps
policy_visuals = {
    "visual_1": {
        "title": "Abortion Bans and Gestational Limits Comparison",
        "fig": create_ban_limit_map(policy_clean),
        "caption": """
            Red indicates total bans, orange indicates heartbeat bans, and purple indicates varying
            gestational limits for obtaining an abortion. Note that in 2018, *Roe v. Wade* was still the law of the land
            and as such, no bans or gestational limitations were permitted.
        """
    },
    "visual_2": {
        "title": "State Legal & Constitutional Protections",
        "fig": create_protection_map(policy_clean),
        "caption": """
            Dark blue reflects explicit constitutional safety; light blue indicates legislative protections. Post-*Dobbs*,
            several states amended their constitions to enshrine the right to abortion while many others codified the right
            into law.
        """
    }
}

with t3:
    st.header("Data Sources")

    t_cdc_er, t_preg, t_birth, t_health, t_pol  = st.tabs([
        "Emergency Room Visits",
        "Pregnancy, Births, and Abortions",
        "Birth Records",
        "Health Rankings",
        "Healthcare Policy"
    ])

    # Emergency Room Data
    with t_cdc_er:    
        data_source_section(
            title=data_view.title_er, 
            source_name=data_view.source_name_er, 
            source_link=data_view.source_link_er,
            api_collect=data_view.api_collect_er, 
            collection_method=data_view.collection_method_er,
            description=data_view.description_er,
            raw=er_raw, 
            clean=er_clean,
            cleaning_steps=data_view.cleaning_steps_er,
            cleaning_code = "https://github.com/amberteetsel/maternal-health/blob/597d1edc47ef13548676ec8e92e0f1ef33a95ab4/src/cleaning/er_cleaning.py",
            api_code=None,
            visuals=data_view.er_visuals,
            data_link="https://github.com/amberteetsel/maternal-health/blob/cebd0bc60d68f180778fcbd9e47e027b2fd5df7a/data/raw/CDC-ER/er_raw.csv"
        )

    # Pregnancy Data
    with t_preg:
        data_source_section(
            title=data_view.title_preg,
            source_name=data_view.source_name_preg,
            source_link=data_view.source_link_preg,
            api_collect=data_view.api_collect_preg,
            collection_method=data_view.collection_method_preg,
            description=data_view.description_preg,
            raw=pregnancy_raw,
            clean=pregnancy_clean,
            visuals=data_view.preg_visuals,
            cleaning_steps=data_view.cleaning_steps_preg,
            cleaning_code="https://github.com/amberteetsel/maternal-health/blob/597d1edc47ef13548676ec8e92e0f1ef33a95ab4/src/cleaning/pregnancy_cleaning.py",
            data_link = "https://github.com/amberteetsel/maternal-health/blob/cebd0bc60d68f180778fcbd9e47e027b2fd5df7a/data/raw/Guttmacher/NatStatePregnancy.csv"
        )

    # Policy Data
    with t_pol:
        data_source_section(
            title=data_view.title_pol,
            source_name=data_view.source_name_pol,
            source_link=data_view.source_link_pol,
            api_collect=data_view.api_collect_pol,
            collection_method=data_view.collection_method_pol,
            description=data_view.description_pol,
            raw=policy_raw,
            clean=policy_clean,
            visuals=policy_visuals,
            cleaning_steps=data_view.cleaning_steps_pol,
            cleaning_code="https://github.com/amberteetsel/maternal-health/blob/597d1edc47ef13548676ec8e92e0f1ef33a95ab4/src/cleaning/policy_cleaning.py",
            data_link="https://github.com/amberteetsel/maternal-health/blob/cebd0bc60d68f180778fcbd9e47e027b2fd5df7a/data/raw/LawAtlas/policy_raw.csv"
        )

    # Health Data
    with t_health:
        data_source_section(
            title=data_view.title_health, 
            source_name=data_view.source_name_health, 
            source_link=data_view.source_link_health,
            api_collect=data_view.api_collect_health, 
            collection_method=data_view.collection_method_health, # Fixed variable typo here too!
            description=data_view.description_health, 
            raw=health_raw, 
            clean=health_clean,
            cleaning_steps=data_view.cleaning_steps_health,
            cleaning_code="https://github.com/amberteetsel/maternal-health/blob/597d1edc47ef13548676ec8e92e0f1ef33a95ab4/src/cleaning/health_cleaning.py",
            api_code=data_view.api_code_health,
            visuals=health_visuals,
            data_link="https://github.com/amberteetsel/maternal-health/blob/cebd0bc60d68f180778fcbd9e47e027b2fd5df7a/data/raw/HealthRankings/raw_api_snapshot.json",
            metrics=metrics_health
        )

    # Birth Data
    with t_birth:
        data_source_section(
            title=data_view.title_birth,
            source_name=data_view.source_name_birth,
            source_link=data_view.source_link_birth,
            api_collect=data_view.api_collect_birth,
            collection_method=data_view.collection_method_birth,
            description=data_view.description_birth,
            raw=birth_raw,
            clean=birth_clean,
            visuals=data_view.birth_visuals,
            cleaning_steps=data_view.cleaning_steps_birth,
            cleaning_code="https://github.com/amberteetsel/maternal-health/blob/597d1edc47ef13548676ec8e92e0f1ef33a95ab4/src/cleaning/birth_cleaning.py",
            data_link="https://github.com/amberteetsel/maternal-health/blob/cebd0bc60d68f180778fcbd9e47e027b2fd5df7a/data/raw/NCHS-Birth/births2024_raw.txt"
        )

############################################################################################################################
# TAB 4: MODELS
############################################################################################################################

# Clustering
import clusters
from clusters import render_clustering_view
from clusters import render_cluster_preg
cluster_res = os.path.join(BASE_DIR, "resources", "clustering")
# PCA
import pca
from pca import render_pca
# Naive Bayes
import nbayes
from nbayes import render_nb
# Decision Tree
import tree_view
from tree_view import render_dt
# SVM
import svm
from svm import render_svm
# NN
import nn
from nn import render_nn

with t4:
    st.header("Modeling Results")

    t_cluster, t_pca, t_nb, t_dt, d_svm, t_nn = st.tabs([
        "Clustering",
        "Principal Component Analysis (PCA)",
        "Naive Bayes",
        "Decision Trees",
        "Support Vector Machines (SVM)",
        "Neural Networks (NN)"
    ])

    # clustering
    with t_cluster:
        with st.expander("State Health Rankings", expanded=False):
            # Run layout call inside app loop structure
            render_clustering_view(
                overview_text=clusters.overview_clust,
                overview_images=clusters.clust_overview_images,
                prep_text=clusters.prep_clust,
                cleaning_code="https://github.com/amberteetsel/maternal-health/blob/7b502ade0260152992815f5a7a3fcd8791a0b3c1/src/models/health_preprocessing.py",
                df_raw_sample=clusters.clust_sample_before_df,
                df_scaled_sample=clusters.clust_sample_after_df,
                data_download_url="https://github.com/amberteetsel/maternal-health/blob/7b502ade0260152992815f5a7a3fcd8791a0b3c1/resources/clustering/cluster_input_data.csv",
                kmeans_code=clusters.kmeans_code,
                hclust_code=clusters.hclust_code,
                kmeans_pipeline=clusters.kmeans_assets,
                hclust_pipeline=clusters.hclust_assets,
                conclusions_text=clusters.conclusions_clustering
            )

        with st.expander("National Pregnancy Trends", expanded=False):
            render_cluster_preg(
                overview_text=clusters.overview_text_preg,
                prep_text=clusters.prep_text_preg,
                pca_text=clusters.pca_text_preg,
                pca_resources=clusters.pca_pipeline_preg,
                raw_data_link=clusters.data_raw_preg_link,
                clean_data_link=clusters.data_processed_preg_link,
                raw_data=clusters.data_raw_preg,
                clean_data=clusters.data_processed_preg,
                model_code_url=clusters.model_code_link_preg,
                model_pipeline=clusters.kmeans_pipeline_preg,
                conclusion_text=clusters.conclusion_text_preg
            )

    # pca
    with t_pca:
        render_pca(
            overview=pca.overview_pca,
            prep_text=pca.prep_pca,
            cleaning_code="https://github.com/amberteetsel/maternal-health/blob/7b502ade0260152992815f5a7a3fcd8791a0b3c1/src/models/health_preprocessing.py",
            df_before=clusters.clust_sample_before_df,
            df_after=pca.pca_df_after,
            data_download_url="https://github.com/amberteetsel/maternal-health/blob/3f403042d1bbd90522052614b4c8c976c54e329f/resources/pca/pca_input_data.csv",
            pipeline_all=pca.pca_all_assets,
            pca_code="https://github.com/amberteetsel/maternal-health/blob/e7ef492c02a7a6f021a7b4c374cff6d7c7d5e6ba/src/models/pca_all.py",
            # pipeline_opt=pca.pca_opt_assets,
            conclusion=pca.pca_conclusion
        )

    # naive bayes
    with t_nb:
        render_nb(overview_text=nbayes.overview_nb,
                prep_text=nbayes.data_prep_nb,
                cleaning_code_url=nbayes.cleaning_code_nb,
                sample_data_url=nbayes.sample_data_url,
                sample_data=nbayes.input_data_nb,
                train_test_text=nbayes.train_test_nb,
                train_data=nbayes.train_data,
                test_data=nbayes.test_data,
                model_code_url=nbayes.model_code_url,
                results_intro=nbayes.results_intro_nb,
                result_table=nbayes.result_table_nb,
                confusion_matrix=nbayes.cm_nb,
                results_text=nbayes.results_interpret_nb,
                conclusion_text=nbayes.conclusion_nb
        )

    # decision trees    
    with t_dt:
        render_dt(
            overview_text=tree_view.overview_dt,
            overview_cols=tree_view.goodness_of_fit,
            prep_text=tree_view.data_prep_dt,
            cleaning_code_url=tree_view.cleaning_code_dt,
            sample_data_url=tree_view.sample_data_url,
            sample_data=tree_view.input_data_dt,
            train_test_text=tree_view.train_test_dt,
            train_data=tree_view.train_data,
            test_data=tree_view.test_data,
            model_code_url=tree_view.model_code_url_dt,
            tree_list=tree_view.three_trees,
            tree_pics=tree_view.tree_images,
            performance_stats=tree_view.performance_df_dt,
            conclusion_text=tree_view.conclusion_text_dt
        )
    
    # support vector machines (svm)
    with d_svm:
        render_svm(
            overview_text=svm.overview_text_svm,
            overview_pipeline=svm.overview_pipeline_svm,
            sample_data_url=svm.sample_data_url_svm,
            data_reqs=svm.data_reqs_svm,
            sample_data=svm.sample_data_svm,
            train_test_text=svm.train_test_text_svm,
            train_data=svm.train_data_svm,
            test_data=svm.test_data_svm,
            model_code_url=svm.model_code_url_svm,
            result_summary=svm.result_summary_svm,
            result_details=svm.result_details_svm,
            result_df=svm.result_df_svm,
            result_viz=svm.result_plot_svm,
            confusion_matrix=svm.confusion_matrices_svm,
            conclusion_text=svm.conclusion_text_svm,
            conclusion_df=svm.feature_mean_df,
            final_takeaway=svm.final_takeaway_svm
        )
    
    with t_nn:
        render_nn(
            overview_text=nn.overview_text_nn,
            sample_data_url=nn.sample_data_url_nn,
            data_reqs=nn.data_reqs_nn,
            sample_data=nn.sample_data_nn,
            train_test_text=nn.train_test_text_nn,
            train_data=nn.train_data_nn,
            test_data=nn.test_data_nn,
            model_code_url=nn.model_code_url_nn,
            result_section=nn.result_section_nn,
            results_pipeline=nn.results_nn,
            conclusion_pipeline=nn.conclusion_pipeline_nn
        )


##############################################################
# TAB 5: REFERENCES
##############################################################

from references import raw_citations
sorted_citations = sorted(raw_citations, key=lambda x: x['footnote'])

with t5:
    st.header("References")
    st.markdown("---")

    # CSS styles
    st.markdown(
        """
        <style>
            .footnote-container {
                display: flex;
                align-items: flex-start;
                margin-bottom: 22px;
                font-family: 'Roboto', sans-serif;
                font-size: 14.5px;
                line-height: 1.6;
            }
            .footnote-number {
                min-width: 32px;
                font-weight: bold;
                color: #b44391; /* Matches your 'flare' palette index [4] color */
            }
            .footnote-body {
                flex-grow: 1;
                padding-left: 4px;
                text-indent: -24px; /* Creates clean secondary alignment hanging indent */
                margin-left: 24px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    for cite in sorted_citations:
        # Safely extract dynamic volume details if they are declared in the map
        details_str = f", {cite['details']}." if cite['details'] else "."
        
        html_block = f"""
        <div class="footnote-container">
            <div class="footnote-number">[{cite['footnote']}]</div>
            <div class="footnote-body">
                {cite['authors']} ({cite['year']}). {cite['title']} 
                <i>{cite['publication']}</i>{details_str} 
                <a href="{cite['url']}" target="_blank">{cite['url']}</a>
            </div>
        </div>
        """
        st.markdown(html_block, unsafe_allow_html=True)

