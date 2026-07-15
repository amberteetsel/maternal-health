#### HELPER FUNCTION TO DISPLAY CLUSTERING RESULTS

# Dependencies
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import inspect

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
cluster_res = os.path.join(BASE_DIR, "resources", "clustering")

# ==========================================================
# RENDER CLUSTERING FUNCTION
# ==========================================================
def render_clustering_view(
    overview_text: str,
    overview_images: list,
    prep_text: str,
    cleaning_code: str,
    df_raw_sample: pd.DataFrame,    # raw dataset sample matrix (with labels)
    df_scaled_sample: pd.DataFrame, # pure numeric, scaled input matrix
    data_download_url: str,
    kmeans_code: str,
    hclust_code: str,
    kmeans_pipeline: list,
    hclust_pipeline: list,
    conclusions_text: str
):
    """
    Renders a side-by-side clustering analysis view: KMeans vs. Agglomerative
    
    Parameters:
    -----------
    overview_text : str -> Markdown explaining partitional vs hierarchical concepts.
    overview_images : list -> List of dicts with keys 'fig' and optionally 'caption' for Section (a).
    prep_text : str -> Explanation of numeric/unlabeled requirements.
    cleaning_code : str -> Link or string pointing to data preprocessing/cleaning steps.
    df_raw_sample : pd.DataFrame -> Raw dataset matrix sample containing labels/strings.
    df_scaled_sample : pd.DataFrame -> Purely numeric, scaled input matrix used for modeling.
    data_download_url : str -> Target dataset source string/link.
    kmeans_code/hclust_code: str --> Link to code on GitHub
    kmeans_pipeline : list -> Visual pipeline dictionaries for the K-Means column.
    hclust_pipeline : list -> Visual pipeline dictionaries for the Hierarchical column.
    conclusions_text : str -> Final evaluation insights.
    """
    
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
            st.image(fig, width='stretch')
            
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
    # OVERVIEW
    # ----------------------------------------------------
    st.subheader("Clustering Overview")
    with st.expander("Read About Clustering", expanded=True):
        st.markdown(overview_text)
        
        if overview_images:
            # automatically split the layout into columns depending on how many images are supplied
            img_cols = st.columns(len(overview_images))
            for idx, col in enumerate(img_cols):
                with col:
                    img_data = overview_images[idx]
                    if isinstance(img_data["fig"], str):
                        st.image(img_data["fig"], width='stretch')
                    else:
                        st.pyplot(img_data["fig"])
                    if img_data.get("caption"):
                        st.caption(img_data["caption"])
                    
    # st.markdown("---")
    
    # ----------------------------------------------------
    # Data preparation
    # ----------------------------------------------------
    st.subheader("Data Preparation")
    st.markdown(prep_text)
    
    st.markdown(f"🔗 **[Download Sample Unlabeled Dataset]({data_download_url})**")
    st.markdown(f"🔍 **[View Cleaning Code]({cleaning_code})**")
    
    # constructing two equal columns for side-by-side data comparison
    prep_col1, prep_col2 = st.columns(2)
    
    with prep_col1:
        st.markdown("📋 **Original Contextually Grounded Dataset**")
        st.dataframe(df_raw_sample, width='stretch')
        
        # schema info rendered within a clean layout expander
        with st.expander("🛠️ View Original Feature Schema Details", expanded=False):
            schema_df1 = pd.DataFrame({
                "Data Type": df_raw_sample.dtypes.astype(str),
                "Non-Null Count": df_raw_sample.notnull().sum(),
                "Null Count": df_raw_sample.isnull().sum()
            })
            st.table(schema_df1)

    with prep_col2:
        st.markdown("🔢 **Mathematical Input Feature Space (Strictly Unlabeled & Scaled)**")
        st.dataframe(df_scaled_sample, width='stretch')
        
        with st.expander("🛠️ View Modeled Vector Space Schema Details", expanded=False):
            schema_df2 = pd.DataFrame({
                "Data Type": df_scaled_sample.dtypes.astype(str),
                "Non-Null Count": df_scaled_sample.notnull().sum(),
                "Null Count": df_scaled_sample.isnull().sum()
            })
            st.table(schema_df2)
            
    st.markdown("---")
    
    # ----------------------------------------------------
    # code/results
    # ----------------------------------------------------
    st.subheader("Model Run & Results")
    
    col_km, col_hc = st.columns(2)
    
    # left column: k-means execution pipeline
    with col_km:
        st.markdown("#### Method 1: Partitional K-Means")
        st.markdown(f"👾 [View Code]({kmeans_code})")
        for i, item in enumerate(kmeans_pipeline):
            _render_pipeline_asset(item, step_index=i, unique_key_prefix="km")
            
    # right column: hierarchical execution pipeline
    with col_hc:
        st.markdown("#### Method 2: Cosine Hierarchical")
        st.markdown(f"👾 [View Code]({hclust_code})")
        for i, item in enumerate(hclust_pipeline):
            _render_pipeline_asset(item, step_index=i, unique_key_prefix="hc")
            
    st.markdown("---")
    
    # ----------------------------------------------------
    # conclusions
    # ----------------------------------------------------
    st.subheader("Conclusions")
    st.markdown(conclusions_text)


def render_cluster_preg(
        overview_text: str,

):
    return

# ==========================================================
# ACTUAL INPUTS - NATIONAL PREGNANCY TRENDS
# ==========================================================
overview_text_preg = """
    This analysis uses data from the Guttmacher Institute to answer the questions:

    ###### How did historical national trends in pregnany, birth, abortion, and miscarriage rates cluster across time? Do these
    temporal clusters align with major federal judicial milestones (*1973 Roe, 1992 Casey, 2022 Dobbs*)?

    Using **K-Means Clustering** on time-series data (treating years as samples and rates as features) will provide insight
    to whether the timeline naturally breaks into distinct historical "eras" corresponding to Supreme Court Decisions.
"""
data_raw_preg = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "NatStatePregnancy.csv"))
data_processed_preg = pd.read_csv(os.path.join(cluster_res, "preg_data_pca.csv"))
prep_text_preg = """
    Clustering requires only unlabeled, scaled, numeric data. To this end, the following data preprocessing steps were implemented:

    * Filtering for national data instead of state-by-state
    * Dropping non-essential, non-numeric columns (`state`, `notes`, `versiondate`)
    * Feature engineering to calculate miscarriage rates (original data only provided counts)
    * Feature selection: restricting input data to specific group of metrics representing Rates per 1,000 women broken down
    sequentially by age bands across four clinical vectors
        * Pregnancy Rates
        * Abortion Rates
        * Birth Rates
        * Miscarriage Rates
    * Linear interpolation to fill missing values and ensure perfectly continuous statistical structure across all years
    * Feature scaling with z-score scaling (`StandardScaler`)
"""
pca_text_preg = """
    In addition, dimensionality was reduced using Principal Component Analysis (PCA). The original data contained dozens of highly
    correlated features (rates mentioned above) across multiple granular age bands over nearly 50 years. PCA was implemented to
    capture the majority of variance and project the information into a lower-dimensional space.

    *For a more robust discussion of Principal Component Analysis, please see the PCA tab.*
"""
pca_pipeline_preg = [
    {
        'title': 'Explained Variance',
        'text': "placeholder",
        'fig': os.path.join(cluster_res, "preg_pca_expvar.png"),
        'caption': "To capture at least 90% of overall variance, 3 principal components are needed. Together they capture almost 95% of overall variance."
    }
]
opt_k_text = """
    The optimal number of clusters was determined to be $k = 4$. Looking at the plot below, there is a clear elbow and
    a local maximum of silhouette score at $k=4$ and $k=7$. Using seven clusters to group about 50 years will result in too-granular
    cluster assignments, so using four clusters is the optimal approach. 
"""
timeline_text = """
    K-Means clustering worked exactly as intended: though the algorithm had absolutely no access to the `Year` column during clustering
    and only received standardized clinical and demographic rates, it partitioned the data into **four perfectly consecutive,
    unbroken chronological eras.** This proves that national reproductive behavior, outcomes, and healthcare trends do not fluctuate
    randomly but exist in distinct, stable historical states.
"""
cluster_text_preg = """
    **Cluster 2 (1973 - 1976): The Immediate Post-*Roe* Era**
    * Following the *Roe v. Wade* ruling in January 1973, the U.S. underwent a massive structural transition. This 4-year block
    represents the initial baseline shift of when legal abortion infrastructure was rapidly established after decades of criminalization.

    **Cluster 3 (1977 - 1996): The Stablization Era**
    * This 20-year era captures the long-term stabilization of national birth, pregnany, abortion, and miscarriage rates. It includes
    the 1992 *Planned Parenthood v. Casey* decision.
    * Rather than showing immediate changes after 1992, the data reveals that the "undue burden" framework introduced by *Casey*
    took years to subtly shift national baseline outcomes.

    **Cluster 1 (1997 - 2010): The Contraceptive Era**
    * This era represents the shift in public health brought characterized by a historic drop in teen pregnancy and abortion rates.
    This was driven by the introduction and widespread adoption of long-acting contraception such as IUDs and implants, as well as
    Clinton/Bush-era public health campaigns targeting teen pregnancy.

    **Cluster 0 (2011 - 2020): The TRAP Era**
    * After the 2010 midterm elections, conservative state legislatures began aggressively passing Targeted Regulation of Abortion
    Providers (TRAP) laws. Common TRAP laws included requiring clinics to meet strict, costly surgical center standards, forcing 
    doctors to have local hospital admitting privileges, and mandating specific unnecessary staff qualifications. This led to large
    numbers of clinic closures, predominantly in the South and the Midwest, and a major divergence in abortion access that set the stage
    for the *Dobbs* decision in 2022.
"""
trend_text_preg = """
    Projecting 48 years of national reproductive metrics onto the first two Principal Components reveals a highly structured,
    non-linear evolutionary path. Rather than random annual fluctuations, the physical movement of the years from the bottom-right
    (1973) to the far-left (2020) maps a clear transformation in the landscape of American public health.

    1. Component Interpretation

    * **Principal Compnent 1 (Horizontal Axis) - Overall Trend & Volume:** Represents the broad, long-term decline in overall
    pregnancy and birth rates across almost all age groups. Years on the far right (1970s, 1980s) are eras of high baseline
    fertility and higher overall volumes of reproductive events. Years on the far left show the midern era of low birth and pregnancy
    rates.

    * **Principal Component 2 (Vertical Axis) - Outcome Divergence:** Captures the relative balance of how pregnancies resolved,
    specifically the ratio of births to legal abortions. Higher values (e.g. peak in early '90s) represent eras where abortion rates
    and ratios reached their historical peak relative to live births. Lower values represent eras of lower relative abortion rates.

    2. Cluster Interpretation

    * **1973 - 1976 (Immediate Post-*Roe* Era):** The vertical climb on the plot reflects rapid legalization and integration of
    abortion services nationwide, causing a large immediate change in national outcome resolution profile (PC2).

    * **1977 - 1996 (Stabilization Era):** The plot reveals that the *Casey* decision in 1992 did not have an immediate effect 
    on national metrics. Instead, 1992 sits near the apex of this cluster, showing the "undue burden" framework took half a decade
    to manifest as a structural break in national trends.

    * **1997 - 2010 (Contraceptive Era):** The sharp left turn in 1997 marks a public health success: a simultaneous decline in
    unintended pregnancies and abortion rates, particularly among teenagers, driven by popularization of highly effective long-acting
    contraceptives.

    * **2011 - 2020 (TRAP Era):** 2011 sees the start of a highly linear, downward-sloping shift to the far left. This represents
    the wave of Targeted Regulation of Abortion Providers (TRAP) laws that forced many clinic closures and greatly reduced abortion
    access for large swathes of the country. 2020 sits at the terminal of this line, showing the final, most extreme state of the
    reproductive landscape that would lead into the *Dobbs* decision in 2022.
"""
kmeans_pipeline_preg = [
    {
        'title': "Finding Optimal Number of Clusters (k)",
        'text': opt_k_text,
        'fig': os.path.join(cluster_res, "preg_kmeans_eval.png"),
        'caption': "Both the Elbow method and Silhouette method were implemented to determine the optimal number of clusters."
    },
    {
        'title': "Final Model",
        'text': timeline_text,
        'fig': os.path.join(cluster_res, "preg_timeline.png"),
        'caption': "Clusters are perfectly chronological despite the model never seeing the year variable."
    },
    {
        'title': "Cluster Interpretation",
        'text': cluster_text_preg,
        'fig': None,
        'caption': None
    },
    {
        'title': "U.S. Reproductive Health Trajectory",
        'text': trend_text_preg,
        'fig': os.path.join(cluster_res, "preg_viz_final.png"),
        'caption': "The timeline flows like a physical trajectory through space, with sharp 'pivot points' that align with major historical shifts"
    }
]
conclusion_text_preg = """
    ##### How did historical national trends in pregnany, birth, abortion, and miscarriage rates cluster across time?

    The historical national trends clustered into four highly distinct, strictly sequential chronological eras rather than fragmenting
    or jumping back and forth over time.

    ##### Do these temporal clusters align with major federal judicial milestones (*1973 Roe, 1992 Casey, 2022 Dobbs*)?

    Yes. Legal and policy climates are not isolated events; they dictate the macro-level behavior of national reproductive
    healthcare. The fact that the years partition so cleanly and sequentially supports that judicial milestones and state legislative
    movements are primary structural forces shaping health outcomes for pregnany people in the United States. That the trends along
    principal component axes are not consistent or linear over the 48-year span proves that the distinct clusters are not simply
    tracking inherent improvements in healthcare that come with time and medical advancements.
"""


# ==========================================================
# ACTUAL INPUTS - STATE HEALTH RANKINGS
# ==========================================================

# CLUSTERING INTRO/OVERVIEW
clust_overview_images = [
    {
        "fig": os.path.join(cluster_res, "iris_kmeans.png"), 
        "caption": "Partitional Clustering Logic: Centroid optimization partitions."
    },
    {
        "fig": os.path.join(cluster_res, "iris_hierarchical.png"),
        "caption": "Hierarchical Dendrogram Logic: Tree branching structures."
    }
]
overview_clust = inspect.cleandoc("""
    Clustering is a core branch of unsupervised machine learning used to discover patterns, structures, or groupings within
    unlabelled datasets. Unlike supervised learning which relies on predefined target classes, clustering algorithms 
    evaluate inherent mathematical relationships between data points to group similar observations together while separating
    dissimilar observations. For studying maternal health infrastructure and outcomes, clustering enables a fresh look
    at state health profiles free from internal biases about certain states or regions.

    For a robust analysis, this study analyzes State Health Ranking data through two different clustering paradigms: 
    **Partitional** and **Hierarchical** clustering.

    ##### 1. Partitional Clustering (e.g. KMeans)

    Partitional clustering algorithms construct a single, flat partition of data points into a user-specified number of clusters ($k$).
    The KMeans algorithm used for this study treats the data space as a geometric landscape and seeks to optimize clusters
    by minimizing the total variance between data points and their respective group centers (centroids). It is based on
    **Euclidean Distance**, which calculates the traditional straight-line distance between two points in space. For this study
    it helps identify states that share similar raw numbers across healthcare measures.

    ##### 2. Hierarchical Clustering (e.g. Agglomerative)

    Hierarchical clustering algorithms build a continuous, nested tree structure known as a dendrogram. The Agglomerative 
    (bottom-up) algorithm used for this study begins by treating every individual state observation as its own distinct cluster.
    The algorithm sequentially merges the closest pairs of clusters based on **Cosine Distance** until all observations 
    are unified into a single global tree. **Cosine Distance** measures the cosine of the angle between two multi-dimensional
    data vectors, shifting analytical focus from magnitude to orientation and proportion.  For this study it helps group
    states together if the relative shape or proportion of their healthcare profile matches.

    This study leverages unsupervised clustering as a means of data-driven discovery. The models will identify
    distinct state health profiles, helping to understand the maternal health crisis in the United States. By plotting clusters
    onto a geographic map of the US, potential spacial dependencies will emerge that either challenge or reinforce
    historical stereotypes (for example, the South is often perceived as "backwards" in terms of gender equality and healthcare.)
""")
prep_clust = inspect.cleandoc("""
    Clustering algorithms require only **unlabeled, numeric data** because its primary goal is classification, not prediction.
    It relies on quantitative distance formulas and so cannot process categorical data without encoding.

    To this end, data from *America's Health Rankings* was further processed to conform with clustering requirements. First
    the DataFrame was melted into wide format.
    Category labels such as `State` and `Year` were stripped away to leave only unlabelled relevant features and values.
    Features were also reconstructed so that for all columns, a higher number indicated a "worse" outcome than a lower number.
    For example, "Adequate Prenatal Care" was converted to "Inadequate Prenatal Care" by flipping the percentage.
    This increases interpretability of results. Z-Score standardization was also used so that varying data magnitudes would
    not have outsized influence on clusters.
""")

clust_sample_before_df = pd.read_csv(os.path.join(BASE_DIR,"data","clean","HealthRankings","health.csv"))
clust_sample_after_df = pd.read_csv(os.path.join(cluster_res, 'cluster_input_data.csv'))

# KMEANS CLUSTERING
kmeans_code = "https://github.com/amberteetsel/maternal-health/blob/8a84b89fd23314b95894810c7781565d1933442e/src/models/clustering_kmeans.py"
kmeans_summary_md = inspect.cleandoc("""
    Optimizing for Silhouette Score lead to the selection of three optimal features: `Maternity Care Desert`, 
    `Maternal Mortality`, and `Patients Per Doctor`. Running the KMeans model with these features produced three distinct clusters.
    **Clusters are framed in terms of risk to potential mothers.** They are labelled according to average value of each feature.

    The **Low Risk** cluster is characterized by below-average rates of women facing care deserts, maternal mortality, and
    patients per doctor. States in this cluster have stronger maternal health ecosystems and better maternal outcomes
    than states in other clusters.

    The **Moderate Risk** cluster has below-average rates of women facing care deserts, but above-average maternal mortality and
    patients per doctor. States in this cluster have decent maternal health ecosystems but suffer from lack of sufficient
    doctors.

    The **High Risk** cluster has above-average rates of women facing care deserts, maternal mortality, and patients per doctor.
    People living in these states are much less likely than average to live in proximity to maternal healthcare services,
    and have fewer available doctors.  

    Review the Snake Plots below for a visual representation of these patterns. 
""")
kmeans_map_interpretation = """
    These results somewhat reinforce traditional stereotypes about "Blue/Democratic" vs. "Red/Republican" states.
    Democratic strongholds in the West, Great Lakes, and Northeast regions are mostly **Low Risk**, while Southern
    and Midwestern regions are mostly **Moderate** or **High Risk.** 
"""
kmeans_cluster_sum = pd.read_csv(os.path.join(cluster_res, 'kmeans_cluster_summary.csv'))
kmeans_assets = [
    {
        "title": "K-Means Parameter Sweep (Silhouette Line)",
        "fig": os.path.join(cluster_res, "kmeans_silhouette_scores.png"),
        "caption": "An examination of different k values revealed that using k = 3 clusters yielded the highest silhouette score."
    },
    {
        'title': 'Cluster Characteristics',
        'fig': kmeans_cluster_sum,
        'interpretation': kmeans_summary_md
    },
    {
        "title": "K-Means Attribute Snake Plot",
        "fig": os.path.join(cluster_res, "kmeans_snake_plot.png"),
        "caption": "Comparing cluster attributes to national average."
    },
    {
        "title": "KMeans: Interactive US Cluster Map",
        "fig": os.path.join(cluster_res, "kmeans_map.html"),
        "caption": "Map of the US by KMeans Cluster. Data is unavailable for Idaho, Maine, Vermont, and Alabama.",
        'interpretation': kmeans_map_interpretation
    }
]


# HIERARCHICAL CLUSTERING
hclust_code = "https://github.com/amberteetsel/maternal-health/blob/8a84b89fd23314b95894810c7781565d1933442e/src/models/clustering_hier.py"
hclust_summary_md = inspect.cleandoc("""
    Optimizing for Silhouette Score lead to the selection of three optimal features: `Maternity Care Desert`, 
    `Unplanned Pregnancy`, and `Patients Per Doctor`. Running the Hierarchical model with these features produced three distinct clusters.
    **Clusters are framed in terms of strengths and weaknesses of maternal healthcare systems.** 
    They are labelled according to average value of each feature.

    The **Strong Health Ecosystem** cluster is characterized by below-average rates of women facing care deserts,
    unplanned pregnancy, and patients per doctor. There is considerable overlap between this cluster and the **Low Risk**
    cluster found during KMeans.

    The **Poor Family Planning** cluster is characterized by above-average rates of unintended pregnancy despite
    fewer women than average residing in maternity care deserts. This suggests states in this cluster do not have strong
    family planning education or resources, but further investigation is needed to determine specific drivers.

    The **Poor Access to Care** cluster is characterized by above-average rates of women residing in maternity care deserts
    and an above-average patient to doctor ratio. This suggests states in this cluster primarily suffer from lack of healthcare
    providers.

    Review the Snake Plots below for a visual representation of these patterns. 
""")
hclust_map_interpretation = """
    These results are less predictable, though there is considerable overlap between **Strong Health Ecosystem** states
    and **Low Risk** states from KMeans clustering. States with **Poor Family Planning** seem to be concentrated in the South,
    while states with **Poor Access to Care** are concentrated in the Midwest.
"""
hclust_cluster_sum = pd.read_csv(os.path.join(cluster_res, 'hclust_cluster_summary.csv'))
hclust_assets = [
    {
        "title": "Hierarchical Parameter Sweep (Silhouette Line)",
        "fig": os.path.join(cluster_res, "hclust_silhouette_scores.png"),
        "caption": "An examination of different k values revealed that using k = 3 clusters yielded the highest silhouette score."
    },
        {
        'title': 'Cluster Characteristics',
        'fig': hclust_cluster_sum,
        'interpretation': hclust_summary_md
    },
    {
        "title": "Hierarchical Attribute Snake Plot",
        "fig": os.path.join(cluster_res, "hclust_snake_plot.png"),
        "caption": "Comparing cluster attributes to national average."
    },
    {
        "title": "Hierarchical: Interactive US Cluster Map",
        "fig": os.path.join(cluster_res, "hclust_map.html"),
        "caption": "Map of the US by Agglomerative Cluster. Data is unavailable for Idaho, Maine, Vermont, and Alabama.",
        'interpretation': hclust_map_interpretation
    },
    {
        "title": "Cosine Linkage Tree Dendrogram",
        "fig": os.path.join(cluster_res, "hclust_dendrogram.png"),
        "caption": "Average Linkage hierarchy branch map using Cophenetic evaluation calculations."
    }, 
]

# CONCLUSIONS
conclusions_clustering = """
* **Model Validation:** Both algorithmic approaches identified an optimal cluster cut-off at $k = 3$. Additionally,
both models chose very similar optimal feature sets. The only difference was that KMeans preferred `Maternal Mortality` to
Agglomerative's `Unintended Pregnancy`.

* **Silhouette Analysis:** Neither model had a particularly strong Silhouette Score (both under 0.5), despite being heavily
optimized to use the best available features and k value. This suggests additional data is needed to shore up the validity
of identified clusters, or that the healthcare crisis is driven by so many diverse levers that robust clustering is not feasible.

* **Geographic Findings:** Assigning clusters to states reveals some interesting patterns that should prove useful in 
subsequent analyses. Hierarchical clustering in particular was useful for providing insight into specific drivers
of suboptimal maternal health conditions, such as care deserts vs. family planning resources.
"""