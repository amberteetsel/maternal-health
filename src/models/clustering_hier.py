# Dependencies
import os
# Restrict the number of threads used by linear algebra libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import random

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go

import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist

# Base Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Load Data
health_data = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "HealthRankings", "health_preprocessed.csv"))

# Get average feature value by state
df_state = health_data.groupby('State').mean().reset_index()

# Prepare for Clustering: Remove categorical labels
cluster_label_index = health_data[['State', 'Year']]
X = df_state.copy().drop(columns=['State', 'Year'])

# Relevant Features
features_all = ['Gender Pay Gap',
                'Poverty',
                'Unemployment',
                'Uninsured Women',
                'WIC Shortfall',
                'Voter Abstainment',
                'Inadequate Prenatal Care',
                'No Cancer Screen',
                'Maternity Care Desert',
                'Maternity Practices Score',
                'No Postpartum Visit',
                'No Preventative',
                'Patients Per Doctor',
                'Unintended Pregnancy',
                'Maternal Mortality',
                'Postpartum Depression',
                'Smoking During Pregnancy']

X_all = X[features_all]

# Feature Selection
def find_best_hierarchical_combination(df, available_features, min_features=3, max_features=6, k_values=[3, 4, 5]):
    """
    Evaluates all possible combinations of features across multiple k values 
    using Agglomerative Hierarchical Clustering to maximize the silhouette score.
    """
    best_score = -1
    best_combo = None
    best_k = None
    
    total_combos = sum(len(list(itertools.combinations(available_features, r))) 
                       for r in range(min_features, max_features + 1))
    print(f"Starting Hierarchical grid search over {total_combos} combinations across k={k_values}...")

    for r in range(min_features, max_features + 1):
        for combo in itertools.combinations(available_features, r):
            combo_list = list(combo)
            X_subset = df[combo_list]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            
            for k in k_values:
                if len(X_scaled) <= k:
                    continue
                    
                # Cosine Similarity
                # 'average' linkage
                hierarchical = AgglomerativeClustering(n_clusters=k,
                                                       metric='cosine',
                                                       linkage='average')
                labels = hierarchical.fit_predict(X_scaled)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels, metric='cosine')
                    
                    if score > best_score:
                        best_score = score
                        best_combo = combo_list
                        best_k = k

    print("\n--- Hierarchical Grid Search Complete ---")
    print(f"Highest Silhouette Score: {best_score:.4f}")
    print(f"Optimal Number of Clusters (k): {best_k}")
    print(f"Optimal Feature Combination ({len(best_combo)} features): {best_combo}")
    
    return {
        'best_score': best_score,
        'best_k': best_k,
        'best_features': best_combo
    }

#########################################################
#  Uncomment section below to run optimization algorithm
#########################################################
# # Find best features
# n_runs = 30
# all_hierarchical_results = []

# for i in range(n_runs):
#     print(f"\n--- RUN {i+1} of {n_runs} ---")
#     current_features = random.sample(features_all, 8)
    
#     run_output = find_best_hierarchical_combination(
#         df=X_all, 
#         available_features=current_features, 
#         min_features=3, 
#         max_features=6, 
#         k_values=[3, 4, 5]
#     )
    
#     all_hierarchical_results.append({
#         'run_index': i + 1,
#         'feature_pool_used': current_features,
#         'best_score': run_output['best_score'],
#         'best_k': run_output['best_k'],
#         'best_features': run_output['best_features']
#     })

# # Extract top results
# sorted_h_results = sorted(all_hierarchical_results, key=lambda x: x['best_score'], reverse=True)
# print("\n" + "="*50)
# print(" TOP HIERARCHICAL RESULT")
# print("="*50)
# print(f"Score: {sorted_h_results[0]['best_score']:.4f}")
# print(f"Optimal k: {sorted_h_results[0]['best_k']}")
# print(f"Optimal Features: {sorted_h_results[0]['best_features']}")

# optimal_features = sorted_h_results[0]['best_features']
# optimal_k = sorted_h_results[0]['best_k']
#########################################################

# Optimized Hierarchical Clustering Using Cosine Distance
optimal_features = ['Maternity Care Desert', 'Unintended Pregnancy', 'Patients Per Doctor']
optimal_k = 3
X_optimal = X_all[optimal_features]

# Scale
scaler = StandardScaler()
X_opt_scaled = scaler.fit_transform(X_optimal)

# CPCC Score
actual_distances = pdist(X_opt_scaled, metric='cosine')
Z_linkage = linkage(X_opt_scaled, method='average', metric='cosine')
cpcc_score, coph_distances = cophenet(Z_linkage, actual_distances)

print(f"=========================================================")
print(f"Cophenetic Correlation Coefficient (CPCC): {cpcc_score:.4f}")
print(f"=========================================================")

# Run Hierarchical Clustering Model
hclust = AgglomerativeClustering(n_clusters=optimal_k,
                                        metric='cosine',
                                        linkage='average')

cluster_labels = hclust.fit_predict(X_opt_scaled)

# Verify the actual unique clusters generated
import numpy as np
print("Actual Unique Clusters:", np.unique(cluster_labels))
print("Total number of clusters:", len(np.unique(cluster_labels)))

# Map Clusters to Descriptive Names, Standardize Color Palette
cluster_map = {
    0: {
        'name': 'Strong Health Ecosystem',
        'color': sns.color_palette("cubehelix")[2]
    },
    1: {
        'name': 'Poor Family Planning',
        'color': sns.color_palette("cubehelix")[4]
    },
    2: {
        'name': 'Poor Access to Care',
        'color': sns.color_palette("cubehelix")[3]
    }
}

cluster_names = {cluster_id: info['name'] for cluster_id, info in cluster_map.items()}
cluster_colors = {info['name']: info['color'] for cluster_id, info in cluster_map.items()}

# Inspect Cluster Characteristics
dfc = df_state.copy()

# Add cluster numbers
dfc['Cluster'] = cluster_labels
dfc['Cluster Label'] = dfc['Cluster'].map(cluster_names)
cols_to_analyze = ['Cluster', "Cluster Label"] + optimal_features

cluster_means = dfc[cols_to_analyze].groupby(['Cluster', "Cluster Label"]).mean().reset_index()
cluster_means.to_csv(os.path.join(BASE_DIR, "resources", "clustering", "hclust_cluster_summary.csv"), index=False)

#########################################################
# --------------------- PLOTTING ---------------------- #
#########################################################

# ------ Silhouette Scores for Varying K Values ------
k_range = [2, 3, 4, 5, 6]
silhouette_scores = []

# Test each k and get score
for k in k_range:
    hclust = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
    cluster_labels_tmp = hclust.fit_predict(X_opt_scaled)

    # calculate silhouette score
    score = silhouette_score(X_opt_scaled, cluster_labels_tmp)
    silhouette_scores.append(score)

# Plot scores
plt.figure(figsize=(9,5))
plt.plot(k_range,
         silhouette_scores,
         marker='o',
         color='gray',
         linewidth=2,
         markersize=5
         )

# Highlight peak
optimal_score = silhouette_scores[k_range.index(optimal_k)]
plt.scatter(
    optimal_k, 
    optimal_score, 
    color="#85d6f4",      
    s=120,
    marker='o', 
    edgecolor='black',
    linewidth=1.5,
    zorder=2,
    label=f'Optimal $k$ ({optimal_k})'
)

plt.annotate(
    f'Optimal Clusters ($k={optimal_k}$)\nHighest Silhouette Score', 
    xy=(optimal_k + 0.05, optimal_score - 0.001),                                              # Coordinates for arrow point
    xytext=(optimal_k + 0.4, optimal_score - 0.02),                             # Coordinates for text
    fontweight='bold',
    fontsize=10,
    color="#56aece",
    arrowprops=dict(
        arrowstyle="->",            
        connectionstyle="arc3,rad=-0.2", 
        color='black',  
        linewidth=1.5   
    ),
    zorder=3
)

# Styling
plt.title("Silhouette Score Analysis for Optimal $k$", fontsize=14, fontweight='bold', loc='left')
plt.xlabel("Number of Clusters ($k$)", fontsize=12, labelpad=10)
plt.ylabel("Average\nSilhouette\nScore", fontsize=12, labelpad=10,
           rotation=0, va='center', ha='right')
plt.xticks(k_range)
plt.grid(axis='both', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "resources", "clustering", "hclust_silhouette_scores.png"))
# plt.show()

# ------ Cluster Snake Plot ------
df_scaled_features = pd.DataFrame(X_opt_scaled, columns=optimal_features)
df_scaled_features['Cluster'] = cluster_labels
df_scaled_features['Cluster'] = df_scaled_features['Cluster'].map(cluster_names)
print("Unique values in array before mapping:", np.unique(cluster_labels))
print("Mapped column head:\n", df_scaled_features['Cluster'].head())
# Convert to long format
df_melted = pd.melt(
    df_scaled_features, 
    id_vars=['Cluster'], 
    value_vars=optimal_features,
    var_name='Feature', 
    value_name='Z-Score'
)
df_melted['Feature_Formatted'] = df_melted['Feature'].str.replace(' ', '\n')
# Plot each cluster's relative characteristics based on Z-Score
plt.figure(figsize=(10,6))
sns.lineplot(
    data=df_melted, 
    x='Feature_Formatted', 
    y='Z-Score', 
    hue='Cluster', 
    marker='o',
    palette=cluster_colors, 
    linewidth=2.5,
    errorbar=None,
    hue_order=["Poor Access to Care", "Poor Family Planning", "Strong Health Ecosystem"]
)

# Add a baseline for the national average
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.text(x = 1.65, y = 0.03,
         s = "National Average")

# Styling
plt.title("Cluster Profile Snake Plot (Relative Characteristics)", fontsize=14, fontweight='bold', loc='left')
plt.ylabel("Standardized Values (Z-Score Relative to Average)", fontsize=12)
plt.xlabel("Optimal Feature Set", fontsize=12, labelpad=10)
plt.xticks(fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.savefig(os.path.join(BASE_DIR, "resources", "clustering", "hclust_snake_plot.png"))
# plt.show()

# ------ Interactive 3D Scatterplot ------
# Prepare data
plot_df = pd.DataFrame(X_opt_scaled, columns=optimal_features)
plot_df['Cluster'] = cluster_labels
plot_df['Cluster'] = plot_df['Cluster'].map(cluster_names)

# Convert colors to hex
cluster_colors_hex = {
    'Strong Health Ecosystem': mcolors.to_hex(sns.color_palette("cubehelix")[2]),
    'Poor Family Planning': mcolors.to_hex(sns.color_palette("cubehelix")[4]),
    'Poor Access to Care': mcolors.to_hex(sns.color_palette("cubehelix")[3])
}

# scatter plot
fig = px.scatter_3d(
    plot_df, 
    x=optimal_features[0], 
    y=optimal_features[1], 
    z=optimal_features[2],
    color='Cluster',
    color_discrete_map=cluster_colors_hex, 
    title="Interactive Hierarchical Clustering (k=3)",
    opacity=0.7,
    category_orders={"Cluster": ["Poor Access to Care", "Poor Family Planning", "Strong Health Ecosystem"]} 
)

# styling
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(
        yanchor="top", 
        y=0.9, 
        xanchor="left", 
        x=0.1,
        traceorder="normal" # Preserves category_orders layout
    )
)

# save figure
fig.write_html(os.path.join(BASE_DIR, "resources", "clustering", "hclust_3d_scatter.html"))
# fig.show()

# ----------------- US Map -------------------
# Cluster Map
df_map = dfc.copy()
# df_map = df_map.groupby('State')['Cluster'].mean().reset_index()
df_map['Cluster'] = np.ceil(df_map['Cluster'])
df_map['Cluster Label'] = df_map['Cluster'].map(cluster_names)

fig = px.choropleth(
    df_map,
    locations='State',
    locationmode="USA-states",
    color = 'Cluster Label',
    color_discrete_map=cluster_colors_hex,
    category_orders={"Cluster Label": ["Low Risk", "Moderate Risk", "High Risk"]},
    scope='usa',
    # title=f"US Health Clusters"
)

fig.update_layout(
    margin={"r":0,"t":50,"l":0,"b":0},
    legend_title_text='Healthcare Status Cluster'
)

fig.write_html(os.path.join(BASE_DIR, "resources", "clustering", f"hclust_map.html"))
# fig.show()

# ----------------- Dendrogram -------------------
# Dendrogram
ddf = X[optimal_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(ddf)
Z_linkage_state = linkage(X_scaled, method='average', metric='cosine')
state_labels = fcluster(Z_linkage_state, t=optimal_k, criterion='maxclust')
plt.figure(figsize=(14,8))
dendrogram(
    Z_linkage_state,
    labels=list(df_state['State']),
    leaf_rotation=0,
    leaf_font_size=9,
    color_threshold=0.12
)
plt.axhline(y=0.7, color='red', linestyle='--', linewidth=1.5, 
            label='k=3 State Cutoff Threshold')
plt.title("State-Level Hierarchical Clustering Dendrogram\n(Aggregated Historical Feature Profiles)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("US States (Unique Profiles)", fontsize=12, labelpad=10)
plt.ylabel("Cosine\nDistance", fontsize=12,
           rotation=0, va='center', ha='right', labelpad=10)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "resources", "clustering", "hclust_dendrogram.png"))
# plt.show()