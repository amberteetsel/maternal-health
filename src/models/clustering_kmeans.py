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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Base Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Load Data
health_data = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "HealthRankings", "health_preprocessed.csv"))

# Prepare for Clustering: Remove categorical labels
cluster_label_index = health_data[['State', 'Year']]
X = health_data.copy().drop(columns=['State', 'Year'])

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

# Custom Function to Find Optimal Features and K Value
def find_best_feature_combination(df, available_features, min_features=3, max_features=6, k_values=[3, 4, 5]):
    """
    Evaluates all possible combinations of features across multiple k values 
    to find the combination and cluster count that maximizes the silhouette score.
    """
    best_score = -1
    best_combo = None
    best_k = None
    
    # Total combinations calculation for user visibility
    total_combos = sum(len(list(itertools.combinations(available_features, r))) 
                       for r in range(min_features, max_features + 1))
    print(f"Starting grid search over {total_combos} unique feature combinations across k={k_values}...")

    # Iterate through the specified range of feature subset sizes
    for r in range(min_features, max_features + 1):
        for combo in itertools.combinations(available_features, r):
            # Extract subset
            combo_list = list(combo)
            X_subset = df[combo_list]
            
            # Scale the features for this specific subset
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            
            # Test each k value on this subset
            for k in k_values:
                # Ensure we don't have fewer samples than clusters
                if len(X_scaled) <= k:
                    continue
                    
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                # Silhouette score requires at least 2 clusters
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    
                    # Update tracking variables if we found a new maximum
                    if score > best_score:
                        best_score = score
                        best_combo = combo_list
                        best_k = k

    print("\n--- Grid Search Optimization Complete ---")
    print(f"Highest Silhouette Score: {best_score:.4f}")
    print(f"Optimal Number of Clusters (k): {best_k}")
    print(f"Optimal Feature Combination ({len(best_combo)} features): {best_combo}")
    
    return {
        'best_score': best_score,
        'best_k': best_k,
        'best_features': best_combo
    }

# # Run function many times and pick best overall results based on Silhouette Score
# n_runs = 30             # number of runs
# top_n_to_keep = 3       # number of choices to display at end

# # List to store results from every run
# all_results = []

# print(f"Starting experiment: Running grid search {n_runs} times...\n")

# for i in range(n_runs):
#     print(f"=========================================")
#     print(f" RUN {i+1} of {n_runs}")
#     print(f"=========================================")
    
#     # Get random sample of 8 features from features_all
#     current_features = random.sample(features_all, 8)
    
#     # Run function
#     run_output = find_best_feature_combination(
#         df=X_all, 
#         available_features=current_features, 
#         min_features=3, 
#         max_features=6, 
#         k_values=[3, 4, 5]
#     )
    
#     # Store inputs and results
#     all_results.append({
#         'run_index': i + 1,
#         'feature_pool_used': current_features,
#         'best_score': run_output['best_score'],
#         'best_k': run_output['best_k'],
#         'best_features': run_output['best_features']
#     })

# # Get Top N Results

# # Sort all runs by best_score descending
# sorted_results = sorted(all_results, key=lambda x: x['best_score'], reverse=True)
# top_3_results = sorted_results[:top_n_to_keep]

# # --- Print Top Results ---
# print("\n" + "="*50)
# print(f" TOP {top_n_to_keep} BEST RESULTS (Sorted by Silhouette Score)")
# print("="*50)

# for rank, result in enumerate(top_3_results, 1):
#     print(f"\n🏆 Rank {rank} (From Run #{result['run_index']})")
#     print(f"  - Silhouette Score: {result['best_score']:.4f}")
#     print(f"  - Optimal Clusters (k): {result['best_k']}")
#     print(f"  - Optimal Feature Subset: {result['best_features']}")
#     print(f"  - Total Pool Sampled From: {result['feature_pool_used']}")

# FINAL KMEANS MODEL WITH OPTIMAL INPUTS
optimal_features = ['Maternity Care Desert', 'Maternal Mortality', 'Patients Per Doctor']
optimal_k = 3
X_optimal = X_all[optimal_features]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_optimal)

# Run Optimized KMeans Model
kmeans = KMeans(n_clusters = optimal_k,
                random_state=42,
                n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Map Clusters to Descriptive Names, Standardize Color Palette
cluster_map = {
    0: {
        'name': 'Low Risk',
        'color': sns.color_palette("RdBu")[-1]
    },
    1: {
        'name': 'Moderate Risk',
        'color': sns.color_palette("RdBu")[1]
    },
    2: {
        'name': 'High Risk',
        'color': sns.color_palette("RdBu")[0]
    }
}

cluster_names = {cluster_id: info['name'] for cluster_id, info in cluster_map.items()}
cluster_colors = {info['name']: info['color'] for cluster_id, info in cluster_map.items()}

# Inspect Cluster Characteristics
dfc = health_data.copy()

# Add cluster numbers
dfc['Cluster'] = cluster_labels
dfc['Cluster Label'] = dfc['Cluster'].map(cluster_names)
cols_to_analyze = ['Cluster', "Cluster Label"] + optimal_features

cluster_means = dfc[cols_to_analyze].groupby(['Cluster', 'Cluster Label']).mean().reset_index()
cluster_means.to_csv(os.path.join(BASE_DIR, "resources", "clustering", "kmeans_cluster_summary.csv"), index=False)
print(cluster_means)

# ==================================================================================
# PLOTTING
# ==================================================================================

# ------ Silhouette Scores for Varying K Values ------
k_range = [2, 3, 4, 5, 6]
silhouette_scores = []

# Test each k and get score
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # calculate silhouette score
    score = silhouette_score(X_scaled, cluster_labels)
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
    xy=(optimal_k + 0.05, optimal_score - 0.001),                               # Coordinates for arrow point
    xytext=(optimal_k + 0.4, optimal_score - 0.01),                             # Coordinates for text
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
plt.savefig(os.path.join(BASE_DIR, "resources", "clustering", "kmeans_silhouette_scores.png"))
# plt.show()

# ------ Cluster Profile Snake Plot ------
df_scaled_features = pd.DataFrame(X_scaled, columns=optimal_features)
df_scaled_features['Cluster'] = cluster_labels
df_scaled_features['Cluster'] = df_scaled_features['Cluster'].map(cluster_names)

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
    hue_order=["High Risk", "Moderate Risk", "Low Risk"]
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
plt.savefig(os.path.join(BASE_DIR, "resources", "clustering", "kmeans_snake_plot.png"))
# plt.show()

# ------ INTERACTIVE 3D SCATTER PLOT ------
# Prepare data
plot_df = pd.DataFrame(X_scaled, columns=optimal_features)
plot_df['Cluster'] = cluster_labels
plot_df['Cluster'] = plot_df['Cluster'].map(cluster_names)

# Convert colors to hex
cluster_colors_hex = {
    'Low Risk': mcolors.to_hex(sns.color_palette("RdBu")[-1]),
    'Moderate Risk': mcolors.to_hex(sns.color_palette("RdBu")[1]),
    'High Risk': mcolors.to_hex(sns.color_palette("RdBu")[0])
}

# scatter plot
fig = px.scatter_3d(
    plot_df, 
    x=optimal_features[0], 
    y=optimal_features[1], 
    z=optimal_features[2],
    color='Cluster',
    color_discrete_map=cluster_colors_hex, 
    title="Interactive K-Means Clustering (k=3)",
    opacity=0.7,
    category_orders={"Cluster": ["High Risk", "Moderate Risk", "Low Risk"]} 
)

# add centroids
centroids = kmeans.cluster_centers_

fig.add_trace(
    go.Scatter3d(
        x=centroids[:, 0], 
        y=centroids[:, 1], 
        z=centroids[:, 2], 
        mode='markers',
        marker=dict(
            size=10,
            color='black',       
            symbol='x',     
            line=dict(width=2, color='white')
        ),
        name='Centroids'
    )
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
fig.write_html(os.path.join(BASE_DIR, "resources", "clustering", "kmeans_3d_scatter.html"))
# fig.show()

# ------ CLUSTER US MAP ------
df_map = dfc.copy()
df_map = df_map.groupby('State')['Cluster'].mean().reset_index()
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
    legend_title_text='Maternal Risk Cluster'
)

fig.write_html(os.path.join(BASE_DIR, "resources", "clustering", f"kmeans_map.html"))
# fig.show()