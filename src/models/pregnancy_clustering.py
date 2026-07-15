import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
from pandas.errors import PerformanceWarning

warnings.filterwarnings('ignore', category=PerformanceWarning)

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
guttmacher_path = os.path.join(BASE_DIR, "data", "raw", "Guttmacher")
cluster_rec = os.path.join(BASE_DIR, "resources", "clustering")

# Load Data
data = pd.read_csv(os.path.join(guttmacher_path, 'NatStatePregnancy.csv'))
print(data.shape)
print(f"Missing Values:\n{data.isna().sum()}")
# data.head()

# Preprocessing
# Copy
tmp = data.copy()
tmp = tmp.drop(columns=['notes', 'versiondate'])

# Filter for National Level Data
tmp = data.loc[data.state.str.upper()=='US'].reset_index()

# Fix Data Types
numeric_cols = [col for col in tmp.columns if col != 'state']
for col in numeric_cols:
    tmp[col] = pd.to_numeric(tmp[col], errors='coerce')

# Calculate Miscarriage Rates
age_groups = {
    '1517': ('miscarriages1517', 'population1517'),
    '1819': ('miscarriages1819', 'population1819'),
    '2024': ('miscarriages2024', 'population2024'),
    '2529': ('miscarriages2529', 'population2529'),
    '3034': ('miscarriages3034', 'population3034'),
    '3539': ('miscarriages3539', 'population3539'),
    '40plus': ('miscarriages40plus', 'population40plus'),
    'total': ('miscarriagestotal', 'population1544')
}
for suffix, (misc_col, pop_col) in age_groups.items():
    rate_name = f'miscarriagerate{suffix}'
    # Compute rate per 1,000 women
    tmp[rate_name] = (tmp[misc_col] / tmp[pop_col]) * 1000

# Get Relevant Rate Columns
selected_features = [
    # --- Pregnancy Rates (Per 1,000) ---
    'pregnancyrate1517', 'pregnancyrate1819', 'pregnancyrate2024', 
    'pregnancyrate2529', 'pregnancyrate3034', 'pregnancyrate3539', 'pregnancyrate40plus',
    'pregnancyratetotal',
    
    # --- Abortion Rates (Per 1,000) ---
    'abortionrate1517', 'abortionrate1819', 'abortionrate2024', 
    'abortionrate2529', 'abortionrate3034', 'abortionrate3539', 'abortionrate40plus',
    'abortionratetotal',
    
    # --- Birth Rates (Per 1,000) ---
    'birthrate1517', 'birthrate1819', 'birthrate2024', 
    'birthrate2529', 'birthrate3034', 'birthrate3539', 'birthrate40plus',
    'birthratetotal',
    
    # --- Miscarriage Rates (Per 1,000) ---
    'miscarriagerate1517', 'miscarriagerate1819', 'miscarriagerate2024', 
    'miscarriagerate2529', 'miscarriagerate3034', 'miscarriagerate3539', 'miscarriagerate40plus',
    'miscarriageratetotal'
]
selected_features = [col for col in selected_features if col in tmp.columns]

# Subset, sort by year
df_prep = tmp[['year'] + selected_features].copy()
df_prep = df_prep.sort_values('year').reset_index(drop=True)
df_prep.set_index('year', inplace=True)

# Interpolate missing values
df_int = df_prep.interpolate(method='linear', limit_direction='both')
print(f"Resolved All Missing Values: {df_int.isna().sum().sum()==0}")

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_int)
X_scaled_df = pd.DataFrame(X_scaled, columns=df_int.columns, index=df_int.index)

print(f"Data prepared! Shape: {X_scaled_df.shape} (Years: {X_scaled_df.index.min()} to {X_scaled_df.index.max()})")

# Dimensionality Reduction - PCA

pca_full = PCA()
pca_full.fit(X_scaled_df)

# cumulative explained var
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

# n components needed to explain 90% variance
n_comp_90 = np.argmax(cum_var >= 0.9) + 1

# scree plot for cumulative variance
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cum_var)+1), cum_var, marker='o', linestyle="--", color='purple')
plt.axhline(y=0.90, color='r', linestyle=':', label='90% Variance Threshold')
plt.axvline(x=n_comp_90, color='g', linestyle=':', label=f'{n_comp_90} Components')

plt.title('PCA Explained Variance', fontsize=12, fontweight='bold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)
# save
plt.savefig(os.path.join(cluster_rec, "preg_pca_expvar.png"), dpi=300)

print(f"To capture 90% of the variance, need to use {n_comp_90} principal components.")

# Transform data into 3-component PCA space
pca = PCA(n_components=n_comp_90, random_state=42)
X_pca = pca.fit_transform(X_scaled_df)
X_pca_df = pd.DataFrame(
    X_pca,
    columns=[f"PC{i+1}" for i in range(n_comp_90)],
    index=X_scaled_df.index
)

# Save processed data snapshot for report
X_pca_df.to_csv(os.path.join(cluster_rec, "preg_data_pca.csv"))

# Cluster Counts - Elbow & Silhouette Methods
cluster_range = range(2,11)
inertia_scores = []
silhouette_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k,
                    random_state=42,
                    n_init=10)
    
    # fit on PCA data
    labels = kmeans.fit_predict(X_pca_df)

    inertia_scores.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca_df, labels))

# Plot
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
ax1.set_ylabel('Inertia (on PCA space)', color=color, fontweight='bold')
ax1.plot(cluster_range, inertia_scores, marker='o', color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.grid(False)
color = 'tab:green'
ax2.set_ylabel('Silhouette Score (on PCA space)', color=color, fontweight='bold')
ax2.plot(cluster_range, silhouette_scores, marker='s', color=color, linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('K-Means Evaluation on PCA-Reduced Data', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig(os.path.join(cluster_rec, "preg_kmeans_eval.png"), dpi=300)

# Final Model

# Final PCA DataFrame
pca_final = PCA(n_components=3, random_state=42)
X_pca_final = pca_final.fit_transform(X_scaled_df)
X_pca_df = pd.DataFrame(
    X_pca_final,
    columns=['PC1', 'PC2', 'PC3'],
    index=X_scaled_df.index
)

# Fit KMeans with 4 clusters
kmeans_final = KMeans(n_clusters=4,
                      random_state=42,
                      n_init=10)
cluster_labels = kmeans_final.fit_predict(X_pca_df)

# Add cluster assignments back to original data
df = df_int.copy()
df['cluster'] = cluster_labels

# Print years for each cluster to see if there's a chronological trend
for cluster_num in sorted(df['cluster'].unique()):
    cyears = df.loc[df.cluster==cluster_num].index.tolist()
    print(f"Cluster {cluster_num} (Years {min(cyears)} - {max(cyears)}):")
    print(f" -> Total Years: {len(cyears)}")
    print(f" -> Years: {cyears}\n")

# Define consistent color palette
colors_list = ['#440154', '#31688e', '#35b779', '#9fd938'] # Viridis-inspired sequence

cluster_names = {
    2: "1973-1976: Post-Roe Baseline",
    3: "1977-1996: Post-Casey Stabilization",
    1: "1997-2010: Contraceptive Revolution",
    0: "2011-2020: TRAP Restriction Era"
}

# Separate color dicts for both plotting structures
timeline_palette = {k: colors_list[i] for i, k in enumerate(cluster_names.keys())}
pca_palette = {cluster_names[k]: colors_list[i] for i, k in enumerate(cluster_names.keys())}

# Plot - Timeline w Clusters
year_plot = X_pca_df.copy()
year_plot['Cluster'] = cluster_labels
year_plot['Year'] = year_plot.index
year_plot = year_plot[['PC1', 'Cluster', 'Year']]

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 4))

# line chart
sns.lineplot(
    data=year_plot,
    x='Year',
    y='PC1',              
    hue='Cluster',
    marker='o',  
    linewidth=2.5,
    markersize=7,
    palette=timeline_palette
)

# Styling
plt.title("Historical Timeline: PC1 Score by Cluster", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Year", labelpad=10)
plt.ylabel("PC1 Score", labelpad=10, rotation=0, ha='right', va='center')

plt.legend(title="K-Means Cluster", loc='best', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(cluster_rec, "preg_timeline.png"), dpi=300)

# Plot - Clusters with PC1 and PC2
# Merge the cluster labels to PCA coordinates DataFrame
df_pca_plot = X_pca_df.copy()
df_pca_plot['Cluster'] = cluster_labels
df_pca_plot['Year'] = df_pca_plot.index

# setup
plt.figure(figsize=(11, 7))
sns.set_theme(style="whitegrid")

# Define labels
cluster_names = {
    2: "1973-1976: Post-Roe Baseline",
    3: "1977-1996: Post-Casey Stabilization",
    1: "1997-2010: Contraceptive Revolution",
    0: "2011-2020: TRAP Restriction Era"
}
df_pca_plot['Era'] = df_pca_plot['Cluster'].map(cluster_names)

# Plot years as points in PCA space, colored by their cluster
ax = sns.scatterplot(
    data=df_pca_plot,
    x='PC1',
    y='PC2',
    hue='Era',
    palette=pca_palette,
    s=120,
    edgecolor='black',
    alpha=0.85,
    legend='full'
)

# connect the dots
plt.plot(df_pca_plot['PC1'], df_pca_plot['PC2'], color='gray', linestyle='-', alpha=0.4, zorder=1)

# annotate key years
key_years = [1973, 1977, 1992, 1997, 2010, 2011, 2020]
for idx, row in df_pca_plot.iterrows():
    if row['Year'] in key_years:
        plt.annotate(
            str(int(row['Year'])),
            (row['PC1'], row['PC2']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontweight='bold',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3, ec="gray")
        )

# styling
plt.title("U.S. Reproductive Health Trajectory (1973 - 2020)\nK-Means Clustering on 3D PCA Space", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Principal Component 1 (Overall Trend & Volume)", fontweight='bold')
plt.ylabel("Principal Component 2 (Outcome Divergence)", fontweight='bold')
plt.legend(title="Identified Historical Eras", loc='best', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig(os.path.join(cluster_rec, "preg_viz_final.png"), dpi=300)

