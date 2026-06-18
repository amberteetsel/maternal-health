### PRINCIPAL COMPONENT ANALYSIS
### HEALTH RANKINGS DATA - ALL FEATURES

# Dependencies
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Base Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# pca_res = os.path.join(BASE_DIR, "resources", "pca")

# Load preprocessed health data
data = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "HealthRankings", "health_preprocessed.csv"))

# Feature Selection
economic_cols = [
    # 'Avoided Care Due to Cost',
    # 'Concentrated Disadvantage',
    # 'Food Insecurity',
    'Gender Pay Gap',
    # 'Infant Child Care Affordability',
    'Poverty',
    'Unemployment',
    'Uninsured Women',
    'WIC Shortfall',
    'Voter Abstainment']

structural_cols = [
    'Inadequate Prenatal Care',
    'No Cancer Screen',
    'Maternity Care Desert',
    'Maternity Practices Score',
    'No Postpartum Visit',
    'No Preventative',
    'Patients Per Doctor',      # choose over maternity care desert
    'Unintended Pregnancy']

medical_cols = [
    'Infant Mortality',
    # 'Low Birth Weight',
    'Maternal Mortality',
    # 'Neonatal Mortality',
    'Postpartum Depression',
    # 'Severe Maternal Morbidity',
    'Smoking During Pregnancy']

features_all = economic_cols+structural_cols+medical_cols

df = data[["State", "Year"] + features_all].copy()

# Save and remove categorical labels State, Year
cluster_label_index = df[['State', 'Year']].copy()
X = df.copy().drop(columns=['State', 'Year'])

# Scaling
scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(X)

# save input to show on site
tmp = pd.DataFrame(columns=features_all, data=X_scaled_all)
output_dir = os.path.join(BASE_DIR, "resources", "pca")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
tmp.to_csv(os.path.join(output_dir, "pca_input_data.csv"), index=False)
del tmp

# initialize and fit PCA
pca_all = PCA(random_state=42)
X_pca_all = pca_all.fit_transform(X_scaled_all)

# new DF with principal components
pca_cols = [f'PC{i+1}' for i in range(pca_all.n_components_)]
df_pca_all_results = pd.DataFrame(X_pca_all, columns=pca_cols)

df_pca_all_results['State'] = cluster_label_index['State'].values           # add back for visuals
df_pca_all_results['Year'] = cluster_label_index['Year'].values             # add back for visuals
df_pca_all_results = df_pca_all_results[['State', 'Year'] + pca_cols]       # re-order columns

# Calculate explained variance ratios
explained_variance_all = pca_all.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_all)

print("Variance breakdown by component:")
for i, (var, cum_var) in enumerate(zip(explained_variance_all, cumulative_variance), 1):
    print(f"  PC{i}: Explains {var*100:.2f}% | Cumulative: {cum_var*100:.2f}%")

######################################################
# PLOTTING
######################################################

# ----------------------------------------------------
# SCREE PLOT
# ----------------------------------------------------

# calculate individual/cumulative variance
exp_var_pct = explained_variance_all * 100
cum_var_pct = np.cumsum(explained_variance_all) * 100
components = range(1, len(exp_var_pct) + 1)

plt.figure(figsize=(10, 5))
plt.plot(components, cum_var_pct, marker='o', linestyle='-', color='darkorange', linewidth=2, label='Cumulative Explained Variance')

# pc2 point
plt.annotate(f'{cum_var_pct[1]:.2f}%', 
             xy=(2, cum_var_pct[1]), 
             xytext=(2.5, cum_var_pct[1] - 8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

# pc5 point (70% threshold)
plt.annotate(f'{cum_var_pct[4]:.2f}%', 
             xy=(5, cum_var_pct[4]), 
             xytext=(5.5, cum_var_pct[4] - 8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

# Layout and Labels
plt.title("PCA Scree Plot: Variance Analysis of All Health Metrics", fontsize=14, fontweight='bold')
plt.xlabel("Principal Components", fontsize=12)
plt.ylabel("Percentage of Variance Explained (%)", fontsize=12)
plt.xticks(components)
plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='70% Information Threshold')
plt.grid(axis='y', alpha=0.3)
plt.legend(loc='center right')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "resources", "pca", "pca_scree_all.png"))

# ----------------------------------------------------
# LOADING PLOT
# ----------------------------------------------------
loadings = pca_all.components_[:2, :].T 

df_loadings = pd.DataFrame(
    loadings, 
    columns=['PC1_Loading', 'PC2_Loading'], 
    index=features_all
).reset_index().rename(columns={'index': 'Feature'})

# Sort by PC1
df_loadings = df_loadings.sort_values(by='PC1_Loading', ascending=False)

# bar plot
plt.figure(figsize=(12, 8))
df_melted = pd.melt(df_loadings, id_vars=['Feature'], value_vars=['PC1_Loading', 'PC2_Loading'],
                    var_name='Component', value_name='Loading Value')

sns.barplot(
    data=df_melted,
    y='Feature',
    x='Loading Value',
    hue='Component',
    palette=['#1f77b4', '#ff7f0e']
)

# Formatting and reference lines
plt.axvline(0, color='black', linestyle='-', alpha=0.4)
plt.axvline(0.3, color='red', linestyle='--', alpha=0.3, label='Strong Positive/Negative Influence (> |0.3|)')
plt.axvline(-0.3, color='red', linestyle='--', alpha=0.3)

plt.title("Feature Loadings: How Raw Metrics Direct PC1 and PC2", fontsize=14, fontweight='bold')
plt.xlabel("Loading Coefficient (Correlation with Component)", fontsize=12)
plt.ylabel("Public Health Features", fontsize=12)
plt.legend(loc='upper left')
plt.grid(axis='x', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "resources", "pca", "pca_loading_all.png"))