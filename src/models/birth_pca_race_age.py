# Dependencies
import pandas as pd
import numpy as np
import os
import zipfile
import subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Base Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
nchs_path = os.path.join(BASE_DIR, "data", "raw", "NCHS-Birth")
pca_res = os.path.join(BASE_DIR, "resources", "pca")

# define target cols/values
target_columns = {

    'year': (8, 12),
    'month': (12, 14),

    # Demographic Characteristics
    "mage": (74, 76),                         # maternal age in years
    "mrace": (106, 107),                      # maternal race, recode 6
    "mhisp": (114, 115),                      # maternal hispanic origin recode
    "med": (123, 124),                        # maternal education
    
    # Risk Factors
    "rf_gdb": (313, 314),                     # gestational diabetes
    "rf_ghyp": (315, 316),                    # gestational hypertension
    "rf_ehyp": (316, 317),                    # hypertension eclampsia

    # Maternal Morbidity
    "mm_trans": (414, 415),                   # maternal transfusion
    "mm_plac": (415, 416),                    # perineal laceration
    "mm_rupt": (416, 417),                    # ruptured uterus
    "mm_uhyst": (417, 418),                   # unplanned hysterectomy
    "mm_icu": (418, 419) ,                    # ICU admission
}

demo_cols = ['mage', 'mrace', 'mhisp', 'med']
clinical_cols = ['rf_gdb', 'rf_ghyp', 'rf_ehyp',
                 'mm_trans', 'mm_plac', 'mm_rupt', 'mm_uhyst', 'mm_icu']


# Extract Data from Zip Files (5% sample)
years = list(range(2018, 2024 + 1))

file_paths = []
raw_snapshot_paths = []

for yr in years:
    path = os.path.join(nchs_path, f"Nat{yr}us.zip")
    file_paths.append(path)

    raw_path = os.path.join(nchs_path, f"births{yr}_raw.txt")
    raw_snapshot_paths.append(raw_path)

col_names = list(target_columns.keys())
col_specs = list(target_columns.values())

# Sampling Rate
SAMPLE_RATE = 0.05  
sampled_chunks = []

# ==============================================================================
# DATA EXTRACTION -- takes a few minutes
# ==============================================================================
for i in range(len(file_paths)):
    p = file_paths[i]
    if not os.path.exists(p):
        print(f"File path missing, skipping: {p}")
        continue

    # Use standard zipfile to grab the metadata and internal file name
    with zipfile.ZipFile(p, 'r') as z:
        internal_file_name = z.namelist()[0]
        zinfo = z.getinfo(internal_file_name)
        
    print(f"\nProcessing file: {internal_file_name}")

    proc = None
    f = None

    # Method 9 = Deflate64
    if zinfo.compress_type == 9:
        print(" -> Detected Deflate64. Streaming via system 'unzip' pipe...")
        # Use macOS native unzip to stream the file straight to stdout
        proc = subprocess.Popen(
            ['unzip', '-p', p, internal_file_name],
            stdout=subprocess.PIPE,
            text=True,
            encoding='ascii'
        )
        f = proc.stdout
    else:
        print(" -> Detected standard compression. Opening with python zipfile...")
        # Open normally using Python zip wrapper
        z_open = zipfile.ZipFile(p, 'r')
        f = z_open.open(internal_file_name)

    try:
        # Process in chunks
        for chunk in pd.read_fwf(
            f, 
            colspecs=col_specs, 
            names=col_names, 
            chunksize=250000, 
            encoding='ascii',
            dtype=str  # Kept as string during ingest to prevent truncation/dropping of codes
        ):
            # Trim leading/trailing whitespace
            for col in chunk.columns:
                chunk[col] = chunk[col].str.strip()

            # Filter out records (excluding 'U' values)
            chunk = chunk[
                (chunk['rf_gdb'] != 'U') &
                (chunk['rf_ghyp'] != 'U') &
                (chunk['rf_ehyp'] != 'U') &
                (chunk['mm_trans'] != 'U') &
                (chunk['mm_plac'] != 'U') &
                (chunk['mm_rupt'] != 'U') &
                (chunk['mm_uhyst'] != 'U') &
                (chunk['mm_icu'] != 'U')
            ]

            # Apply uniform random sampling to the chunk
            if not chunk.empty:
                chunk_sampled = chunk.sample(frac=SAMPLE_RATE, random_state=42)
                sampled_chunks.append(chunk_sampled)
                
    finally:
        # Clean up files/processes properly
        if proc:
            # Kill subprocess stream if it's still running
            proc.stdout.close()
            proc.wait()
        else:
            f.close()
            z_open.close()

# ==============================================================================
# CONSOLIDATE, DIAGNOSTICS & SUMMARY
# ==============================================================================
# Combine all sampled slices
df_final = pd.concat(sampled_chunks, ignore_index=True)

print("\n" + "="*60)
print(f"SUCCESS: Final consolidated dataframe shape: {df_final.shape}")
print("="*60)

print("\nSampled Births Breakdown by Year:")
print(df_final.groupby('year').size())

# CLEANING - Clinical Columns
data = df_final.copy()

# mapping dictionary for CDC's Y/N/U encoding
clinical_mapping = {
    '1': 1.0, 'Y': 1.0, 'y': 1.0,
    '2': 0.0, 'N': 0.0, 'n': 0.0,
    '9': 0.0, 'U': 0.0, 'u': 0.0
}

for col in clinical_cols:
    if col in data.columns:
        data[col] = data[col].map(clinical_mapping).fillna(0.0)
    else:
        print(f"Warning: '{col} not found in data.")

# CLEANING - Demographic Columns

# maternal education
med_map = {
    '1': "HS_Nongrad",
    '2': "HS_Nongrad",
    '3': "HS_Grad_GED",
    '4': "College_Nongrad",
    '5': "College_Nongrad",
    '6': "Bachelors",
    '7': "Masters",
    '8': "Advanced",
    '9': "Unknown"
}
data['med'] = data['med'].map(med_map)

data['mage'] = pd.to_numeric(data['mage'], errors='coerce')

# race mapping
race_map = {
    '1': 'NH_White',
    '2': 'NH_Black',
    '3': 'AIAN',
    '4': 'Asian',
    '5': 'NHOPI',
    '6': 'Multi'
}

def resolve_race_ethnicity(row):
    try:
        hisp = str(row['mhisp']).strip()
        # CDC code: 0 = Non-Hispanic, 9 = Unknown/Not Stated
        if hisp not in ['0', '9', 'U', '']:
            return 'Hispanic'
        
        race_code = str(row['mrace']).strip()
        return race_map.get(race_code, 'Other / Unknown')
    except Exception:
        return 'Other / Unknown'
    
data['race_eth'] = data.apply(resolve_race_ethnicity, axis=1)

# Drop missing/unknown
data = data.dropna(subset=['mage'])
data = data[data['race_eth'] != 'Other / Unknown'].reset_index(drop=True)

# save copy of data before preprocessing for modeling
tmp = data.drop(columns=['med']).reset_index(drop=True)
tmp = tmp.head(100)
tmp.to_csv(os.path.join(pca_res, "birth_data_raw.csv"), index=False)

data.head()

# PLOT 1
# Race/Ethnicity Frequency
reth_map = {
    'NH_White': 'Non-Hispanic White',
    'Hispanic': 'Hispanic',
    'NH_Black': 'Non-Hispanic Black',
    'Asian': 'Asian',
    'Multi': 'Multiple Races',
    'AIAN': 'American Indian / Alaska Native',
    'NHOPI': 'Native Hawaiian / Pacific Islander'
}

means = data['race_eth'].value_counts()/len(data)

df_means = pd.DataFrame({
    'raw_key': means.index,
    'mean_value': means.values
})
df_means['feature_name'] = df_means['raw_key'].map(reth_map)
df_means = df_means.sort_values(by='mean_value', ascending=True)

# Plot
plt.figure(figsize=(10, 6))

# plot bars
bars = plt.barh(df_means['feature_name'], 
                df_means['mean_value'], 
                color=sns.color_palette('flare', 7),
                # edgecolor='steelblue',
                # height=0.6
                )

# labels
for bar, name, val in zip(bars, df_means['feature_name'], df_means['mean_value']):
    width = bar.get_width()
    y_pos = bar.get_y() + bar.get_height() / 2.0
    
    # data val outside bar
    label_text = f"{val:.1%}"
    plt.text(x=width + (0.01),
             y=y_pos, 
             s=label_text, 
             va='center', 
             ha='left', 
             color='dimgray', 
             fontweight='bold',
             fontsize=10)

# Styling
plt.title("Distribution of Maternal Race/Ethnicity", fontweight='bold', pad=15)
plt.xlabel("Frequency", labelpad=10)

# borders
for spine in ['top', 'right', 'left']:
    plt.gca().spines[spine].set_visible(False)

# gridlines
plt.grid(axis='x', linestyle='--', alpha=0.3)

# expand xlim to fit labels
plt.xlim(0, df_means['mean_value'].max() * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(pca_res, "race_distribution.png"), dpi=300)

# PLOT 2
feature_mapping = {
    'rf_gdb': "Gestational Diabetes",
    'rf_ghyp': "Gestational Hypertension",
    'rf_ehyp': "Hypertension Eclampsia",
    'mm_trans': "Maternal Transfusion",
    'mm_plac': "Perineal Laceration",
    'mm_rupt': "Ruptured Uterus",
    'mm_uhyst': "Unplanned Hysterectomy",
    'mm_icu': "ICU Admission"
}
type_map = {
    'rf_gdb': "Risk Factor",
    'rf_ghyp': "Risk Factor",
    'rf_ehyp': "Risk Factor",
    'mm_trans': "Complication/Morbidity",
    'mm_plac': "Complication/Morbidity",
    'mm_rupt': "Complication/Morbidity",
    'mm_uhyst': "Complication/Morbidity",
    'mm_icu': "Complication/Morbidity",
}

color_palette = {
    "Risk Factor": "#E07A5F",            # reddish
    "Complication/Morbidity": "#3D5A80"   # deep blue
}

means = data[clinical_cols].mean()

df_means = pd.DataFrame({
    'raw_key': means.index,
    'mean_value': means.values
})
df_means['feature_name'] = df_means['raw_key'].map(feature_mapping)
df_means = df_means.sort_values(by='mean_value', ascending=True)
df_means['type'] = df_means['raw_key'].map(type_map)
df_means['color'] = df_means['type'].map(color_palette)

# Plot
plt.figure(figsize=(10, 6))

# plot bars
bars = plt.barh(df_means['feature_name'], 
                df_means['mean_value'], 
                color=df_means['color'].to_list(), 
                # edgecolor='steelblue',
                # height=0.6
                )

# labels
for bar, name, val in zip(bars, df_means['feature_name'], df_means['mean_value']):
    width = bar.get_width()
    y_pos = bar.get_y() + bar.get_height() / 2.0
    
    # data val outside bar
    label_text = f"{val:.2%}"
    plt.text(x=width + (0.001),
             y=y_pos, 
             s=label_text, 
             va='center', 
             ha='left', 
             color='dimgray', 
             fontweight='bold',
             fontsize=10)

# Styling
plt.title("Prevalence of Clinical Risk Factors and Maternal Morbidities", fontweight='bold', pad=15)
plt.xlabel("Mean Value (Prevalence)", labelpad=10)

# remove yticks
# plt.gca().set_yticklabels([])
# plt.gca().tick_params(axis='y', which='both', length=0) # Hide tick marks on y-axis

# borders
for spine in ['top', 'right', 'left']:
    plt.gca().spines[spine].set_visible(False)

# gridlines
plt.grid(axis='x', linestyle='--', alpha=0.3)

legend_handles = [Patch(facecolor=color, label=label) for label, color in color_palette.items()]
plt.legend(handles=legend_handles, loc='lower right', frameon=True, facecolor='white', edgecolor='none')

# expand xlim to fit labels
plt.xlim(0, df_means['mean_value'].max() * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(pca_res, "birth_clinical_prevalence.png"), dpi=300)

# MORE CLEANING
# Isolate & Standardize Features
X_raw = data[clinical_cols].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_scaled_df = pd.DataFrame(X_scaled,
                           columns=clinical_cols,
                           index=data.index)

# Save clean snapshot for report
X_scaled_df.head(100).to_csv(os.path.join(pca_res, "birth_data_processed.csv"), index=False)

# matching metadata for later plotting
df_meta = data[['year', 'month', 'mage', 'race_eth', 'med']].copy()

print("\n" + "="*60)
print("CLEANING COMPLETE!")
print(f"Clinical Matrix Shape (for PCA): {X_scaled_df.shape}")
print(f"Metadata Matrix Shape (for grouping): {df_meta.shape}")
print("="*60)

# PCA

# fit on clinical variables
pca_maternal = PCA(random_state=42)
X_pca_maternal = pca_maternal.fit_transform(X_scaled_df)

# calculate explained variance
explained_variance_ratio = pca_maternal.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# print("="*60)
# print("PCA EXPLAINED VARIANCE RATIOS")
# print("="*60)
# for idx, ratio in enumerate(explained_variance_ratio):
#     print(f"PC{idx+1}: {ratio*100:.2f}% of variance explained (Cumulative: {cumulative_variance[idx]*100:.2f}%)")

# 3. Extract the loadings (weights) for the first two components
loadings_df = pd.DataFrame(
    pca_maternal.components_.T,
    columns=[f'PC{i+1}' for i in range(len(clinical_cols))],
    index=clinical_cols
)

# print("\n" + "="*60)
# print("CLINICAL COMPONENT LOADINGS (WEIGHTS)")
# print("="*60)
# # Display PC1 and PC2 side-by-side sorted by PC1 influence
# print(loadings_df[['PC1', 'PC2']].sort_values(by='PC1', ascending=False))

# Cumulative Variance DataFrame for Report
pcs = [f"PC{i+1}" for i in list(range(len(explained_variance_ratio)))]
cum_var_df = pd.DataFrame({'Principal Component': pcs,
                           'Explained Variance Ratio': explained_variance_ratio,
                           'Cumulative Variance': cumulative_variance})

cum_var_df['Explained Variance Ratio'] = cum_var_df['Explained Variance Ratio'].apply('{:.2%}'.format)
cum_var_df['Cumulative Variance'] = cum_var_df['Cumulative Variance'].apply('{:.2%}'.format)
cum_var_df.to_csv(os.path.join(pca_res, "birth_cum_var.csv"), index=False)

# PLOT 3
# Cumulative Variance Plot

# calculate individual and cumulative percentages
exp_var_pct = explained_variance_ratio * 100
cum_var_pct = cumulative_variance * 100
components = range(1, len(exp_var_pct)+1)

# plot
plt.figure(figsize=(10,5))

# individual variance
bars = plt.bar(components,
        exp_var_pct,
        label="Individual Explained Variance")

# cumulative variance
plt.plot(components,
         cum_var_pct,
         marker='o',
         linestyle='-',
         color='darkorange',
         linewidth=2,
         label="Cumulative Explained Variance")

# data labels for bars
for bar in bars:
    height = bar.get_height()
    x_pos = bar.get_x() + bar.get_width()/2.0
    y_pos = 1.5
    label_text = f"{height:.1f}%"
    plt.text(x_pos,
             y_pos,
             label_text,
             ha='center',
             va='bottom',
             color='white',
             fontweight='bold',
             fontsize=10)

# styling
plt.title("PCA Scree Plot: Variance Analysis of Clinical Components", fontweight='bold', pad=15)
plt.xlabel("Principal Components", labelpad=10)
plt.ylabel("Percentage of Variance Explained (%)", labelpad=10)
plt.xticks(components)
plt.grid(axis='y', alpha=0.3)
plt.legend(loc='center right')
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(pca_res, "birth_scree_plot.png"), dpi=300)

# Get Loading DF ready for report
feature_mapping = {
    'rf_gdb': "Gestational Diabetes",
    'rf_ghyp': "Gestational Hypertension",
    'rf_ehyp': "Hypertension Eclampsia",
    'mm_trans': "Maternal Transfusion",
    'mm_plac': "Perineal Laceration",
    'mm_rupt': "Ruptured Uterus",
    'mm_uhyst': "Unplanned Hysterectomy",
    'mm_icu': "ICU Admission"
}

tmp = loadings_df[['PC1', 'PC2']].copy()
tmp.reset_index(inplace=True)
tmp['index'] = tmp['index'].map(feature_mapping)
tmp = tmp.rename(columns={'index': 'Risk Factor / Complication'})
tmp = tmp.sort_values(by="PC1", ascending=False)
tmp.to_csv(os.path.join(pca_res, "birth_pc1_pc2.csv"), index=False)

# Create a final analysis DataFrame
df_analysis = df_meta.copy()

# get both components of interest
df_analysis['PC1_acute_trauma'] = X_pca_maternal[:, 0]
df_analysis['PC2_cardio_metabolic'] = X_pca_maternal[:, 1]

# calc overall averages to get top 2, bottom 1
pc1_means = df_analysis.groupby('race_eth')['PC1_acute_trauma'].mean().sort_values(ascending=False)
pc1_top2 = pc1_means.index[:2].tolist()
pc1_bottom1 = pc1_means.index[-1:].tolist()

# calc overall averages to get top 2, bottom 1
pc2_means = df_analysis.groupby('race_eth')['PC2_cardio_metabolic'].mean().sort_values(ascending=False)
pc2_top2 = pc2_means.index[:2].tolist()
pc2_bottom1 = pc2_means.index[-1:].tolist()

# PLOT 4
df_cons_pc1 = df_analysis.copy()
df_cons_pc2 = df_analysis.copy()

# dynamic labelling to get top 2, bottom 1 defined above
df_cons_pc1['group_split'] = df_cons_pc1['race_eth'].apply(
    lambda x: x if (x in pc1_top2 or x in pc1_bottom1) else 'Other'
)

df_cons_pc2['group_split'] = df_cons_pc2['race_eth'].apply(
    lambda x: x if (x in pc2_top2 or x in pc2_bottom1) else 'Other'
)

# mean trends by maternal age
df_grouped_pc1 = df_cons_pc1.groupby(['group_split', 'mage'])['PC1_acute_trauma'].mean().reset_index()
df_grouped_pc2 = df_cons_pc2.groupby(['group_split', 'mage'])['PC2_cardio_metabolic'].mean().reset_index()

# plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

global_color_palette = {
    # PC1 Spotlight Group Colors
    'AIAN': '#d62728',       # Crimson Red (PC1 Top 1)
    'NHOPI': "#f8a964",      # Bright Orange (PC1 Top 2)
    
    # PC2 Spotlight Group Colors
    'AIAN': '#d62728',   # Crimson Red (PC2 Top 1)
    'Asian': "#f7b1e2",      # Vibrant Orchid/Pink (PC2 Top 2)
    
    # Bottom Group Color (shared or distinct)
    'Hispanic': '#1f77b4',   # Cool Steel Blue (Low-Risk Baseline)
    
    # Shared Neutral Baseline
    'Other': "#b0b0b0"       # Neutral Slate Gray
}

# ------------------------------------------------------------------------------
# PANEL 1: PC1 Consolidated (Acute Systemic Trauma)
# ------------------------------------------------------------------------------
sns.lineplot(
    data=df_grouped_pc1,
    x='mage',
    y='PC1_acute_trauma',
    hue='group_split',
    palette=global_color_palette,
    hue_order=[pc1_top2[0], pc1_top2[1], pc1_bottom1[0], 'Other'],
    ax=axes[0],
    linewidth=2.5,
    alpha=0.9
)
axes[0].set_title(f"Panel A: PC1 (Acute Trauma Index)\nSpotlight: {pc1_top2[0]} & {pc1_top2[1]} vs. National Averages", fontsize=11, fontweight='bold', pad=10)
axes[0].set_xlabel("Maternal Age (Years)", fontweight='bold')
axes[0].set_ylabel("Mean PC1 Score (Standardized)", fontweight='bold')
axes[0].legend(title="Race / Ethnicity", loc='upper left', frameon=True)

# ------------------------------------------------------------------------------
# PANEL 2: PC2 Consolidated (Cardio-Metabolic Profile)
# ------------------------------------------------------------------------------
sns.lineplot(
    data=df_grouped_pc2,
    x='mage',
    y='PC2_cardio_metabolic',
    hue='group_split',
    palette=global_color_palette,
    hue_order=[pc2_top2[0], pc2_top2[1], pc2_bottom1[0], 'Other'],
    ax=axes[1],
    linewidth=2.5,
    alpha=0.9
)
axes[1].set_title(f"Panel B: PC2 (Gestational Cardio-Metabolic Profile)\nSpotlight: {pc2_top2[0]} & {pc2_top2[1]} vs. National Averages", fontsize=11, fontweight='bold', pad=10)
axes[1].set_xlabel("Maternal Age (Years)", fontweight='bold')
axes[1].set_ylabel("Mean PC2 Score (Standardized)", fontweight='bold')
axes[1].legend(title="Race / Ethnicity", loc='upper left', frameon=True)

plt.suptitle("U.S. Maternal Risk Profiles (CDC Birth Records 2021-2022)\nConsolidated Extremes vs. National Baselines (Distinct Panel Colors)",
             fontsize=15,
             fontweight='bold',
             y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(pca_res, "birth_pca_race_plots.png"), dpi=300)
