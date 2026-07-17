# Dependencies
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyreadstat
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Root Directory for File Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
er_raw_path = os.path.join(BASE_DIR, "data", "raw", "CDC-ER")
er_clean_path = os.path.join(BASE_DIR, "data", "clean", "CDC-ER")
nb_res = os.path.join(BASE_DIR, "resources", "nbayes")

# Load file and metadata
df18, metadata18 = pyreadstat.read_dta(os.path.join(er_raw_path, "ED2018-stata.dta"), apply_value_formats=True)
df19, metadata19 = pyreadstat.read_dta(os.path.join(er_raw_path, "ED2019-stata.dta"), apply_value_formats=True)
df20, metadata20 = pyreadstat.read_dta(os.path.join(er_raw_path, "ed2020-stata.dta"), apply_value_formats=True)
df21, metadata21 = pyreadstat.read_dta(os.path.join(er_raw_path, "ed2021-stata.dta"), apply_value_formats=True)
df22, metadata22 = pyreadstat.read_dta(os.path.join(er_raw_path, "ed2022-stata.dta"), apply_value_formats=True)
print("Successfully loaded data/metadata from 2018-2022")

# ICD-10-CM Codes & Descriptions
icd_path = os.path.join(er_raw_path, "icd10cm-04012026.txt")
icd_map = {}

with open(icd_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        code = line[:8].strip()
        description = line[8:].strip()

        icd_map[code] = description
print(f"Successfully loaded {len(icd_map)} codes into mapping dictionary.")

# Bundle metadata (diagnoses) by year
diag_metadata = {
    2018: metadata18.variable_value_labels.get("DIAG1"),
    2019: metadata19.variable_value_labels.get("DIAG1"),
    2020: metadata20.variable_value_labels.get("DIAG1"),
    2021: metadata21.variable_value_labels.get("DIAG1"),
    2022: metadata22.variable_value_labels.get("DIAG1"),
}

# Filter Data
cols_to_keep = ['SEX', 'AGE', 'RACER', 'RACERETH', 'REGION',
                'DIAG1', 'DIAG2', 'DIAG3', 'DIAG4', 'DIAG5',
                ]

df18f = df18[cols_to_keep]
df19f = df19[cols_to_keep]
df20f = df20[cols_to_keep]
df21f = df21[cols_to_keep]
df22f = df22[cols_to_keep]

data = pd.concat([df18f, df19f, df20f, df21f, df22f], axis=0)
print(data.shape)
print(f"Missing Values:\n{data.isna().sum()}")

# Save sample for 'raw' snapshot on report
data.head(100).to_csv(os.path.join(nb_res, "er_data_raw.csv"), index=False)

# ==============================================
# CLEANING
# ==============================================
df = data.copy()

def clean_age_column(val):
    val_str = str(val).strip().lower()
    
    # handle non numeric strings
    if "under one year" in val_str or "less than" in val_str:
        return 0
    elif "90 years and older" in val_str or "90 and older" in val_str:
        return 90
    
    # get numeric ages
    try:
        # Remove any non-numeric text left over (like " years")
        numeric_part = ''.join(filter(str.isdigit, val_str))
        return int(numeric_part) if numeric_part else np.nan
    except ValueError:
        return np.nan
df['AGE'] = df.AGE.apply(clean_age_column)
df = df.dropna(subset=['AGE']).reset_index(drop=True)
df['AGE'] = df['AGE'].astype(int)

# Filter for sex=female, childbearing age (15 - 49)
df = df.loc[(df.SEX=='Female')&(df.AGE>=15)&(df.AGE<=49)].copy().reset_index(drop=True)
print(f"Filtered cohort (Females, Ages 15-49): {df.shape[0]:,}")

# Bin Ages
age_bins = [14, 19, 34, 49]
age_labels = ['Teens (15-19)', 'Average Maternal Age (20-34)', 'Advanced Maternal Age (35-49)']
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=age_bins, labels=age_labels)

# Map and filter regions
region_mapping = {
        'Northeast': 'Highly Protected (Northeast)',
        'South': 'Highly Restricted (South)'
    }
df = df.loc[df.REGION.isin(['Northeast', 'South'])].reset_index(drop=True)
df['MACRO_REGION'] = df.REGION.map(region_mapping)
print(f"Retained records in Northeast/South: {df.shape[0]:,}")

df = df.rename(columns={'RACERETH': 'RACE_ETHNICITY'})

# Flag severe pregnancy related conditions
diag_cols = [col for col in df.columns if col.startswith('DIAG')]
for col in diag_cols:
    df[col] = df[col].astype(str).str.strip().str.upper()
# ICD-10 codes for severe maternal morbidity (smm)
smm_code_map = {
    'O00': 'Ectopic pregnancy',
    'O11': 'Pre-existing hypertension with pre-eclampsia',
    'O14': 'Pre-eclampsia',
    'O15': 'Eclampsia',
    'O20': 'Hemorrhage in early pregnancy',
    'O44': 'Placenta previa',
    'O45': 'Premature separation of placenta (abruptio placentae)',
    'O46': 'Antepartum hemorrhage, not elsewhere classified',
    'O67': 'Labor and delivery complicated by intrapartum hemorrhage, not elsewhere classified',
    'O71': 'Other obstetric trauma',
    'O72': 'Portpartum hemorrhage',
    'O85': 'Puerperal sepsis',
    'O88': 'Obstetric embolism'
}
smm_codes = list(smm_code_map.keys())
df['SMM_FLAG'] = 0
for col in diag_cols:
    matches_smm = df[col].str.startswith(tuple(smm_codes))
    df.loc[matches_smm, 'SMM_FLAG'] = 1

# Isolate features for modeling
features_to_keep = ['AGE_GROUP', 'RACE_ETHNICITY', 'MACRO_REGION', 'SMM_FLAG']
df_prep = df[features_to_keep].dropna()

for col in ['AGE_GROUP', 'RACE_ETHNICITY', 'MACRO_REGION']:
    df_prep[col] = df_prep[col].astype(str)

print(f"Final dataset dimensions: {df_prep.shape[0]} patients, {df_prep.shape[1]} columns")
print(f"Target distribution (Severe Maternal Morbidity): \n{df_prep['SMM_FLAG'].value_counts(normalize=True) * 100}")

# ==============================================
# ENCODING
# ==============================================
df_model = df_prep.copy()

# explicit category mapping
age_map = {
    'Teens (15-19)': 0,
    'Average Maternal Age (20-34)': 1,
    'Advanced Maternal Age (35-49)': 2
}

race_map = {
    'Non-Hispanic White': 0,
    'Non-Hispanic Black': 1,
    'Hispanic': 2,
    'Non-Hispanic Other': 3
}

region_map = {
    'Highly Protected (Northeast)': 0,
    'Highly Restricted (South)': 1
}

# apply mappings
df_model['AGE_CODE'] = df_model['AGE_GROUP'].map(age_map)
df_model['RACE_CODE'] = df_model['RACE_ETHNICITY'].map(race_map)
df_model['REGION_CODE'] = df_model['MACRO_REGION'].map(region_map)

# save snapshot for report
df_model[['AGE_CODE', 'RACE_CODE', 'REGION_CODE']].head(100).to_csv(os.path.join(nb_res, 'er_data_clean.csv'), index=False)

# ==============================================
# TRAIN/TEST SPLIT
# ==============================================
# features and target
X = df_model[['AGE_CODE', 'RACE_CODE', 'REGION_CODE']].values
y = df_model['SMM_FLAG'].values

# Train/Test Split
## stratify by y to maintain same class imbalance within training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==============================================
# MODELING
# ==============================================
# Fit CategoricalNB model
nb_model = CategoricalNB(alpha=1.0)
nb_model.fit(X_train, y_train)

# Prior probabilities
priors = np.exp(nb_model.class_log_prior_)
print("--- PRIOR PROBABILITIES ---")
print(f"P(No SMM) = {priors[0]:.4f}")
print(f"P(SMM)    = {priors[1]:.4f}\n")

# REGIONAL CONDITIONAL RISK SCENARIOS

scenarios = []

# loop thru parameters to calculate for each combo of features
for age_str, age_val in age_map.items():
    for race_str, race_val in race_map.items():
        
        # Profile 1: Northeast (Region Code 0)
        profile_ne = np.array([[age_val, race_val, 0]])
        prob_ne = nb_model.predict_proba(profile_ne)[0][1]
        
        # Profile 2: South (Region Code 1)
        profile_south = np.array([[age_val, race_val, 1]])
        prob_south = nb_model.predict_proba(profile_south)[0][1]
        
        relative_risk = prob_south / prob_ne
        
        scenarios.append({
            'Age Group': age_str,
            'Race/Ethnicity': race_str,
            'P(SMM | Northeast)': f"{prob_ne * 100:.2f}%",
            'P(SMM | South)': f"{prob_south * 100:.2f}%",
            'Relative Risk (South vs NE)': f"{relative_risk:.2f}x"
        })

# Create dataframe and display
df_scenarios = pd.DataFrame(scenarios)

# Sort by highest SMM risk in the South to see who is most vulnerable
df_scenarios = df_scenarios.sort_values(by='P(SMM | South)', ascending=False)
# print("--- PROBABILISTIC REGIONAL RISK PROFILES ---")
# print(df_scenarios.to_string(index=False))
df_scenarios.to_csv(os.path.join(nb_res, 'er_scenarios_prob.csv'), index=False)

# ==============================================
# PLOTTING
# ==============================================

# COLOR PALETTE
COLOR_NE = '#31688e'     # Blue tone for Protected Northeast
COLOR_SOUTH = '#fde725'  # Yellow/Gold tone for Restricted South
custom_palette = {'Northeast': COLOR_NE, 'South': COLOR_SOUTH}

# Set global plotting style
sns.set_theme(style="whitegrid")

# Copying df_scenarios to be safe
df_plot = df_scenarios.copy()

# strings to floats for plot labels
df_plot['P(SMM | Northeast)'] = df_plot['P(SMM | Northeast)'].str.rstrip('%').astype(float)
df_plot['P(SMM | South)'] = df_plot['P(SMM | South)'].str.rstrip('%').astype(float)

# rename columns for readability
df_plot = df_plot.rename(columns={
    'P(SMM | Northeast)': 'Northeast',
    'P(SMM | South)': 'South',
    'Age Group': 'Age_Group',
    'Race/Ethnicity': 'Race_Ethnicity'
})

# melt to long format
df_melted = pd.melt(
    df_plot, 
    id_vars=['Age_Group', 'Race_Ethnicity'], 
    value_vars=['Northeast', 'South'],
    var_name='Region', 
    value_name='SMM_Probability'
)

# individual charts by age
age_groups = df_melted['Age_Group'].unique()

for age in age_groups:
    # Filter dataset for age 
    df_age_filtered = df_melted[df_melted['Age_Group'] == age]
    
    plt.figure(figsize=(7, 5))
    
    # barplot
    ax = sns.barplot(
        data=df_age_filtered,
        x='Race_Ethnicity',
        y='SMM_Probability',
        hue='Region',
        palette=custom_palette,
        edgecolor='black',
        alpha=0.9
    )
    
    # data labs
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', padding=3, fontsize=11, fontweight='bold')
    
    # styling
    plt.title(f"Conditional SMM Risk Profile by Region\nCohort: {age}", fontsize=13, fontweight='bold', pad=15,
              loc='left')
    plt.xlabel("Patient Race / Ethnicity", fontweight='bold', labelpad=10)
    plt.ylabel("P(SMM | Profile)", fontweight='bold', labelpad=10)
    
    # format yaxis ticks (%)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
    
    # expand y to not cut off content
    plt.ylim(0, df_melted['SMM_Probability'].max() + 0.8)
    
    plt.legend(title="Region", loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    
    # save for report
    file_safe_name = age.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    file_safe_name_final = f"smm_risk_{file_safe_name}"
    plt.savefig(os.path.join(nb_res, file_safe_name_final), dpi=300)


# PLOT 2 - RACIAL DISPARITY
DISPARITY_PALETTE = {
    'Non-Hispanic Other': '#440154',
    'Hispanic': "#C600F8",
    'Non-Hispanic Black': "#E288F9",
    'Non-Hispanic White': "#EEC9F8"
}

# average baselines
df_race_disparity = df_plot.groupby('Race_Ethnicity')[['Northeast', 'South']].mean().reset_index()
df_race_disparity['Overall_SMM_Probability'] = df_race_disparity[['Northeast', 'South']].mean(axis=1)
df_race_disparity = df_race_disparity.sort_values(by='Overall_SMM_Probability', ascending=False)

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    data=df_race_disparity,
    x='Race_Ethnicity',
    y='Overall_SMM_Probability',
    hue='Race_Ethnicity',
    palette=DISPARITY_PALETTE,
    edgecolor='black',
    alpha=1,
    legend=False
)

# data labs
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%', padding=3, fontsize=10, fontweight='bold')

# styling
plt.title("Systemic Racial Disparities in Emergency Department SMM Prevalence\nOverall Probabilistic Baseline Across Combined Contexts",
          fontsize=13,
          fontweight='bold',
          pad=15,
          loc='left')
plt.xlabel("Patient Race / Ethnicity", fontweight='bold', labelpad=10)
plt.ylabel("Overall P(SMM) Baseline", fontweight='bold', labelpad=10)

# yaxis as %
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
plt.ylim(0, df_race_disparity['Overall_SMM_Probability'].max() + 0.5)
plt.tight_layout()
plt.savefig(os.path.join(nb_res, "smm_racial_disparities.png"), dpi=300)