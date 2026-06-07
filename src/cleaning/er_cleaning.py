# Dependencies
import os
import pandas as pd
import numpy as np
import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# Base Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
er_path = os.path.join(BASE_DIR, "data", "raw", "CDC-ER")

# Load files, metadata
df18, metadata18 = pyreadstat.read_dta(os.path.join(er_path, "ED2018-stata.dta"))
df19, metadata19 = pyreadstat.read_dta(os.path.join(er_path, "ED2019-stata.dta"))
df20, metadata20 = pyreadstat.read_dta(os.path.join(er_path, "ed2020-stata.dta"))
df21, metadata21 = pyreadstat.read_dta(os.path.join(er_path, "ed2021-stata.dta"))
df22, metadata22 = pyreadstat.read_dta(os.path.join(er_path, "ed2022-stata.dta"))
print("Successfully loaded data/metadata from 2018-2022")

# ICD-10-CM Codes & Descriptions
icd_path = os.path.join(er_path, "icd10cm-04012026.txt")
icd_map = {}

with open(icd_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        code = line[:8].strip()
        description = line[8:].strip()

        icd_map[code] = description
print(f"Successfully loaded {len(icd_map)} codes into mapping dictionary.")

# Bundle metadata by year
rfv1_metadata = {
    2018: metadata18.variable_value_labels.get("RFV1"),
    2019: metadata19.variable_value_labels.get("RFV1"),
    2020: metadata20.variable_value_labels.get("RFV1"),
    2021: metadata21.variable_value_labels.get("RFV1"),
    2022: metadata22.variable_value_labels.get("RFV1"),
}

# Keep Relevant Columns
cols_to_keep = ["YEAR", "VMONTH", "VDAYR", "ARRTIME", "WAITTIME", "LOV", 
                "AGE", "SEX", "RACEUN", "RACER", "RACERETH",
                "ARREMS", "AMBTRANSFER", "NOPAY", "PAYTYPER",
                "PAYPRIV", "PAYMCARE", "PAYMCAID", "PAYWKCMP", "PAYSELF", "PAYNOCHG", "PAYOTH",
                "IMMEDR", "RFV1", "RFV2", "RFV3", "RFV4", "RFV5", "DIAG1", "DIAG2", "DIAG3", "DIAG4", "DIAG5",
                'TOTCHRON', 'DIAGSCRN', 'TOTDIAG', "PROC", "TOTPROC", 'NUMDIS',
                'DOA', 'DIEDED', 
                "ADMIT", "LOS", "HDDIAG1", "HDDIAG2", "HDDIAG3", "HDDIAG4", "HDDIAG5", "HDSTAT", 
                "HOSPCODE", "REGION", "MSA"]

df18f = df18[cols_to_keep]
df19f = df19[cols_to_keep]
df20f = df20[cols_to_keep]
df21f = df21[cols_to_keep]
df22f = df22[cols_to_keep]

dfp = pd.concat([df18f, df19f, df20f, df21f, df22f], axis=0)

# Export raw snapshot
dfp.to_csv(os.path.join(er_path, "er_raw.csv"), index=False)

# filter for pregnancy-related ER visits
target_cols = ["RFV1", "RFV2", "RFV3", "RFV4", "RFV5"]
target_vals = [17900, 17901, 17902, 17903, 17910, 27350, 35150, 35200]

target_rows = dfp[target_cols].isin(target_vals).any(axis=1)
df = dfp[target_rows].reset_index(drop=True)

# MAP RFV CODES TO TEXT DESCRIPTION
composite_map = {}
for year, code_dict in rfv1_metadata.items():
    if code_dict: # Make sure the dictionary isn't None
        for code, label in code_dict.items():
            composite_map[(year, code)] = label

# temp zip, map appropriate code text
df['RFV1T'] = list(zip(df['YEAR'], df['RFV1']))
df['RFV1T'] = df['RFV1T'].map(composite_map)

df['RFV2T'] = list(zip(df['YEAR'], df['RFV2']))
df['RFV2T'] = df['RFV2T'].map(composite_map)

df['RFV3T'] = list(zip(df['YEAR'], df['RFV3']))
df['RFV3T'] = df['RFV3T'].map(composite_map)

df['RFV4T'] = list(zip(df['YEAR'], df['RFV4']))
df['RFV4T'] = df['RFV4T'].map(composite_map)

df['RFV5T'] = list(zip(df['YEAR'], df['RFV5']))
df['RFV5T'] = df['RFV5T'].map(composite_map)


df[["YEAR", "RFV1", "RFV2", "RFV3", "RFV4", "RFV5", "RFV1T", "RFV2T", "RFV3T", "RFV4T", "RFV5T"]].head(10)

# MAP DIAG CODES TO TEXT DESCRIPTION
# Custom Function
def icd_match(code, map_dict):

    # Clean input string
    code = str(code).strip().upper()

    # Handling missing values
    if not code or code.lower() in ['nan', 'none', '.'] or code=="-9":
        return "Blank"

    if code == "-7":
        return "Not admitted"
    
    # Try to find direct match
    if code in map_dict:
        return map_dict[code]
    
    # Clean formatting elements (trailing dash)
    clean_code = code.rstrip('-')

    # Find closest parent prefix
    while len(clean_code) > 0:
        if clean_code in map_dict:
            return map_dict[clean_code]
        
        matching_keys = [k for k in map_dict.keys() if k.startswith(clean_code)]
        if matching_keys:
            return map_dict[sorted(matching_keys)[0]]
        
        # Remove last character and try again
        clean_code = clean_code[:-1]
    
    return "Unknown Code"

target_cols = ["DIAG1", "DIAG2", "DIAG3", "DIAG4", "DIAG5"]

for col in target_cols:
    txt_col = f"{col}T"

    df[txt_col] = df[col].apply(lambda x: icd_match(x, icd_map))

# MAP HOSPITAL DIAGNOSTIC CODES TO TEXT DESCRIPTION
target_cols = ["HDDIAG1", "HDDIAG2", "HDDIAG3", "HDDIAG4", "HDDIAG5"]

for col in target_cols:
    txt_col = f"{col}T"

    df[txt_col] = df[col].apply(lambda x: icd_match(x, icd_map))

# Export clean data
output_dir = os.path.abspath(os.path.join(BASE_DIR, "data", "clean", "CDC-ER"))
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

df.to_csv(os.path.join(output_dir, "er.csv"), index=False)

########################### VISUALS ###########################

# Path
viz_path = os.path.abspath(os.path.join(BASE_DIR, "resources", "visuals_eda"))
if not os.path.exists(viz_path):
    os.makedirs(viz_path, exist_ok=True)

# Top Reasons for Visit
fig, ax = plt.subplots(figsize=(8, 4))
# Filter out empty codes and plot top 5 reasons
top_rfv = df[df['RFV1T'] != 'Blank']['RFV1T'].value_counts().head(5)
sns.barplot(x=top_rfv.values, y=top_rfv.index, ax=ax, palette="flare", hue=top_rfv.index, legend=False)
ax.set_xlabel("Number of Visits")
ax.set_ylabel("Primary Reason\n for Visit", rotation=0, labelpad=10, va='center', ha='right')

# Gridlines
ax.grid(axis='x', color='lightgray', linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Hide default y-axis ticks/labels
ax.set_yticks([])

# Remove borders (spines)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)

# Add offset to align labels
x_offset = ax.get_xlim()[1] * 0.02 

# Add text inside bars
for i, (label, value) in enumerate(zip(top_rfv.index, top_rfv.values)):
    # Wrap text automatically
    wrapped_label = "\n".join(textwrap.wrap(label, width=30))
    
    # Place text inside the bar with white color and vertical centering
    ax.text(x_offset, i, wrapped_label, 
            va='center', 
            ha='left', 
            color='white', 
            fontsize=8.5, 
            fontweight='bold')

fig.tight_layout()

# Save
fig.savefig(os.path.join(viz_path, "er_v1.png"))

# Top Diagnoses
fig, ax = plt.subplots(figsize=(8, 4))
# Filter out empty codes and plot top 5 reasons
top_diag = df[df['DIAG1T'] != 'Blank']['DIAG1T'].value_counts().head(5)
sns.barplot(x=top_diag.values, y=top_diag.index, ax=ax, palette="flare", hue=top_diag.index, legend=False)
ax.set_xlabel("Number of Visits")
ax.set_ylabel("Primary\nDiagnosis", rotation=0, labelpad=10, va='center', ha='right')

# Gridlines
ax.grid(axis='x', color='lightgray', linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Hide default y-axis ticks/labels
ax.set_yticks([])

# Remove borders (spines)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)

# Add offset to align labels
x_offset = ax.get_xlim()[1] * 0.02 

# Add text inside bars
for i, (label, value) in enumerate(zip(top_diag.index, top_diag.values)):
    # Wrap text automatically
    wrapped_label = "\n".join(textwrap.wrap(label, width=30))
    
    # Place text inside the bar with white color and vertical centering
    ax.text(x_offset, i, wrapped_label, 
            va='center', 
            ha='left', 
            color='white', 
            fontsize=8, 
            fontweight='bold')

fig.tight_layout()

# Save
fig.savefig(os.path.join(viz_path, "er_v2.png"))