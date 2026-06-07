import pandas as pd
import numpy as np
import os
import zipfile_deflate64 as zipfile
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

# Base Dir
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
nchs_path = os.path.join(BASE_DIR, "data", "raw", "NCHS-Birth")
if not os.path.exists:
    os.makedirs(nchs_path, exist_ok=True)

## 2024 DATA
# Get zip file
zip_file_path = os.path.join(nchs_path, 'Nat2024us.zip')
raw_snapshot_path = os.path.join(nchs_path, "births2024_raw.txt")

target_columns = {
    # Basic Characteristics
    "birth_month": (12, 14),
    "hospital_birth": (49, 50),
    "birth_place": (31, 32),
    "maternal_age": (78, 79),
    "maternal_race": (106, 107),

    # Prenatal
    "prenatal_visits": (237, 239),

    # Risk Factors
    "eclampsia": (316, 317),

    # Maternal Morbidity
    "maternal_transfusion": (414, 415),
    "perineal_lac": (415, 416),
    "ruptured_uterus": (416, 417),
    "unplanned_hysterectomy": (417, 418),
    "icu_admit": (418, 419),
    "no_morbidity": (426, 427),

    # Infant Abnormalities
    "no_conditional_abnormality": (530, 531),
    "no_congenital_abnormality": (560, 561),
}

col_names = list(target_columns.keys())
col_specs = list(target_columns.values())
chunks = []

# Decode zip file and get relevant data
with zipfile.ZipFile(zip_file_path, 'r') as z:
    internal_file_name = z.namelist()[0]
    print(f"Reading internal file: {internal_file_name}")
    
    # Save Raw Snapshot
    with z.open(internal_file_name) as f_raw:
        with open(raw_snapshot_path, 'w', encoding='ascii') as snapshot_out:
            for _ in range(100):
                line = f_raw.readline()
                if not line:
                    break
                snapshot_out.write(line.decode('ascii'))

    # Read data chunks into pandas
    with z.open(internal_file_name) as f:
        for chunk in pd.read_fwf(
            f, 
            colspecs=col_specs, 
            names=col_names, 
            chunksize=100000, 
            encoding='ascii'
        ):
            chunks.append(chunk)
            
df_natality = pd.concat(chunks, ignore_index=True)
print(f"Successfully loaded dataframe shape: {df_natality.shape}")

# Data Definitions
hospital_birth_map = {1: "True", 2: "False", 3: "Unknown"}

birth_place_map = {
    1: "Hospital",
    2: "Freestanding Birth Center",
    3: "Home (intended)",
    4: "Home (not intended)",
    5: "Home (unknown if intended)",
    6: "Clinic/Doctor's Office",
    7: "Other",
    9: "Unknown"
}

maternal_age_map = {
    1: "Under 15 years",
    2: "15-19 years",
    3: "20-24 years",
    4: "25-29 years",
    5: "30-34 years",
    6: "35-39 years",
    7: "40-44 years",
    8: "45-49 years",
    9: "50-54 years"
}

maternal_race_map = {
    1: "White",
    2: "Black",
    3: "AIAN",
    4: "Asian",
    5: "NHOPI",
    6: "More than one race"
}

df_natality['birth_place'] = df_natality['birth_place'].map(birth_place_map)
df_natality['hospital_birth'] = df_natality['hospital_birth'].map(hospital_birth_map)
df_natality['maternal_age'] = df_natality['maternal_age'].map(maternal_age_map)
df_natality['maternal_race'] = df_natality['maternal_race'].map(maternal_race_map)

# Convert Y/N to binary
binary_cols = ['eclampsia', 'maternal_transfusion', 'perineal_lac', 'ruptured_uterus', 'unplanned_hysterectomy', 'icu_admit']
for col in binary_cols:
    df_natality[col] = np.where(df_natality[col] == "N", 0, 1)

# Export clean data
output_dir = os.path.abspath(os.path.join(BASE_DIR, "data", "clean", "NCHS-Birth"))
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

target_zip_path = os.path.join(output_dir, "births2024.csv.zip")

df_natality.to_csv(target_zip_path, index=False,
                   compression={
                       'method': 'zip',
                       'archive_name': 'births2024.csv'
                   })

########## visuals ##########

# Path
viz_path = os.path.abspath(os.path.join(BASE_DIR, "resources", "visuals_eda"))
if not os.path.exists(viz_path):
    os.makedirs(viz_path, exist_ok=True)

# Visual 1, maternal age distribution
fig, ax = plt.subplots(figsize=(8, 6))

# reorder
chronological_order = [
    "Under 15 years", "15-19 years", "20-24 years", "25-29 years",
    "30-34 years", "35-39 years", "40-44 years", "45-49 years", "50-54 years"
]
age_dist = df_natality['maternal_age'].value_counts().reindex(chronological_order).dropna()

# clean labels
clean_labels = [label.replace(" years", "") for label in age_dist.index]

# plot
sns.barplot(x=clean_labels, y=age_dist.values, palette="crest", hue=clean_labels, legend=False, ax=ax)

# gridlines
ax.yaxis.grid(True, linestyle='--', color='lightgray', alpha=0.7)
ax.set_axisbelow(True)

# annotations
inside_white_groups = {"15-19", "20-24", "25-29", "35-39", "40-44"}
max_val = age_dist.max()

for bar, label in zip(ax.patches, clean_labels):
    height = bar.get_height()
    if pd.isna(height) or height == 0:
        continue
    
    # format numbers
    text_label = f"{int(height):,}"
    
    if label in inside_white_groups:
        # Data labels inside the bars (bold white font)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height - (max_val * 0.02),  # Centered vertically inside the bar
            text_label,
            ha='center',
            va='center',
            color='white',
            fontweight='bold',
            fontsize=9
        )
    else:
        # Data labels on top of the bars (black font)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max_val * 0.01),  # Positioned slightly above the bar crest
            text_label,
            ha='center',
            va='bottom',
            color='black',
            fontsize=9
        )

ax.set_xlabel("Maternal Age Cohort")
ax.set_ylabel("Live Birth\nVolume", rotation=0, va='center', ha='right')

# horizontal x axis labels
plt.xticks(rotation=0)
fig.tight_layout()

# Save
fig.savefig(os.path.join(viz_path, "birth_v1.png"))

# Visual 2, icu admissions by maternal age
fig, ax = plt.subplots(figsize=(8, 6))

chronological_order = [
    "Under 15 years", "15-19 years", "20-24 years", "25-29 years",
    "30-34 years", "35-39 years", "40-44 years", "45-49 years", "50-54 years"
]

# Calculate admission rate per 1,000 births
icu_rate = df_natality.groupby('maternal_age')['icu_admit'].mean().reset_index()

icu_rate['sort_idx'] = icu_rate['maternal_age'].map({cat: i for i, cat in enumerate(chronological_order)})
icu_rate = icu_rate.sort_values('sort_idx').dropna(subset=['sort_idx']).reset_index(drop=True)

icu_rate['clean_age'] = icu_rate['maternal_age'].str.replace(" years", "")
icu_rate['rate_per_1k'] = icu_rate['icu_admit'] * 1000

# coloring
flare_color = "#d04961"
sns.lineplot(data=icu_rate, x='clean_age', y='rate_per_1k', marker="o", color=flare_color, linewidth=2.5, ax=ax, alpha=0.8)

# gridlines
ax.yaxis.grid(True, linestyle='--', color='lightgray', alpha=0.7)
ax.set_axisbelow(True)

ax.set_ylim(0, 27.5)

# data labels
max_val = icu_rate['rate_per_1k'].max()
for i, row in icu_rate.iterrows():
    ax.text(
        i,                                # x-coordinate matching the category position index
        row['rate_per_1k'] + (max_val * 0.02), # y-coordinate slightly above the marker point
        f"{row['rate_per_1k']:.2f}",      # Format rate with 2 decimal places
        ha='center',
        va='bottom',
        color='black',
        fontsize=9
    )

ax.set_xlabel("Maternal Age Cohort")
ax.set_ylabel("ICU Admissions\nper 1,000 Births", rotation=0, va='center', ha='right')

plt.xticks(rotation=0)
fig.tight_layout()

# Save
fig.savefig(os.path.join(viz_path, "birth_v2.png"))