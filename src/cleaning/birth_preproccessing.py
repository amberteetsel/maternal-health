import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
data = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "NCHS-Birth", "birth_icu.csv"))

# Encoded with Y/N/U
yesno_cols = ["maternal_transfusion", "perineal_lac", "ruptured_uterus", "unplanned_hysterectomy", "icu_admit", 'infant_alive',
                  'diabetes', 'hypertension', 'eclampsia', 'induction', 'augmentation', 'steroids', 'antibiotics',
                  'anesthesia']

cols_to_drop = ['year', 'birth_month', 'hospital_birth', 'birth_place', 'maternal_age_yrs', 'prenatal_start_mo']
df = data.copy().drop(columns=cols_to_drop, axis=1)

# Mappings
maternal_age_map = {
    1: "<15",
    2: "15-19 years",
    3: "20-24 years",
    4: "25-29 years",
    5: "30-34 years",
    6: "35-39 years",
    7: "40-44 years",
    8: "45-49 years",
    9: "50-54 years"
}

df['maternal_age_group'] = df['maternal_age_group'].map(maternal_age_map)
df['maternal_age_group'] = df['maternal_age_group'].str.replace(" years", "")

# Binning prenatal visits
df['prenatal_visits'] = df['prenatal_visits'].replace(99, np.nan)  # 99 indicates unknown
visit_bins = [-1, 0, 4, 9, 14, 100]
visit_labels = ['0_visits', '1-4_visits', '5-9_visits', '10-14_visits', '15+_visits']
df['binned_visits'] = pd.cut(df['prenatal_visits'], bins=visit_bins, labels=visit_labels).astype(str)
df['binned_visits'] = df['binned_visits'].fillna("UNKNOWN")

# Binning cigarettes
for col in ['cigT1', 'cigT2', 'cigT3']:
    df[col] = df[col].replace(99, 0) # unknown or missing
max_cigarettes = df[['cigT1', 'cigT2', 'cigT3']].max(axis=1)
cig_bins = [-1, 0, 5, 10, 100]
cig_labels = ['Nonsmoker', 'Light (1-5)', 'Moderate (6-10)', 'Heavy (11+)']
df['binned_cigs'] = pd.cut(max_cigarettes, bins=cig_bins, labels=cig_labels).astype(str)

# Y/N/U columns (yesno_cols)
for col in yesno_cols:
    df[col] = df[col].replace(['U', 'u', ' ', '', None], "UNKNOWN")

# Handle Unknown Placeholders (9, 99)
for col in ['maternal_race', 'maternal_ed', 'delivery_method', 'pay_source']:
    df[col] = df[col].replace([9], 99).fillna(99).astype(str)

# Keep only feature columns and target column
use_cols = ['icu_admit', 'maternal_age_group', 'maternal_race', 'maternal_ed', 'binned_visits', 'binned_cigs',
                'pay_source', 'delivery_method',
                'diabetes', 'hypertension', 'eclampsia', 'induction', 'augmentation', 'steroids', 'antibiotics', 'anesthesia',
                'maternal_transfusion', 'perineal_lac', 'ruptured_uterus', 'unplanned_hysterectomy']

df = df[use_cols]

# Export clean data, can encode and train/test/split in modeling code
output_dir = os.path.join(BASE_DIR, "data", "clean", "NCHS-Birth")
df.to_csv(os.path.join(output_dir, "birth_icu_processed.csv"), index=False)

