########## Preprocessing Health Rankings Data for Clustering and PCA ##########

# Load dependencies
import os
import pandas as pd
import numpy as np

# Base Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Load data
health_data = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "HealthRankings", "health.csv"))

## Additional Cleaning Steps

# Remove Washington DC and aggregated (nationwide) entries
df_health = health_data.loc[~health_data.State.isin(["DC", "ALL"])].reset_index(drop=True)

measure_to_drop = ['Low-Risk Cesarean Delivery']
df_health = df_health.loc[~df_health.Measure.isin(measure_to_drop)].reset_index(drop=True)

# Add missing data based on outside research
missing = {
    'Maternal Mortality': {
        'DE': 18.9
    },
    'Unintended Pregnancy': {
        'CA': 31.2,
        'SC': 37.0,
        'TX': 53.0,
        'NV': 52.0
    },
    'Postpartum Depression': {
        'CA': 13.0
    },
    'Postpartum Visit': {
        'CA': 87.5
    }
}

print(f"# Missing Values Before Update: {len(df_health.loc[df_health.Value==9999])}")

for k in list(missing.keys()):
    states = list(missing.get(k).keys())
    for s in states:
        df_health.loc[(df_health.Measure==k)&(df_health.Value==9999)&(df_health.State==s), 'Value'] = missing.get(k).get(s)

print(f"# Missing Values After Update: {len(df_health.loc[df_health.Value==9999])}")

# FOR NOW, DROP ALL RECORDS FROM STATES WITH ANY MISSING DATA
print(f"Dropping states with missing values: {list(df_health.loc[df_health.Value==9999].State.unique())}")
df_health = df_health.loc[~df_health.State.isin(list(df_health.loc[df_health.Value==9999].State.unique()))]

# Pivot to wide format
df = df_health.pivot_table(
    index=['State', 'Year'],
    columns='Measure',
    values='Value',
    aggfunc='mean'
).reset_index()

# Restructure Column Logic/Values so that higher numbers = bad, lower numbers = good
better_when_high_cols = {
    "Adequate Prenatal Care": "Inadequate Prenatal Care",       # percentage
    "Cervical Cancer Screening": "No Cancer Screen",            # percentage
    "Gender Pay Gap": "Gender Pay Gap",                         # percentage
    "Maternity Practices Score": "Maternity Practices Score",   # score
    "Postpartum Visit": "No Postpartum Visit",                  # percentage
    "Voter Participation (Average)": "Voter Abstainment",       # percentage
    "WIC Coverage": "WIC Shortfall",                            # percentage
    "Well-Woman Visit": "No Preventative",                      # percentage
    "Women's Health Providers": "Patients Per Doctor"           # number per 100,000
}

percent_cols = ['Adequate Prenatal Care', 'Cervical Cancer Screening', 'Gender Pay Gap',
                'Postpartum Visit', 'Voter Participation (Average)', 'WIC Coverage', 'Well-Woman Visit']
for col in percent_cols:
    df[col] = 100 - df[col]

df['Maternity Practices Score'] = -1 * df['Maternity Practices Score']
df["Women's Health Providers"] = 100000/df["Women's Health Providers"]

df = df.rename(columns=better_when_high_cols)

# Export cleaned data
df.to_csv(os.path.join(BASE_DIR, "data", "clean", "HealthRankings", "health_preprocessed.csv"), index=False)