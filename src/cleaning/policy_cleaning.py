import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
policy_data_path = os.path.join(BASE_DIR, "data", "raw", "LawAtlas")

# Statutory and Constitutional Abortion Protections
df1 = pd.read_excel(os.path.join(policy_data_path, "AbortionRights.xlsx"),
                   sheet_name="Statistical Data")

df1 = df1.rename(columns={'Jurisdictions': 'State',
                        'Effective Date': 'Start',
                        'Valid Through Date': 'End',
                        'StatConst_AbortionProtections_Law': 'LegalProtection',
                        'StatConst_StateConstitution ': 'ConstProtection',
                        'StatConst_ViabilityExceptions_Protect life ': 'LifeException',
                        'StatConst_ViabilityExceptions_Protect health': 'HealthException',
                        'StatConst_ViabilityExceptions_Fetal anomaly': 'FetalAnomalyException'})

df1 = df1[['State', 'Start', 'End', 'LegalProtection', 'ConstProtection',
         'LifeException', 'HealthException', 'FetalAnomalyException']]

for col in ['LegalProtection', 'ConstProtection', 'LifeException', 'HealthException', 'FetalAnomalyException']:
    df1[col] = np.where(df1[col] == 1, 1, 0)

# Post-Dobbs Abortion Restrictions
df2 = pd.read_excel(os.path.join(policy_data_path, "PostDobbs.xlsx"),
                   sheet_name="Statistical Data")

# Export raw snapshot
df2.to_csv(os.path.join(policy_data_path, "policy_raw.csv"), index=False)

df2 = df2.rename(columns={
    'Jurisdiction': 'State',
    'Effective Date': 'Start',
    'Valid Through Date': 'End',
    'legalstat': 'LegalRestriction',
    'protection': 'LegalProtection',
    'abortion-restrictions_Total ban': 'TotalBan',
    'abortion-restrictionsBanned with limited exceptions': 'BanException',
    'abortion-restrictions_6 week (LMP) gestational limit': 'GestLimit6',
    'abortion-restrictions_7-14 week (LMP) gestational limit': 'GestLimit7_14',
    'abortion-restrictions_15-20 week (LMP) gestational limit': 'GestLimit15_20',
    'abortion-restrictions_Fetal heartbeat ban': 'HeartbeatBan',
    'medication-abortion': 'MedBan',
    'telehealth': 'TeleBan',
    'penaltiesCriminal': 'CriminalPenalty',
    'penaltiesCivil': 'CivilPenalty',
    'penaltiesLicensing': 'LicensePenalty',
    'who-penaltyMedical providers': 'ProviderPenalty',
    'who-penalty_Pregnant person': 'PatientPenalty'
})

df2 = df2[['State', 'Start', 'End',
         'LegalRestriction', 'LegalProtection','TotalBan', 'BanException',
         'GestLimit6', 'GestLimit7_14', 'GestLimit15_20', 'HeartbeatBan',
         'MedBan', 'TeleBan', 
         'CriminalPenalty', 'CivilPenalty', 'LicensePenalty',
         'ProviderPenalty', 'PatientPenalty']]

for col in ['LegalRestriction', 'LegalProtection','TotalBan', 'BanException',
       'GestLimit6', 'GestLimit7_14', 'GestLimit15_20', 'HeartbeatBan',
       'MedBan', 'TeleBan', 'CriminalPenalty', 'CivilPenalty',
       'LicensePenalty', 'ProviderPenalty', 'PatientPenalty']:
    df2[col] = np.where(df2[col] == 1, 1, 0)

# Public Funding Restrictions
df3 = pd.read_excel(os.path.join(policy_data_path, "PublicFundingRestrictions.xlsx"),
                   sheet_name="Statistical Data")

df3 = df3.rename(columns={
    'Effective Date': 'Start',
    'Valid Through Date': 'End',
    'RestrPublic_Restrict': 'FundRestriction',
    'RestrPublic_FundsExcept_Life endangerment': 'FundLifeException',
    'RestrPublic_FundsExcept_Serious health risk': 'FundHealthException1',
    'RestrPublic_FundsExcept_Health': 'FundHealthException2',
    'RestrPublic_FundsExcept_Fetal anomaly': 'FundFetalAnomalyException'
})

cols_to_keep = ['State', 'Start', 'End', 'FundRestriction',
         'FundLifeException', 'FundHealthException1', 'FundHealthException2',
         'FundFetalAnomalyException']

df3 = df3[cols_to_keep]

for col in ['FundRestriction', 'FundLifeException', 'FundHealthException1', 'FundHealthException2', 'FundFetalAnomalyException']:
    df3[col] = np.where(df3[col] == 1, 1, 0)

df3['FundHealthException'] = np.where((df3['FundHealthException1'] == 1) | (df3['FundHealthException2'] == 1), 1, 0)

df3 = df3.drop(columns=['FundHealthException1', 'FundHealthException2'])


# Combine DataFrames with Cross-Dataset Policy Forward-Filling
# standardize date formats
for d in [df1, df2, df3]:
    d['Start'] = pd.to_datetime(d.Start)
    d['End'] = pd.to_datetime(d.End)

# get all timeline breakpoints
all_dates = pd.concat([
    df1[['State', 'Start']].rename(columns={'Start': 'Date'}),
    df1[['State', 'End']].rename(columns={'End': 'Date'}),
    df2[['State', 'Start']].rename(columns={'Start': 'Date'}),
    df2[['State', 'End']].rename(columns={'End': 'Date'}),
    df3[['State', 'Start']].rename(columns={'Start': 'Date'}),
    df3[['State', 'End']].rename(columns={'End': 'Date'})
]).drop_duplicates().sort_values(['State', 'Date'])

all_dates['Next_Date'] = all_dates.groupby('State')['Date'].shift(-1)
df = all_dates.dropna(subset=['Next_Date']).copy()
df = df.rename(columns={'Date': 'Start', 'Next_Date': 'End'})

# unique policy features
all_policy_vars = set()
for d in [df1, df2, df3]:
    vars_in_df = [c for c in d.columns if c not in ['State', 'Start', 'End']]
    all_policy_vars.update(vars_in_df)
all_policy_vars = list(all_policy_vars)

# Initialize all unique tracking columns with NaN (to distinguish "no data" from an explicit 0)
for var in all_policy_vars:
    df[var] = np.nan

# add data to skeleton
for target_df in [df1, df2, df3]:
    policy_vars = [c for c in target_df.columns if c not in ['State', 'Start', 'End']]
    
    for state, group in target_df.groupby('State'):
        for _, row in group.iterrows():
            # Find the master skeleton intervals that intersect this row's active window
            overlap_mask = (
                (df['State'] == state) & 
                (df['Start'] < row['End']) & 
                (df['End'] > row['Start'])
            )
            
            # Layer the values into the mask (ignoring completely empty rows)
            if overlap_mask.any():
                df.loc[overlap_mask, policy_vars] = row[policy_vars].values

# sort chronologically and forward-fill values by State boundary
df = df.sort_values(['State', 'Start']).reset_index(drop=True)
df[all_policy_vars] = df.groupby('State')[all_policy_vars].ffill()

# handle NAs
df[all_policy_vars] = df[all_policy_vars].fillna(0).astype(int)

# Export clean data
output_dir = os.path.abspath(os.path.join(BASE_DIR, "data", "clean", "LawAtlas"))
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

df.to_csv(os.path.join(output_dir, "policy.csv"), index=False)