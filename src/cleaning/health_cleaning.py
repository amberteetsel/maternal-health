import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Base Dir
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
target_dir = os.path.join(BASE_DIR, "data", "raw", "HealthRankings", "health.csv")
if not os.path.exists(target_dir):
    os.makedirs(target_dir, exist_ok=True)

# Securely initialize API credentials
load_dotenv()
API_KEY = os.getenv("AHR_API_SUBSCRIPTION_KEY")

if not API_KEY:
    raise ValueError("Error: API key failed to load out of local environment.")

url = 'https://api.americashealthrankings.org/graphql'
headers = {
    'Content-Type': 'application/json',
    'X-Api-Key': API_KEY
}

# Specific metrics of interest from annual HWC Reports
target_measures = [
    "Concentrated Disadvantage", "Food Insecurity",
    "Gender Pay Gap", "Poverty", "Unemployment", "College Graduate", "Infant Child Care Affordability",
    "Voter Participation (Average)", "Adequate Prenatal Care", "Avoided Care Due to Cost",
    "Maternity Care Desert", "Uninsured Women", "Women's Health Providers", "Cervical Cancer Screening",
    "Postpartum Visit", "Well-Woman Visit", "Low-Risk Cesarean Delivery", "Maternity Practices Score",
    "Unintended Pregnancy", "Smoking During Pregnancy", "Postpartum Depression",
    "Maternal Mortality", "Mortality Rate", "Severe Maternal Morbidity", "WIC Coverage",
    "Infant Mortality", "Neonatal Mortality", "Low Birth Weight"
]

# Build query string
query = """
query GetReportDataByMeasures($measureNames: [String!]) {
  measures_A(where: { name: { in: $measureNames } }) {
    name
    source {
      name
    }
    data {
      state
      dateLabel
      value
    }
  }
}
"""

variables = {
    "measureNames": target_measures
}

print("🚀 Querying America's Health Rankings GraphQL API via measures schema...")

try:
    response = requests.post(
        url, 
        json={'query': query, 'variables': variables}, 
        headers=headers, 
        timeout=60
    )
    response.raise_for_status()
    payload = response.json()

    if 'errors' in payload:
        print("❌ GraphQL Gatekeeper compilation or syntax error:")
        print(payload['errors'])
        
    elif 'data' in payload and payload['data'] is not None and 'measures_A' in payload['data']:

        # Save raw snapshot of JSON data
        if "target_dir" not in locals() and "target_dir" not in globals():
            target_dir = os.getcwd()  # Fallback to current workspace if undefined

        # Resolve path to target_dir/../raw_api_snapshot.json securely
        snapshot_path = os.path.abspath(
            os.path.join(target_dir, "..", "raw_api_snapshot.json")
        )

        print(f"💾 Saving raw snapshot of the payload to: {snapshot_path}")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            # indent=4 formats it with clean, readable nesting lines
            json.dump(payload, f, indent=4, ensure_ascii=False)

        measures_list = payload['data']['measures_A']
        
        # Array to hold flattened individual data dictionaries
        flattened_records = []
        
        # Loop through parent measures to extract child observation arrays
        for measure_node in measures_list:
            m_name = measure_node.get('name')
            m_source = measure_node.get('source', {}).get('name') if measure_node.get('source') else None
            data_points = measure_node.get('data', [])
            
            for dp in data_points:
                flattened_records.append({
                    'Measure': m_name,
                    'Source': m_source,
                    'State': dp.get('state'),
                    'Data Year(s)': dp.get('dateLabel'),
                    'Value': dp.get('value'),
                    'Score': dp.get('score'),
                    'Rank': dp.get('rank')
                })
        
        if len(flattened_records) > 0:
            df_raw = pd.DataFrame(flattened_records)
            print(f"✅ Download Successful! Aggregated {df_raw.shape[0]} matching data records.")
            
            # Add hardcoded report identifier column
            df_raw['Report'] = "Health of Women and Children Report"
            
            # filter for 2018 - 2025
            df_raw['_extracted_year'] = df_raw['Data Year(s)'].str.extract(r'(\d{4})')
            df_raw['_extracted_year'] = pd.to_numeric(df_raw['_extracted_year'], errors='coerce')
            
            timeline_mask = (df_raw['_extracted_year'] >= 2018) & (df_raw['_extracted_year'] <= 2025)
            df_filtered = df_raw[timeline_mask].copy()
            
            # target columns for final df
            target_columns = ['Report', 'Measure', 'State', 'Value', 'Source', 'Data Year(s)']
            df_final_long = df_filtered[target_columns].copy()
            
            print("\n" + "="*60 + "\nIN-MEMORY DATAFRAME GENERATION COMPLETE\n" + "="*60)
            print(f"📊 Dimensions of clean matrix: {df_final_long.shape}")
            print("\nPreview of active records:")
            print(df_final_long.head(5))
            
        else:
            print("⚠️ Server processed request but zero data rows were returned for those specific measures.")
    else:
        print("❌ Unexpected payload dictionary format nested inside query trace response.")

except requests.exceptions.HTTPError as http_err:
    print(f"🛑 Handshake Failure: {http_err}")
    if 'response' in locals() and response.text:
        print(f"Server response details: {response.text}")

df = df_final_long.copy()
df['Value'] = df.Value.fillna(9999)

# Convert to clean format, 6 rows per State/Measure combination for 2018-2023 respectively
def get_yr_from_string(year_str):

    if pd.isna(year_str):
        return []
    
    year_str = (
        str(year_str)
        .replace(" Publication", "")
        .replace(" Report", "")
        .replace("September ", "")
    )
    year_str = year_str.strip()

    # Hyphenated Spans
    if "-" in year_str:
        try:
            start, end = map(int, year_str.split("-"))
            return list(range(start, end+1))
        except ValueError:
            pass
    
    # Slashes in Year String
    if "/" in year_str:
        try:
            years = [int(y) for y in year_str.split("/")]
            if len(years) == 2 and years[1] - years[0] <= 4:
                return list(range(years[0], years[1]+1))
            return years
        except ValueError:
            pass

    # Standard single year string
    try:
        return [int(year_str)]
    except ValueError:
        return []
    
# Clean and Expand Ranges
df['Parsed_Years'] = df['Data Year(s)'].apply(get_yr_from_string)

df_exploded = df.explode("Parsed_Years")
df_exploded = df_exploded.dropna(subset=['Parsed_Years'])
df_exploded['Year'] = df_exploded["Parsed_Years"].astype(int)

# Drop Data outside of 2018-2023
df_target = df_exploded[df_exploded['Year'].between(2018,2023)].copy()

# Handle Duplicates: Keep record with closest proximity to historical average

# Flag rows where the exact State + Measure + Year combination appears more than once
df_target['is_dup_year'] = df_target.duplicated(subset=['State', 'Measure', 'Year'], keep=False)

# compute historical baseline average using only the clean, non-duplicate years
clean_numeric_rows = df_target[~df_target['is_dup_year'] & (df_target['Value'] != 9999)]
state_measure_means = clean_numeric_rows.groupby(['State', 'Measure'])['Value'].mean().reset_index()
state_measure_means = state_measure_means.rename(columns={'Value': 'Baseline_Avg'})

# Merge the baseline averages back into the target dataframe
df_target = pd.merge(df_target, state_measure_means, on=['State', 'Measure'], how='left')

# # Edge Case: Every State/Measure/Year is duplicated
# global_means = df_target[df_target['Value'] != 9999].groupby(['State', 'Measure'])['Value'].mean()
# df_target['Baseline_Avg'] = df_target['Baseline_Avg'].fillna(df_target.set_index(['State', 'Measure']).index.map(global_means).to_series())

# calculate the absolute distance from the baseline average
df_target['Dist_From_Avg'] = np.where(
    df_target['Value'] == 9999, 
    np.inf, 
    (df_target['Value'] - df_target['Baseline_Avg']).abs()
)

# sort rows so closest values (smallest Dist_From_Avg) on top
df_target = df_target.sort_values(['State', 'Measure', 'Year', 'Dist_From_Avg'])

# drop duplicates, keeping first measure (smallest Dist_From_Avg)
df_target = df_target.drop_duplicates(subset=['State', 'Measure', 'Year'], keep='first')

# drop tmp columns
df_target = df_target.drop(columns=['is_dup_year', 'Baseline_Avg', 'Dist_From_Avg'])

# Build template (6 rows per state/measure)
states = df_target.State.unique()
measures = df_target.Measure.unique()
years = list(range(2018, 2023+1))

# Master index
mx = pd.MultiIndex.from_product(
    [states, measures, years], names=['State', 'Measure', 'Year']
)
template = pd.DataFrame(index=mx).reset_index()

# Add data
final_df = pd.merge(
    template,
    df_target[['State', 'Measure', 'Year', 'Value']],
    on=["State", "Measure", "Year"],
    how="left"
)

# Interpolate Missing Values
final_df['Value'] = final_df['Value'].replace(9999, np.nan)

def impute_group(group):
    group = group.sort_values('Year')

    group['Value'] = group['Value'].interpolate(method='linear', limit_direction='both')

    return group

final_df = final_df.groupby(['State', 'Measure'], group_keys=False).apply(impute_group)

# Fill any remaining NAs with 9999
final_df.Value = final_df.Value.fillna(9999)

final_df = final_df.sort_values(['State', 'Measure', 'Year']).reset_index(drop=True)
# final_df.head(12)

# Export Clean DataFrame
output_dir = os.path.abspath(os.path.join(BASE_DIR, "data", "clean", "HealthRankings", "health.csv"))
final_df.to_csv(output_dir, index=False)