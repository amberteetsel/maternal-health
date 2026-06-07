import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
guttmacher_path = os.path.join(BASE_DIR, "data", "raw", "Guttmacher")

# Read CSV
df_wide = pd.read_csv(os.path.join(guttmacher_path, "NatStatePregnancy.csv"))

# Convert to long format, keeping only relevant column date and age groups

# Define mappings for clean labels and raw column strings
age_map = {
    # "<15": "lt15",
    "15-19": "1519",
    "20-24": "2024",
    "25-29": "2529",
    "30-34": "3034",
    "35-39": "3539",
    "40+": "40plus",
}

metric_map = {
    "Pregnancy": "pregnancies",
    "Abortion": "abortions",
    "Birth": "births",
    "Miscarriage": "miscarriages",
}

# Convert count
count_records=[]
for age_lab, age_suff in age_map.items():
    for metric_lab, metric_pre in metric_map.items():
        col_name = f"{metric_pre}{age_suff}"

        if col_name in df_wide.columns:
            tmp = df_wide[['state', 'year', col_name]].copy()
            tmp['metric'] = metric_lab
            tmp['age_group'] = age_lab
            tmp = tmp.rename(columns={col_name: 'count'})
            count_records.append(tmp)

df_counts_long = pd.concat(count_records, ignore_index=True)

# Convert population
pop_records = []
for age_lab, age_suff in age_map.items():
    pop_col_name = f"population{age_suff}"

    if pop_col_name in df_wide.columns:
        tmp = df_wide[['state', 'year', pop_col_name]].copy()
        tmp['age_group'] = age_lab
        tmp = tmp.rename(columns={pop_col_name: 'population'})
        pop_records.append(tmp)

df_pop_long = pd.concat(pop_records, ignore_index=True)

# Join
df_long_final = pd.merge(
    df_counts_long,
    df_pop_long,
    on=['state', 'year', 'age_group'],
    how='left'
)

target_cols = ['state', 'year', 'metric', 'age_group', 'count', 'population']
df_long_final = df_long_final[target_cols]
df_long_final['rate'] = (df_long_final['count']/df_long_final['population'])*100000
df_long_final = df_long_final.sort_values(by=['state', 'year', 'metric', 'age_group']).reset_index(drop=True)

# Drop DC (all NAs)
df_long_final = df_long_final.loc[df_long_final.state!="DC"].reset_index(drop=True)

# Export clean data
output_dir = os.path.join(BASE_DIR, "data", "clean", "Guttmacher")
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

df_long_final.to_csv(os.path.join(output_dir, "pregnancy.csv"), index=False)

############# VISUALS #############
# Path
viz_path = os.path.abspath(os.path.join(BASE_DIR, "resources", "visuals_eda"))
if not os.path.exists(viz_path):
    os.makedirs(viz_path, exist_ok=True)

df_pregnancy = df_long_final.copy()

# Visual 1: rates over time
fig, ax = plt.subplots(figsize=(8.5, 5))

national_trends = df_pregnancy.groupby(['year', 'metric'])['rate'].mean().reset_index()

metrics = national_trends['metric'].unique()
colors = sns.color_palette("Set2", n_colors=len(metrics))
palette_dict = dict(zip(metrics, colors))

sns.lineplot(
    data=national_trends, 
    x='year', 
    y='rate', 
    hue='metric', 
    palette=palette_dict,
    marker='o',          
    markersize=6,
    linewidth=2, 
    ax=ax,
    legend=False         # No default legend
)

ax.set_xlabel("Historical Statistical Year")
ax.set_ylabel("Rate (per\n1,000 Women)", rotation=0, va='center', ha='right')

ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# gridlines
ax.grid(axis='y', color='lightgray', linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# borders
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)

# custom coordinates for labels
y_min, y_max = ax.get_ylim()
y_height = y_max - y_min

y_tweaks = {
    'Miscarriage': y_height * 0.03,   # Shifts up by 5% of chart height
    'Abortion': -(y_height * 0.03),   # Shifts down by 5% of chart height
}

# text labels
for metric in metrics:
    metric_data = national_trends[national_trends['metric'] == metric].sort_values('year')
    line_color = palette_dict[metric]
    
    last_row = metric_data.iloc[-1]
    
    # Calculate the tweaked y-position if specified
    tweak = y_tweaks.get(metric, 0.0)
    final_y = last_row['rate'] + tweak
    
    ax.text(
        x=last_row['year'] + 0.5,  # Increased offset to shift ALL text labels further right
        y=final_y, 
        s=metric.capitalize(),     
        color=line_color, 
        va='center', 
        ha='left', 
        fontweight='bold',
        fontsize=10
    )

# Push the x-axis limit
ax.set_xlim(national_trends['year'].min(), national_trends['year'].max() + 2)

fig.tight_layout()

# Save
fig.savefig(os.path.join(viz_path, "preg_v1.png"))

# Visual 2: abortions and miscarriages per pregnancy by age group
fig, ax = plt.subplots(figsize=(8, 6))

df_grouped = df_pregnancy.groupby(['age_group', 'metric'])['rate'].mean().unstack()
total_pregnancies = df_grouped.sum(axis=1) 

# Divide the target metrics by total pregnancies to get the proportion/rate per pregnancy
df_rate_per_preg = df_grouped[['Abortion', 'Miscarriage']].div(total_pregnancies, axis=0)

# long format
age_metrics = df_rate_per_preg.reset_index().melt(
    id_vars='age_group', 
    value_vars=['Abortion', 'Miscarriage'], 
    var_name='metric', 
    value_name='rate_per_pregnancy'
)

# color palette
set2_palette = sns.color_palette("Set2", n_colors=4) 
custom_palette = {
    'Abortion': set2_palette[0],      # Green
    'Miscarriage': set2_palette[2]    # Grayish-Blue
}

# plot
sns.barplot(
    data=age_metrics, 
    x='age_group', 
    y='rate_per_pregnancy', 
    hue='metric', 
    palette=custom_palette, 
    ax=ax
)

# axis labels
ax.set_xlabel("Maternal Age Cohort")
ax.set_ylabel("Mean\nHistorical Rate\n(per Pregnancy)", rotation=0, va='center', ha='right')

# format y-axis ticks
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0%}'))

# Remove the legend title
legend = ax.legend()
legend.set_title(None)

# gridlines
ax.grid(axis='y', color='lightgray', linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Remove borders
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)

# data labels inside bars
for p in ax.patches:
    height = p.get_height()
    if height > 0: 
        width = p.get_width()
        x = p.get_x()
        y = p.get_y()
        
        ax.text(
            x + width / 2.0, 
            y + height - (ax.get_ylim()[1] * 0.025), 
            f"{height:,.1%}",
            ha='center', 
            va='top', 
            color='black',  
            fontsize=8.5
        )

fig.tight_layout()

# Save
fig.savefig(os.path.join(viz_path, "preg_v2.png"))
