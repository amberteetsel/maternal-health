import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def generate_stacked_us_maps(df, measure_name, color_scale, title_text):
    """
    Generates vertically stacked US choropleth maps (2018 on top, 2023 on bottom).
    Missing values or 9999 are converted to NaN so that those states are grayed out.
    """
    # 1. Filter for the requested measure
    measure_df = df[df['Measure'] == measure_name].copy()
    
    # 2. Convert missing data flags (like 9999) to NaN so Plotly treats them as empty
    measure_df['Value'] = measure_df['Value'].replace(9999, np.nan)
    
    # 3. Separate the data by target years
    df_2018 = measure_df[measure_df['Year'] == 2018]
    df_2023 = measure_df[measure_df['Year'] == 2023]
    
    # 4. Find the common min/max range across both years to ensure a synchronized color bar
    valid_vals = measure_df['Value'].dropna()
    v_min = valid_vals.min() if not valid_vals.empty else 0
    v_max = valid_vals.max() if not valid_vals.empty else 100

    # 5. Build subplots stacked vertically (2 Rows, 1 Column)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.05,
        specs=[[{"type": "choropleth"}], [{"type": "choropleth"}]],
        subplot_titles=("2018", "2023")
    )

    # Configuration to ensure missing states show up as light gray land
    geo_config = dict(
        scope='usa',
        projection_type='albers usa',
        showland=True,
        landcolor='#e0e0e0',      # 🌟 Gray out states with no data/NaN
        showlakes=True,
        lakecolor='white'
    )

    # --- Top Map: 2018 ---
    fig.add_trace(
        go.Choropleth(
            locations=df_2018['State'],
            z=df_2018['Value'],
            locationmode='USA-states',
            colorscale=color_scale,
            zmin=v_min,
            zmax=v_max,
            colorbar_title="Value",
            hovertemplate="State: %{location}<br>Value: %{z}<extra></extra>"
        ),
        row=1, col=1
    )

    # --- Bottom Map: 2023 ---
    fig.add_trace(
        go.Choropleth(
            locations=df_2023['State'],
            z=df_2023['Value'],
            locationmode='USA-states',
            colorscale=color_scale,
            zmin=v_min,
            zmax=v_max,
            showscale=False,  # Avoid duplicate color bars
            hovertemplate="State: %{location}<br>Value: %{z}<extra></extra>"
        ),
        row=2, col=1
    )

    # 6. Apply layouts and shared map geo parameters
    fig.update_layout(
        title_text=title_text,
        title_x=0,
        height=750,
        margin=dict(l=10, r=10, t=60, b=10),
        geo=geo_config,
        geo2=geo_config
    )
    
    return fig