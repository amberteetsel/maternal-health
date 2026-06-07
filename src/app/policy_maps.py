import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_ban_limit_map(df, title_text="Total Bans & Gestational Limits"):
    # 1. Standardize and copy dates
    df_map = df.copy()
    df_map['Start'] = pd.to_datetime(df_map['Start'])
    
    df_2018 = df_map[df_map['Start'] <= pd.to_datetime('2018-12-01')].copy()
    df_2023 = df_map[df_map['Start'] <= pd.to_datetime('2023-05-31')].copy()

    # Sort to preserve highest severity order
    df_2018 = df_2018.sort_values(['Start', 'TotalBan', 'HeartbeatBan', 'GestLimit6']).groupby('State').last().reset_index()
    df_2023 = df_2023.sort_values(['Start', 'TotalBan', 'HeartbeatBan', 'GestLimit6']).groupby('State').last().reset_index()

    def calculate_metrics(target_df):
        values = []
        labels = []
        for _, row in target_df.iterrows():
            if row.get('TotalBan') == 1:
                values.append(5)
                labels.append("Total Ban")
            elif row.get('HeartbeatBan') == 1:
                values.append(4)
                labels.append("Heartbeat Ban")
            elif row.get('GestLimit6') == 1 or row.get('GestLimt6') == 1:
                values.append(3)
                labels.append("Gestational Limit: 6 Weeks")
            elif row.get('GestLimit7_14') == 1:
                values.append(2)
                labels.append("Gestational Limit: 7-14 Weeks")
            elif row.get('GestLimit15_20') == 1:
                values.append(1)
                labels.append("Gestational Limit: 15-20 Weeks")
            else:
                values.append(0)
                labels.append("No Restrictive Bans or Limits")
        target_df['MapValue'] = values
        target_df['PolicyLabel'] = labels
        return target_df

    df_2018 = calculate_metrics(df_2018)
    df_2023 = calculate_metrics(df_2023)

    ban_colorscale = [
        [0.0, '#f7f7f7'], [0.166, '#f7f7f7'],
        [0.166, '#e8e8f3'], [0.333, '#e8e8f3'],
        [0.333, '#b2abd2'], [0.5, '#b2abd2'],
        [0.5, '#5e3c99'], [0.666, '#5e3c99'],
        [0.666, '#fdb863'], [0.833, '#fdb863'],
        [0.833, '#b2182b'], [1.0, '#b2182b']
    ]

    # FIX: Initialize clean layout without subplot_titles parameter
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "choropleth"}], [{"type": "choropleth"}]]
    )

    fig.add_trace(
        go.Choropleth(
            locations=df_2018['State'], locationmode='USA-states',
            z=df_2018['MapValue'], colorscale=ban_colorscale,
            zmin=0, zmax=5, showscale=False,
            text=df_2018['PolicyLabel'],
            hovertemplate="<b>%{location}</b><br>%{text}<extra></extra>"
        ), row=1, col=1
    )

    fig.add_trace(
        go.Choropleth(
            locations=df_2023['State'], locationmode='USA-states',
            z=df_2023['MapValue'], colorscale=ban_colorscale,
            zmin=0, zmax=5, showscale=False,
            text=df_2023['PolicyLabel'],
            hovertemplate="<b>%{location}</b><br>%{text}<extra></extra>"
        ), row=2, col=1
    )

    # Clean layout annotations instead of using broken tuple parameters
    fig.update_layout(
        title_text=title_text,
        geo=dict(scope='usa', showlakes=True, lakecolor='white'),
        geo2=dict(scope='usa', showlakes=True, lakecolor='white'),
        height=850,
        margin=dict(t=100, b=20, l=20, r=20),
        annotations=[
            dict(text="As of 2018-12-01", x=0.5, y=1.02, xref="paper", yref="paper", showarrow=False, font=dict(size=14, bold=True)),
            dict(text="As of 2023-05-31", x=0.5, y=0.47, xref="paper", yref="paper", showarrow=False, font=dict(size=14, bold=True))
        ]
    )

    return fig

# def create_ban_limit_map(df, title_text="Total Bans & Gestational Limits"):
#     """
#     Visual 1: Total Bans/Gestational Limits stacked maps.
#     Dates: 2018-12-01 (Top), 2023-05-31 (Bottom).
    
#     3 Distinct Color Categories:
#     - Base State: Light Gray (0)
#     - Gestational Limits: Purple Scale (1 = 15-20w Lightest, 2 = 7-14w Medium, 3 = 6w Darkest)
#     - Heartbeat Ban: Vibrant Amber/Orange (4)
#     - Total Ban: Deep Crimson Red (5)
#     """
#     # Force copy to avoid modification warnings
#     df_map = df.copy()
    
#     # CRITICAL FIX 1: Force true datetime conversion to prevent string evaluation failures
#     df_map['Start'] = pd.to_datetime(df_map['Start'])
    
#     # Filter out snapshots using true timestamps
#     df_2018 = df_map[df_map['Start'] <= pd.to_datetime('2018-12-01')].copy()
#     df_2023 = df_map[df_map['Start'] <= pd.to_datetime('2023-05-31')].copy()

#     # CRITICAL FIX 2: Sort by priority columns *before* taking .last() 
#     # This guarantees that if a state has BOTH a GestLimit and a HeartbeatBan on the same date,
#     # the higher restriction takes precedence and isn't randomly squashed.
#     df_2018 = df_2018.sort_values(['Start', 'TotalBan', 'HeartbeatBan', 'GestLimit6'])
#     df_2018 = df_2018.groupby('State').last().reset_index()
    
#     df_2023 = df_2023.sort_values(['Start', 'TotalBan', 'HeartbeatBan', 'GestLimit6'])
#     df_2023 = df_2023.groupby('State').last().reset_index()

#     # 2. Assign strict metric hierarchy directly scanning your clean columns
#     def calculate_metrics(target_df):
#         values = []
#         labels = []
#         for _, row in target_df.iterrows():
#             if row.get('TotalBan') == 1:
#                 values.append(5)
#                 labels.append("Total Ban")
#             elif row.get('HeartbeatBan') == 1:
#                 values.append(4)
#                 labels.append("Heartbeat Ban")
#             elif row.get('GestLimit6') == 1 or row.get('GestLimt6') == 1:
#                 values.append(3)
#                 labels.append("Gestational Limit: 6 Weeks (Darkest)")
#             elif row.get('GestLimit7_14') == 1:
#                 values.append(2)
#                 labels.append("Gestational Limit: 7-14 Weeks")
#             elif row.get('GestLimit15_20') == 1:
#                 values.append(1)
#                 labels.append("Gestational Limit: 15-20 Weeks (Lightest)")
#             else:
#                 values.append(0)
#                 labels.append("No Restrictive Bans or Limits")
        
#         target_df['MapValue'] = values
#         target_df['PolicyLabel'] = labels
#         return target_df

#     df_2018 = calculate_metrics(df_2018)
#     df_2023 = calculate_metrics(df_2023)

#     # 3. Discrete hard-stepped block colorscale (zmin=0, zmax=5)
#     ban_colorscale = [
#         [0.0, '#f7f7f7'], [0.166, '#f7f7f7'],       # 0 = Light Gray (Base)
#         [0.166, '#e8e8f3'], [0.333, '#e8e8f3'],     # 1 = Light Purple (GestLimit 15-20)
#         [0.333, '#b2abd2'], [0.5, '#b2abd2'],       # 2 = Medium Purple (GestLimit 7-14)
#         [0.5, '#5e3c99'], [0.666, '#5e3c99'],       # 3 = Dark Purple (GestLimit 6)
#         [0.666, '#fdb863'], [0.833, '#fdb863'],     # 4 = Vibrant Amber/Orange (Heartbeat Ban)
#         [0.833, '#b2182b'], [1.0, '#b2182b']        # 5 = Deep Crimson Red (Total Ban)
#     ]

#     fig = make_subplots(
#         rows=2, cols=1,
#         specs=[[{"type": "choropleth"}], [{"type": "choropleth"}]],
#         subplot_titles=("As of 2018-12-01", "As of 2023-05-31")
#     )

#     # Top Map: 2018
#     fig.add_trace(
#         go.Choropleth(
#             locations=df_2018['State'], locationmode='USA-states',
#             z=df_2018['MapValue'], colorscale=ban_colorscale,
#             zmin=0, zmax=5, showscale=False,
#             text=df_2018['PolicyLabel'],
#             hovertemplate="<b>%{location}</b><br>%{text}<extra></extra>"
#         ), row=1, col=1
#     )

#     # Bottom Map: 2023
#     fig.add_trace(
#         go.Choropleth(
#             locations=df_2023['State'], locationmode='USA-states',
#             z=df_2023['MapValue'], colorscale=ban_colorscale,
#             zmin=0, zmax=5, showscale=False,
#             text=df_2023['PolicyLabel'],
#             hovertemplate="<b>%{location}</b><br>%{text}<extra></extra>"
#         ), row=2, col=1
#     )

#     fig.update_layout(
#         title_text=title_text,
#         geo=dict(scope='usa', showlakes=True, lakecolor='white'),
#         geo2=dict(scope='usa', showlakes=True, lakecolor='white'),
#         height=850,
#         margin=dict(t=80, b=20, l=20, r=20)
#     )

#     return fig


def create_protection_map(df, title_text="Legal & Constitutional Protections"):
    """
    Visual 2: Legal/Constitutional Protections stacked maps.
    Dates: 2018-12-01 (Top), 2023-05-31 (Bottom).
    """
    df_2018 = df[df['Start'] <= '2018-12-01'].sort_values('Start').groupby('State').last().reset_index()
    df_2023 = df[df['Start'] <= '2023-05-31'].sort_values('Start').groupby('State').last().reset_index()

    def calculate_protection_metrics(target_df):
        values = []
        labels = []
        for _, row in target_df.iterrows():
            if row.get('ConstProtection') == 1:
                values.append(2)  
                labels.append("Constitutional Protection (Solid)")
            elif row.get('LegalProtection') == 1:
                values.append(1)  
                labels.append("Legal Protection Only")
            else:
                values.append(0)  
                labels.append("No Explicit Protections Documented")

        target_df['MapValue'] = values
        target_df['PolicyLabel'] = labels
        return target_df

    df_2018 = calculate_protection_metrics(df_2018)
    df_2023 = calculate_protection_metrics(df_2023)

    # Distinct Blueprint Colorscale
    protection_colorscale = [
        [0.0, '#ffffff'], [0.33, '#ffffff'],       # 0 = White (No protection)
        [0.33, '#9ecae1'], [0.66, '#9ecae1'],     # 1 = Light Blue (Legal protection)
        [0.66, '#08519c'], [1.0, '#08519c']       # 2 = Deep Royal Blue (Constitutional)
    ]

    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "choropleth"}], [{"type": "choropleth"}]],
        subplot_titles=("As of 2018-12-01", "As of 2023-05-31")
    )

    # Top Map: 2018
    fig.add_trace(
        go.Choropleth(
            locations=df_2018['State'], locationmode='USA-states',
            z=df_2018['MapValue'], colorscale=protection_colorscale,
            zmin=0, zmax=2, showscale=False,
            text=df_2018['PolicyLabel'],
            hovertemplate="<b>%{location}</b><br>Protection: %{text}<extra></extra>"
        ), row=1, col=1
    )

    # Bottom Map: 2023
    fig.add_trace(
        go.Choropleth(
            locations=df_2023['State'], locationmode='USA-states',
            z=df_2023['MapValue'], colorscale=protection_colorscale,
            zmin=0, zmax=2, showscale=False,
            text=df_2023['PolicyLabel'],
            hovertemplate="<b>%{location}</b><br>Protection: %{text}<extra></extra>"
        ), row=2, col=1
    )

    fig.update_layout(
        title_text=title_text, 
        height=850, 
        geo=dict(scope='usa', showlakes=True, lakecolor='white'), 
        geo2=dict(scope='usa', showlakes=True, lakecolor='white')
    )
    return fig