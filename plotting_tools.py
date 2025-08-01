import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.colors

def plot_cost_demand_facets(data_dict, width=1600, height=900):
    # Prepare records from the nested dictionary
    records = []
    iterations = sorted(data_dict.keys())
    for k in iterations:
        df = data_dict[k]['df']
        for country in df.index:
            for scenario in ['-', '0', '+']:
                cost = df.at[country, scenario]
                demand_col = f'demand_{scenario}'
                demand = df.at[country, demand_col] if demand_col in df.columns else None
                records.append({
                    'Iteration': k,
                    'Country': country,
                    'Scenario': scenario,
                    'Cost': cost,
                    'Demand': demand
                })

    df_all = pd.DataFrame(records)

    # Manual color map for scenarios
    plotly_colors = plotly.colors.qualitative.Plotly
    scenario_colors = {
        '-': plotly_colors[0],
        '0': plotly_colors[1],
        '+': plotly_colors[2]
    }

    countries = df_all['Country'].unique()
    scenarios = ['-', '0', '+']
    cols = 4
    rows = (len(countries) + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=countries,
        shared_xaxes=False, shared_yaxes=False,
        vertical_spacing=0.12, horizontal_spacing=0.05,
        specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
    )

    for i, country in enumerate(countries):
        row = i // cols + 1
        col = i % cols + 1
        for scenario in scenarios:
            subset = df_all[(df_all['Country'] == country) & (df_all['Scenario'] == scenario)]
            color = scenario_colors[scenario]
            fig.add_trace(go.Scatter(
                x=subset['Iteration'], y=subset['Cost'],
                mode='lines+markers', name=f"{scenario} Cost",
                legendgroup=scenario, showlegend=(i == 0),
                line=dict(color=color)
            ), row=row, col=col, secondary_y=False)
            fig.add_trace(go.Scatter(
                x=subset['Iteration'], y=subset['Demand'],
                mode='markers', name=f"{scenario} Demand",
                legendgroup=scenario, showlegend=(i == 0),
                marker=dict(color=color, symbol='circle', size=8)
            ), row=row, col=col, secondary_y=True)

    fig.update_layout(
        height=height, width=width,
        title_text="Cost (Line) and Demand (Dots) per Country and Scenario",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    fig.update_yaxes(title_text="Cost", secondary_y=False)
    fig.update_yaxes(title_text="Demand", secondary_y=True)
    fig.show()

def plot_demand_transmission(data, width=1200, height=900):
    iterations = sorted(data.keys())
    records = []

    for k in iterations:
        df = data[k]['df']
        tdf = data[k].get('transmission_df', pd.DataFrame())

        for country in df.index:
            demand = df.at[country, f'demand_0'] if f'demand_0' in df.columns else 0.0
            imports = -tdf.loc[(tdf['start_country'] == country) & (tdf['exchange'] < 0), 'exchange'].sum()
            exports = tdf.loc[(tdf['start_country'] == country) & (tdf['exchange'] > 0), 'exchange'].sum()
            imports = imports if not pd.isna(imports) else 0.0
            exports = exports if not pd.isna(exports) else 0.0
            export_within_demand = min(exports, demand)
            records.append({
                'Iteration': k,
                'Country': country,
                'Demand': demand,
                'Import': imports,
                'Export': export_within_demand
            })

    df_tx = pd.DataFrame(records)
    countries = df_tx['Country'].unique()
    cols = 4
    rows = (len(countries) + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=countries,
        shared_xaxes=False, shared_yaxes=False,
        vertical_spacing=0.12, horizontal_spacing=0.05
    )

    for i, country in enumerate(countries):
        row = i // cols + 1
        col = i % cols + 1
        subset = df_tx[df_tx['Country'] == country].sort_values('Iteration')

        fig.add_trace(go.Bar(
            x=subset['Iteration'], y=subset['Demand'],
            name='Internal Demand',
            marker_color='blue', showlegend=(i == 0),
            offsetgroup='demand', legendgroup='Demand'
        ), row=row, col=col)

        fig.add_trace(go.Bar(
            x=subset['Iteration'], y=subset['Export'],
            name='Export (from demand)',
            marker=dict(color='blue', pattern=dict(shape='\\')),
            offsetgroup='demand', base=0,
            showlegend=(i == 0), legendgroup='Export'
        ), row=row, col=col)

        fig.add_trace(go.Bar(
            x=subset['Iteration'], y=subset['Import'],
            name='Import',
            marker_color='green', opacity=0.8,
            offsetgroup='import', base=subset['Demand'],
            showlegend=(i == 0), legendgroup='Import'
        ), row=row, col=col)

    fig.update_layout(
        height=height, width=width,
        title_text=f"Demand + Transmission (Patterned Export) per Country ",
        barmode='overlay',
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    fig.update_yaxes(title_text="Energy")
    fig.show()

def analyze_transmission_symmetry(width, height, timeslice_data, title_prefix="Transmission Flow"):
    """
    Analyze and plot transmission flow symmetry over iterations for a given timeslice.

    Parameters:
    - width (int): Width of the plot in pixels
    - height (int): Height of the plot in pixels
    - timeslice_data (dict): dataframes[year][timeslice], containing per-iteration transmission_df entries
    - title_prefix (str): Custom title prefix for the plot

    Returns:
    - fig (plotly.graph_objs._figure.Figure): The generated plotly figure
    - df_mismatches (pd.DataFrame or None): DataFrame of symmetry mismatches (or None if all OK)
    """
    iterations = sorted(timeslice_data.keys())
    mismatches = []
    records = []

    for k in iterations:
        tdf = timeslice_data[k].get('transmission_df', pd.DataFrame())

        # Build flow map
        flow_map = {(row['start_country'], row['end_country']): row['exchange']
                    for _, row in tdf.iterrows()}

        # Check for mismatches
        for (start, end), value in flow_map.items():
            reverse_key = (end, start)
            reverse_value = flow_map.get(reverse_key, None)

            if reverse_value is None:
                mismatches.append({
                    'Iteration': k,
                    'Issue': 'Missing reverse flow',
                    'Start': start, 'End': end,
                    'Value': value, 'ReverseValue': None
                })
            elif not abs(value + reverse_value) < 1e-3:
                mismatches.append({
                    'Iteration': k,
                    'Issue': 'Mismatch in reverse flow',
                    'Start': start, 'End': end,
                    'Value': value, 'ReverseValue': reverse_value
                })

        # Collect data for plotting
        for (start, end), value in flow_map.items():
            records.append({
                'Iteration': k,
                'Start': start,
                'End': end,
                'Exchange': value
            })

    df_tx_all = pd.DataFrame(records)
    df_mismatches = pd.DataFrame(mismatches) if mismatches else None

    # Print mismatches
    if df_mismatches is not None:
        print("❌ Transmission symmetry mismatches found:")
        print(df_mismatches)
    else:
        print("✅ All transmission flows are symmetric across iterations.")

    # Plot
    fig = px.line(df_tx_all, x='Iteration', y='Exchange', color='Start',
                  facet_col='End', facet_col_wrap=4,
                  title=f"{title_prefix} — {len(iterations)} Iterations")
    fig.update_layout(height=height, width=width)
    fig.show()

    return df_mismatches



def plot_mc_and_marginals(data_dict, width=1600, height=900):
    """
    Plot marginal cost of import/export and marginal demand per country over iterations.
    
    Parameters:
    - data_dict: dictionary with structure like dataframes[year][timeslice]
    - width: plot width in px
    - height: plot height in px
    """
    records = []
    iterations = sorted(data_dict.keys())
    
    for k in iterations:
        df = data_dict[k]['df']
        for country in df.index:
            record = {
                'Iteration': k,
                'Country': country,
                'Cost': df.loc[country, '0'] if '0' in df.columns else None,
                'MC_import': df.loc[country, 'MC_import'] if 'MC_import' in df.columns else None,
                'MC_export': df.loc[country, 'MC_export'] if 'MC_export' in df.columns else None,
                'MarginalDemand': df.loc[country, 'marginal_demand'] if 'marginal_demand' in df.columns else None
            }
            records.append(record)
    
    df_all = pd.DataFrame(records)
    countries = df_all['Country'].unique()
    cols = 4
    rows = (len(countries) + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=countries,
        shared_xaxes=True, shared_yaxes=False,
        vertical_spacing=0.12, horizontal_spacing=0.05,
        specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
    )

    for i, country in enumerate(countries):
        row = i // cols + 1
        col = i % cols + 1
        subset = df_all[df_all['Country'] == country]

        # Cost line
        fig.add_trace(go.Scatter(
            x=subset['Iteration'], y=subset['Cost'],
            mode='lines+markers', name='Cost',
            legendgroup='Cost', showlegend=(i == 0),
            line=dict(color='black')
        ), row=row, col=col, secondary_y=False)

        # MC_import
        fig.add_trace(go.Scatter(
            x=subset['Iteration'], y=subset['MC_import'],
            mode='lines+markers', name='MC Import',
            legendgroup='MC Import', showlegend=(i == 0),
            line=dict(color='blue', dash='dot')
        ), row=row, col=col, secondary_y=True)

        # MC_export
        fig.add_trace(go.Scatter(
            x=subset['Iteration'], y=subset['MC_export'],
            mode='lines+markers', name='MC Export',
            legendgroup='MC Export', showlegend=(i == 0),
            line=dict(color='red', dash='dot')
        ), row=row, col=col, secondary_y=True)

        # Marginal Demand
        fig.add_trace(go.Scatter(
            x=subset['Iteration'], y=subset['MarginalDemand'],
            mode='markers+lines', name='Marginal Demand',
            legendgroup='Marginal Demand', showlegend=(i == 0),
            line=dict(color='green')
        ), row=row, col=col, secondary_y=True)

    fig.update_layout(
        height=height, width=width,
        title_text="Cost, MC Import/Export and Marginal Demand per Country",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )

    fig.update_yaxes(title_text="Cost", secondary_y=False)
    fig.update_yaxes(title_text="Marginal Values", secondary_y=True)

    fig.show()