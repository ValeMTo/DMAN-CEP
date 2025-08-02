import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px
import plotly.colors

def plot_demand_mc_marginals(data_dict, width=1600, height=900):
    """
    Plot demand (-, 0, +), marginal demand, MC_import, and MC_export per country over iterations.
    """
    records = []
    def try_int(x):
        try:
            return int(x)
        except Exception:
            return x

    iterations = sorted(data_dict.keys(), key=try_int)
    for k in iterations:
        df = data_dict[k]['df']
        if 'MC_import' in df.columns:
            df['MC_import'] = pd.to_numeric(df['MC_import'], errors='coerce')
        if 'MC_export' in df.columns:
            df['MC_export'] = pd.to_numeric(df['MC_export'], errors='coerce')
        for country in df.index:
            record = {
                'Iteration': try_int(k),
                'Country': country,
                'Demand_-': df.at[country, 'demand_-'] if 'demand_-' in df.columns else None,
                'Demand_0': df.at[country, 'demand_0'] if 'demand_0' in df.columns else None,
                'Demand_+': df.at[country, 'demand_+'] if 'demand_+' in df.columns else None,
                'MarginalDemand': df.at[country, 'marginal_demand'] if 'marginal_demand' in df.columns else None,
                'MC_import': df.at[country, 'MC_import'] if 'MC_import' in df.columns else None,
                'MC_export': df.at[country, 'MC_export'] if 'MC_export' in df.columns else None,
            }
            records.append(record)

    df_all = pd.DataFrame(records)
    df_all = df_all.sort_values('Iteration')
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

        # Demand lines
        fig.add_trace(go.Scatter(
            x=subset['Iteration'], y=subset['Demand_-'],
            mode='lines+markers', name='Demand -',
            legendgroup='Demand -', showlegend=(i == 0),
            line=dict(color='blue', dash='dot')
        ), row=row, col=col, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=subset['Iteration'], y=subset['Demand_0'],
            mode='lines+markers', name='Demand 0',
            legendgroup='Demand 0', showlegend=(i == 0),
            line=dict(color='blue', dash='solid')
        ), row=row, col=col, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=subset['Iteration'], y=subset['Demand_+'],
            mode='lines+markers', name='Demand +',
            legendgroup='Demand +', showlegend=(i == 0),
            line=dict(color='blue', dash='dash')
        ), row=row, col=col, secondary_y=False)

        # Marginal Demand
        fig.add_trace(go.Scatter(
            x=subset['Iteration'], y=subset['MarginalDemand'],
            mode='lines+markers', name='Marginal Demand',
            legendgroup='Marginal Demand', showlegend=(i == 0),
            line=dict(color='green')
        ), row=row, col=col, secondary_y=True)

        # MC_import
        fig.add_trace(go.Scatter(
            x=subset['Iteration'], y=subset['MC_import'],
            mode='lines+markers', name='MC Import',
            legendgroup='MC Import', showlegend=(i == 0),
            line=dict(color='orange', dash='dot')
        ), row=row, col=col, secondary_y=True)

        # MC_export
        fig.add_trace(go.Scatter(
            x=subset['Iteration'], y=subset['MC_export'],
            mode='lines+markers', name='MC Export',
            legendgroup='MC Export', showlegend=(i == 0),
            line=dict(color='red', dash='dot')
        ), row=row, col=col, secondary_y=True)

    fig.update_layout(
        height=height, width=width,
        title_text="Demand (-, 0, +), Marginal Demand, MC Import/Export per Country",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    fig.update_yaxes(title_text="Demand", secondary_y=False)
    fig.update_yaxes(title_text="Marginal / MC", secondary_y=True)
    fig.show()

def plot_cost_demand_facets(data_dict, width=1600, height=900):
    # Prepare records from the nested dictionary
    records = []
    def try_int(x):
        try:
            return int(x)
        except Exception:
            return x

    iterations = sorted(data_dict.keys(), key=try_int)
    for k in iterations:
        df = data_dict[k]['df']
        for country in df.index:
            for scenario in ['-', '0', '+']:
                cost = df.at[country, scenario]
                demand_col = f'demand_{scenario}'
                demand = df.at[country, demand_col] if demand_col in df.columns else None
                records.append({
                    'Iteration': try_int(k),
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

    # Sort iterations numerically if possible
    def try_int(x):
        try:
            return int(x)
        except Exception:
            return x

    iterations = sorted(data.keys(), key=try_int)
    records = []

    for k in iterations:
        df = data[k]['df']
        tdf = data[k].get('transmission_df', pd.DataFrame())

        for country in df.index:
            demand = df.at[country, f'demand_0'] if f'demand_0' in df.columns else 0.0
            imports = -tdf.loc[(tdf['start_country'] == country) & (tdf['exchange'] < 0), 'exchange'].sum()
            exports = tdf.loc[(tdf['start_country'] == country) & (tdf['exchange'] > 0), 'exchange'].sum()
            total_demand = demand + exports
            imports = imports if not pd.isna(imports) else 0.0
            exports = exports if not pd.isna(exports) else 0.0
            export_within_demand = min(exports, demand)
            cost = df.at[country, '0'] if '0' in df.columns else None
            cost_per_unit = cost / total_demand if total_demand and cost is not None else None
            records.append({
                'Iteration': try_int(k),
                'Country': country,
                'Demand': demand,
                'Import': imports,
                'Export': export_within_demand,
                'CostPerUnit': cost_per_unit
            })

    df_tx = pd.DataFrame(records)
    df_tx = df_tx.sort_values('Iteration')
    countries = df_tx['Country'].unique()
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
        subset = df_tx[df_tx['Country'] == country].sort_values('Iteration')

        fig.add_trace(go.Bar(
            x=subset['Iteration'], y=subset['Demand'],
            name='Internal Demand',
            marker_color='blue', showlegend=(i == 0),
            offsetgroup='demand', legendgroup='Demand'
        ), row=row, col=col, secondary_y=False)

        fig.add_trace(go.Bar(
            x=subset['Iteration'], y=subset['Export'],
            name='Export (from demand)',
            marker=dict(color='blue', pattern=dict(shape='\\')),
            offsetgroup='demand', base=0,
            showlegend=(i == 0), legendgroup='Export'
        ), row=row, col=col, secondary_y=False)

        fig.add_trace(go.Bar(
            x=subset['Iteration'], y=subset['Import'],
            name='Import',
            marker_color='green', opacity=0.8,
            offsetgroup='import', base=subset['Demand'],
            showlegend=(i == 0), legendgroup='Import'
        ), row=row, col=col, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=subset['Iteration'], y=subset['CostPerUnit'],
            mode='lines+markers', name='Cost per Unit',
            line=dict(color='red', dash='solid'),
            showlegend=(i == 0), legendgroup='CostPerUnit'
        ), row=row, col=col, secondary_y=True)

    fig.update_layout(
        height=height, width=width,
        title_text=f"Demand + Transmission (Patterned Export) per Country with Cost per Unit",
        barmode='overlay',
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    fig.update_yaxes(title_text="Energy", secondary_y=False)
    fig.update_yaxes(title_text="Cost per Unit", secondary_y=True)
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
    def try_int(x):
        try:
            return int(x)
        except Exception:
            return x

    iterations = sorted(timeslice_data.keys(), key=try_int)
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
                'Iteration': try_int(k),
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
    def try_int(x):
        try:
            return int(x)
        except Exception:
            return x

    iterations = sorted(data_dict.keys(), key=try_int)
    
    for k in iterations:
        df = data_dict[k]['df']
        if 'MC_import' in df.columns:
            df['MC_import'] = pd.to_numeric(df['MC_import'], errors='coerce').round(0)
        if 'MC_export' in df.columns:
            df['MC_export'] = pd.to_numeric(df['MC_export'], errors='coerce').round(0)
        for country in df.index:
            record = {
                'Iteration': try_int(k),
                'Country': country,
                'Cost': df.loc[country, '0'] if '0' in df.columns else None,
                'MC_import': df.loc[country, 'MC_import'] if 'MC_import' in df.columns else None,
                'MC_export': df.loc[country, 'MC_export'] if 'MC_export' in df.columns else None,
                'MarginalDemand': df.loc[country, 'marginal_demand'] if 'marginal_demand' in df.columns else None
            }
            records.append(record)
    
    df_all = pd.DataFrame(records)
    df_all = df_all.sort_values('Iteration')
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

def plot_total_cost_variation_over_time(data_dict, width=1200, height=700):
    """
    Plot total cost per country and the variation (delta) per iteration, plus total sum and its variation.

    Parameters:
    - data_dict: dictionary with structure like dataframes[year][timeslice]
    - width: plot width in px
    - height: plot height in px
    """
    records = []
    def try_int(x):
        try:
            return int(x)
        except Exception:
            return x

    iterations = sorted(data_dict.keys(), key=try_int)
    for k in iterations:
        df = data_dict[k]['df']
        for country in df.index:
            cost = df.loc[country, '0'] if '0' in df.columns else None
            records.append({
                'Iteration': try_int(k),
                'Country': country,
                'Cost': cost
            })

    df_all = pd.DataFrame(records)
    df_all = df_all.sort_values(['Country', 'Iteration'])

    # Compute per-country cost delta
    df_all['CostDelta'] = df_all.groupby('Country')['Cost'].diff()

    # Compute total cost and its delta per iteration
    total_cost = df_all.groupby('Iteration')['Cost'].sum().reset_index()
    total_cost['Country'] = 'Total'
    total_cost = total_cost.sort_values('Iteration')
    total_cost['CostDelta'] = total_cost['Cost'].diff()

    # Combine for plotting
    df_plot = pd.concat([df_all, total_cost], ignore_index=True)
    countries = df_all['Country'].unique().tolist() + ['Total']

    import plotly.graph_objects as go

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=["Total Cost per Country and Overall", "Cost Variation (Delta) per Iteration"]
    )

    # Line plot for cost
    for country in countries:
        subset = df_plot[df_plot['Country'] == country]
        fig.add_trace(go.Scatter(
            x=subset['Iteration'],
            y=subset['Cost'],
            mode='lines+markers',
            name=f"{country} Cost",
            legendgroup=country,
            showlegend=True if country != 'Total' else False,
            line=dict(width=2)
        ), row=1, col=1)
    # Add 'Total' to legend
    subset = df_plot[df_plot['Country'] == 'Total']
    fig.add_trace(go.Scatter(
        x=subset['Iteration'],
        y=subset['Cost'],
        mode='lines+markers',
        name="Total Cost",
        legendgroup='Total',
        showlegend=True,
        line=dict(width=3, color='black', dash='dash')
    ), row=1, col=1)

    # Bar plot for cost delta
    for country in countries:
        subset = df_plot[df_plot['Country'] == country]
        fig.add_trace(go.Bar(
            x=subset['Iteration'],
            y=subset['CostDelta'],
            name=f"{country} ΔCost",
            legendgroup=country,
            showlegend=False if country != 'Total' else True,
            opacity=0.6 if country != 'Total' else 1.0
        ), row=2, col=1)
    # Add 'Total' to legend for delta
    subset = df_plot[df_plot['Country'] == 'Total']
    fig.add_trace(go.Bar(
        x=subset['Iteration'],
        y=subset['CostDelta'],
        name="Total ΔCost",
        legendgroup='Total',
        showlegend=True,
        marker_color='black',
        opacity=1.0
    ), row=2, col=1)

    fig.update_layout(
        width=width,
        height=height,
        title="Total Cost and Cost Variation per Country and Overall Across Iterations",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    fig.update_yaxes(title_text="Cost", row=1, col=1)
    fig.update_yaxes(title_text="ΔCost (Change)", row=2, col=1)
    fig.show()