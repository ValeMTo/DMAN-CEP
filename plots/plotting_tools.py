import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
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
            total_demand = demand
            imports = imports if not pd.isna(imports) else 0.0
            exports = exports if not pd.isna(exports) else 0.0
            cost = df.at[country, '0'] if '0' in df.columns else None
            cost_per_unit = cost / total_demand if total_demand and cost is not None else None
            cost_per_unit_with_exchange = df.at[country, 'total_cost_after_exchange'] / (total_demand + imports) if total_demand and 'total_cost_after_exchange' in df.columns else None
            records.append({
                'Iteration': try_int(k),
                'Country': country,
                'Demand': demand,
                'Import': imports,
                'Export': exports,
                'CostPerUnit': cost_per_unit,
                'CostPerUnitWithExchange': cost_per_unit_with_exchange,
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

        fig.add_trace(go.Scatter(
            x=subset['Iteration'], y=subset['CostPerUnitWithExchange'],
            mode='lines+markers', name='Cost per Unit (with Exchange)',
            line=dict(color='red', dash='dash'),
            showlegend=(i == 0), legendgroup='CostPerUnitWithExchange'
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

    return df_tx

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



def plot_mc_and_marginals(data_dict, width=1600, height=900, exchange_cost=True):
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
    
    cost_column = "0"
    cost_name = "Cost"
    if exchange_cost:
        cost_column = "total_cost_after_exchange"
        cost_name = "Cost + Exchange"

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
                cost_name: df.loc[country, cost_column] if cost_column in df.columns else None,
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
            x=subset['Iteration'], y=subset[cost_name],
            mode='lines+markers', name=cost_name,
            legendgroup=cost_name, showlegend=(i == 0),
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


def plot_final_costs_by_country(data_dict, width=900, height=600):
    """
    Plot the final iteration cost per country and the total sum, with and without exchange.

    Parameters:
    - data_dict: dictionary with structure like dataframes[year][timeslice]
    - width: plot width in px
    - height: plot height in px
    """
    def try_int(x):
        try:
            return int(x)
        except Exception:
            return x

    # Find the last iteration key
    iterations = sorted(data_dict.keys(), key=try_int)
    last_k = iterations[-1]
    df = data_dict[last_k]['df']

    countries = df.index.tolist()
    cost_no_ex = [df.loc[c, '0'] if '0' in df.columns else None for c in countries]
    cost_with_ex = [df.loc[c, 'total_cost_after_exchange'] if 'total_cost_after_exchange' in df.columns else None for c in countries]

    total_no_ex = sum([v for v in cost_no_ex if v is not None])
    total_with_ex = sum([v for v in cost_with_ex if v is not None])

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=countries,
        y=cost_no_ex,
        name='Cost (No Exchange)',
        marker_color='blue'
    ))
    fig.add_trace(go.Bar(
        x=countries,
        y=cost_with_ex,
        name='Cost (With Exchange)',
        marker_color='orange'
    ))

    # Add total as annotation
    fig.add_trace(go.Bar(
        x=['Total'],
        y=[total_no_ex],
        name='Total (No Exchange)',
        marker_color='blue',
        opacity=0.5
    ))
    fig.add_trace(go.Bar(
        x=['Total'],
        y=[total_with_ex],
        name='Total (With Exchange)',
        marker_color='orange',
        opacity=0.5
    ))

    fig.update_layout(
        width=width,
        height=height,
        title="Final Iteration Cost per Country and Total (With/Without Exchange)",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    fig.update_yaxes(title_text="Cost")
    fig.show()

def plot_total_cost_per_timeslice_last_iter(timeslice_dict_by_year, width=900, height=600):
    """
    Plot total cost per timeslice (sum of all countries) at the last iteration.

    Parameters:
    - timeslice_dict_by_year: dict of {timeslice: {k: {'df': ...}}}
    - width, height: figure dimensions
    """
    def timeslice_sort_key(ts):
        match = re.match(r"[MW](\d+)", str(ts))
        return int(match.group(1)) if match else ts

    records = []
    for timeslice, iterations in timeslice_dict_by_year.items():
        if not iterations:
            continue
        last_k = max(iterations.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        df = iterations[last_k]['df']
        total_cost = df['0'].sum() if '0' in df.columns else None
        total_cost_ex = df['total_cost_after_exchange'].sum() if 'total_cost_after_exchange' in df.columns else None
        records.append({
            'Timeslice': timeslice,
            'TotalCost': total_cost,
            'TotalCostWithExchange': total_cost_ex
        })

    df_total = pd.DataFrame(records)
    # Sort timeslices numerically (e.g., w1, w2, ..., w10, ...)
    df_total['Timeslice'] = pd.Categorical(
        df_total['Timeslice'],
        categories=sorted(df_total['Timeslice'].unique(), key=timeslice_sort_key),
        ordered=True
    )
    df_total = df_total.sort_values('Timeslice')

    average_total_cost = df_total['TotalCost'].sum() / len(df_total)

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_total['Timeslice'],
        y=df_total['TotalCost'],
        name='Total Cost (No Exchange)',
        marker_color='blue'
    ))
    fig.add_trace(go.Bar(
        x=df_total['Timeslice'],
        y=df_total['TotalCostWithExchange'],
        name='Total Cost (With Exchange)',
        marker_color='orange'
    ))

    # Plot average total cost as a horizontal line
    fig.add_shape(
        type="line",
        x0=df_total['Timeslice'].min(),
        x1=df_total['Timeslice'].max(),
        y0=average_total_cost,
        y1=average_total_cost,
        line=dict(color="red", dash="dash"),
        xref="x",
        yref="y"
    )
    fig.add_trace(go.Scatter(
        x=[df_total['Timeslice'].min(), df_total['Timeslice'].max()],
        y=[average_total_cost, average_total_cost],
        mode='lines',
        name='Average Total Cost',
        line=dict(color='red', dash='dash'),
        showlegend=True
    ))

    fig.update_layout(
        width=width,
        height=height,
        title="Total Cost per Timeslice (Last Iteration, Sum of All Countries)",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    fig.update_yaxes(title_text="Total Cost")
    fig.update_xaxes(title_text="Timeslice")
    fig.show()

    return df_total

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
            cost_with_exchange = df.loc[country, 'total_cost_after_exchange'] if 'total_cost_after_exchange' in df.columns else None
            records.append({
                'Iteration': try_int(k),
                'Country': country,
                'Cost': cost,
                'CostWithExchange': cost_with_exchange,
            })

    df_all = pd.DataFrame(records)
    df_all = df_all.sort_values(['Country', 'Iteration'])

    # Compute per-country cost delta
    df_all['CostDelta'] = df_all.groupby('Country')['Cost'].diff()
    df_countries_all = df_all.copy()

    # Compute total cost and its delta per iteration (both with and without exchange)
    total_cost = df_all.groupby('Iteration')['Cost'].sum().reset_index()
    total_cost['Country'] = 'Total'
    total_cost = total_cost.sort_values('Iteration')
    total_cost['CostDelta'] = total_cost['Cost'].diff()

    total_cost_with_ex = df_all.groupby('Iteration')['CostWithExchange'].sum().reset_index()
    total_cost_with_ex['Country'] = 'TotalWithExchange'
    total_cost_with_ex = total_cost_with_ex.sort_values('Iteration')
    total_cost_with_ex['CostDelta'] = total_cost_with_ex['CostWithExchange'].diff()

    # Combine for plotting
    df_plot = pd.concat([df_all, total_cost, total_cost_with_ex], ignore_index=True)
    countries = df_all['Country'].unique().tolist() + ['Total', 'TotalWithExchange']

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
        if country == 'Total':
            fig.add_trace(go.Scatter(
                x=subset['Iteration'],
                y=subset['Cost'],
                mode='lines+markers',
                name="Total Cost (No Exchange)",
                legendgroup='Total',
                showlegend=True,
                line=dict(width=3, color='black', dash='solid')
            ), row=1, col=1)
        elif country == 'TotalWithExchange':
            fig.add_trace(go.Scatter(
                x=subset['Iteration'],
                y=subset['CostWithExchange'],
                mode='lines+markers',
                name="Total Cost (With Exchange)",
                legendgroup='TotalWithExchange',
                showlegend=True,
                line=dict(width=3, color='orange', dash='dash')
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=subset['Iteration'],
                y=subset['Cost'],
                mode='lines+markers',
                name=f"{country} Cost",
                legendgroup=country,
                showlegend=True,
                line=dict(width=2)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=subset['Iteration'],
                y=subset['CostWithExchange'],
                mode='lines+markers',
                name=f"{country} CostWithExchange",
                legendgroup=country,
                showlegend=True,
                line=dict(width=2, dash='dot')
            ), row=1, col=1)

    # Bar plot for cost delta
    for country in countries:
        subset = df_plot[df_plot['Country'] == country]
        if country == 'Total':
            fig.add_trace(go.Bar(
                x=subset['Iteration'],
                y=subset['CostDelta'],
                name="Total ΔCost (No Exchange)",
                legendgroup='Total',
                showlegend=True,
                marker_color='black',
                opacity=1.0
            ), row=2, col=1)
        elif country == 'TotalWithExchange':
            fig.add_trace(go.Bar(
                x=subset['Iteration'],
                y=subset['CostDelta'],
                name="Total ΔCost (With Exchange)",
                legendgroup='TotalWithExchange',
                showlegend=True,
                marker_color='orange',
                opacity=1.0
            ), row=2, col=1)
        else:
            fig.add_trace(go.Bar(
                x=subset['Iteration'],
                y=subset['CostDelta'],
                name=f"{country} ΔCost",
                legendgroup=country,
                showlegend=False,
                opacity=0.6
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

    return df_countries_all

def plot_net_demand_across_timeslices(timeslice_dict_by_year, width=1600, height=900):
    """
    Plot a single histogram per country across timeslices:
    net demand = demand_0 + imports - exports

    Also plots the difference between demand_0 at iteration 0 and net_demand.

    Parameters:
    - timeslice_dict_by_year: dataframes[year], i.e., dict of {timeslice: {k: {'df': ..., ...}}}
    - width, height: figure dimensions
    """

    def timeslice_sort_key(ts):
        match = re.match(r"[MW](\d+)", str(ts))
        return int(match.group(1)) if match else ts

    records = []
    diff_records = []

    for timeslice, iterations in timeslice_dict_by_year.items():
        if not iterations:
            continue
        # Get last iteration for net_demand
        last_k = max(iterations.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        df_last = iterations[last_k]['df']
        tdf_last = iterations[last_k].get('transmission_df', pd.DataFrame())

        # Get iteration 0 for demand_0
        iter0_k = min(iterations.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        df_iter0 = iterations[iter0_k]['df']

        for country in df_last.index:
            demand = df_last.loc[country, 'demand_0'] if 'demand_0' in df_last.columns else 0.0
            imports = -tdf_last.loc[(tdf_last['start_country'] == country) & (tdf_last['exchange'] < 0), 'exchange'].sum() if not tdf_last.empty else 0.0
            exports = tdf_last.loc[(tdf_last['start_country'] == country) & (tdf_last['exchange'] > 0), 'exchange'].sum() if not tdf_last.empty else 0.0
            imports = imports if not pd.isna(imports) else 0.0
            exports = exports if not pd.isna(exports) else 0.0
            net_demand = demand + imports - exports

            # demand_0 at iteration 0
            demand0_iter0 = df_iter0.loc[country, 'demand_0'] if 'demand_0' in df_iter0.columns else 0.0
            diff = demand0_iter0 - net_demand

            records.append({
                'Timeslice': timeslice,
                'Country': country,
                'NetDemand': net_demand
            })
            diff_records.append({
                'Timeslice': timeslice,
                'Country': country,
                'Demand0_Iter0': demand0_iter0,
                'NetDemand': net_demand,
                'Difference': diff
            })

    df_all = pd.DataFrame(records)
    df_all['Timeslice'] = pd.Categorical(
        df_all['Timeslice'],
        categories=sorted(df_all['Timeslice'].unique(), key=timeslice_sort_key),
        ordered=True
    )
    countries = df_all['Country'].unique()
    cols = 4
    rows = (len(countries) + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=countries,
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.12, horizontal_spacing=0.05,
        specs=[[{} for _ in range(cols)] for _ in range(rows)]
    )

    for i, country in enumerate(countries):
        row = i // cols + 1
        col = i % cols + 1
        subset = df_all[df_all['Country'] == country].sort_values('Timeslice')

        fig.add_trace(go.Bar(
            x=subset['Timeslice'], y=subset['NetDemand'],
            name='Net Demand',
            marker_color='purple',
            legendgroup='NetDemand', showlegend=(i == 0),
            offsetgroup='netdemand',
            base=0
        ), row=row, col=col)

    fig.update_layout(
        height=height, width=width,
        title_text="Net Demand (demand_0 + imports - exports) per Timeslice and Country",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )

    fig.update_yaxes(title_text="Net Demand")
    fig.show()

    # --- Plot difference between demand_0 at iteration 0 and net_demand ---
    df_diff = pd.DataFrame(diff_records)
    df_diff['Timeslice'] = pd.Categorical(
        df_diff['Timeslice'],
        categories=sorted(df_diff['Timeslice'].unique(), key=timeslice_sort_key),
        ordered=True
    )

    fig2 = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=countries,
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.12, horizontal_spacing=0.05,
        specs=[[{} for _ in range(cols)] for _ in range(rows)]
    )

    for i, country in enumerate(countries):
        row = i // cols + 1
        col = i % cols + 1
        subset = df_diff[df_diff['Country'] == country].sort_values('Timeslice')

        fig2.add_trace(go.Bar(
            x=subset['Timeslice'], y=subset['Difference'],
            name='Demand0_Iter0 - NetDemand',
            marker_color='orange',
            legendgroup='Diff', showlegend=(i == 0),
            offsetgroup='diff',
            base=0
        ), row=row, col=col)

    fig2.update_layout(
        height=height, width=width,
        title_text="Difference: demand_0 at Iteration 0 minus Net Demand (final) per Timeslice and Country",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )

    fig2.update_yaxes(title_text="Demand0_Iter0 - NetDemand")
    fig2.show()

    return df_all, df_diff

def plot_demand_across_timeslices(timeslice_dict_by_year, width=1600, height=900):
    """
    Plot demand_0, imports, and exports across timeslices using only the last iteration per timeslice.
    Exports are stacked on top of demand with a pattern, imports are stacked on top of demand in green.

    Parameters:
    - timeslice_dict_by_year: dataframes[year], i.e., dict of {timeslice: {k: {'df': ..., ...}}}
    - width, height: figure dimensions
    """

    def timeslice_sort_key(ts):
        match = re.match(r"[MW](\d+)", str(ts))
        return int(match.group(1)) if match else ts

    records = []

    for timeslice, iterations in timeslice_dict_by_year.items():
        if not iterations:
            continue
        last_k = max(iterations.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        df = iterations[last_k]['df']
        tdf = iterations[last_k].get('transmission_df', pd.DataFrame())

        for country in df.index:
            demand_0 = df.loc[country, 'demand_0'] 
            # Imports: sum of negative exchanges where end_country == country
            imports = -tdf.loc[(tdf['start_country'] == country) & (tdf['exchange'] < 0), 'exchange'].sum() if not tdf.empty else 0.0
            # Exports: sum of positive exchanges where start_country == country
            exports = tdf.loc[(tdf['start_country'] == country) & (tdf['exchange'] > 0), 'exchange'].sum() if not tdf.empty else 0.0
            imports = imports if not pd.isna(imports) else 0.0
            exports = exports if not pd.isna(exports) else 0.0
            records.append({
                'Timeslice': timeslice,
                'Country': country,
                'Demand': demand_0,
                'Import': imports,
                'Export': exports
            })

    df_all = pd.DataFrame(records)
    df_all['Timeslice'] = pd.Categorical(
        df_all['Timeslice'],
        categories=sorted(df_all['Timeslice'].unique(), key=timeslice_sort_key),
        ordered=True
    )
    countries = df_all['Country'].unique()
    cols = 4
    rows = (len(countries) + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=countries,
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.12, horizontal_spacing=0.05,
        specs=[[{} for _ in range(cols)] for _ in range(rows)]
    )

    for i, country in enumerate(countries):
        row = i // cols + 1
        col = i % cols + 1
        subset = df_all[df_all['Country'] == country].sort_values('Timeslice')

        # Calculate export portions

        # Demand bar (full demand)
        fig.add_trace(go.Bar(
            x=subset['Timeslice'], y=subset['Demand'],
            name='Demand',
            marker_color='blue',
            legendgroup='Demand', showlegend=(i == 0),
            offsetgroup='demand',
            base=0
        ), row=row, col=col)

        # Import stacked on top of demand, in green
        fig.add_trace(go.Bar(
            x=subset['Timeslice'], y=subset['Import'],
            name='Import',
            marker_color='green',
            legendgroup='Import', showlegend=(i == 0),
            offsetgroup='import',
            base=subset['Demand']
        ), row=row, col=col)

        # Overlay export portion (patterned blue) up to demand
        fig.add_trace(go.Bar(
            x=subset['Timeslice'], y=subset['Export'],
            name='Export (in Demand)',
            marker=dict(color='rgba(0,0,255,0.2)', pattern=dict(shape='\\')),
            legendgroup='Export', showlegend=(i == 0),
            offsetgroup='export',
            base=0,
            opacity=0.7
        ), row=row, col=col)

    fig.update_layout(
        height=height, width=width,
        title_text="Final Iteration Demand, Export (patterned), and Import (green) per Timeslice and Country",
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )

    fig.update_yaxes(title_text="Demand / Import / Export")
    fig.show()

    return df_all

def plot_country_results_across_timeslices(timeslice_dict_by_year, width=1600, height=900, cost_name='0'):
    """
    Plot cost, marginal values and MCs across timeslices using only the last iteration per timeslice.
    
    Parameters:
    - timeslice_dict_by_year: dataframes[year], i.e., dict of {timeslice: {k: {'df': ..., ...}}}
    - width, height: figure dimensions
    """

    def timeslice_sort_key(ts):
        # Extract numeric part after 'm' or 'w' (e.g., m1 -> 1, w2 -> 2, m10 -> 10)
        match = re.match(r"[MW](\d+)", str(ts))
        return int(match.group(1)) if match else ts

    records = []

    for timeslice, iterations in timeslice_dict_by_year.items():
        if not iterations:
            continue
        last_k = max(iterations.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        df = iterations[last_k]['df']

        for country in df.index:
            records.append({
                'Timeslice': timeslice,
                'Country': country,
                'Cost': df.loc[country, cost_name],
                'MarginalDemand': df.loc[country, 'marginal_demand'],
                'CostPerUnit': df.loc[country, 'cost_per_unit'],
                'MC_import': df.loc[country, 'MC_import'],
                'MC_export': df.loc[country, 'MC_export']
            })

    df_all = pd.DataFrame(records)
    # Sort timeslices numerically
    df_all['Timeslice'] = pd.Categorical(
        df_all['Timeslice'],
        categories=sorted(df_all['Timeslice'].unique(), key=timeslice_sort_key),
        ordered=True
    )
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
        subset = df_all[df_all['Country'] == country].sort_values('Timeslice')

        # Bar for cost
        fig.add_trace(go.Bar(
            x=subset['Timeslice'], y=subset['Cost'],
            name='Cost',
            marker_color='black',
            legendgroup='Cost', showlegend=(i == 0),
        ), row=row, col=col, secondary_y=False)

        # Marginal Demand
        fig.add_trace(go.Scatter(
            x=subset['Timeslice'], y=subset['MarginalDemand'],
            mode='markers+lines', name='Marginal Demand',
            marker=dict(color='green', symbol='circle'), line=dict(dash='solid'),
            legendgroup='Marginal Demand', showlegend=(i == 0)
        ), row=row, col=col, secondary_y=True)

        # Cost per unit
        fig.add_trace(go.Scatter(
            x=subset['Timeslice'], y=subset['CostPerUnit'],
            mode='markers+lines', name='Cost per Unit',
            marker=dict(color='blue', symbol='square'), line=dict(dash='dot'),
            legendgroup='Cost per Unit', showlegend=(i == 0)
        ), row=row, col=col, secondary_y=True)

        # MC Import
        fig.add_trace(go.Scatter(
            x=subset['Timeslice'], y=subset['MC_import'],
            mode='markers+lines', name='MC Import',
            marker=dict(color='orange', symbol='diamond'), line=dict(dash='dash'),
            legendgroup='MC Import', showlegend=(i == 0)
        ), row=row, col=col, secondary_y=True)

        # MC Export
        fig.add_trace(go.Scatter(
            x=subset['Timeslice'], y=subset['MC_export'],
            mode='markers+lines', name='MC Export',
            marker=dict(color='red', symbol='x'), line=dict(dash='dashdot'),
            legendgroup='MC Export', showlegend=(i == 0)
        ), row=row, col=col, secondary_y=True)

    fig.update_layout(
        height=height, width=width,
        title_text="Final Iteration Results per Timeslice — Cost and Marginals per Country",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )

    fig.update_yaxes(title_text="Total Cost", secondary_y=False)
    fig.update_yaxes(title_text="Marginal Values", secondary_y=True)

    fig.show()

def plot_cost_difference_across_timeslices(timeslice_dict_by_year, width=1600, height=900):
    """
    Plot the difference between cost '0' and 'total_cost_after_exchange' across timeslices for each country.

    Parameters:
    - timeslice_dict_by_year: dict of {timeslice: {k: {'df': ...}}}
    - width, height: figure dimensions
    """
    def timeslice_sort_key(ts):
        match = re.match(r"[MW](\d+)", str(ts))
        return int(match.group(1)) if match else ts

    records = []
    for timeslice, iterations in timeslice_dict_by_year.items():
        if not iterations:
            continue
        last_k =  max(iterations.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        df = iterations[last_k]['df']
        for country in df.index:
            cost_0 = df.loc[country, '0'] if '0' in df.columns else None
            cost_ex = df.loc[country, 'total_cost_after_exchange'] if 'total_cost_after_exchange' in df.columns else None
            diff = cost_ex - cost_0 if cost_0 is not None and cost_ex is not None else None
            records.append({
                'Timeslice': timeslice,
                'Country': country,
                'Cost_0': cost_0,
                'CostAfterExchange': cost_ex,
                'CostDifference': diff
            })

    df_all = pd.DataFrame(records)
    df_all['Timeslice'] = pd.Categorical(
        df_all['Timeslice'],
        categories=sorted(df_all['Timeslice'].unique(), key=timeslice_sort_key),
        ordered=True
    )
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
        subset = df_all[df_all['Country'] == country].sort_values('Timeslice')

        fig.add_trace(go.Bar(
            x=subset['Timeslice'], y=subset['Cost_0'],
            name='Cost 0',
            marker_color='blue',
            legendgroup='Cost 0', showlegend=(i == 0),
        ), row=row, col=col, secondary_y=False)

        fig.add_trace(go.Bar(
            x=subset['Timeslice'], y=subset['CostAfterExchange'],
            name='Cost After Exchange',
            marker_color='orange',
            legendgroup='Cost After Exchange', showlegend=(i == 0),
        ), row=row, col=col, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=subset['Timeslice'], y=subset['CostDifference'],
            mode='markers+lines', name='Cost Difference',
            marker=dict(color='red', symbol='circle'), line=dict(dash='dash'),
            legendgroup='Cost Difference', showlegend=(i == 0)
        ), row=row, col=col, secondary_y=True)

    fig.update_layout(
        height=height, width=width,
        title_text="Cost 0, Cost After Exchange, and Their Difference per Timeslice and Country",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    fig.update_yaxes(title_text="Cost", secondary_y=False)
    fig.update_yaxes(title_text="Cost Difference", secondary_y=True)
    fig.show()

def plot_installed_capacity_per_timeslice_by_technology(capacity_summary_df, width=1200, height=500):
    """
    Plot stacked bar of installed capacity per timeslice by technology, with timeslices ordered numerically.

    Parameters:
    - capacity_summary_df: DataFrame with columns ['timeslice', 'technology', 'Installed Capacity']
    - width: plot width in px
    - height: plot height in px
    """

    # Unpack tuple if necessary
    if isinstance(capacity_summary_df, tuple):
        capacity_summary_df = capacity_summary_df[0]

    def timeslice_sort_key(ts):
        match = re.match(r"[MW](\d+)", str(ts))
        return int(match.group(1)) if match else ts

    # Ensure timeslice is ordered numerically (e.g., w1, w2, ..., w10, ...)
    timeslices = sorted(capacity_summary_df['timeslice'].unique(), key=timeslice_sort_key)
    capacity_summary_df['timeslice'] = pd.Categorical(
        capacity_summary_df['timeslice'],
        categories=timeslices,
        ordered=True
    )
    capacity_summary_df = capacity_summary_df.sort_values('timeslice')
    # Remove technologies that are always zero for all timeslices
    tech_totals = capacity_summary_df.groupby('technology')['Installed Capacity'].sum()
    nonzero_techs = tech_totals[tech_totals != 0].index
    capacity_summary_df = capacity_summary_df[capacity_summary_df['technology'].isin(nonzero_techs)]

    fig = px.bar(
        capacity_summary_df,
        x='timeslice',
        y='Installed Capacity',
        color='technology',
        labels={'timeslice': 'Timeslice', 'Installed Capacity': 'Installed Capacity'},
        title='Installed Capacity per Timeslice by Technology',
        width=width,
        height=height,
        barmode='stack'
    )
    fig.show()

def plot_annual_capacity_per_technology(data_dict, width=1000, height=700):
    """
    Plot installed capacity (max per year) per technology and country.

    Parameters:
    - data_dict: dataframes[year], i.e., {timeslice: {k: {'tech_df': ...}}}
    - width, height: dimensions of the figure
    Returns:
    - capacity_df: aggregated dataframe
    - fig: plotly figure
    """
    tech_records = []

    for timeslice, iterations in data_dict.items():
        if not iterations:
            continue
        last_k =  max(iterations.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        tech_df = iterations[last_k].get('tech_df')
        tech_df['timeslice'] = timeslice

        if tech_df is None or tech_df.empty:
            continue

        # Only keep demand_type == '0'
        tech_df_filtered = tech_df[tech_df['demand_type'] == '0'].copy()
        tech_records.append(tech_df_filtered)

    if not tech_records:
        print("No technology data available with demand_type == '0'.")
        return pd.DataFrame(), None

    df_all = pd.concat(tech_records, ignore_index=True)

    # Remove rows where technology == 'load'
    df_all = df_all[df_all['technology'] != 'load']
    
    # Group and aggregate for capacity (max per tech/timeslice/country over all timeslices)
    capacity_df = df_all.groupby(['technology', 'country'], as_index=False).agg({
        'Installed Capacity': 'max',
        'Supply': 'sum',
        'Capital Expenditure': 'max',
        'Operational Expenditure': 'sum',
        'Capacity Factor': 'mean',
        'Curtailment': 'sum'
    })

    countries = capacity_df['country'].unique()
    cols = 3
    rows = (len(countries) + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=countries,
        shared_xaxes=False, shared_yaxes=True,
        vertical_spacing=0.12, horizontal_spacing=0.05,
        specs=[[{} for _ in range(cols)] for _ in range(rows)]
    )

    for i, country in enumerate(countries):
        row = i // cols + 1
        col = i % cols + 1

        subset_cap = capacity_df[capacity_df['country'] == country]
        fig.add_trace(go.Bar(
            x=subset_cap['technology'],
            y=subset_cap['Installed Capacity'],
            name='Installed Capacity',
            marker_color='blue',
            showlegend=(i == 0),
            legendgroup='Capacity',
            offsetgroup='capacity'
        ), row=row, col=col)

    fig.update_layout(
        height=height, width=width,
        title_text="Installed Capacity (max per year, blue) per Technology and Country",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    fig.update_yaxes(title_text="Capacity (MW or unit)")

    fig.show()

    return df_all

def plot_annual_new_installed_per_technology(data_dict, width=1000, height=700):
    """
    Plot new installed capacity (max per year minus min_capacity) per technology and country.

    Parameters:
    - data_dict: dataframes[year], i.e., {timeslice: {k: {'tech_df': ...}}}
    - width, height: dimensions of the figure
    Returns:
    - new_installed_df: aggregated dataframe
    - fig: plotly figure
    """
    tech_records = []

    for timeslice, iterations in data_dict.items():
        if not iterations:
            continue
        last_k =  max(iterations.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        tech_df = iterations[last_k].get('tech_df')
        if tech_df is None or tech_df.empty:
            continue

        # Only keep demand_type == '0'
        tech_df_filtered = tech_df[tech_df['demand_type'] == '0'].copy()
        tech_records.append(tech_df_filtered)

    if not tech_records:
        print("No technology data available with demand_type == '0'.")
        return pd.DataFrame(), None

    df_all = pd.concat(tech_records, ignore_index=True)

    # Group and aggregate for capacity (max per tech/country over all timeslices)
    agg = df_all.groupby(['technology', 'country'], as_index=False).agg({
        'Installed Capacity': 'max',
        'Supply': 'sum',
        'Capital Expenditure': 'max',
        'Operational Expenditure': 'sum',
        'Capacity Factor': 'mean',
        'Curtailment': 'sum'
    })
    agg['new_installed'] = agg['Installed Capacity'] - agg['min_capacity']

    countries = agg['country'].unique()
    cols = 3
    rows = (len(countries) + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=countries,
        shared_xaxes=False, shared_yaxes=True,
        vertical_spacing=0.12, horizontal_spacing=0.05,
        specs=[[{} for _ in range(cols)] for _ in range(rows)]
    )

    for i, country in enumerate(countries):
        row = i // cols + 1
        col = i % cols + 1

        subset_new = agg[agg['country'] == country]
        fig.add_trace(go.Bar(
            x=subset_new['technology'],
            y=subset_new['new_installed'],
            name='New Installed Capacity',
            marker_color='orange',
            showlegend=(i == 0),
            legendgroup='NewInstalled',
            offsetgroup='newinstalled'
        ), row=row, col=col)

    fig.update_layout(
        height=height, width=width,
        title_text="New Installed Capacity (max - min, orange) per Technology and Country",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    fig.update_yaxes(title_text="New Installed Capacity (MW or unit)")

    fig.show()

    return agg


def plot_annual_activity_stacked(data_dict, width=1200, height=800):
    """
    Plot stacked rate of activity per timeslice and technology for each country.

    Parameters:
    - data_dict: dataframes[year], i.e., {timeslice: {k: {'tech_df': ...}}}
    - width, height: dimensions of the figure
    Returns:
    - activity_df: aggregated dataframe
    - fig: plotly figure
    """
    tech_records = []

    for timeslice, iterations in data_dict.items():
        if not iterations:
            continue
        last_k =  max(iterations.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        tech_df = iterations[last_k].get('tech_df')
        if tech_df is None or tech_df.empty:
            continue

        # Only keep demand_type == '0'
        tech_df_filtered = tech_df[tech_df['demand_type'] == '0'].copy()
        tech_df_filtered['timeslice'] = timeslice
        tech_records.append(tech_df_filtered)

    if not tech_records:
        print("No technology data available with demand_type == '0'.")
        return pd.DataFrame(), None

    df_all = pd.concat(tech_records, ignore_index=True)

    # Group and aggregate for rate of activity (sum per tech/country/timeslice)
    activity_df = df_all.groupby(['technology', 'country', 'timeslice'], as_index=False).agg({
        'rate_activity': 'sum'
    })

    countries = activity_df['country'].unique()
    cols = 3
    rows = (len(countries) + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=countries,
        shared_xaxes=False, shared_yaxes=True,
        vertical_spacing=0.12, horizontal_spacing=0.05,
        specs=[[{} for _ in range(cols)] for _ in range(rows)]
    )

    for i, country in enumerate(countries):
        row = i // cols + 1
        col = i % cols + 1

        subset_act = activity_df[activity_df['country'] == country]
        techs = subset_act['technology'].unique()
        timeslices = sorted(subset_act['timeslice'].unique())
        for j, tech in enumerate(techs):
            tech_data = subset_act[subset_act['technology'] == tech]
            fig.add_trace(go.Bar(
                x=tech_data['timeslice'],
                y=tech_data['rate_activity'],
                name=f'Activity: {tech}',
                legendgroup=f'Activity_{tech}',
                showlegend=(i == 0),
                marker_color=px.colors.qualitative.Plotly[j % len(px.colors.qualitative.Plotly)],
                offsetgroup=tech,
                base=None
            ), row=row, col=col)

    fig.update_layout(
        height=height, width=width,
        title_text="Stacked Rate of Activity per Timeslice and Technology per Country",
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    fig.update_yaxes(title_text="Rate of Activity (stacked, per timeslice)")

    fig.show()

    return activity_df

def plot_capacity_over_time_from_csv(filepath, width=1200, height=700):
    # --- Load and clean ---
    df = pd.read_csv(filepath)
    # Remove rows where all columns except 'TECHNOLOGY' and 'COUNTRY' are NaN
    cols_to_check = [col for col in df.columns if col not in ['TECHNOLOGY', 'COUNTRY']]
    df_clean = df.dropna(axis=0, how='all', subset=cols_to_check)

    # --- Extract year columns ---
    year_cols = [col for col in df_clean.columns if col.startswith("capacity_")]

    # --- Plot 2: Capacity - Baseline, Faceted by Country ---
    df_norm = df_clean.copy()
    df_norm['baseline'] = df_norm['baseline'].fillna(0)
    for col in year_cols:
        df_norm[col] = df_norm[col] - df_norm['baseline']
        # Remove rows where all year columns are zero after normalization
        df_norm = df_norm.loc[~(df_norm[year_cols].abs().sum(axis=1) == 0)]
    
    df_norm = df_norm.drop(columns=['baseline'])

    # Melt the dataframe to long format for plotting
    df_long = df_norm.melt(id_vars=['TECHNOLOGY', 'COUNTRY'], value_vars=year_cols,
                            var_name='Year', value_name='Capacity')

    # Clean up 'Year' column to just the year number
    df_long['Year'] = df_long['Year'].str.replace('capacity_', '').astype(int)

    # Assign colors by technology[2:]
    tech_labels = df_long['TECHNOLOGY'].apply(lambda x: x[2:] if isinstance(x, str) and len(x) > 2 else x)
    unique_tech_labels = tech_labels.unique()
    color_map = {label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                    for i, label in enumerate(unique_tech_labels)}

    countries = df_long['COUNTRY'].unique()
    cols = 3
    rows = (len(countries) + cols - 1) // cols

    if len(countries) == 0 or rows == 0:
        print("No data available to plot capacity over time.")
        return

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=countries,
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.12, horizontal_spacing=0.05,
        specs=[[{} for _ in range(cols)] for _ in range(rows)]
    )

    for i, country in enumerate(countries):
        row = i // cols + 1
        col = i % cols + 1
        subset = df_long[df_long['COUNTRY'] == country]
        for tech in subset['TECHNOLOGY'].unique():
            tech_label = tech[2:] if isinstance(tech, str) and len(tech) > 2 else tech
            tech_data = subset[subset['TECHNOLOGY'] == tech]
            fig.add_trace(go.Scatter(
                x=tech_data['Year'],
                y=tech_data['Capacity'],
                mode='lines+markers',
                name=tech_label,
                legendgroup=tech_label,
                showlegend=(i == 0),
                line=dict(color=color_map[tech_label])
            ), row=row, col=col)

    fig.update_layout(
        height=height, width=width,
        title_text="Installed Capacity Over Time per Technology (Faceted by Country)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    fig.update_yaxes(title_text="Installed Capacity (MW)")
    fig.update_xaxes(title_text="Year")
    fig.show()
