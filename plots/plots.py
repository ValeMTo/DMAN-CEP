import re
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pandas as pd
import os

SAVING_PATH = "./plots/images"

def try_int(x):
    try:
        return int(x)
    except Exception:
        return x

def timeslice_sort_key(ts):
    match = re.match(r"[MW](\d+)", str(ts))
    return int(match.group(1)) if match else ts

def get_years_and_timeslices_from_dataframes(dataframes):
    years = sorted(dataframes.keys())
    if not years:
        return [], []
    # Get timeslices from the first year
    first_year = years[0]
    timeslices = sorted(dataframes[first_year].keys())
    return years, timeslices

def extract_k_max_per_timeslice(timeslices, dataframes, YEAR):
    k_max_per_timeslice = {}
    for ts in timeslices:
        if ts in dataframes[YEAR]:
            k_values = [ts_data['k'] for ts_data in dataframes[YEAR][ts].values() if 'k' in ts_data]
        if k_values:
            k_max_per_timeslice[ts] = max(k_values)
        else:
            k_max_per_timeslice[ts] = None  # or np.nan if you prefer

    # Sort timeslices and corresponding k_max values using timeslice_sort_key
    sorted_items = sorted(k_max_per_timeslice.items(), key=lambda item: timeslice_sort_key(item[0]))
    #x_vals = [item[0] for item in sorted_items]
    #y_vals = [item[1] for item in sorted_items]

    return sorted_items


def plot_convergence_k(yearly_dataframes, monthly_dataframes, weekly_dataframes, height=400, width=1200):
    years_yearly, yearly_timeslices = get_years_and_timeslices_from_dataframes(yearly_dataframes)
    years_monthly, monthly_timeslices = get_years_and_timeslices_from_dataframes(monthly_dataframes)
    years_weekly, weekly_timeslices = get_years_and_timeslices_from_dataframes(weekly_dataframes)

    years = set(years_yearly) | set(years_monthly) | set(years_weekly)

    for year in years:

        # Extract k_max values
        yearly_k = extract_k_max_per_timeslice(yearly_timeslices, yearly_dataframes, year)
        monthly_k = extract_k_max_per_timeslice(monthly_timeslices, monthly_dataframes, year)
        weekly_k = extract_k_max_per_timeslice(weekly_timeslices, weekly_dataframes, year)

        # Prepare x and y values
        yearly_x, yearly_y = zip(*yearly_k) if yearly_k else ([], [])
        monthly_x, monthly_y = zip(*monthly_k) if monthly_k else ([], [])
        weekly_x, weekly_y = zip(*weekly_k) if weekly_k else ([], [])

        yearly_y = [y + 1 for y in yearly_y]
        monthly_y = [y + 1 for y in monthly_y]
        weekly_y = [y + 1 for y in weekly_y]

        # Create subplots with reduced space and shorter titles
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("Yearly", "Monthly", "Weekly"),
            shared_yaxes=True,
            column_widths=[1/13, 3/13, 9/13],
            horizontal_spacing=0.03  # reduce space between subplots
        )

        def get_color(y_values):
            # If any value in y_values is >= 100, use red, else black
            return ['red' if y is not None and y >= 100 else 'black' for y in y_values]

        fig.add_trace(go.Scatter(
            x=yearly_x,
            y=yearly_y,
            mode='lines+markers',
            name='Yearly',
            line=dict(color='black'),
            marker=dict(color=get_color(yearly_y))
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=monthly_x,
            y=monthly_y,
            mode='lines+markers',
            name='Monthly',
            line=dict(color='black'),
            marker=dict(color=get_color(monthly_y))
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=weekly_x,
            y=weekly_y,
            mode='lines+markers',
            name='Weekly',
            line=dict(color='black'),
            marker=dict(color=get_color(weekly_y))
        ), row=1, col=3)

        fig.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40),  # increase top margin for annotation
            width=width,
            height=height,
        )
        fig.update_layout(
            xaxis=dict(domain=[0, 0.13]),   # reduce yearly width
            xaxis2=dict(domain=[0.16, 0.40]),
            xaxis3=dict(domain=[0.43, 1.0])
        )

        fig.update_yaxes(title_text="k_max", row=1, col=1)

        # Remove default subplot titles
        fig.layout.annotations = []

        # Add centered subplot titles manually (shorter, more descriptive)
        fig.add_annotation(
            text="Year",
            x=0.065, y=1.08, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16), xanchor="center"
        )
        fig.add_annotation(
            text="Month",
            x=0.28, y=1.08, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16), xanchor="center"
        )
        fig.add_annotation(
            text="Week",
            x=0.72, y=1.08, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16), xanchor="center"
        )

        fig.update_layout(showlegend=False)

        fig.write_html(f"{SAVING_PATH}/convergence_k_year_{year}.html", include_plotlyjs='cdn')


def plot_demand_transmission(data, width=1200, height=900, timeslice=None, year=None):

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
    # Use the same y-axis range for all subplots    #
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
            name='Production',
            marker_color='blue', showlegend=(i == 0),
            offsetgroup='demand', legendgroup='Demand'
        ), row=row, col=col, secondary_y=False)

        fig.add_trace(go.Bar(
            x=subset['Iteration'], y=subset['Export'],
            name='Export',
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
            x=subset['Iteration'], y=subset['CostPerUnitWithExchange'],
            mode='lines+markers', name='Cost per Unit (with Exchange)',
            line=dict(color='red', dash='dash'),
            showlegend=(i == 0), legendgroup='CostPerUnitWithExchange'
        ), row=row, col=col, secondary_y=True)

    fig.update_layout(
        height=height, width=width,
        barmode='overlay',
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    #fig.update_yaxes(title_text="Energy", secondary_y=False)
    #fig.update_yaxes(title_text="Cost per Unit", secondary_y=True)

    fig.update_yaxes(title_text="MWh", row=2, col=1)
    fig.update_yaxes(title_text="$/MWh", secondary_y=True, row=2, col=4)

    os.makedirs(f"{SAVING_PATH}/convergence", exist_ok=True)
    fig.write_html(
        f"{SAVING_PATH}/convergence/exchange_over_iterations_{timeslice}_{year}.html",
        include_plotlyjs='cdn'
    )



def plot_last_iteration_timeslices(
    dataframes_yearly,
    dataframes_monthly,
    dataframes_weekly,
    year=None,
    width=1400,
    height_per_country=320,
    max_iteration=99,
    countries=None
):
    # ---------- helpers
    def try_int(x):
        try:
            return int(x)
        except Exception:
            return x

    def pick_year(dy, dm, dw, year):
        all_years = set()
        for d in (dy, dm, dw):
            if isinstance(d, dict):
                all_years.update([y for y in d.keys() if isinstance(y, (int, str))])
        if not all_years:
            raise ValueError("No years found in the provided dataframes.")
        all_years_int = [try_int(y) for y in all_years]
        if year is None:
            # default: max available year
            chosen = max(all_years_int)
        else:
            chosen = try_int(year)
            if chosen not in all_years_int:
                raise ValueError(f"Requested year {year} not found. Available: {sorted(all_years_int)}")
        return int(chosen)

    def last_iteration_block(droot, year, timescale_key):
        """
        For 'Y': return the dict for the last iteration under droot[year]['Y'] (should be a single 'Y' key).
        For 'M' or 'W': return a dict mapping timeslice (e.g., 'M1', 'M2', ...) to the last iteration's block for each timeslice.
        """
        if not droot or year not in droot:
            return None
        year_block = droot[year]
        if timescale_key == 'Y':
            if 'Y' not in year_block:
                return None
            iterations_block = year_block['Y']
            if not iterations_block:
                return None
            last_iter_key = sorted(iterations_block.keys(), key=try_int)[-1]
            return iterations_block[last_iter_key]
        elif timescale_key in ('M', 'W'):
            # Collect all timeslices for this timescale
            ts_prefix = timescale_key
            ts_blocks = {}
            for ts_key, ts_val in year_block.items():
                if isinstance(ts_key, str) and ts_key.startswith(ts_prefix):
                    # Each ts_val is a dict of iterations
                    if not ts_val:
                        ts_blocks[ts_key] = None
                        continue
                    last_iter_key = sorted(ts_val.keys(), key=try_int)[-1]
                    ts_blocks[ts_key] = ts_val[last_iter_key]
            return ts_blocks
        else:
            return None

    def collect_countries_from_df(df):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return []
        return list(df.index)

    def compute_imp_exp_for_country(tdf, country):
        """Imports: sum of exchange > 0 where end_country == country.
           Exports: sum of exchange > 0 where start_country == country.
        """
        if tdf is None or not isinstance(tdf, pd.DataFrame) or tdf.empty:
            return 0.0, 0.0
        exp = tdf.loc[(tdf['start_country'] == country) & (tdf['exchange'] > 0), 'exchange'].sum()
        imp = tdf.loc[(tdf['end_country'] == country) & (tdf['exchange'] > 0), 'exchange'].sum()
        imp = 0.0 if pd.isna(imp) else float(imp)
        exp = 0.0 if pd.isna(exp) else float(exp)
        return imp, exp

    def demand0(df, country):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return 0.0
        if 'demand_0' in df.columns:
            val = df.at[country, 'demand_0'] if country in df.index else 0.0
            return 0.0 if pd.isna(val) else float(val)
        # fallback if needed
        for c in ('demand', 'demand0'):
            if c in df.columns:
                val = df.at[country, c] if country in df.index else 0.0
                return 0.0 if pd.isna(val) else float(val)
        return 0.0

    # ---------- collect last-iteration data blocks for the chosen year
    Y = pick_year(dataframes_yearly, dataframes_monthly, dataframes_weekly, year)

    y_last = last_iteration_block(dataframes_yearly, Y, 'Y')
    m_last = last_iteration_block(dataframes_monthly, Y, 'M')
    w_last = last_iteration_block(dataframes_weekly, Y, 'W')

    # Yearly: expect a single df + tdf
    y_df   = y_last.get('df') if isinstance(y_last, dict) else None
    y_tdf  = y_last.get('transmission_df') if isinstance(y_last, dict) else None
    y_k = y_last.get('k') if isinstance(y_last, dict) else None
    # Monthly/Weekly can be stored per timeslice inside the iteration block. We’ll try to normalize:
    # We accept either:
    #   m_last = {'M1': {'df':..., 'transmission_df':...}, 'M2': {...}, ...}
    # or       = list/iterable of dicts with a 'timeslice' key
    def normalize_timeslice_map(block, prefix):
        ts_map = {}
        if block is None:
            return ts_map
        # dict keyed by timeslice
        if isinstance(block, dict) and any(k.startswith(prefix) for k in block.keys() if isinstance(k, str)):
            for ts, payload in block.items():
                if isinstance(payload, dict) and 'df' in payload:
                    ts_map[ts] = {
                        'df': payload.get('df'),
                        'tdf': payload.get('transmission_df'),
                        'k': payload.get('k')
                    }
        else:
            # maybe a flat dict with many entries having 'timeslice' key
            if isinstance(block, dict):
                candidates = block.values()
            elif isinstance(block, (list, tuple)):
                candidates = block
            else:
                candidates = []

            for item in candidates:
                if isinstance(item, dict):
                    ts = item.get('timeslice')
                    if isinstance(ts, str) and ts.startswith(prefix):
                        ts_map[ts] = {
                            'df': item.get('df'),
                            'tdf': item.get('transmission_df'),
                            'k': item.get('k')
                        }
        return ts_map

    m_ts_map = normalize_timeslice_map(m_last, 'M')
    w_ts_map = normalize_timeslice_map(w_last, 'W')

    # Ensure complete sets of timeslices with None if missing
    months = [f"M{i}" for i in range(1, 13)]
    weeks  = [f"W{i}" for i in range(1, 53)]
    for m in months:
        m_ts_map.setdefault(m, {'df': None, 'tdf': None, 'k': None})
    for w in weeks:
        w_ts_map.setdefault(w, {'df': None, 'tdf': None, 'k': None})

    # ---------- use the provided countries list for row order and titles
    if not countries:
        raise ValueError("No countries found across yearly/monthly/weekly last iterations.")
        # ---------- prep the subplot grid with custom column widths (1:12:52)        #
        rows = len(countries)

    rows = len(countries)
    cols = 3  # Y | M | W
    total = 1 + 12 + 52
    col_widths = [1/total, 12/total, 52/total]

    # Titles: one per subplot, in row-major order (Y, M, W for each country)
    subplot_titles = []
    for i, c in enumerate(countries):
        if i == 0:
            subplot_titles.extend([
                f"<span style='font-size:14px'>Yearly</span>",
                f"<span style='font-size:14px'>Monthly</span>",
                f"<span style='font-size:14px'>Weekly</span>"
            ])
        else:
            subplot_titles.extend(["", "", ""])

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.02,
        column_widths=col_widths
    )

        # Add country names as y-axis titles for the first (yearly) subplot in each row
    yaxis_titles = {i + 1: c for i, c in enumerate(countries)}
    for row, country in yaxis_titles.items():
        fig.update_yaxes(title_text=country, row=row, col=1, title_standoff=10, automargin=True)

    # colors for the three series
    demand_color = "#1f77b4"  # default Plotly blue
    import_color = "#2ca02c"  # green
    export_color = "#ff7f0e"  # orange
    # ---------- build per-country panels (one row per country, columns: yearly | monthly | weekly)
    for i, country in enumerate(countries):
        row = i + 1

        # ---- YEARLY PANEL (single timeslice 'Y')
        y_x = ['Y']
        y_dem = [demand0(y_df, country)]
        y_imp, y_exp = compute_imp_exp_for_country(y_tdf, country)
        y_imports = [y_imp]
        y_exports = [y_exp]

        yearly_subset = pd.DataFrame({
            'Iteration': y_x,
            'Demand': y_dem,
            'Import': y_imports,
            'Export': y_exports
        })

        fig.add_trace(go.Bar(
            x=yearly_subset['Iteration'], y=yearly_subset['Demand'],
            name='Production',
            marker_color='blue', showlegend=(row == 1),
            offsetgroup='demand', legendgroup='Demand'
        ), row=row, col=1, secondary_y=False)

        fig.add_trace(go.Bar(
            x=yearly_subset['Iteration'], y=yearly_subset['Export'],
            name='Export',
            marker=dict(color='blue', pattern=dict(shape='\\')),
            offsetgroup='demand', base=0,
            showlegend=(row == 1), legendgroup='Export'
        ), row=row, col=1, secondary_y=False)

        fig.add_trace(go.Bar(
            x=yearly_subset['Iteration'], y=yearly_subset['Import'],
            name='Import',
            marker_color='green', opacity=0.8,
            offsetgroup='import', base=yearly_subset['Demand'],
            showlegend=(row == 1), legendgroup='Import'
        ), row=row, col=1, secondary_y=False)

        if y_k >= max_iteration:
            fig.add_vrect(
                x0=-0.5, x1=0.5,
                fillcolor="gray", opacity=0.5, line_width=0,
                row=row, col=1
            )

        # ---- MONTHLY PANEL (M1..M12)
        m_x = months
        m_dem, m_imp, m_exp = [], [], []
        for m in months:
            df_m = m_ts_map[m]['df']
            tdf_m = m_ts_map[m]['tdf']
            m_dem.append(demand0(df_m, country))
            mi, me = compute_imp_exp_for_country(tdf_m, country)
            m_imp.append(mi)
            m_exp.append(me)

        monthly_subset = pd.DataFrame({
            'Iteration': m_x,
            'Demand': m_dem,
            'Import': m_imp,
            'Export': m_exp
        })

        fig.add_trace(go.Bar(
            x=monthly_subset['Iteration'], y=monthly_subset['Demand'],
            name='Production',
            marker_color='blue', showlegend=False,
            offsetgroup='demand', legendgroup='Demand'
        ), row=row, col=2, secondary_y=False)

        fig.add_trace(go.Bar(
            x=monthly_subset['Iteration'], y=monthly_subset['Export'],
            name='Export',
            marker=dict(color='blue', pattern=dict(shape='\\')),
            offsetgroup='demand', base=0,
            showlegend=False, legendgroup='Export'
        ), row=row, col=2, secondary_y=False)

        fig.add_trace(go.Bar(
            x=monthly_subset['Iteration'], y=monthly_subset['Import'],
            name='Import',
            marker_color='green', opacity=0.8,
            offsetgroup='import', base=monthly_subset['Demand'],
            showlegend=False, legendgroup='Import'
        ), row=row, col=2, secondary_y=False)

        # Add gray rectangle if monthly iteration exceeds max_iteration
        for idx, m in enumerate(months):
            m_k = m_ts_map[m]['k']
            if m_k is not None and m_k >= max_iteration:
                fig.add_vrect(
                    x0=idx - 0.5, x1=idx + 0.5,
                    fillcolor="gray", opacity=0.5, line_width=0,
                    row=row, col=2
                )

        # ---- WEEKLY PANEL (W1..W52)
        w_x = weeks
        w_dem, w_imp, w_exp = [], [], []
        for w in weeks:
            df_w = w_ts_map[w]['df']
            tdf_w = w_ts_map[w]['tdf']
            w_dem.append(demand0(df_w, country))
            wi, we = compute_imp_exp_for_country(tdf_w, country)
            w_imp.append(wi)
            w_exp.append(we)

        weekly_subset = pd.DataFrame({
            'Iteration': w_x,
            'Demand': w_dem,
            'Import': w_imp,
            'Export': w_exp
        })

        fig.add_trace(go.Bar(
            x=weekly_subset['Iteration'], y=weekly_subset['Demand'],
            name='Production',
            marker_color='blue', showlegend=False,
            offsetgroup='demand', legendgroup='Demand'
        ), row=row, col=3, secondary_y=False)

        fig.add_trace(go.Bar(
            x=weekly_subset['Iteration'], y=weekly_subset['Export'],
            name='Export',
            marker=dict(color='blue', pattern=dict(shape='\\')),
            offsetgroup='demand', base=0,
            showlegend=False, legendgroup='Export'
        ), row=row, col=3, secondary_y=False)

        fig.add_trace(go.Bar(
            x=weekly_subset['Iteration'], y=weekly_subset['Import'],
            name='Import',
            marker_color='green', opacity=0.8,
            offsetgroup='import', base=weekly_subset['Demand'],
            showlegend=False, legendgroup='Import'
        ), row=row, col=3, secondary_y=False)

        # Add gray rectangle if weekly iteration exceeds max_iteration
        for idx, w in enumerate(weeks):
            w_k = w_ts_map[w]['k']
            if w_k is not None and w_k >= max_iteration:
                fig.add_vrect(
                    x0=idx - 0.5, x1=idx + 0.5,
                    y0=0, y1=1.31,
                    fillcolor="gray", opacity=0.5, line_width=0,
                    row=row, col=3
                )

    # Update layout for less space between columns and shared x-axes
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        width=width,
        height=max(500, rows * height_per_country),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.8),
        margin=dict(t=90, b=60, l=40, r=20)
    )

    # Reduce horizontal spacing between columns
    #fig.update_layout(horizontal_spacing=0.01)

    # Shared x-axes for each column
    for col in range(1, 4):
        fig.update_xaxes(matches=f'x{col}', row=1, col=col)

    # Only show x-axis labels at the last row
    for row in range(1, rows + 1):
        showticklabels = (row == rows)
        for col in range(1, 4):
            fig.update_xaxes(showticklabels=showticklabels, row=row, col=col)
            if showticklabels:
                if col == 1:
                    fig.update_xaxes(row=row, col=col, tickangle=35)
                elif col == 2:
                    fig.update_xaxes(row=row, col=col, tickangle=45)
                elif col == 3:
                    fig.update_xaxes(row=row, col=col, tickangle=45)
            else:
                fig.update_xaxes(title_text=None, row=row, col=col)
                fig.update_xaxes(title_text=None, row=row, col=col)

    # Make weekly axis labels a bit lighter to avoid clutter
    fig.update_xaxes(tickangle=0, tickfont=dict(size=9), row=1, col=3)

    # Use to_image for interactive environments, or write_image for file output.
    # If write_image is too slow or times out, try reducing figure size or complexity.
    try:
        # Try using kaleido (default) for static image export, but fallback to interactive HTML if the image looks wrong.
        #fig.write_image(f"{SAVING_PATH}/overall_exchange_{year}.png", width=width, scale=1)
        # Optionally, also save as interactive HTML for full fidelity
        fig.write_html(f"{SAVING_PATH}/overall_exchange_{year}.html", include_plotlyjs='cdn')
    except Exception as e:
        print(f"Could not save image due to: {e}")
        # As a fallback, show the figure in an interactive window (if running in a notebook or GUI)
        try:
            fig.show()
        except Exception as show_e:
            print(f"Could not display figure interactively: {show_e}")

    