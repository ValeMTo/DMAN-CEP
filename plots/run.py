import yaml
from pathlib import Path
import os
import pickle
import pandas as pd
import plots as pt
from tqdm import tqdm

def read_parameters():
    params_path = Path(__file__).parent / "parameters.yaml"

    with open(params_path, "r") as f:
        parameters = yaml.safe_load(f)

    return parameters['countries'], parameters['yearly'], parameters['monthly'], parameters['weekly'], parameters['centralized']


def get_metadata_from_folder(folder_path):
    # Get years and timeslices from the folder structure
    folder_path = os.path.join(folder_path, 'DCOP')
    years = [int(year) for year in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, year))]
    years.sort()
    # Assume all years have the same timeslices, get from the first year
    first_year_folder = os.path.join(folder_path, str(years[0]))
    timeslices = [ts for ts in os.listdir(first_year_folder) if os.path.isdir(os.path.join(first_year_folder, ts))]
    timeslices.sort()

    return years, timeslices

def get_data_from_folder(folder_path):

    years, timeslices = get_metadata_from_folder(folder_path)

    dataframes = {}
    folder_path = os.path.join(folder_path, 'DCOP')
    for year in years:
        dataframes[year] = {}
        year_folder = os.path.join(folder_path, str(year))
        for timeslice in timeslices:
            dataframes[year][timeslice] = {}
            timeslice_folder = os.path.join(year_folder, timeslice)
            if not os.path.exists(timeslice_folder):
                raise FileNotFoundError(f"No data for year {year} and timeslice {timeslice} in folder {timeslice_folder}")
            else:
                for file_k in os.listdir(timeslice_folder):
                    if file_k.endswith('.pkl'):
                        dataframes[year][timeslice][file_k.split('_')[1]] = {}
                        file_path = os.path.join(timeslice_folder, file_k)
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    dataframes[year][timeslice][file_k.split('_')[1]][key] = value
                            else:
                                raise ValueError(f"Data in {file_k} is not a dict")
                            f.close()

    for year in years:
        for timeslice in timeslices:
            ts_data = dataframes[year][timeslice]
            k_keys_sorted = sorted(ts_data.keys(), key=lambda x: int(x))
            for k in k_keys_sorted:
                k_data = ts_data[k]
                k_data['df']['exchange_cost'] = k_data['df']['total_cost_after_exchange'] - k_data['df']['0']
                dataframes[year][timeslice][k] = k_data

            cost_with_exchange = {country: 0 for country in dataframes[year][timeslice][k]['df'].index}
            k_keys_sorted = sorted(ts_data.keys(), key=lambda x: int(x))
            for k in k_keys_sorted:
                df = dataframes[year][timeslice][k]['df']
                for country in df.index:
                    cost = df.loc[country, '0'] if '0' in df.columns else None
                    cost_with_exchange[country] += df.loc[country, 'exchange_cost'] if 'exchange_cost' in df.columns else 0
                    df.loc[country, 'total_cost_after_exchange'] = cost + cost_with_exchange[country] * 0.95
                
    return dataframes

def get_centralized_data(solved_network_path):
    import pypsa
    if not os.path.exists(solved_network_path):
        raise FileNotFoundError(f"Network file {solved_network_path} does not exist.")
        
    n = pypsa.Network(solved_network_path)

    statistics = n.statistics()
    total_cost = statistics["Capital Expenditure"].sum() + statistics["Operational Expenditure"].sum()

    rows = []
    for (component, technology), row in n.statistics().iterrows():
        if component not in ["Generator", "Link", "StorageUnit", "Store"]:
            continue
        rows.append({
            "technology": technology,
            "country": "SAPP",
            "Installed Capacity": row.get("Installed Capacity", 0),
            "Supply": row.get("Supply", 0),
            "Capital Expenditure": row.get("Capital Expenditure", 0),
            "Operational Expenditure": row.get("Operational Expenditure", 0),
            "Capacity Factor": row.get("Capacity Factor", 0),
            "Curtailment": row.get("Curtailment", 0),
        })
    tech_df = pd.DataFrame(rows, columns=['technology','country', 'Installed Capacity', 'Supply', 'Capital Expenditure', 'Operational Expenditure', 'Capacity Factor', 'Curtailment'])

    dataframes_centralized = {2030: {'C': {'0': {'k': 0, 't': 'C', 'year': 2030, 'tech_df': tech_df}}}}
    return dataframes_centralized

if __name__ == "__main__":

    countries, yearly, monthly, weekly, centralized = read_parameters()

    dataframes_yearly = get_data_from_folder(yearly['folder'])
    dataframes_monthly = get_data_from_folder(monthly['folder'])
    dataframes_weekly = get_data_from_folder(weekly['folder'])
    dataframes_centralized = get_centralized_data(centralized['network_path'])

    pt.plot_convergence_k(dataframes_yearly, dataframes_monthly, dataframes_weekly, height=333, width=2000)
    pt.plot_last_iteration_timeslices(
        dataframes_yearly,
        dataframes_monthly,
        dataframes_weekly,
        year=2030,                
        width=2000,
        height_per_country=100,
        max_iteration=99,
        countries=countries,
    )

    for year in dataframes_yearly.keys():
        for timeslice in tqdm(dataframes_yearly[year].keys(), desc=f"Yearly {year} timeslices"):
            pt.plot_demand_transmission(
                dataframes_yearly[year][timeslice],
                timeslice=timeslice,
                year=year,
                width=2000,
                height=700
            )

    for year in dataframes_monthly.keys():
        for timeslice in tqdm(dataframes_monthly[year].keys(), desc=f"Monthly {year} timeslices"):
            pt.plot_demand_transmission(
                dataframes_monthly[year][timeslice],
                timeslice=timeslice,
                year=year,
                width=2000,
                height=700
            )

    for year in dataframes_weekly.keys():
        for timeslice in tqdm(dataframes_weekly[year].keys(), desc=f"Weekly {year} timeslices"):
            pt.plot_demand_transmission(
                dataframes_weekly[year][timeslice],
                timeslice=timeslice,
                year=year,
                width=2000,
                height=700
            )

    

    
