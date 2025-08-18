import pandas as pd
import os
from datetime import datetime, timedelta
import pypsa
import re
from snakemake import snakemake


class EnergyAgentClass:
    def __init__(self, country, extra_name, logger, year, timeslice, index_time, yearly_split, demand, opts, original_demand, config):
        self.name = country
        self.extra_name = extra_name
        self.logger = logger
        self.year = year
        self.timeslice = timeslice
        self.index_time = index_time
        self.yearly_split = yearly_split
        self.demand = demand
        self.opts = opts
        self.original_demand = original_demand
        self.config = config
        self.logger.debug(f"EnergyAgentClass initialized for {country}")

        self.run_name = f"{self.year}_{self.timeslice}_{self.name}"

    def build_demand_profile(self):
        self.logger.debug(f"Building demand profile for {self.name} in year {self.year} and timeslice {self.timeslice}")
        self.run_pypsa_snakemake(
            work_dir=f"./pypsa_earth",
            targets=["build_demand_profiles"],
            config={
                    "countries": [self.name],
                    "enable": {
                        "download_osm_data":"true",
                        "not_from_demand_profiles": "true"
                    },
                    "run": {"name": f"{self.run_name}___{self.extra_name}"},
                    "scenario": {"planning_horizons": [self.year]},
                    "snapshots": self.calculate_snapshots(),
                    "costs": {"year": self.year},
                    "convergence_factor": 1.0,  
                    "demand": None,  # Exchange value for the agent
                },
            cores=4,
            dryrun=False,
        )

        return self.get_demand_profile()
    
    def adjust_emission_cap(self):
        if 'eletricity' in self.config:
            if 'co2limit' in self.config['eletricity']:
                self.config['eletricity']['co2limit'] *= self.demand / self.original_demand  # Adjust cap by demand ratio
                return 
            if 'automatic_emission_base_year' in self.config['eletricity']:
                new_opts = []
                for o in self.config['scenario']['opts']:
                    if 'Co2L' in o:
                        m = re.findall(r"Co2L([0-9]*\.?[0-9]+)", o)
                        o = f"Co2L{float(m[0]) * self.demand / self.original_demand}"
                    new_opts.append(o)
                self.config['scenario']['opts'] = new_opts
                self.opts['opts'] = new_opts

    def solve(self, scale, complete=True):
        self.adjust_emission_cap()
        if complete:
            self.run_pypsa_snakemake(
                work_dir=f"./pypsa_earth",
                targets=["solve_all_networks"],
                config=self.merge_configs({
                    "countries": [self.name],
                    "enable": {
                        "download_osm_data":"true",
                        #"retrieve_cost_data":"true",
                        "not_from_demand_profiles": "true"
                    }, 
                    "run": {"name": f"{self.run_name}___{self.extra_name}"},
                    "scenario": {"planning_horizons": [self.year]},
                    "snapshots": self.calculate_snapshots(),
                    "load_options": {
                         "prediction_year": self.year 
                         },
                    #"costs": {"year": self.year},
                    "convergence_factor": scale,  
                    "demand": self.demand,  # Exchange value for the agent
                },self.config),
                cores=4,
                dryrun=False,
            )
        else:
            self.run_pypsa_snakemake(
                work_dir=f"./pypsa_earth",
                targets=["solve_all_networks"],
                config=self.merge_configs({
                    "countries": [self.name],
                    "enable": {"not_from_demand_profiles": "true"}, 
                    "run": {"name": f"{self.run_name}___{self.extra_name}"},
                    "scenario": {"planning_horizons": [self.year]},
                    "snapshots": self.calculate_snapshots(),
                    "load_options": {
                        "prediction_year": self.year 
                         },
                    #"costs": {"year": self.year},
                    "convergence_factor": scale,  
                    "demand": self.demand,  # Demand value for the agent
                }, self.config),
                cores=4,
                dryrun=False,
            )

        return self.process_results()
    
    def run_pypsa_snakemake(
        self,
        work_dir: str,
        targets: list,
        config: dict = dict(),
        cores: int = 1,
        dryrun: bool = False,
    ):
        success = snakemake(
            snakefile="pypsa_earth/Snakefile",
            workdir=work_dir,
            config=config,
            targets=targets,
            cores=cores,
            use_conda=True,
            conda_prefix="pypsa-earth-model", 
            conda_base_path="/opt/miniconda3", 
            printreason=True,
            printshellcmds=True,
            dryrun=dryrun,
        )
        if not success:
            raise RuntimeError(f"Snakemake run failed for {self.name} in timeslice {self.timeslice}")
        print("✅ Workflow completed successfully.")

    def merge_configs(self, config1, config2):
        merged = config1.copy()
        merged.update(config2)
        return merged

    def calculate_snapshots(self):
        self.logger.debug(f"Calculating snapshots for {self.name} in year {self.year} and timeslice {self.timeslice}")

        days = 365 * self.yearly_split

        first_start_date = datetime(2013, 1, 1)
        start_date = first_start_date + timedelta(days=round(days * self.index_time))
        end_date = first_start_date + timedelta(days=round(days*(self.index_time+1)))
        # Cap the end date at 2014-01-01
        if end_date > datetime(2014, 1, 1):
            end_date = datetime(2014, 1, 1)

        return {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
        }

    def get_demand_profile(self):
        self.logger.debug(f"Getting demand profile for {self.name} in year {self.year} and timeslice {self.timeslice}")
        demand_profiles_path = f"./pypsa_earth/resources/{self.run_name}___{self.extra_name}/demand_profiles.csv"
        if not os.path.exists(demand_profiles_path):
            raise FileNotFoundError(f"Demand profiles file {demand_profiles_path} does not exist.")
        df = pd.read_csv(demand_profiles_path, index_col=0)
        original_demand = df.sum().sum()
        return original_demand

    def process_results(self):

        solved_network_path = f"./pypsa_earth/results/{self.run_name}___{self.extra_name}/networks/elec_s{self.opts['simpl'][0]}_{self.opts['clusters'][0]}_ec_l{self.opts['ll'][0]}_{self.opts['opts'][0]}.nc"
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
                "country": self.name,
                "Installed Capacity": row.get("Installed Capacity", 0),
                "Supply": row.get("Supply", 0),
                "Capital Expenditure": row.get("Capital Expenditure", 0),
                "Operational Expenditure": row.get("Operational Expenditure", 0),
                "Capacity Factor": row.get("Capacity Factor", 0),
                "Curtailment": row.get("Curtailment", 0),
            })
        tech_df = pd.DataFrame(rows, columns=['technology','country', 'Installed Capacity', 'Supply', 'Capital Expenditure', 'Operational Expenditure', 'Capacity Factor', 'Curtailment'])

        return total_cost, tech_df
