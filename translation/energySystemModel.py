from translation.parsers.configParser import ConfigParserClass
from translation.parsers.dataParser import dataParserClass
from translation.xmlGenerator import XMLGeneratorClass
from translation.energyAgentModel import EnergyAgentClass
from translation.transmissionModel import TransmissionModelClass
from translation.energyReader import EnergyReader
from translation.worker import ThreadManager
import pandas as pd
import logging
import os
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from collections import defaultdict
from threading import Thread
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

def build_profile_args(args):
    country, logger, year, t, time_resolution, yearly_split, opts = args
    return (
        country,
        EnergyAgentClass(
            country=country,
            logger=logger,
            year=year,
            timeslice=t,
            index_time=time_resolution.index(t),
            yearly_split=yearly_split,
            demand=None,
            opts=opts,
        ).build_demand_profile()
    )

def solve_country_optimization_wrapper(args):
            (country, time, year, logger, index_time, yearly_split, demand, opts, delta_marginal_cost, first_optimization) = args
            energy_country_class = EnergyAgentClass(
                country=country,
                logger=logger,
                year=year,
                timeslice=time,
                index_time=index_time,
                yearly_split=yearly_split,
                demand=demand,
                opts=opts,
            )
            # This is a placeholder; you may need to adapt how results are stored/returned
            return (
                country,
                energy_country_class.solve(scale=1.0 + delta_marginal_cost, complete=not first_optimization),
                energy_country_class.solve(scale=1.0 - delta_marginal_cost, complete=False),
                energy_country_class.solve(scale=1.0, complete=False)
            )
class EnergyModelClass:
    def __init__(self):
        self.config_parser = ConfigParserClass(file_path='config.yaml')
        self.logger = self.create_logger(*self.config_parser.get_log_info())
        self.config_parser.set_logger(self.logger)

        base_output_path = self.config_parser.get_output_file_path()
        output_path = base_output_path
        k = 1
        while os.path.exists(output_path):
            output_path = f"{base_output_path}+{k}"
            k += 1
        self.config_parser.set_output_file_path(output_path)

        self.countries = self.config_parser.get_countries()
        self.name = self.config_parser.get_problem_name()
        self.time_resolution = self.config_parser.get_annual_time_resolution()
        self.years = self.config_parser.get_years()
        self.opts = self.get_model_options()

        self.results_df = None
        self.marginal_costs_df = None
        self.country_data_per_time = {}

        self.data_parser = dataParserClass(logger=self.logger)

        self.transmission_data_capacity = self.data_parser.get_transmission_data(self.countries, yearly=True)
        self.transmission_data_max_capacity = self.transmission_data_capacity.copy()
        self.transmission_data = pd.DataFrame()
        self.yearly_split = self.build_yearly_split_map()
        self.demand_map = {}

        self.max_iteration = self.config_parser.get_max_iteration()
        self.delta_marginal_cost = self.config_parser.get_delta_marginal_cost()
        self.marginal_cost_tolerance = self.config_parser.get_marginal_cost_tolerance()

        self.convergence_iteration = 0
        self.logger.info("Energy model initialized")


    def get_model_options(self):
        with open('pypsa_earth/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config["scenario"]

    def create_logger(self, log_level, log_file):
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(
            filename=log_file,
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)

    def build_yearly_split_map(self):
        self.logger.info(f"Building yearly split map")
        yearly_split =  (365 / len(self.time_resolution)) / 365

        self.transmission_data_max_capacity['capacity'] = self.transmission_data_max_capacity['capacity'] * yearly_split
        return yearly_split

    def build_demand_map(self):
        self.logger.info("Building demand map")
        demand_map = {}
        for t in self.time_resolution:
            demand_map[t] = {}
            for c in self.countries:
                demand_map[t][c] = {
                    'demand': None,
                }
        self.demand_map = demand_map

    def solve(self):
        self.logger.info("Solving the energy model")
        for year in self.years:
            self.solve_year(year)
        self.logger.info("Energy model solved")
        #self.results_df.to_csv(os.path.join(self.config_parser.get_output_file_path(), 'results.csv'), index=False)
    
    def solve_year(self, year):
        self.logger.info(f"Solving the energy model for {year}")
        self.build_demand_map()
        for t in tqdm(self.time_resolution, desc=f"Solving year {year}"):
            self.calculate_demand_profiles(t, year)
            self.first_optimization = {country: False for country in self.countries}
            for k in tqdm(range(self.max_iteration), desc=f"Solving timeslice {t} for year {year}"):
                self.reader = self.prepare_reader(k=k, t=t, year=year)
                marginal_costs_df = self.solve_internal_problem(t, year)
                if self.check_convergence(marginal_costs_df):
                    self.logger.info(f"Convergence reached for time {t} and year {year} after {k} iterations")
                    self.solve_transmission_problem(t, year)
                    self.reader.save(folder=os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{t}"), transmission_data=self.transmission_data)
                    break
                self.solve_transmission_problem(t, year)
                self.reader.save(folder=os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{t}"), transmission_data=self.transmission_data)
            #self.update_data(t, year)
            if k == self.max_iteration:
                self.logger.warning(f"Maximum iterations reached for time {t} and year {year}")
            marginal_costs_df = None
            self.reset()
    
    def reset(self):
        self.transmission_data_max_capacity = self.transmission_data_capacity.copy()  # Reset transmission data for the next time slice
        self.transmission_data = pd.DataFrame() # Reset transmission data for the next time slice

    def prepare_reader(self, k, t, year):
        self.logger.debug(f"Preparing XML reader for iteration {k}, time {t}, year {year}")

        return EnergyReader(
            k=k,
            t=t,
            year=year,
            countries=self.countries,
            delta=self.delta_marginal_cost
        )

    def solve_internal_problem(self, t, year):
        self.logger.info(f"Calculating marginal costs for time {t} and year {year}")

        if not os.path.exists(os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{t}/internal/problems")):
            os.makedirs(os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{t}/internal/problems"))

        args_list = [
            (
                country,
                t,
                year,
                self.logger,
                self.time_resolution.index(t),
                self.yearly_split,
                self.demand_map[t][country]['demand'],
                self.opts,
                self.delta_marginal_cost,
                self.first_optimization[country]
            )
            for country in self.countries
        ]

        for country in self.countries:
            self.reader.set_demand(
                self.demand_map[t][country]['demand'],
                demand_type='0',
                country=country
            )

        # Use solve_country_optimization_wrapper in parallel, collect results, and store in parent method
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(args_list))) as executor:
            futures = {executor.submit(solve_country_optimization_wrapper, args): args[0] for args in args_list}
            for future in as_completed(futures):
                country, plus_result, minus_result, zero_result = future.result()
                self.reader.store(*plus_result, demand_type='+', country=country)
                self.reader.store(*minus_result, demand_type='-', country=country)
                self.reader.store(*zero_result, demand_type='0', country=country)
                self.first_optimization[country] = True

        marginal_costs_df = self.calculate_marginal_costs(t, year)
        self.logger.info(f"Marginal costs calculated for time {t} and year {year}")
        return marginal_costs_df

    def calculate_marginal_costs(self, t, year):
        self.logger.info(f"Calculating marginal costs for time {t} and year {year}")

        self.reader.calculate_marginal_costs()
        output_df = self.reader.get_total_cost_table()

        return output_df

    def calculate_demand_profiles(self, t, year):
        self.logger.info(f"Calculating demand profiles for time {t} and year {year}")

        args_list = [
            (country, self.logger, year, t, self.time_resolution, self.yearly_split, self.opts)
            for country in self.countries
        ]
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(args_list))) as executor:
            futures = {executor.submit(build_profile_args, args): args[0] for args in args_list}
            for future in as_completed(futures):
                country, profile = future.result()
                self.demand_map[t][country]['demand'] = profile
                self.demand_map[t][country]['marginal_demand'] = profile * self.delta_marginal_cost

    def solve_transmission_problem(self, time, year):
        self.logger.info(f"Solving transmission problem for time {time}")

        transmission_solver = TransmissionModelClass(
            countries=self.countries,
            data=self.transmission_data_max_capacity.copy(),
            delta_demand_map=self.demand_map[time].copy(),
            marginal_costs_df=self.marginal_costs_df[['MC_import', 'MC_export']],
            cost_transmission_line=self.config_parser.get_cost_transmission_line(),
            logger=self.logger,
            xml_file_path=os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{time}/transmission/problems"),
            expansion_enabled=self.config_parser.get_expansion_enabled(),
        )

        self.reader.set_transmission_outputs(transmission_solver.solve())
        self.update_demand(time)
        self.logger.debug(f"Transmission problem solved for time {time}")

    def check_convergence(self, marginal_costs_df):
        self.logger.info("Checking convergence of marginal costs")
        if self.marginal_costs_df is None:
            self.marginal_costs_df = marginal_costs_df
            return False
        distance = (self.marginal_costs_df[['MC_import', 'MC_export']] - marginal_costs_df[['MC_import', 'MC_export']]).abs().max().max() 
        self.logger.debug(f"Maximum Distance between marginal costs: {distance}")
        if distance < self.marginal_cost_tolerance:
            self.logger.info(f"Convergence reached with distance {distance} < tolerance {self.marginal_cost_tolerance}")
            if self.convergence_iteration < self.config_parser.get_min_convergence_iterations():
                self.convergence_iteration += 1
                self.marginal_costs_df = marginal_costs_df
                return False
            else:
                self.convergence_iteration = 0
                return True
        self.convergence_iteration = 0
        self.marginal_costs_df = marginal_costs_df
        return False
    
    def update_demand(self, time):
        self.logger.info("Updating demand based on transmission problem results")

        transmission_outputs = self.reader.get_transmission_outputs()
        if not transmission_outputs.empty:
            for start_country, exchange in transmission_outputs.groupby('start_country')['exchange'].sum().items():
                self.demand_map[time][start_country]['demand'] += exchange

        for c in self.countries:    
            self.demand_map[time][c]['marginal_demand'] = self.demand_map[time][c]['demand'] * self.delta_marginal_cost

        if not transmission_outputs.empty:
            for start_country, end_country, exchange, price in transmission_outputs.itertuples(index=False):
                if start_country != end_country:
                    self.transmission_data_max_capacity.loc[
                        (self.transmission_data_max_capacity['start_country'] == start_country) & 
                        (self.transmission_data_max_capacity['end_country'] == end_country), 
                        'capacity'
                    ] -= abs(exchange)
        if self.transmission_data.empty and not transmission_outputs.empty:
            self.transmission_data = transmission_outputs
        else:
            if not transmission_outputs.empty:
                self.transmission_data = pd.concat([self.transmission_data, transmission_outputs], ignore_index=True)
                self.transmission_data = self.transmission_data.groupby(['start_country', 'end_country'], as_index=False).sum()
                self.logger.info("Exchange updated based on transmission problem results")


    def update_data(self, t, year):
        self.logger.debug(f"Updating data for year {year} and timeslice {t}")

        outputs_df = self.reader.get_internal_outputs()
        outputs_df = outputs_df[outputs_df['demand_type'] == '0']
        outputs_df = outputs_df[['technology', 'capacity', ]]

        if self.results_df is None:
            self.results_df = self.data[['TECHNOLOGY', 'MIN_INSTALLED_CAPACITY']].copy()
            self.results_df.rename(columns={'MIN_INSTALLED_CAPACITY': 'baseline'}, inplace=True)

        if f'capacity_{year}' not in self.results_df.columns:
            self.results_df = self.results_df.merge(
                outputs_df, 
                left_on='TECHNOLOGY',
                right_on='technology',
                how='left'
            )
            self.results_df.rename(columns={'capacity': f'capacity_{year}'}, inplace=True)
        else:
            merged_df = self.results_df.merge(
                outputs_df, 
                left_on='TECHNOLOGY',
                right_on='technology',
                how='left'
            )
            self.results_df[f'capacity_{year}'] = merged_df[[f'capacity_{year}', 'capacity']].max(axis=1)
            self.results_df.drop(columns=['capacity'], inplace=True, errors='ignore')
        self.results_df.drop(columns=['technology'], inplace=True, errors='ignore')
        self.results_df['COUNTRY'] = self.results_df['TECHNOLOGY'].apply(lambda x: x[:2])

        if self.config_parser.get_expansion_enabled():
            raise NotImplementedError("Expansion is not implemented yet.")

    def solve_country_optimization(self, country, time, year, logger, index_time, yearly_split, demand, opts, delta_marginal_cost, first_optimization):

        energy_country_class = EnergyAgentClass(
            country=country,
            logger=logger,
            year=year,
            timeslice=time,
            index_time=index_time,
            yearly_split=yearly_split,
            demand=demand,
            opts=opts,
        )
        self.reader.store(*energy_country_class.solve(scale=1.0 + delta_marginal_cost, complete= not first_optimization), demand_type='+', country=country)
        self.reader.store(*energy_country_class.solve(scale=1.0 - delta_marginal_cost, complete=False), demand_type='-', country=country)
        self.reader.store(*energy_country_class.solve(scale=1.0, complete=False), demand_type='0', country=country)
        self.first_optimization[country] = True
        

