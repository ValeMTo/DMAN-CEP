from translation.parsers.configParser import ConfigParserClass
from translation.parsers.osemosysDataParser import osemosysDataParserClass
from translation.xmlGenerator import XMLGeneratorClass
from translation.energyAgentModel import EnergyAgentClass
from translation.transmissionModel import TransmissionModelClass
from translation.xmlEnergyReader import xmlEnergyReader
from translation.worker import ThreadManager
from deprecated import deprecated
import pandas as pd
import logging
import os
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from collections import defaultdict
from threading import Thread


class EnergyModelClass:
    def __init__(self):
        self.config_parser = ConfigParserClass(file_path='config.yaml')
        self.logger = self.create_logger(*self.config_parser.get_log_info())
        self.config_parser.set_logger(self.logger)
        self.countries = self.config_parser.get_countries()
        self.name = self.config_parser.get_problem_name()
        self.time_resolution = self.config_parser.get_annual_time_resolution()
        self.years = self.config_parser.get_years()

        self.results_df = None
        self.marginal_costs_df = None
        self.demand_map = {}
        self.country_data_per_time = {}

        self.thread_manager = ThreadManager()
        self.data_parser = osemosysDataParserClass(logger = self.logger, file_path=self.config_parser.get_file_path())
        self.thread_manager.run_parallel(
            name=f'data_parser_load_{self.years[0]}',
            func=self.data_parser.load_data,
            args=(self.years[0], self.countries)
        )
        self.transmission_data_capacity = self.data_parser.get_transmission_data(self.countries, yearly=True)
        self.transmission_data_max_capacity = self.transmission_data_capacity.copy()
        self.transmission_data = pd.DataFrame()
        self.xml_generator = XMLGeneratorClass(logger = self.logger)

        self.logger.info("Energy model initialized")
        self.max_iteration = self.config_parser.get_max_iteration()
        self.delta_marginal_cost = self.config_parser.get_delta_marginal_cost()
        self.marginal_cost_tolerance = self.config_parser.get_marginal_cost_tolerance()

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
    
    def build_demand_map(self, year, t): 
        self.logger.info(f"Building demand map for year {year}")
        self.demand_map[t] = {}
        for country in self.countries:
            year_split, demand = self.data_parser.load_demand(year, country, t)
            self.demand_map[t][country] = {
                        'demand': demand,
                        'marginal_demand': demand * self.delta_marginal_cost,
                    }
        self.demand_map[t]['year_split'] = year_split # year_split is the same for all countries for timeslice t
        self.transmission_data_max_capacity['capacity'] = self.transmission_data_capacity['capacity'] * self.demand_map[t]['year_split']

    def solve(self):
        self.logger.info("Solving the energy model")
        i = 0
        while i < len(self.years):
            year = self.years[i]
            self.thread_manager.wait_for(f'data_parser_load_{year}')
            self.data_parser.add_previous_installed_capacity(year, self.results_df)
            if i + 1 < len(self.years):
                self.thread_manager.run_parallel(
                    name=f'data_parser_load_{self.years[i+1]}',
                    func=self.data_parser.load_data,
                    args=(self.years[i+1], self.countries)
                )
            self.solve_year(year)
            i += 1

        self.logger.info("Energy model solved")
    
    def solve_year(self, year):
        self.logger.info(f"Solving the energy model for {year}")
        
        for t in tqdm(self.time_resolution, desc=f"Solving year {year}"):
            self.build_demand_map(year, t)
            self.extract_data(year, t)
            for k in tqdm(range(self.max_iteration), desc=f"Solving timeslice {t} for year {year}"):
                self.reader = self.prepare_reader(k=k, t=t, year=year)
                marginal_costs_df = self.solve_internal_DCOP(t, year)
                if self.check_convergence(marginal_costs_df):
                    self.logger.info(f"Convergence reached for time {t} and year {year} after {k} iterations")
                    self.solve_transmission_problem(t, year)
                    self.reader.save(folder=os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{t}"), transmission_data=self.transmission_data)
                    break
                self.solve_transmission_problem(t, year)
                self.reader.save(folder=os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{t}"), transmission_data=self.transmission_data)
            self.update_data(t, year)
            if k == self.max_iteration:
                self.logger.warning(f"Maximum iterations reached for time {t} and year {year}")
            marginal_costs_df = None
            self.reset()
    
    def reset(self):
        self.transmission_data_max_capacity = self.transmission_data_capacity.copy()  # Reset transmission data for the next time slice
        self.transmission_data = pd.DataFrame() # Reset transmission data for the next time slice

    def prepare_reader(self, k, t, year):
        self.logger.debug(f"Preparing XML reader for iteration {k}, time {t}, year {year}")

        return xmlEnergyReader(
            k=k,
            t=t,
            year=year,
            countries=self.countries,
        )

    def solve_internal_DCOP(self, t, year):
        self.logger.info(f"Calculating marginal costs for time {t} and year {year}")

        if not os.path.exists(os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{t}/internal/problems")):
            os.makedirs(os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{t}/internal/problems"))

        threads = []
        for country in self.countries:
            thread = Thread(target=self.create_internal_DCOP, args=(country, t, year))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        countries_check, output_folder_path = self.solve_folder(os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{t}/internal"))
        marginal_costs_df = self.calculate_marginal_costs(t, year, output_folder_path, countries_check)

        self.logger.info(f"Marginal costs calculated for time {t} and year {year}")
        return marginal_costs_df
    
    def calculate_marginal_costs(self, t, year, output_folder_path, countries_check):
        self.logger.info(f"Calculating marginal costs for time {t} and year {year} at {output_folder_path}")
        data = {
            country: {
            f"-{self.delta_marginal_cost}": None,
            "0": None,
            f"+{self.delta_marginal_cost}": None,
            }
            for country in self.countries
        }

        output_df = self.reader.get_internal_outputs()
        output_df = output_df.merge(self.data, left_on='technology', right_on='TECHNOLOGY', how='left')

        output_df['total_cost'] = (
            output_df['capacity'] * (output_df['CAPITAL_COST'] - output_df['MIN_INSTALLED_CAPACITY']) +
            output_df['rateActivity'] * output_df['VARIABLE_COST'] +
            output_df['FIXED_COST'] * output_df['year_split']
        )

        agg = output_df.groupby(['country', 'demand_type'], as_index=False)['total_cost'].sum()
        pivot = agg.pivot(index='country', columns='demand_type', values='total_cost').fillna(0)

        pivot['marginal_demand'] = [self.demand_map[t][c]['marginal_demand'] for c in pivot.index]

        pivot['MC_import'] = (pivot["0"] - pivot[f"-{self.delta_marginal_cost}"]) / pivot['marginal_demand']
        pivot['MC_export'] = (pivot[f"+{self.delta_marginal_cost}"] - pivot["0"]) / pivot['marginal_demand']
        pivot[['MC_import', 'MC_export']] = pivot[['MC_import', 'MC_export']].clip(lower=0)

        self.reader.set_general_picture(pivot)

        return pivot

    def solve_transmission_problem(self, time, year):
        self.logger.info(f"Solving transmission problem for time {time}")

        transmission_solver = TransmissionModelClass(
            countries=self.countries,
            data=self.transmission_data_max_capacity,
            delta_demand_map=self.demand_map[time],
            marginal_costs_df=self.marginal_costs_df[['MC_import', 'MC_export']],
            cost_transmission_line=self.config_parser.get_cost_transmission_line(),
            logger=self.logger,
            xml_file_path=os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{time}/transmission/problems"),
            expansion_enabled=self.config_parser.get_expansion_enabled(),
        )

        transmission_solver.generate_xml()
        transmission_solver.print_xml(
            name=f"transmission_problem_{time}.xml",
        )

        output_folder = self.solve_folder(os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{time}/transmission"))
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
            return True
        self.marginal_costs_df = marginal_costs_df
        return False
    
    def update_demand(self, time):
        self.logger.info("Updating demand based on transmission problem results")

        transmission_outputs = self.reader.get_transmission_outputs()
        for start_country, exchange in transmission_outputs.groupby('start_country')['exchange'].sum().items():
            self.demand_map[time][start_country]['demand'] += exchange

        for c in self.countries:    
            self.demand_map[time][c]['marginal_demand'] = self.demand_map[time][c]['demand'] * self.delta_marginal_cost

        for start_country, end_country, exchange in transmission_outputs.itertuples(index=False):
            if start_country != end_country:
                self.transmission_data_max_capacity.loc[
                    (self.transmission_data_max_capacity['start_country'] == start_country) & 
                    (self.transmission_data_max_capacity['end_country'] == end_country), 
                    'capacity'
                ] -= abs(exchange)
        if self.transmission_data.empty:
            self.transmission_data = self.reader.get_transmission_outputs()
        else:
            self.transmission_data = pd.concat([self.transmission_data, self.reader.get_transmission_outputs()], ignore_index=True)
            self.transmission_data = self.transmission_data.groupby(['start_country', 'end_country'], as_index=False).sum()
            self.logger.info("Demand updated based on transmission problem results")


    def update_data(self, t, year):
        self.logger.debug(f"Updating data for year {year} and timeslice {t}")

        outputs_df = self.reader.get_internal_outputs()
        outputs_df = outputs_df[outputs_df['demand_type'] == '0']
        outputs_df = outputs_df[['technology', 'capacity', ]]

        if self.results_df is None:
            self.results_df = self.data[['TECHNOLOGY', 'MIN_INSTALLED_CAPACITY']].copy()
            self.results_df.rename(columns={'MIN_INSTALLED_CAPACITY': 'baseline'}, inplace=True)
        else:
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

        if self.config_parser.get_expansion_enabled():
            raise NotImplementedError("Expansion is not implemented yet.")
    
    def extract_data(self, year, time):
        for country in self.countries:
            self.country_data_per_time[country] = self.data_parser.get_country_data(country, time)
        self.extract_costs(self.country_data_per_time, time)

    def extract_costs(self, data, time):
        self.logger.debug("Extracting costs from the data")

        all_data = []
        for c in self.countries:
            country_data = data[c][[ 'CAPITAL_COST', 'VARIABLE_COST', 'FIXED_COST', 'MIN_INSTALLED_CAPACITY', 'OPERATIONAL_LIFETIME']].copy()
            country_data['year_split'] = self.demand_map[time]['year_split']
            all_data.append(country_data)
        self.data = pd.concat(all_data, ignore_index=False)
        self.data['TECHNOLOGY'] = self.data.index
        self.data.reset_index(drop=True, inplace=True)

    def create_internal_DCOP(self, country, time, year):
        self.logger.debug(f"Creating internal DCOP for {country} at time {time} and year {year}")

        energy_country_class = EnergyAgentClass(
            country=country,
            logger=self.logger,
            data=self.country_data_per_time[country],
            year_split=self.demand_map[time]['year_split'],
            demand=self.demand_map[time][country]['demand'],
            xml_file_path=os.path.join(self.config_parser.get_output_file_path(), f"DCOP/{year}/{time}/internal/problems")
        )
        energy_country_class.generate_xml(
            domains=self.create_domains(country=country, time=time, problem_type='internal')
        )
        energy_country_class.print_xml(f"{country}_0.xml")
        energy_country_class.change_demand(demand_variation_percentage=self.delta_marginal_cost)
        energy_country_class.print_xml(f"{country}_+{self.delta_marginal_cost}.xml")
        energy_country_class.change_demand(demand_variation_percentage=-self.delta_marginal_cost)
        energy_country_class.print_xml(f"{country}_-{self.delta_marginal_cost}.xml")

    def solve_DCOP(self, input_path, output_path):
        self.logger.debug(f"Solving DCOP for {input_path} and saving to {output_path}")
        java_command = [
            'java', 
            '-Xmx2G', 
            '-ea',
            '-cp', 
            'frodo2.19.jar:jdom2-2.0.6.jar:jacop-4.7.0.jar', 
            'frodo2.algorithms.AgentFactory', 
            '-timeout', 
            str(self.config_parser.get_timeout_time_steps()*1000), 
            input_path, 
            'agents/DPOP/DPOPagentJaCoP.xml', 
            '-o', 
            output_path,
        ]
        java_process = subprocess.Popen(java_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = java_process.communicate()
        
        # Log both stdout and stderr from the Java process
        if stdout:
            self.logger.info(f"Java stdout for {input_path}:\n{stdout.decode()}")
        if stderr:
            self.logger.error(f"Java stderr for {input_path}:\n{stderr.decode()}")
        if java_process.returncode == 0:
            self.logger.info(f"Java program finished successfully for {input_path}.")
        else:
            self.logger.error(f"Java program encountered an error for {input_path}.")

    def solve_folder(self, folder):
        problem_folder = os.path.join(folder, f"problems")
        output_folder = os.path.join(folder, f"outputs")

        if not os.path.exists(problem_folder):
            self.logger.error(f"Problem folder {problem_folder} does not exist.")
            raise FileNotFoundError(f"Problem folder {problem_folder} does not exist.")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        def process_file(file_name):
            if file_name.endswith(".xml"):
                input_path = os.path.join(problem_folder, file_name)
                output_path = os.path.join(output_folder, f"{file_name.replace('.xml', '')}_output.xml")
                self.solve_DCOP(input_path, output_path)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_file, file_name) for file_name in os.listdir(problem_folder)]
            for future in as_completed(futures):
                future.result()  # Wait for each thread to complete

        if 'transmission' in folder:
            return self.reader.load(type='transmission', output_folder_path=output_folder)
        
        countries_check = self.reader.load(type='internal', output_folder_path=output_folder)
        return countries_check, output_folder
    
    def create_domains(self, country, time, problem_type='internal'):
        self.logger.debug(f"Creating domains for {problem_type} problem in the model")
        domains = self.config_parser.get_domains()

        if problem_type == 'internal':
            if country is None or time is None:
                self.logger.error("Country and year must be provided for internal problem type.")
                raise ValueError("Country and year must be provided for internal problem type.")
            
            domains_mapping = {
                'capacity_domain': list(range(
                    domains['capacity']['min'],
                    domains['capacity']['max'] + 1,
                    domains['capacity']['step']
                )),
                'rateActivity_domain': list(range(
                    0,
                    round(self.demand_map[time][country]['demand'] + self.demand_map[time][country]['demand']*0.10),
                    min(round(self.demand_map[time][country]['demand']/self.config_parser.get_steps_rate_activity()), 100)
                ))
            }
            return domains_mapping
        elif problem_type == 'transmission':
            domains_mapping = {
                'capacity_domain': list(range(
                    domains['transmission']['min'],
                    domains['transmission']['max'] + 1,
                    domains['transmission']['step'], 
                ))
            }
            return domains_mapping

        self.logger.warning(f"Problem type {problem_type} not recognized. Returning empty domains.")
        raise ValueError(f"Problem type {problem_type} in create_domains method not recognized.")

