from translation.parsers.configParser import ConfigParserClass
from translation.parsers.osemosysDataParser import localDataParserClass
from translation.xmlGenerator import XMLGeneratorClass
from deprecated import deprecated
import pandas as pd
import logging
import os
from itertools import product
class EnergyModelClass:
    def __init__(self):
        self.config_parser = ConfigParserClass(file_path='config.yaml')
        self.log_level, log_file = self.config_parser.get_log_info()
        self.logger = self.create_logger(self.log_level, log_file)
        self.config_parser.set_logger(self.logger)

        self.data_parser = localDataParserClass(logger = self.logger, file_path=self.config_parser.get_file_path())
        self.xml_generator = XMLGeneratorClass(logger = self.logger)

        self.name = self.config_parser.get_problem_name()
        self.countries = self.config_parser.get_countries()
        self.time_resolution = self.config_parser.get_annual_time_resolution()
        self.year = self.config_parser.get_year()

        self.logger.info("Energy model initialized")

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
    
    @deprecated(reason="Olf function to fix the collected data")
    def add_extra_power_tech(self, extra_power_tech):
        self.power_tech = [tech for tech in self.power_tech if tech[-1] == 'N' or tech[-1] == 'X']
        self.power_tech.extend(extra_power_tech)
        self.power_tech = list(set(self.power_tech))
    
    def filter_data(self, data, only_powerplants=True):
        if 'COUNTRY' not in data.columns: 
            raise ValueError("Data does not have a 'COUNTRY' column")
        data = data.loc[data['COUNTRY'].isin(self.countries)]

        if only_powerplants:
            if 'TECH' in data.columns:
                data = data.loc[data['TECH'].isin(self.power_tech)]
            
            if 'TECHNOLOGY' in data.columns:
                data = data.loc[data['TECHNOLOGY'].map(lambda x: str(x)[2:]).isin(self.power_tech)]
        return data
        
    @deprecated(reason="Data directly connected in generate_xml function")
    def collect_data(self):
        self.logger.debug("Collecting data...")
        self.already = self.data_parser.get_already_installed_powerplants_data(self.countries)
        self.logger.debug("Powerplants data collected")

        self.demand_data = self.data_parser.get_annual_demand_data(self.year, self.countries)
        self.logger.debug("Annual demand data collected")

    def generate_xml(self):
        self.logger.debug("Generating XML...")
        
        self.xml_generator.add_presentation(name=self.name, maximize='False') # For some reason i get a lower cost value in this case
        self.xml_generator.add_agents(self.countries)

        domains = self.generate_domains()
        self.xml_generator.add_domains(domains)

        power_technologies = pd.read_csv('data/input_data/power_tech.csv')
        self.power_tech = power_technologies['power_tech'].unique()
        self.power_tech = [tech for tech in self.power_tech if 'BACKSTOP' not in tech]

        input_data=self.filter_data(self.data_parser.extract_technologies_per_country(impose_one_mode=True))
        selected_technologies = input_data['TECHNOLOGY'].unique()
        timeslice_technologies_modes = input_data['VARIABLE'].values
        modes = input_data['MODE_OF_OPERATION'].unique()
        self.variables = self.xml_generator.add_variable_from_name(
            variables = timeslice_technologies_modes,
            technologies=selected_technologies, 
            agents=self.countries
        )

        residual_capacity_df = self.filter_data(self.data_parser.extract_minimum_installed_capacity(year=self.year, unit='MW'), only_powerplants=False)
        residual_capacity_df = residual_capacity_df[(residual_capacity_df['MIN_INSTALLED_CAPACITY'] > 0) & (residual_capacity_df['TECHNOLOGY'].isin(selected_technologies))]
        self.add_extra_power_tech(residual_capacity_df['TECHNOLOGY'].unique())


        # Minimum installed capacity constraint    
        residual_capacity_df = self.filter_data(residual_capacity_df)
        for index, row in residual_capacity_df.iterrows():
            self.xml_generator.add_minimum_capacity_constraint(
                variable_name=f"{row['TECHNOLOGY']}_capacity", 
                min_capacity=round(row['MIN_INSTALLED_CAPACITY'])
            )

        # Maximum rate of activity constraint based on the installed capacity
        factors_df = self.collect_factors(selected_technologies)
        self.xml_generator.add_maximum_rate_of_activity_per_all_technology_constraint(modes=modes, factors_df=factors_df)
        self.xml_generator.add_maximum_annual_activity_rate_per_timeslice_constraint(modes=modes, factors_df=factors_df)
        #self.xml_generator.add_minimum_annual_activity_rate_per_timeslice_constraint(modes=modes, factors_df=factors_df, non_dispatchable_technologies=['HYDMS01X', 'HYDMS02X', 'HYDMS03X', 'SOC1P00X', 'SOC2P00X'])

        
        input_output_activity_ratio_df, specified_annual_demand_df, specified_demand_profile_df, year_split_df = self.collect_ratio_annual_demand()

        #TODO: check again
        # self.xml_generator.add_minimum_rate_of_activity_constraint(
        #     input_output_activity_ratio_df=input_output_activity_ratio_df,
        #     specified_annual_demand_df=specified_annual_demand_df,
        #     specified_demand_profile_df=specified_demand_profile_df,
        #     year_split_df = year_split_df
        # )

        #TODO: substitute the following code with the previous one
        # Easy version - Energy balance A & B (only electricity without input and output activity ratio)
        self.xml_generator.add_minimum_respecting_demand(
            timeslice_technologies_modes=timeslice_technologies_modes,
            specified_demand_profile_df=specified_demand_profile_df,
            specified_annual_demand_df=specified_annual_demand_df,
            year_split_df=year_split_df
        )   

        # Total annual maximum capacity constraint
        max_capacity_installable_df = self.filter_data(self.data_parser.extract_total_annual_max_capacity(year=self.year, unit='MW'))
        for index, row in max_capacity_installable_df.iterrows():
            self.xml_generator.add_maximum_capacity_constraint(
                variable_name=f"{row['TECHNOLOGY']}_capacity", 
                max_capacity=round(row['TOTAL_ANNUAL_CAPACITY'])
            )

        # Annual activity constraint
        # TODO: It is doing sth else
        # upper_limit_technological_demand_df = self.filter_data(self.data_parser.extract_total_technology_annual_activity_upper_limit(year=self.year, unit='TJ'))
        # lower_limit_technological_demand_df = self.filter_data(self.data_parser.extract_total_technology_annual_activity_lower_limit(year=self.year, unit='TJ'))
        # limit_technological_demand_df = upper_limit_technological_demand_df.merge(lower_limit_technological_demand_df, on=['COUNTRY', 'TECHNOLOGY'], how='outer')
        # for index, row in limit_technological_demand_df.iterrows():
        #     self.xml_generator.add_min_max_total_technology_annual_activity_constraint(
        #         modes=modes,
        #         year_split_df=year_split_df,
        #         technology=row['TECHNOLOGY'], 
        #         upper_limit=row['TOTAL_ANNUAL_ACTIVITY_UPPER_LIMIT'],
        #         lower_limit=row['TOTAL_ANNUAL_ACTIVITY_LOWER_LIMIT'] 
        #     )

        #Emission accounting
        #TODO: note that some emission factor can be negative - atm there is no management for negative values
        # emission_factors_df = self.filter_data(self.data_parser.extract_emission_activity_ratio(year=self.year))
        # emission_annual_limit_df = self.filter_data(self.data_parser.extract_annual_emission_limit(year=self.year))
        # for index, row in emission_annual_limit_df.iterrows():
        #     variables_per_country_emission = [f"{var}_rateActivity" for var in timeslice_technologies_modes if row['COUNTRY'] == var.split('_')[1][:2]]
        #     self.xml_generator.add_emission_accounting_constraint(
        #         emission_name = row['EMISSION'],
        #         max_emission_limit = row['ANNUAL_EMISSION_LIMIT'],
        #         variables = variables_per_country_emission,
        #         modes = modes,
        #         emission_factors_df=emission_factors_df,
        #         year_split_df=year_split_df
        #     )

        #Soft constraint: Operating cost minimization
        capital_costs_df = self.filter_data(self.data_parser.extract_capital_costs(year=self.year, unit='M$'))
        lifetime_df = self.filter_data(self.data_parser.extract_technology_operational_life())
        amortized_capital_costs_df = capital_costs_df.merge(lifetime_df, on=['COUNTRY', 'TECHNOLOGY'], how='left')
        amortized_capital_costs_df['AMORTIZED_CAPITAL_COST'] = amortized_capital_costs_df['CAPITAL_COST'] / amortized_capital_costs_df['OPERATIONAL_LIFETIME']
        amortized_capital_costs_df = amortized_capital_costs_df.merge(residual_capacity_df, on=['COUNTRY', 'TECHNOLOGY'], how='left')
        amortized_capital_costs_df['AMORTIZED_CAPITAL_COST'] = amortized_capital_costs_df['AMORTIZED_CAPITAL_COST'].apply(lambda x: min(x, 2**31 - 1))

        full_index = pd.DataFrame(
            list(product(self.countries, selected_technologies)),
            columns=['COUNTRY', 'TECHNOLOGY']
        )

        amortized_capital_costs_df = full_index.merge(amortized_capital_costs_df, on=['COUNTRY', 'TECHNOLOGY'], how='left')
        amortized_capital_costs_df['MIN_INSTALLED_CAPACITY'] = amortized_capital_costs_df['MIN_INSTALLED_CAPACITY'].fillna(0)
        amortized_capital_costs_df['AMORTIZED_CAPITAL_COST'] = amortized_capital_costs_df['AMORTIZED_CAPITAL_COST'].fillna(4999)
        amortized_capital_costs_df = self.filter_data(amortized_capital_costs_df)
        amortized_capital_costs_df.sort_values(by=['AMORTIZED_CAPITAL_COST'], ascending=False, inplace=True)

        fixed_costs_df = self.filter_data(self.data_parser.extract_fixed_costs(year=self.year, unit='M$'))
        fixed_costs_df['FIXED_COST'] = fixed_costs_df['FIXED_COST'].apply(lambda x: min(x, 2**31 - 1))
        fixed_costs_df = full_index.merge(fixed_costs_df, on=['COUNTRY', 'TECHNOLOGY'], how='left')
        fixed_costs_df['FIXED_COST'] = fixed_costs_df['FIXED_COST'].fillna(4999)
        fixed_costs_df = self.filter_data(fixed_costs_df)

        amortized_capital_costs_df = amortized_capital_costs_df.merge(fixed_costs_df, on=['COUNTRY', 'TECHNOLOGY'], how='left')


        for index, row in amortized_capital_costs_df.iterrows():
            capacity_variable = f"{row['TECHNOLOGY']}_capacity"
            self.xml_generator.add_installing_cost_minimization_constraint(
                weight=1,
                variable_capacity_name=capacity_variable,
                previous_installed_capacity=int(row["MIN_INSTALLED_CAPACITY"]),
                cost_per_MW=round(row['AMORTIZED_CAPITAL_COST'] + row['FIXED_COST']),
                extra_name = 'amortized'
            )


        # for index, row in fixed_costs_df.iterrows():
        #     capacity_variable = f"{row['TECHNOLOGY']}_capacity"
        #     self.xml_generator.add_installing_cost_minimization_constraint(
        #         weight=1,
        #         variable_capacity_name=capacity_variable,
        #         previous_installed_capacity=0,
        #         cost_per_MW=round(row['FIXED_COST']),
        #         extra_name = 'fixedCost'
        #     )


        # variable_costs_df = self.filter_data(self.data_parser.extract_variable_costs(year=self.year, unit='$'))
        # variable_costs_df['VARIABLE_COST'] = variable_costs_df['VARIABLE_COST'].apply(lambda x: min(x, 2**31 - 1))
        # variable_costs_df = full_index.merge(variable_costs_df, on=['COUNTRY', 'TECHNOLOGY'], how='left')
        # variable_costs_df['VARIABLE_COST'] = variable_costs_df['VARIABLE_COST'].fillna(9999)
        # variable_costs_df = self.filter_data(variable_costs_df)
        # for index, row in variable_costs_df.iterrows():
        #     if row['VARIABLE_COST'] != 0:
        #         self.xml_generator.add_minimizing_operating_cost_constraint(
        #             weight=1,
        #             rateActivity_variables=[f"{l}_{row['TECHNOLOGY']}_{m}_rateActivity" for l in year_split_df['TIMESLICE'].unique() for m in modes],
        #             cost_per_unit_of_activity=round(row['VARIABLE_COST']), 
        #             year_split_df=year_split_df,
        #         )

        self.xml_generator.print_xml(output_file=self.config_parser.get_output_file_path())
        self.logger.info("XML generated")
    
    def collect_factors(self, selected_technologies):
        capacity_factor_df = self.filter_data(self.data_parser.extract_capacity_factors(year=self.year, timeslices=True))
        availability_factor_df = self.filter_data(self.data_parser.extract_availability_factors(year=self.year))
        conversion_factor_df = self.data_parser.extract_capacity_to_activity_unit()
        year_split_df = self.data_parser.extract_year_split(year=self.year)
        factors_df = capacity_factor_df.set_index(['COUNTRY', 'TECHNOLOGY']).join(
            availability_factor_df.set_index(['COUNTRY', 'TECHNOLOGY']), how='outer'
        ).join(
            conversion_factor_df.set_index(['COUNTRY', 'TECHNOLOGY']), how='outer'
        ).reset_index()
        factors_df = factors_df[factors_df['TECHNOLOGY'].isin(selected_technologies)]
        #TODO: how should i treat the NaN values? - At the moment filling them with 1
        factors_df['CAPACITY_FACTOR'] = factors_df['CAPACITY_FACTOR'].fillna(1)
        factors_df['AVAILABILITY_FACTOR'] = factors_df['AVAILABILITY_FACTOR'].fillna(1)
        factors_df = factors_df[~factors_df['TIMESLICE'].isna()]

        countries = factors_df['COUNTRY'].unique()
        timeslices = year_split_df['TIMESLICE'].unique()

        from itertools import product
        full_index = pd.DataFrame(
            list(product(countries, selected_technologies, timeslices)),
            columns=['COUNTRY', 'TECHNOLOGY', 'TIMESLICE']
        )

        factors_df = full_index.merge(factors_df, on=['COUNTRY', 'TECHNOLOGY', 'TIMESLICE'], how='left')
        factors_df['CAPACITY_FACTOR'] = factors_df['CAPACITY_FACTOR'].fillna(0)
        factors_df['AVAILABILITY_FACTOR'] = factors_df['AVAILABILITY_FACTOR'].fillna(0)
        factors_df['CAPACITY_TO_ACTIVITY_UNIT'] = factors_df['CAPACITY_TO_ACTIVITY_UNIT'].fillna(31.536)
        factors_df = factors_df.merge(year_split_df, on='TIMESLICE', how='left')
        factors_df['YEAR_SPLIT'] = factors_df['YEAR_SPLIT'].map(lambda x: 1/round(1/x))
 
        return factors_df
    
    def collect_ratio_annual_demand(self):
        output_activity_ratio_df = self.filter_data(self.data_parser.extract_output_activity_ratio(year=self.year))
        input_activity_ratio_df = self.filter_data(self.data_parser.extract_input_activity_ratio(year=self.year))
        input_output_activity_ratio_df = input_activity_ratio_df.merge(output_activity_ratio_df, on=['COUNTRY', 'TECHNOLOGY', 'FUEL', 'MODE_OF_OPERATION'], how='outer')
        #input_output_activity_ratio_df = input_output_activity_ratio_df.dropna(subset=['INPUT_ACTIVITY_RATIO', 'OUTPUT_ACTIVITY_RATIO'])
        input_output_activity_ratio_df = input_output_activity_ratio_df.fillna(1)
        # input_output_activity_ratio_df['Input/Output-Egual'] = abs(input_output_activity_ratio_df['INPUT_ACTIVITY_RATIO'] - input_output_activity_ratio_df['OUTPUT_ACTIVITY_RATIO']) < 0.001
        # input_output_activity_ratio_df = input_output_activity_ratio_df[input_output_activity_ratio_df['Input/Output-Egual'] == False]
        # input_output_activity_ratio_df = input_output_activity_ratio_df.drop(columns=['Input/Output-Egual'])
        specified_annual_demand_df = self.filter_data(self.data_parser.extract_specified_annual_demand(year=self.year, unit='TJ'))
        specified_demand_profile_df = self.filter_data(self.data_parser.extract_specified_demand_profile(year=self.year, timeslices=True))
        year_split_df = self.data_parser.extract_year_split(year=self.year)

        return input_output_activity_ratio_df, specified_annual_demand_df, specified_demand_profile_df, year_split_df
        
    def generate_domains(self):
        #TODO: Do not hard code this values
        self.logger.debug("Generating domains...")
        domains = {}
        domains["rate_activity_domain"] = range(0, 2000000, 5000) #TJ/year
        domains["installable_capacity_domain"] = range(0, 40000, 500) #MW 
        #domains["trasferable_capacity_domain"] = range(0, 3000, 500) #TJ/year
        return domains
    
