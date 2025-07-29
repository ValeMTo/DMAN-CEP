import pandas as pd
from translation.parsers.dataParser import dataParserClass
from deprecated import deprecated
from itertools import product
class osemosysDataParserClass(dataParserClass):
    def __init__(self, logger, file_path): #tech_path, fuel_path):
        self.logger = logger
        self.logger.info("Local Data parser initialized")
        self.data_file_path = file_path
        #self.tech_file_path = tech_path
        #self.fuel_file_path = fuel_path

        self.merged_df = None

    def convert_fromGW_capacity_unit(self, data, unit):
        if unit == 'GW':
            data = data
        elif unit == 'MW':
            data = data * 1000
        else:
            raise ValueError("Unit must be 'GW' or 'MW'")
        return data
    
    def load_demand(self, year, country, timeslice):
        self.logger.debug("Loading demand data")
        specified_annual_demand_df = self.extract_specified_annual_demand(year=year, unit='TJ')
        specified_demand_profile_df = self.extract_specified_demand_profile(year=year, timeslices=True)
        demand_df = specified_annual_demand_df.merge(specified_demand_profile_df, on=['FUEL', 'COUNTRY'])
        demand_df['DEMAND_PER_TIMESLICE'] = demand_df['SPECIFIED_ANNUAL_DEMAND'] * demand_df['SPECIFIED_DEMAND_PROFILE']
        try:
            specified_demand = demand_df[(demand_df['COUNTRY'] == country) & (demand_df['TIMESLICE'] == timeslice)]['DEMAND_PER_TIMESLICE'].values[0] 
        except IndexError:
            self.logger.error("No demand data found for country %s and timeslice %s", country, timeslice)
            raise ValueError("No demand data found for country %s and timeslice %s" % (country, timeslice))
        

        year_split_df = self.extract_year_split(year=year)
        year_split_coefficient = year_split_df[year_split_df['TIMESLICE'] == timeslice]['YEAR_SPLIT'].values[0] 

        return year_split_coefficient, specified_demand 
    
    def load_data(self, year, countries):
        self.logger.debug("Loading data from Excel file")
        
        dfs = {}

        dfs['minimum_installed_capacity'] = self.extract_minimum_installed_capacity(year=year, unit='MW')
        dfs['capacity_factors'] = self.extract_capacity_factors(year=year, timeslices=True)
        dfs['availability_factors'] = self.extract_availability_factors(year=year)
        dfs['capacity_to_activity_unit'] = self.extract_capacity_to_activity_unit()
        #dfs['specified_annual_demand'] = self.extract_specified_annual_demand(year=year)
        #dfs['specified_demand_profile'] = self.extract_specified_demand_profile(year=year, timeslices=True)
        #dfs['year_split'] = self.extract_year_split(year=year)
        #dfs['accumulated_annual_demand'] = self.extract_accumulated_annual_demand(year=year) #Fuel only
        dfs['capital_costs'] = self.extract_capital_costs(year=year, unit='$')
        dfs['fixed_costs'] = self.extract_fixed_costs(year=year, unit='$')
        dfs['variable_costs'] = self.extract_variable_costs(year=year, unit='$')
        #dfs['discount_rate'] = self.extract_discount_rate()
        dfs['operational_lifetime'] = self.extract_technology_operational_life()
        dfs['total_annual_max_capacity'] = self.extract_total_annual_max_capacity(year=year, unit='MW')
        dfs['total_technology_annual_activity_upper_limit'] = self.extract_total_technology_annual_activity_upper_limit(year=year, unit='TJ')
        dfs['total_technology_annual_activity_lower_limit'] = self.extract_total_technology_annual_activity_lower_limit(year=year, unit='TJ')
        #dfs['emission_activity_ratio'] = self.extract_emission_activity_ratio(year=year)
        #dfs['emissions_penalty'] = self.extract_emissions_penalty(year=year)
        #dfs['annual_emission_limit'] = self.extract_annual_emission_limit(year=year)
        dfs['output_activity_ratio'] = self.extract_output_activity_ratio(year=year)
        dfs['input_activity_ratio'] = self.extract_input_activity_ratio(year=year)

        technologies_df = self.extract_technologies(all=True).drop_duplicates()
        emissions_df = self.extract_emissions().drop_duplicates()    
        fuels_df = self.extract_fuels().drop_duplicates()
        timeslices_df = self.extract_timeslices().drop_duplicates()

        combinations = []
        for country in countries:
            techs = technologies_df[technologies_df['COUNTRY'] == country]['TECHNOLOGY']
            emis = emissions_df[emissions_df['COUNTRY'] == country]['EMISSION']
            fuels = fuels_df[fuels_df['COUNTRY'] == country]['FUEL']
            timeslices = timeslices_df['TIMESLICE']
            combinations.extend(product([country], techs, emis, fuels, timeslices))

        self.merged_df = pd.DataFrame(combinations, columns=['COUNTRY', 'TECHNOLOGY', 'EMISSION', 'FUEL', 'TIMESLICE'])

        for key, df in dfs.items():
            self.merged_df = self.merged_df.merge(df, on=[col for col in df.columns if col in self.merged_df.columns], how='left')

        self.logger.debug("Data loaded successfully for year %s", year)

    def add_previous_installed_capacity(self, year, new_installable_capacity_df=None):
        if new_installable_capacity_df is not None:
            self.merged_df = self.merged_df.merge(new_installable_capacity_df[['COUNTRY', 'TECHNOLOGY', f'capacity_{year-1}']], on=['COUNTRY', 'TECHNOLOGY'], how='left')
            self.merged_df['MIN_INSTALLED_CAPACITY'] = self.merged_df[f'capacit_{year-1}'].fillna(0)
            self.merged_df.drop(columns=[f'capacity_{year-1}'], inplace=True)
    
    def get_technologies(self):
        self.logger.debug("Getting technologies")
        if self.merged_df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        return self.merged_df[['TECHNOLOGY']].drop_duplicates().set_index('TECHNOLOGY')

    def get_country_data(self, country, time):
        self.logger.debug("Getting data for %s in %s", country, time)

        country_data = self.merged_df[(self.merged_df['COUNTRY'] == country) & (self.merged_df['TIMESLICE'] == time)]
        return country_data.copy().drop_duplicates(subset=['TECHNOLOGY'], keep='first').set_index('TECHNOLOGY')

    def extract_AHA_dataset(self, year):
        aha_df = pd.read_excel("./osemosys_data/input_data/African_Hydropower_Atlas_v2-0_PoliTechM.xlsx", sheet_name='6 - Inputs code and GIS')
        aha_df['Country'] = aha_df['Country'].map(lambda x: x.lower())
        countrycode_df = pd.read_csv('./osemosys_data/input_data/countrycode.csv')
        countrycode_df['Country Name'] = countrycode_df['Country Name'].map(lambda x: x.lower())
        aha_df = aha_df.merge(countrycode_df, left_on='Country', right_on='Country Name', how='inner')
        aha_df['COUNTRY'] = aha_df['Country code']
        
        aha_df = aha_df[(aha_df['First Year'] <= year) & (aha_df['First Year'] >= year - 100)]
        aha_df = aha_df[['COUNTRY', 'Capacity', 'Size Type']]
        aha_df = aha_df.groupby(['COUNTRY', 'Size Type'], as_index=False).agg({'Capacity': 'sum'})
        
        aha_df['TECH'] = aha_df['Size Type'].map({'Large': 'HYDMS03X', 'Middle': 'HYDMS02X', 'Small': 'HYDMS01X'})

        if aha_df['TECH'].isna().any():
            raise ValueError("NaN values found in TECHNOLOGY column")
        
        aha_df['Capacity'] = aha_df['Capacity'] / 1000 # GW
        aha_df['TECHNOLOGY'] = aha_df['COUNTRY'] + aha_df['TECH']
        aha_df.drop(columns=['Size Type', 'TECH'], inplace=True)
        return aha_df

    def extract_minimum_installed_capacity(self, year, unit='GW'):
        aha_df = self.extract_AHA_dataset(year)
        residualCapacity_df = pd.read_excel(self.data_file_path, sheet_name="ResidualCapacity")
        residualCapacity_df['COUNTRY'] = residualCapacity_df['TECHNOLOGY'].map(lambda x: x[:2])
        residualCapacity_df['TECH'] = residualCapacity_df['TECHNOLOGY'].map(lambda x: x[2:])
        new_df = residualCapacity_df[['COUNTRY', 'TECHNOLOGY', year]].rename(columns={year: 'MIN_INSTALLED_CAPACITY'})
        new_df['MIN_INSTALLED_CAPACITY'] = pd.to_numeric(new_df['MIN_INSTALLED_CAPACITY'], errors='coerce')

        new_df = new_df.merge(aha_df, left_on=['COUNTRY', 'TECHNOLOGY'], right_on=['COUNTRY', 'TECHNOLOGY'], how='left')
        new_df['MIN_INSTALLED_CAPACITY'] = new_df['MIN_INSTALLED_CAPACITY'] + new_df['Capacity'].fillna(0)
        new_df.drop(columns=['Capacity'], inplace=True)
        
        new_df['MIN_INSTALLED_CAPACITY'] = self.convert_fromGW_capacity_unit(new_df['MIN_INSTALLED_CAPACITY'], unit)

        return new_df
    
    def extract_capacity_factors(self, year, timeslices=False):
        capacity_factors_df = pd.read_excel(self.data_file_path, sheet_name="CapacityFactor")
        capacity_factors_df['COUNTRY'] = capacity_factors_df['TECHNOLOGY'].map(lambda x: x[:2])
        capacity_factors_df['TECH'] = capacity_factors_df['TECHNOLOGY'].map(lambda x: x[2:])

        new_df = capacity_factors_df[['COUNTRY', 'TECHNOLOGY', 'TIMESLICE', year]].rename(columns={year: 'CAPACITY_FACTOR'})
        new_df['CAPACITY_FACTOR'] = pd.to_numeric(new_df['CAPACITY_FACTOR'], errors='coerce')

        if not timeslices:
            # Select only numeric columns before applying mean
            numeric_cols = ['CAPACITY_FACTOR']
            new_df = new_df.groupby(['COUNTRY', 'TECHNOLOGY'], as_index=False)[numeric_cols].mean()

        return new_df
    
    def extract_availability_factors(self, year):
        availability_factors_df = pd.read_excel(self.data_file_path, sheet_name="AvailabilityFactor")
        availability_factors_df['COUNTRY'] = availability_factors_df['TECHNOLOGY'].map(lambda x: x[:2])
        availability_factors_df['TECH'] = availability_factors_df['TECHNOLOGY'].map(lambda x: x[2:])

        new_df = availability_factors_df[['COUNTRY', 'TECHNOLOGY', year]].rename(columns={year: 'AVAILABILITY_FACTOR'})
        new_df['AVAILABILITY_FACTOR'] = pd.to_numeric(new_df['AVAILABILITY_FACTOR'], errors='coerce')

        return new_df
    
    def extract_capacity_to_activity_unit(self):
        capacity_to_activity_unit_df = pd.read_excel(self.data_file_path, sheet_name="CapacityToActivityUnit", )
        capacity_to_activity_unit_df['COUNTRY'] = capacity_to_activity_unit_df['TECHNOLOGY'].map(lambda x: x[:2])
        capacity_to_activity_unit_df['TECH'] = capacity_to_activity_unit_df['TECHNOLOGY'].map(lambda x: x[2:])

        new_df = capacity_to_activity_unit_df[['COUNTRY', 'TECHNOLOGY', 'Value']].rename(columns={'Value': 'CAPACITY_TO_ACTIVITY_UNIT'})
        new_df['CAPACITY_TO_ACTIVITY_UNIT'] = pd.to_numeric(new_df['CAPACITY_TO_ACTIVITY_UNIT'], errors='coerce')

        return new_df
    
    def extract_specified_annual_demand(self, year, unit='PJ'):
        #Assuming that we are interesting only to the electricity demand
        specified_annual_demand_df = pd.read_excel(self.data_file_path, sheet_name="SpecifiedAnnualDemand")
        specified_annual_demand_df['COUNTRY'] = specified_annual_demand_df['FUEL'].map(lambda x: x[:2])

        new_df = specified_annual_demand_df[['COUNTRY', 'FUEL', year]].rename(columns={year: 'SPECIFIED_ANNUAL_DEMAND'})
        new_df['SPECIFIED_ANNUAL_DEMAND'] = pd.to_numeric(new_df['SPECIFIED_ANNUAL_DEMAND'], errors='coerce')
        
        if unit == 'TJ':
            new_df['SPECIFIED_ANNUAL_DEMAND'] = new_df['SPECIFIED_ANNUAL_DEMAND'] * 1000
        elif unit == 'PJ':
            new_df['SPECIFIED_ANNUAL_DEMAND'] = new_df['SPECIFIED_ANNUAL_DEMAND']
        else:
            raise ValueError("Unit must be 'PJ' or 'TJ'")

        return new_df

    def extract_specified_demand_profile(self, year, timeslices=False):
        specifiedDemandProfile_df = pd.read_excel(self.data_file_path, sheet_name="SpecifiedDemandProfile")
        specifiedDemandProfile_df['COUNTRY'] = specifiedDemandProfile_df['FUEL'].map(lambda x: x[:2])

        new_df = specifiedDemandProfile_df[['COUNTRY', 'FUEL', 'TIMESLICE', year]].rename(columns={year: 'SPECIFIED_DEMAND_PROFILE'})
        new_df['SPECIFIED_DEMAND_PROFILE'] = pd.to_numeric(new_df['SPECIFIED_DEMAND_PROFILE'], errors='coerce')

        if not timeslices:
            # Select only numeric columns before applying sum
            numeric_cols = ['SPECIFIED_DEMAND_PROFILE']
            new_df = new_df.groupby(['COUNTRY', 'FUEL_NAME'], as_index=False)[numeric_cols].sum()

        return new_df
    
    def extract_year_split(self, year):
        year_split_df = pd.read_excel(self.data_file_path, sheet_name="YearSplit")
        year_split_df.rename(columns={'Unnamed: 0': 'TIMESLICE'}, inplace=True)

        new_df = year_split_df[['TIMESLICE', year]].rename(columns={year: 'YEAR_SPLIT'})
        return new_df
    
    def extract_accumulated_annual_demand(self, year):
        accumulated_annual_demand_df = pd.read_excel(self.data_file_path, sheet_name="AccumulatedAnnualDemand")
        accumulated_annual_demand_df['COUNTRY'] = accumulated_annual_demand_df['FUEL'].map(lambda x: x[:2])
        accumulated_annual_demand_df['FUEL_NAME'] = accumulated_annual_demand_df['FUEL'].map(lambda x: x[2:])

        new_df = accumulated_annual_demand_df[['COUNTRY', 'FUEL_NAME', year]].rename(columns={year: 'ACCUMULATED_ANNUAL_DEMAND'})
        new_df['ACCUMULATED_ANNUAL_DEMAND'] = pd.to_numeric(new_df['ACCUMULATED_ANNUAL_DEMAND'], errors='coerce')

        return new_df
    
    def convert_fromMdollars_cost_unit(self, data, unit):
        if unit == 'M$':
            data = data
        elif unit == 'k$':
            data = data * 1000
        elif unit == '$':
            data = data * 1000000
        elif unit == 'B$':
            data = data / 1000
        else:
            raise ValueError("Unit must be 'M$', 'k$', or '$'")
        return data
    
    def extract_capital_costs(self, year, unit='M$'):
        capital_costs_df = pd.read_excel(self.data_file_path, sheet_name="CapitalCost")
        capital_costs_df['COUNTRY'] = capital_costs_df['TECHNOLOGY'].map(lambda x: x[:2])
        capital_costs_df['TECHNOLOGY'] = capital_costs_df['TECHNOLOGY']

        new_df = capital_costs_df[['COUNTRY', 'TECHNOLOGY', year]].rename(columns={year: 'CAPITAL_COST'})
        new_df['CAPITAL_COST'] = pd.to_numeric(new_df['CAPITAL_COST'], errors='coerce')
        new_df['CAPITAL_COST'] = new_df['CAPITAL_COST']/1000 # The cost in the Excel is for GW not MW of capacity

        new_df['CAPITAL_COST'] = self.convert_fromMdollars_cost_unit(new_df['CAPITAL_COST'], unit)

        return new_df
    
    def extract_fixed_costs(self, year, unit='M$'):
        fixed_costs_df = pd.read_excel(self.data_file_path, sheet_name="FixedCost")
        fixed_costs_df['COUNTRY'] = fixed_costs_df['TECHNOLOGY'].map(lambda x: x[:2])
        fixed_costs_df['TECHNOLOGY'] = fixed_costs_df['TECHNOLOGY']

        new_df = fixed_costs_df[['COUNTRY', 'TECHNOLOGY', year]].rename(columns={year: 'FIXED_COST'})
        new_df['FIXED_COST'] = pd.to_numeric(new_df['FIXED_COST'], errors='coerce')
        new_df['FIXED_COST'] = new_df['FIXED_COST']/1000 # The cost in the Excel is for GW not MW of capacity

        new_df['FIXED_COST'] = self.convert_fromMdollars_cost_unit(new_df['FIXED_COST'], unit)

        return new_df
    
    def extract_variable_costs(self, year, unit='M$'):
        variable_costs_df = pd.read_excel(self.data_file_path, sheet_name="VariableCost")
        variable_costs_df['COUNTRY'] = variable_costs_df['TECHNOLOGY'].map(lambda x: x[:2])
        variable_costs_df['TECHNOLOGY'] = variable_costs_df['TECHNOLOGY']

        new_df = variable_costs_df[['COUNTRY', 'TECHNOLOGY','MODEOFOPERATION', year]].rename(columns={year: 'VARIABLE_COST', 'MODEOFOPERATION': 'MODE_OF_OPERATION'})
        new_df['VARIABLE_COST'] = pd.to_numeric(new_df['VARIABLE_COST'], errors='coerce')
        new_df['VARIABLE_COST'] = new_df['VARIABLE_COST']/1000 # The cost in the Excel is for GW not MW of capacity

        new_df['VARIABLE_COST'] = self.convert_fromMdollars_cost_unit(new_df['VARIABLE_COST'], unit)

        return new_df
    
    def extract_discount_rate(self):
        discount_rate_df = pd.read_excel(self.data_file_path, sheet_name="DiscountRate", header=None)
        return discount_rate_df.iloc[0, 0]
    
    def extract_technology_operational_life(self):
        operational_lifetime_df = pd.read_excel(self.data_file_path, sheet_name="OperationalLife", )
        operational_lifetime_df['COUNTRY'] = operational_lifetime_df['TECHNOLOGY'].map(lambda x: x[:2])
        operational_lifetime_df['TECHNOLOGY'] = operational_lifetime_df['TECHNOLOGY']

        new_df = operational_lifetime_df[['COUNTRY', 'TECHNOLOGY', 'VALUE']].rename(columns={'VALUE': 'OPERATIONAL_LIFETIME'})
        new_df['OPERATIONAL_LIFETIME'] = pd.to_numeric(new_df['OPERATIONAL_LIFETIME'], errors='coerce')

        return new_df
    
    def extract_total_annual_max_capacity(self, year, unit='GW'):
        total_annual_capacity_df = pd.read_excel(self.data_file_path, sheet_name="TotalAnnualMaxCapacity")
        total_annual_capacity_df['COUNTRY'] = total_annual_capacity_df['TECHNOLOGY'].map(lambda x: x[:2])
        total_annual_capacity_df['TECHNOLOGY'] = total_annual_capacity_df['TECHNOLOGY'].map(lambda x: x[2:])

        new_df = total_annual_capacity_df[['COUNTRY', 'TECHNOLOGY', year]].rename(columns={year: 'TOTAL_ANNUAL_CAPACITY'})
        new_df['TOTAL_ANNUAL_CAPACITY'] = pd.to_numeric(new_df['TOTAL_ANNUAL_CAPACITY'], errors='coerce')
        new_df = new_df[new_df['TOTAL_ANNUAL_CAPACITY'] != 99999999]

        new_df['TOTAL_ANNUAL_CAPACITY'] = self.convert_fromGW_capacity_unit(new_df['TOTAL_ANNUAL_CAPACITY'], unit)

        return new_df
    
    def convert_fromPJ_energy_unit(self, data, unit):
        if unit == 'PJ':
            data = data
        elif unit == 'TJ':
            data = data * 1000
        else:
            raise ValueError("Unit must be 'PJ' or 'TJ'")
        return data

    def extract_total_technology_annual_activity_upper_limit(self, year, unit='PJ'):
        total_annual_activity_upper_limit_df = pd.read_excel(self.data_file_path, sheet_name="TotalTechnologyAnnualActivityUp")
        total_annual_activity_upper_limit_df['COUNTRY'] = total_annual_activity_upper_limit_df['TECHNOLOGY'].map(lambda x: x[:2])
        total_annual_activity_upper_limit_df['TECHNOLOGY'] = total_annual_activity_upper_limit_df['TECHNOLOGY']

        new_df = total_annual_activity_upper_limit_df[['COUNTRY', 'TECHNOLOGY', year]].rename(columns={year: 'TOTAL_ANNUAL_ACTIVITY_UPPER_LIMIT'})
        new_df['TOTAL_ANNUAL_ACTIVITY_UPPER_LIMIT'] = pd.to_numeric(new_df['TOTAL_ANNUAL_ACTIVITY_UPPER_LIMIT'], errors='coerce')

        new_df['TOTAL_ANNUAL_ACTIVITY_UPPER_LIMIT'] = self.convert_fromPJ_energy_unit(new_df['TOTAL_ANNUAL_ACTIVITY_UPPER_LIMIT'], unit)
        
        return new_df
    
    def extract_total_technology_annual_activity_lower_limit(self, year, unit='PJ'):
        total_annual_activity_upper_limit_df = pd.read_excel(self.data_file_path, sheet_name="TotalTechnologyAnnualActivityLo")
        total_annual_activity_upper_limit_df['COUNTRY'] = total_annual_activity_upper_limit_df['TECHNOLOGY'].map(lambda x: x[:2])
        total_annual_activity_upper_limit_df['TECHNOLOGY'] = total_annual_activity_upper_limit_df['TECHNOLOGY']

        new_df = total_annual_activity_upper_limit_df[['COUNTRY', 'TECHNOLOGY', year]].rename(columns={year: 'TOTAL_ANNUAL_ACTIVITY_LOWER_LIMIT'})
        new_df['TOTAL_ANNUAL_ACTIVITY_LOWER_LIMIT'] = pd.to_numeric(new_df['TOTAL_ANNUAL_ACTIVITY_LOWER_LIMIT'], errors='coerce')

        new_df['TOTAL_ANNUAL_ACTIVITY_LOWER_LIMIT'] = self.convert_fromPJ_energy_unit(new_df['TOTAL_ANNUAL_ACTIVITY_LOWER_LIMIT'], unit)

        return new_df
    
    def extract_emission_activity_ratio(self, year):
        emission_activity_ratio_df = pd.read_excel(self.data_file_path, sheet_name="EmissionActivityRatio", )
        emission_activity_ratio_df['COUNTRY_TECH'] = emission_activity_ratio_df['TECHNOLOGY'].map(lambda x: x[:2])
        emission_activity_ratio_df['TECHNOLOGY'] = emission_activity_ratio_df['TECHNOLOGY']
        emission_activity_ratio_df['COUNTRY_EMI'] = emission_activity_ratio_df['EMISSION'].map(lambda x: x[:2])
        emission_activity_ratio_df['EMISSION'] = emission_activity_ratio_df['EMISSION']

        #TODO: check if it makes sense to filter only the rows where the country of the technology is the same as the country of the emission
        emission_activity_ratio_df = emission_activity_ratio_df[emission_activity_ratio_df['COUNTRY_TECH'] == emission_activity_ratio_df['COUNTRY_EMI']]

        new_df = emission_activity_ratio_df[['COUNTRY_TECH', 'TECHNOLOGY', 'EMISSION', 'MODEOFOPERATION', year]].rename(columns={year: 'EMISSION_ACTIVITY_RATIO', 'COUNTRY_TECH': 'COUNTRY'})
        new_df['EMISSION_ACTIVITY_RATIO'] = pd.to_numeric(new_df['EMISSION_ACTIVITY_RATIO'], errors='coerce')

        return new_df

    def extract_emissions_penalty(self, year):
        emissions_penalty_df = pd.read_excel(self.data_file_path, sheet_name="EmissionsPenalty")
        emissions_penalty_df['COUNTRY'] = emissions_penalty_df['EMISSION'].map(lambda x: x[:2])
        emissions_penalty_df['EMISSION'] = emissions_penalty_df['EMISSION'].map(lambda x: x[2:])

        new_df = emissions_penalty_df[['COUNTRY', 'EMISSION', year]].rename(columns={year: 'EMISSIONS_PENALTY'})
        new_df['EMISSIONS_PENALTY'] = pd.to_numeric(new_df['EMISSIONS_PENALTY'], errors='coerce')

        return new_df

    def extract_annual_emission_limit(self, year):
        annual_emission_limit_df = pd.read_excel(self.data_file_path, sheet_name="AnnualEmissionLimit")
        annual_emission_limit_df['COUNTRY'] = annual_emission_limit_df['EMISSION'].map(lambda x: x[:2])
        annual_emission_limit_df['EMISSION'] = annual_emission_limit_df['EMISSION']

        new_df = annual_emission_limit_df[['COUNTRY', 'EMISSION', year]].rename(columns={year: 'ANNUAL_EMISSION_LIMIT'})
        new_df['ANNUAL_EMISSION_LIMIT'] = pd.to_numeric(new_df['ANNUAL_EMISSION_LIMIT'], errors='coerce')

        new_df = new_df[(new_df['ANNUAL_EMISSION_LIMIT'] != 999) & (new_df['ANNUAL_EMISSION_LIMIT'] != 0)]

        return new_df
    
    @deprecated
    def extract_technologies_per_country(self, country):
        print("Deprecated: Use extract_technologies_per_country instead")
        technologies_df = pd.read_excel(self.data_file_path, sheet_name="TECHNOLOGY", header=None)
        technologies_df['COUNTRY'] = technologies_df[0].map(lambda x: x[:2])
        technologies_df['TECHNOLOGY'] = technologies_df[0]
  
        technologies_df = technologies_df[technologies_df['COUNTRY'] == country]
        technologies_df = technologies_df[['TECHNOLOGY']]

        return technologies_df
    
    def extract_technologies(self, all=False):
        technologies_df = pd.read_excel(self.data_file_path, sheet_name="TECHNOLOGY", header=None)
        technologies_df['COUNTRY'] = technologies_df[0].map(lambda x: x[:2])
        technologies_df['TECHNOLOGY'] = technologies_df[0].map(lambda x: x[2:])

        if all:
            technologies_df['TECHNOLOGY'] = technologies_df['COUNTRY'] + technologies_df['TECHNOLOGY']
            return technologies_df[['COUNTRY', 'TECHNOLOGY']]
        power_plants_df = pd.read_csv("./osemosys_data/techcodes(in).csv")
        power_plants_df = power_plants_df[power_plants_df['Group'] == 'Power_plants']
        power_plants_df = power_plants_df[['code (Old)']].rename(columns={'code (Old)': 'TECHNOLOGY'})

        technologies_df = technologies_df.merge(power_plants_df, left_on='TECHNOLOGY', right_on='TECHNOLOGY', how='inner')
        technologies_df['TECHNOLOGY'] = technologies_df['COUNTRY'] + technologies_df['TECHNOLOGY']
        return technologies_df[['COUNTRY', 'TECHNOLOGY']]
    
    def extract_emissions(self):
        emissions_df = pd.read_excel(self.data_file_path, sheet_name="EMISSION", header=None)
        emissions_df['COUNTRY'] = emissions_df[0].map(lambda x: x[:2])
        return emissions_df[['COUNTRY', 0]].rename(columns={0: 'EMISSION'})
    
    def extract_fuels(self):
        fuels_df = pd.read_excel(self.data_file_path, sheet_name="FUEL", header=None)
        fuels_df['COUNTRY'] = fuels_df[0].map(lambda x: x[:2])
        fuels_df['FUEL'] = fuels_df[0]
        return fuels_df[['COUNTRY', 'FUEL']]
    
    def extract_timeslices(self):
        timeslices_df = pd.read_excel(self.data_file_path, sheet_name="TIMESLICE", header=None)
        timeslices_df['TIMESLICE'] = timeslices_df[0]
        return timeslices_df[['TIMESLICE']]
    
    def extract_technologies_per_country(self, country):
        technologies_df = pd.read_csv("./osemosys_data/techcodes(in).csv")
        technologies_df = technologies_df[technologies_df['Group'] == 'Power_plants']
        technologies_df = technologies_df[['code (Old)']].rename(columns={'code (Old)': 'TECHNOLOGY'})
        technologies_df['COUNTRY'] = country
        technologies_df['TECHNOLOGY'] = technologies_df['COUNTRY'] + technologies_df['TECHNOLOGY']

        return technologies_df
    
    def extract_output_activity_ratio(self, year):
        technologies_df = pd.read_excel(self.data_file_path, sheet_name="OutputActivityRatio")
        technologies_df['COUNTRY'] = technologies_df['TECHNOLOGY'].map(lambda x: x[:2])
        technologies_df['TECHNOLOGY'] = technologies_df['TECHNOLOGY']
        technologies_df = technologies_df[['COUNTRY', 'TECHNOLOGY', 'FUEL', 'MODEOFOPERATION', year]].rename(columns={year: 'OUTPUT_ACTIVITY_RATIO', 'MODEOFOPERATION': 'MODE_OF_OPERATION'})
        technologies_df['OUTPUT_ACTIVITY_RATIO'] = pd.to_numeric(technologies_df['OUTPUT_ACTIVITY_RATIO'], errors='coerce')

        return technologies_df
    
    def extract_input_activity_ratio(self, year):
        technologies_df = pd.read_excel(self.data_file_path, sheet_name="InputActivityRatio")
        technologies_df['COUNTRY'] = technologies_df['TECHNOLOGY'].map(lambda x: x[:2])
        technologies_df['TECHNOLOGY'] = technologies_df['TECHNOLOGY']
        technologies_df = technologies_df[['COUNTRY', 'TECHNOLOGY', 'FUEL', 'MODEOFOPERATION', year]].rename(columns={year: 'INPUT_ACTIVITY_RATIO', 'MODEOFOPERATION': 'MODE_OF_OPERATION'})
        technologies_df['INPUT_ACTIVITY_RATIO'] = pd.to_numeric(technologies_df['INPUT_ACTIVITY_RATIO'], errors='coerce')

        return technologies_df
    
    def extract_fuels(self):
        fuels_df = pd.read_excel(self.data_file_path, sheet_name="FUEL", header=None)
        fuels_df['FUEL'] = fuels_df[0]
        fuels_df['COUNTRY'] = fuels_df[0].map(lambda x: x[:2])
        fuels_df.drop(columns=[0], inplace=True)
        return fuels_df