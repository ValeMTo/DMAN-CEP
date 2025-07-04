
from translation.xmlGenerator import XMLGeneratorClass
from deprecated import deprecated
import pandas as pd
import logging
import os
from itertools import product

class EnergyAgentClass:
    def __init__(self, country, logger, data, year_split, demand, xml_file_path):
        self.name = country
        self.logger = logger
        self.data = data
        self.year_split = year_split
        self.demand = demand # Total demand in the country for the year in that time resolution
        self.demand_constraint = None
        self.xml_generator = XMLGeneratorClass(logger = self.logger)
        self.logger.debug(f"EnergyAgentClass initialized for {country}")
        self.xml_file_path = xml_file_path
    
    def generate_xml(self, domains):
        self.logger.debug(f"Generating XML for {self.name}")
        
        self.xml_generator.add_presentation(name=self.name, maximize='False')
        self.xml_generator.add_agents([self.name])
        self.xml_generator.add_domains(domains)
        
        self.tech_df = self.data
        total_technologies = len(self.tech_df)
        self.tech_df['AVAILABILITY_FACTOR'] = self.tech_df['AVAILABILITY_FACTOR'].fillna(0.9)  # Default to 1 if NaN
        self.tech_df['FACTOR'] = self.tech_df['CAPACITY_FACTOR'] * self.tech_df['AVAILABILITY_FACTOR'] * self.tech_df['CAPACITY_TO_ACTIVITY_UNIT'] * self.year_split
        self.tech_df['FACTOR'] = self.tech_df['FACTOR'].fillna(0)  # Replace NaN factors with zero
        self.tech_df = self.tech_df[self.tech_df['FACTOR'] > 0]
        self.tech_df = self.tech_df[
            self.tech_df['CAPITAL_COST'].notna() &
            self.tech_df['VARIABLE_COST'].notna() &
            self.tech_df['FIXED_COST'].notna()
        ]
        
        #self.tech_df['VARIABLE_COST'] = self.tech_df['VARIABLE_COST'].apply(lambda x: min(x, 999) if pd.notna(x) else x)

        if total_technologies < len(self.tech_df):
            self.logger.warning(f"Filtered out {total_technologies - len(self.tech_df)} technologies with zero factor or missing costs.")
            self.logger.warning(f"Remaining technologies used for {self.name} optimization: {len(self.tech_df)}")

        for var in self.tech_df.index:
            self.xml_generator.add_variable(name=f"{var}_capacity", domain='capacity_domain', agent=self.name)
            self.xml_generator.add_variable(name=f"{var}_rateActivity", domain='rateActivity_domain', agent=self.name)

        for technology, row in self.tech_df.iterrows():
            if pd.notna(row['MIN_INSTALLED_CAPACITY']) and round(row['MIN_INSTALLED_CAPACITY']) > 0:
                self.xml_generator.add_minimum_capacity_constraint(
                    variable_name=f"{technology}_capacity",
                    min_capacity=round(row['MIN_INSTALLED_CAPACITY'])
                )
            # Assuming $
            self.xml_generator.add_cost_constraint(
                variable_capacity_name=f"{technology}_capacity",
                rateActivity_variable=f"{technology}_rateActivity",
                previous_installed_capacity=round(row['MIN_INSTALLED_CAPACITY']) if pd.notna(row['MIN_INSTALLED_CAPACITY']) else 0,
                capital_cost=round((row['CAPITAL_COST']/row['OPERATIONAL_LIFETIME'])/10**3), # convert to thousand currency units
                variable_cost=round(row['VARIABLE_COST']),
                fixed_cost=round((row['FIXED_COST'] * self.year_split))  
            )
            self.xml_generator.add_maximum_activity_rate_constraint(
                cap_variable=f"{technology}_capacity",
                capActivity_variable=f"{technology}_rateActivity",
                factor=row['FACTOR']*100,
                div_weight=100,
            )

        self.demand_constraint = self.xml_generator.add_minimum_demand_constraint(
            variables = [f"{var}_rateActivity" for var in self.tech_df.index],
            demand = round(self.demand),
            extra_name = f"0"
        )

    def print_xml(self, name):
        self.logger.debug(f"Printing XML for {self.name}")
        self.xml_generator.print_xml(output_file=os.path.join(self.xml_file_path, name))
        self.logger.debug(f"XML generated for {self.name} at {os.path.join(self.xml_file_path, name)}")
    
    def change_demand(self, demand_variation_percentage):
        if self.demand_constraint is not None:
            self.logger.debug(f"Changing demand for {self.name} by {demand_variation_percentage}%")
            self.xml_generator.remove_constraint(self.demand_constraint)

            new_demand = round(self.demand * (1 + demand_variation_percentage))

            # Add the updated demand constraint
            self.demand_constraint = self.xml_generator.add_minimum_demand_constraint(
                variables=[f"{var}_rateActivity" for var in self.tech_df.index],
                demand=new_demand,
                extra_name=f"{demand_variation_percentage}"
            )
        else:
            raise NotImplementedError("XML instance missing")