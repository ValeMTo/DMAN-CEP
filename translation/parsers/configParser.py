import yaml
import pandas as pd
from deprecated import deprecated

class ConfigParserClass:
    def __init__(self, file_path='config.yaml'):

        try:
            with open(file_path, 'r') as file:
                self.config = yaml.safe_load(file)
                self.config = self.config['config']
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} not found")
        
    def get_file_path(self):
        return self.config['outline']['data_file_path']
    
    def get_output_file_path(self):
        return self.config['output_file_path']
    
    def set_output_file_path(self, path):
        """
        Sets the output folder path in the configuration.
        :param path: str
        """
        self.config['output_file_path'] = path
        self.logger.info(f"Output folder path set to {path}")
    
    def get_problem_name(self):
        return self.config['name']
    
    def get_log_info(self):
        return self.config['logging']['level'], self.config['logging']['file']
    
    def set_logger(self, logger):
        self.logger = logger
        self.logger.info("Logger set in config parser")

    def get_annual_time_resolution(self):
        """
        Returns the number of representation per year inside the model.
        :return: int
        """
        self.logger.debug("Annual time resolution set in config parser")
        return self.config['outline']['representation_per_year']
    
    def get_max_iteration(self):
        """
        Returns the maximum number of iterations for the model.
        :return: int
        """
        self.logger.debug("Max iteration set in config parser")
        return self.config['outline']['max_iterations']

    def get_min_convergence_iterations(self):
        """
        Returns the minimum number of convergence iterations for the model.
        :return: int
        """
        self.logger.debug("Min convergence iterations set in config parser")
        return self.config['outline']['min_convergence_iterations']

    def get_delta_marginal_cost(self):
        """
        Returns the delta for calculating the marginal cost for the model.
        :return: float
        """
        self.logger.debug("Demand variation set in config parser")
        return self.config['outline']['delta_marginal_cost']
    
    def get_marginal_cost_tolerance(self):
        """
        Returns the tolerance for the marginal cost for the model.
        :return: float
        """
        self.logger.debug("Marginal cost tolerance set in config parser")
        return self.config['outline']['marginal_cost_tolerance']
        
    def get_countries(self):
        return self.config['outline']['countries']
    
    def get_years(self):
        return self.config['outline']['years']
    
    def get_steps_rate_activity(self):
        """
        Returns the steps rate activity for the model.
        :return: int
        """
        self.logger.debug("Steps rate activity set in config parser")
        return self.config['outline']['steps_rate_activity']
    
    def get_domains(self):
        """
        Returns the domains for the model.
        :return: list
        """
        self.logger.debug("Domains set in config parser")
        return self.config['outline']['domains']
    
    def get_cost_transmission_line(self):
        """
        Returns the cost of the transmission line for the model.
        :return: float
        """
        self.logger.debug("Cost transmission line set in config parser")
        return self.config['outline']['cost_transmission_line']
    
    def get_missing_computation(self):
        """
        Returns whether the missing computation is enabled for the model.
        :return: bool
        """
        self.logger.debug("Missing computation set in config parser")
        return self.config['outline']['missing_computation']
    
    
    def get_timeout_time_steps(self):
        """
        Returns the timeout for the model.
        :return: int
        """
        self.logger.debug("Timeout set in config parser")
        return self.config['outline']['timeout_time_step']
    
    def get_expansion_enabled(self):
        """
        Returns the expansion enabled for the model.
        :return: bool
        """
        self.logger.debug("Expansion enabled set in config parser")
        return self.config['outline']['expansion_enabled']

    @deprecated(reason="Data extracted by dataParser class")
    def get_powerplants_data(self):
        powerplants = []
        for plant, details in self.powerplants_config.items():
            for country, capacity in details["max_installable_capacity_MW"].items():
                powerplants.append({
                    "Technology": plant,
                    "Installation Cost ($/MW)": details["installation_cost_per_MW"],
                    "Operating Cost ($/MWh)": details["operating_cost_per_MWh"],
                    "Fuel Cost ($/MWh)": details["fuel_cost_per_MWh"],
                    "CO2 Emissions (tCO2/MWh yearly)": round(details["emissions"]["CO2"] * 87.60),
                    "N2O Emissions (tN2O/MWh yearly)": round(details["emissions"]["N2O"] * 87.60),
                    "CH4 Emissions (tCH4/MWh yearly)": round(details["emissions"]["CH4"] * 87.60),
                    "CFC Emissions (tCFC/MWh yearly)": round(details["emissions"]["CFCs"] * 87.60),
                    "Mean Capacity Factor": details["capacity_factor"]["mean"],
                    "Country": country,
                    "Max Installable Capacity (MW)": capacity
                })

        df = pd.DataFrame(powerplants)
        return df
    

