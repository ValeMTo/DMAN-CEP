import os
import xml.etree.ElementTree as ET
import pandas as pd
import pickle

class xmlEnergyReader():
    def __init__(self, k, t, year, countries):

        self.k = k
        self.t = t
        self.year = year

        self.countries_check = {}
        self.countries = countries

        self.df = pd.DataFrame()
        self.tech_df = pd.DataFrame(columns=['technology','country', 'demand_type', 'capacity', 'rateActivity'])

        self.transmission_df = pd.DataFrame(columns=['start_country', 'end_country', 'exchange'])


    def load(self, type='internal', output_folder_path=None):
        if output_folder_path is None:
            raise ValueError("output_folder_path must be provided")
        self.output_folder_path = output_folder_path

        if type == 'internal':
            return self.load_internal()
        elif type == 'transmission':
            return self.load_transmission()
        else:
            raise ValueError(f"Unsupported type: {type}")

    def load_internal(self):
        self.countries_check = {country: {'-': -1, '+': -1, '0': -1} for country in self.countries}
        rows = []
        for file_name in os.listdir(self.output_folder_path):
            if file_name.endswith("_output.xml"):
                file_path = os.path.join(self.output_folder_path, file_name)
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    country_name = file_name.split("_")[0]
                    demand_type = file_name.split("_")[1]
                    if "valuation" not in root.attrib:
                        self.countries_check[country_name][file_name.split('_')[1][0]] = 0
                    elif root.attrib.get("valuation") == "infinity":
                        self.countries_check[country_name][file_name.split('_')[1][0]] = float('inf')
                        raise ValueError(f"File {file_name} has an 'infinity' valuation. UNFEASIBLE")
                    else: 
                        self.countries_check[country_name][file_name.split('_')[1][0]] = int(root.attrib["valuation"])    

                        for assignment in root.findall("assignment"):
                            var = assignment.attrib["variable"]
                            value = int(assignment.attrib["value"])
                            tech, attr = var.split("_")
                            rows.append({
                                'technology': tech,
                                'country': country_name,
                                'demand_type': demand_type,
                                'capacity': value if attr == "capacity" else 0,
                                'rateActivity': value if attr == "rateActivity" else 0,
                            })
            
                except ET.ParseError as e:
                    raise ValueError(f"Error parsing XML file {file_name}: {e}")
                
        self.tech_df = pd.DataFrame(rows)
        if not self.tech_df.empty:
            self.tech_df = self.tech_df.groupby(['technology', 'country', 'demand_type'], as_index=True).sum().reset_index()

        return self.countries_check
                
    def fill_missing_countries(self):
        for country in self.countries:
            for demand_type in ['-', '+', '0']:
                if self.countries_check[country][demand_type] == 0:
                    fallback_order = {
                    '-': ['0', '+'],
                    '+': ['0', '-'],
                    '0': ['+', '-']
                    }.get(demand_type, ['0', '+', '-'])  # default fallback order

                    for fallback in fallback_order:
                        fallback_val = self.countries_check[country].get(fallback)
                        if fallback_val != -1 and fallback_val != 0:
                            self.countries_check[country][demand_type] = fallback_val
                            break

        # After filling, check for any remaining zeros and raise an error if found
        for country in self.countries:
            for demand_type in ['-', '+', '0']:
                if self.countries_check[country][demand_type] == 0 or self.countries_check[country][demand_type] == -1:
                    raise ValueError(f"Zero still present for country '{country}' and demand type '{demand_type}' after attempting to fill missing values.")
        

    def load_transmission(self):
        for file_name in os.listdir(self.output_folder_path):
            if file_name.endswith("_output.xml"):
                file_path = os.path.join(self.output_folder_path, file_name)
                tree = ET.parse(file_path)
                root = tree.getroot()

                if "valuation" in root.attrib:

                    rows = []
                    for assignment in root.findall("assignment"):
                        var = assignment.attrib["variable"]
                        value = int(assignment.attrib["value"])
                        start_country, end_country = var.split("_")[1:3]
                        rows.append({
                            'start_country': start_country,
                            'end_country': end_country,
                            'exchange': int(value)
                        })
                    self.transmission_df = pd.DataFrame(rows)
                else:
                    raise ValueError(f"File {file_name} does not have a 'valuation' parameter")
    
    def get_internal_outputs(self):
        if self.tech_df.empty:
            raise ValueError("No internal outputs available. Please load the internal data first.")
        return self.tech_df.copy()

    def get_transmission_outputs(self):
        if self.transmission_df.empty:
            raise ValueError("No transmission outputs available. Please load the transmission data first.")  
        return self.transmission_df.copy()
    
    def set_general_picture(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        self.df = df

    def save(self, folder, transmission_data):
        if not transmission_data.empty:
            transmission_data = transmission_data.groupby(['start_country', 'end_country'], as_index=False).sum()
            self.transmission_df = transmission_data

        file_path = os.path.join(folder, f"xmlEnergyReader_{self.k}_{self.t}_{self.year}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump({
                'k': self.k,
                't': self.t,
                'year': self.year,
                'countries_check': self.countries_check,
                'df': self.df,
                'tech_df': self.tech_df,
                'transmission_df': self.transmission_df
            }, f)

    def unpickle(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.k = data['k']
            self.t = data['t']
            self.year = data['year']
            self.countries_check = data['countries_check']
            self.df = data['df']
            self.tech_df = data['tech_df']
            self.transmission_df = data['transmission_df']