from translation.xmlGenerator import XMLGeneratorClass
from deprecated import deprecated
import pandas as pd
import logging
import numpy as np
import os
from itertools import product
from enum import Enum

class TransmissionModelClass:
    def __init__(self, countries, data, delta_demand_map, marginal_costs_df, cost_transmission_line, logger, xml_file_path, expansion_enabled):
        self.logger = logger
        self.countries = countries
        self.data = data
        self.delta_demand_map = delta_demand_map
        self.marginal_costs_df = marginal_costs_df
        self.cost_transmission_line = cost_transmission_line
        self.xml_file_path = xml_file_path
        self.xml_generator = XMLGeneratorClass(logger=self.logger)
        self.expansion_enabled = expansion_enabled

        self.logger.debug('Marginal costs DataFrame initialized')
        self.logger.debug(self.marginal_costs_df)

    def solve(self):

        # Update self.data['capacity'] to be the minimum of the rounded marginal demands of the two countries and the original capacity
        self.data['capacity'] = self.data.copy().apply(
            lambda row: min(
            round(self.delta_demand_map[row['start_country']]['marginal_demand']),
            round(self.delta_demand_map[row['end_country']]['marginal_demand']),
            row['capacity']
            ),
            axis=1
        )

        agents = {}
        for country in self.countries:
            agents[country] = TransmissionAgentClass(
                country=country,
                logger=self.logger,
                data = self.data[self.data['start_country'] == country],
                MC_import = self.marginal_costs_df.loc[country, 'MC_import'],
                MC_export = self.marginal_costs_df.loc[country, 'MC_export'],
                transmission_cost = self.cost_transmission_line,
                marginal_demand = self.delta_demand_map[country]['marginal_demand']
            )
        
        for name, agent in agents.items():
            bids = agent.send_bids()
            for bid in bids:
                agents[bid.start_country].receive_bid(bid)
        self.logger.debug("All bids sent and received")

        for name, agent in agents.items():
            agent.process_bids()
        self.logger.debug("All bids processed")
        
        rows = []
        for agent in agents.values():
            for bid in agent.outbox:  # Assuming each agent stores sent bids in an outbox
                if bid.status_sender == TransmissionBidStatus.ACCEPTED and bid.status_receiver == TransmissionBidStatus.ACCEPTED:
                    rows.append({
                        'start_country': bid.start_country,
                        'end_country': bid.end_country,
                        'exchange': bid.capacity
                    })
                    rows.append({
                        'start_country': bid.end_country,
                        'end_country': bid.start_country,
                        'exchange': -bid.capacity
                    })

        return pd.DataFrame(rows)

    def generate_xml(self):
        self.logger.debug("Generating XML for transmission model")

        self.xml_generator.add_presentation(name="TransmissionModel", maximize='False')
        self.xml_generator.add_agents(self.countries)
        domains = self.generate_domains()
        self.xml_generator.add_domains(domains)

        for idx, row in self.data.iterrows():
            self.xml_generator.add_variable(name=f"transmission_{row['start_country']}_{row['end_country']}", domain=f"{row['start_country']}_domain", agent=row['start_country'])
        
        already_inside = {}
        for idx, row in self.data.iterrows():   
            if f"{row['end_country']}_{row['start_country']}" not in already_inside:
                self.xml_generator.add_symmetry_constraint(
                    extra_name=f"{row['start_country']}_{row['end_country']}",
                    var1=f"transmission_{row['start_country']}_{row['end_country']}",
                    var2=f"transmission_{row['end_country']}_{row['start_country']}"
                )
                already_inside[f"{row['start_country']}_{row['end_country']}"] = True

        for country in self.countries:
            # Constraint: Power balance per country
            transmission_variables = [f"transmission_{row['start_country']}_{row['end_country']}" for idx, row in self.data.iterrows() if row['start_country'] == country]
            marginal_demand=max(round(self.delta_demand_map[country]['marginal_demand']), 0)
            self.xml_generator.add_power_balance_constraint(
                extra_name = f"{country}",
                flow_variables = transmission_variables,
                delta=marginal_demand,
            )
            # Soft constraint: Utility function for transmission costs
            self.xml_generator.add_utility_function_constaint(
                extra_name=f"{country}",
                variables= transmission_variables,
                import_marginal_costs = [round(self.marginal_costs_df.loc[row['end_country'], 'MC_import']) for idx, row in self.data.iterrows() if row['start_country'] == country],
                export_marginal_costs = [round(self.marginal_costs_df.loc[country, 'MC_export'])]*len(transmission_variables),
                cost = round(self.cost_transmission_line),
            )

        # Constraint: Transmission capacity should be less than or equal to the maximum capacity
        for idx, row in self.data.iterrows():
            marginal_demand=max(round(self.delta_demand_map[row['start_country']]['marginal_demand']), 0)
            self.xml_generator.add_maximum_capacity_constraint(
                variable_name=f"transmission_{row['start_country']}_{row['end_country']}",
                max_capacity=min(round(row['capacity']*0.7), marginal_demand),
            )

    def reduce_marginal_cost_magnitude(self, df):
        self.logger.debug("Reducing marginal cost magnitude")
        df = df.copy()
        max_value = max(abs(df['MC_import']).max(), abs(df['MC_export']).max())
        df['MC_import'] = pd.to_numeric((df['MC_import'] / max_value * 500)).round().astype(int)
        df['MC_export'] = pd.to_numeric((df['MC_export'] / max_value * 500)).round().astype(int)
        return df
    
    def generate_domains(self):
        self.logger.debug("Generating domains for transmission model")
        domains = {}
        for country in self.countries:
            max_domain = round(self.delta_demand_map[country]['marginal_demand'])
            positive_domain = range(0, max_domain, 50)
            negative_domain = [ -var for var in positive_domain]
            domain_values = set(negative_domain + list(positive_domain))
            domain_values.add(0)
            domains[f"{country}_domain"] =  sorted(domain_values)
        return domains
  
    def print_xml(self, name):
        self.logger.debug(f"Printing XML for {name}")

        if not os.path.exists(self.xml_file_path):
            os.makedirs(self.xml_file_path)

        self.xml_generator.print_xml(output_file=os.path.join(self.xml_file_path, name))
        self.logger.debug(f"XML generated for at {os.path.join(self.xml_file_path, name)}")
    
class TransmissionAgentClass:
    def __init__(self, country, logger, data, MC_import, MC_export, transmission_cost, marginal_demand):
        self.country = country
        self.logger = logger
        self.data = data
        self.MC_import = MC_import
        self.MC_export = MC_export
        self.transmission_cost = transmission_cost
        self.marginal_demand = marginal_demand

        self.inbox = []
        self.outbox = []

    def send_bids(self):
        self.logger.debug(f"Sending bids from {self.country}")
        if self.MC_import > self.transmission_cost:
            for idx, row in self.data.iterrows():
                if row['capacity'] > 0:
                    bid = TransmissionBidClass(
                        sender=self.country,
                        start_country=row['end_country'],
                        end_country=row['start_country'],
                        capacity=row['capacity'],
                        price=(self.MC_import - self.transmission_cost)
                    )
                    self.outbox.append(bid)
        return self.outbox
    
    def receive_bid(self, bid):
        self.logger.debug(f"Receiving bid from {bid.sender} to {bid.end_country}")
        bid.set_MC_exporter(self.MC_export)
        self.inbox.append(bid)

    def process_bids(self):
        self.logger.debug(f"Processing bids for {self.country}")

        # Gather all bids: incoming (inbox) and outgoing (outbox)
        all_bids = []
        for bid in self.inbox:
            # As importer
            utility = (bid.price - self.MC_export) * bid.capacity
            all_bids.append((utility, 'exporter', bid))
            self.logger.debug(f"Bid from {bid.sender} to {bid.end_country} with utility {utility} as exporter")
        for bid in self.outbox:
            # As exporter
            utility = (bid.price - bid.MC_exporter) * bid.capacity
            all_bids.append((utility, 'importer', bid))
            self.logger.debug(f"Bid from {self.country} to {bid.end_country} with utility {utility} as importer")

        # Sort by utility descending
        all_bids.sort(key=lambda x: x[0], reverse=True)

        # Track used links and capacities
        used_links = set()
        capacity_used = 0
        for utility, role, bid in all_bids:
            accepted = False
            if capacity_used + bid.capacity <= self.marginal_demand and utility > 0:
                if bid.start_country == self.country and bid.end_country not in used_links:
                    used_links.add(bid.end_country)
                    capacity_used += bid.capacity
                    accepted = True
                elif bid.end_country == self.country and bid.start_country not in used_links:
                    used_links.add(bid.start_country)
                    capacity_used += bid.capacity
                    accepted = True
            if accepted:
                if role == 'exporter':
                    bid.change_status_receiver(TransmissionBidStatus.ACCEPTED)
                elif role == 'importer':
                    bid.change_status_sender(TransmissionBidStatus.ACCEPTED)
            else:
                if role == 'exporter':
                    bid.change_status_receiver(TransmissionBidStatus.REJECTED)
                elif role == 'importer':
                    bid.change_status_sender(TransmissionBidStatus.REJECTED)
        

class TransmissionBidClass:
    def __init__(self,
            sender,
            start_country,
            end_country,
            capacity,
            price):
        self.sender = sender
        self.start_country = start_country
        self.end_country = end_country
        self.capacity = capacity
        self.price = price
        self.MC_exporter = np.inf 
        self.status_sender = TransmissionBidStatus.PENDING
        self.status_receiver = TransmissionBidStatus.PENDING
    
    def change_status_sender(self, new_status):
        if isinstance(new_status, TransmissionBidStatus):
            self.status_sender = new_status
        else:
            raise ValueError("Status must be an instance of TransmissionBidStatus Enum")

    def change_status_receiver(self, new_status):
        if isinstance(new_status, TransmissionBidStatus):
            self.status_receiver = new_status
        else:
            raise ValueError("Status must be an instance of TransmissionBidStatus Enum")
        
    def set_MC_exporter(self, MC_exporter):
        if isinstance(MC_exporter, (int, float, np.integer, np.floating)):
            self.MC_exporter = MC_exporter
        else:
            raise ValueError("MC_exporter must be a numeric value")
class TransmissionBidStatus(Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"
