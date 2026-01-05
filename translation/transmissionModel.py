from translation.xmlGenerator import XMLGeneratorClass
from deprecated import deprecated
import pandas as pd
import logging
import numpy as np
import os
from itertools import product
from enum import Enum

class TransmissionModelClass:
    def __init__(self, countries, data, delta_demand_map, marginal_costs_df, cost_transmission_line, cost_threshold, logger, xml_file_path, expansion_enabled):
        self.logger = logger
        self.countries = countries
        self.data = data
        self.delta_demand_map = delta_demand_map
        self.marginal_costs_df = marginal_costs_df
        self.cost_transmission_line = cost_transmission_line
        self.cost_percentage_threshold = cost_threshold
        self.xml_file_path = xml_file_path
        self.xml_generator = XMLGeneratorClass(logger=self.logger)
        self.expansion_enabled = expansion_enabled

        self.logger.debug('Marginal costs DataFrame initialized')
        self.logger.debug(self.marginal_costs_df)

    def update_data(self):
        # Update self.data['capacity'] to be the minimum of the rounded marginal demands of the two countries and the original capacity
        if not self.data.empty:
            self.data['capacity'] = self.data.apply(
                lambda row: min(
                round(self.delta_demand_map[row['start_country']]['marginal_demand']),
                round(self.delta_demand_map[row['end_country']]['marginal_demand']),
                row['capacity']
                ),
                axis=1
            )

    def solve(self):
        self.update_data()
        agents = {}
        for country in self.countries:
            agents[country] = TransmissionAgentClass(
                country=country,
                logger=self.logger,
                data = self.data[self.data['start_country'] == country],
                MC_import = self.marginal_costs_df.loc[country, 'MC_import'],
                MC_export = self.marginal_costs_df.loc[country, 'MC_export'],
                cost_percentage_threshold = self.cost_percentage_threshold,
                transmission_cost = self.cost_transmission_line,
                marginal_demand = round(self.delta_demand_map[country]['marginal_demand'])
            )

        rows = []
        while len(agents) > 1:
            for name, agent in agents.items():
                bids = agent.send_bids()
                for bid in bids:
                    agents[bid.start_country].receive_bid(bid)
            self.logger.debug("All bids sent and received")

            for name, agent in agents.items():
                agent.process_bids()
            self.logger.debug("All bids processed")
            
            for agent in agents.values():
                for bid in agent.outbox:  # Assuming each agent stores sent bids in an outbox
                    if bid.status_sender == TransmissionBidStatus.ACCEPTED and bid.status_receiver == TransmissionBidStatus.ACCEPTED:
                        rows.append({
                            'start_country': bid.start_country,
                            'end_country': bid.end_country,
                            'exchange': bid.capacity,
                            'price': bid.price
                        })
                        rows.append({
                            'start_country': bid.end_country,
                            'end_country': bid.start_country,
                            'exchange': -bid.capacity,
                            'price': bid.price
                        })
                        self.delta_demand_map[bid.start_country]['marginal_demand'] -= abs(bid.capacity)
                        self.delta_demand_map[bid.end_country]['marginal_demand'] -= abs(bid.capacity)
                        self.data.loc[(self.data['start_country'] == bid.start_country) & (self.data['end_country'] == bid.end_country), 'capacity'] -= abs(bid.capacity)
                        self.data.loc[(self.data['start_country'] == bid.end_country) & (self.data['end_country'] == bid.start_country), 'capacity'] -= abs(bid.capacity)

                        agent.add_trade(capacity=abs(bid.capacity))
                        agents[bid.start_country].add_trade(capacity=abs(bid.capacity))
            self.update_data()
            # Remove agents that are satisfied or excluded
            to_remove_agents = [name for name, agent in agents.items() if agent.process_status() == 'satisfied' or agent.process_status() == 'excluded']
            for name in to_remove_agents:
                del agents[name]
                self.logger.debug(f"Removed agent {name} from the model")
            self.data = self.data[~self.data['start_country'].isin(to_remove_agents) & ~self.data['end_country'].isin(to_remove_agents)]
            for agent in agents.values():
                agent.erase_messages()
                agent.erase_neighbours(to_remove_agents)
                agent.update_data(marginal_demand=self.delta_demand_map[agent.country]['marginal_demand'], data=self.data[self.data['start_country'] == agent.country])

                self.logger.debug(f"Erased messages for agent {agent.country}")

        return pd.DataFrame(rows)
    
class TransmissionAgentClass:
    def __init__(self, country, logger, data, MC_import, MC_export, cost_percentage_threshold, transmission_cost, marginal_demand):
        self.country = country
        self.logger = logger
        self.data = data
        self.MC_import = MC_import
        self.MC_export = MC_export
        self.cost_percentage_threshold = cost_percentage_threshold
        self.transmission_cost = transmission_cost
        self.marginal_demand = marginal_demand
        self.status = 'unsatisfied'

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
                        price=round((self.MC_import) * (1 + self.cost_percentage_threshold), 2)
                    )
                    self.outbox.append(bid)
        return self.outbox
    
    def receive_bid(self, bid):
        self.logger.debug(f"Receiving bid from {bid.sender} to {bid.end_country}")
        bid.set_MC_exporter(self.MC_export)
        self.inbox.append(bid)
    
    def process_status(self):
        if self.marginal_demand <= 0:
            self.status = 'satisfied'
        return self.status
    
    def add_trade(self, capacity):
        self.logger.debug(f"Adding trade for {self.country} with capacity {capacity}")
        self.marginal_demand = round(self.marginal_demand - capacity)
    
    def erase_messages(self):
        self.inbox = []
        self.outbox = []
        self.logger.debug(f"Erased messages for {self.country}")

    def erase_neighbours(self, neighbours):
        self.logger.debug(f"Erasing neighbours for {self.country}: {neighbours}")
        self.data = self.data[~self.data['end_country'].isin(neighbours)]

    def update_data(self, marginal_demand, data):
        self.logger.debug(f"Adding trade for {self.country} with marginal demand {marginal_demand}")
        self.marginal_demand = round(marginal_demand)
        self.data = data
        
    def process_bids(self):
        self.logger.debug(f"Processing bids for {self.country}")

        # Gather all bids: incoming (inbox) and outgoing (outbox)
        all_bids = []
        for bid in self.inbox:
            utility = (bid.price + self.transmission_cost - self.MC_export * (1 + self.cost_percentage_threshold)) * bid.capacity
            all_bids.append((utility, 'exporter', bid))
            self.logger.debug(f"Bid from {bid.sender} to {bid.end_country} with utility {utility} as exporter")
        for bid in self.outbox:
            utility = (bid.price + self.transmission_cost - bid.MC_exporter * (1 + self.cost_percentage_threshold)) * bid.capacity
            all_bids.append((utility, 'importer', bid))
            self.logger.debug(f"Bid from {self.country} to {bid.end_country} with utility {utility} as importer")

        # Sort by utility descending
        all_bids.sort(key=lambda x: x[0], reverse=True)

        if len(all_bids) > 0 and all_bids[0][0] <= 0 or len(all_bids) == 0:
            self.status = 'excluded'
            return

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
