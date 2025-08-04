from translation.xmlGenerator import XMLGeneratorClass
from pyomo.opt import SolverFactory
from deprecated import deprecated
import pandas as pd
import logging
import os
import io
import traceback
from itertools import product
from pyomo.environ import *
import contextlib



class EnergyAgentClass:
    def __init__(self, country, logger, data, year_split, demand):
        self.name = country
        self.logger = logger
        self.data = data
        self.year_split = year_split
        self.demand = demand # Total demand in the country for the year in that time resolution
        self.model = None
        self.logger.debug(f"EnergyAgentClass initialized for {country}")

    def run(self):
        self.preprocess_data()
        self.build_model()
        return self.solve()

    def solve(self):
        self.logger.debug(f"Solving model for {self.name}")
        try:
            solver = SolverFactory('gurobi')

            # Capture stdout from Gurobi
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                results = solver.solve(self.model, tee=True)  # tee=True prints to redirected stdout

            self.logger.info(f"Solver output for {self.name}:\n{stream.getvalue()}")
            return self.process_results(results)

        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Solver failed: {e}\n{tb}")
            raise RuntimeError("Solver failed")

    def build_model(self):
        def min_capacity_rule(m, k):
            return m.capacity[k] >= m.min_capacity[k]
        def max_activity_rule(m, k):
            return m.rate_activity[k] <= (m.capacity[k] * m.factor[k])
        def demand_rule(m):
            return sum(m.rate_activity[k] for k in m.TECHS) >= self.demand
        def total_cost_rule(m):
            return sum(
                (m.capacity[k] - m.min_capacity[k]) * m.capital_cost[k] +
                m.capacity[k] * m.fixed_cost[k] +
                m.rate_activity[k] * m.variable_cost[k]
                for k in m.TECHS
            )
        def max_emission_rule(m):
            pass

        self.logger.debug(f"Building model for {self.name}")

        model = ConcreteModel(name=f"{self.name}_model")

        model.TECHS = Set(initialize=list(self.tech_df.index))
        model.capacity = Var(model.TECHS, domain=NonNegativeReals)
        model.rate_activity= Var(model.TECHS, domain=NonNegativeReals)

        model.capital_cost = Param(model.TECHS, initialize=lambda m, t: self.tech_df.loc[t, 'CAPITAL_COST'] / self.tech_df.loc[t, 'OPERATIONAL_LIFETIME'])
        model.variable_cost = Param(model.TECHS, initialize=lambda m, t: self.tech_df.loc[t, 'VARIABLE_COST'])
        model.fixed_cost = Param(model.TECHS, initialize=lambda m, t: self.tech_df.loc[t, 'FIXED_COST'] * self.year_split)
        model.factor = Param(model.TECHS, initialize=lambda m, t: self.tech_df.loc[t, 'FACTOR'])
        model.min_capacity = Param(model.TECHS, initialize=lambda m, t: self.tech_df.loc[t, 'MIN_INSTALLED_CAPACITY'] if pd.notna(self.tech_df.loc[t, 'MIN_INSTALLED_CAPACITY']) else 0)
        model.max_capacity = Param(model.TECHS, initialize=lambda m, t: self.tech_df.loc[t, 'TOTAL_ANNUAL_CAPACITY'] if pd.notna(self.tech_df.loc[t, 'TOTAL_ANNUAL_CAPACITY']) else float('inf'))
        model.max_annual_rate = Param(model.TECHS, initialize=lambda m, t: self.tech_df.loc[t, 'TOTAL_ANNUAL_ACTIVITY_UPPER_LIMIT'] * self.year_split if pd.notna(self.tech_df.loc[t, 'TOTAL_ANNUAL_ACTIVITY_UPPER_LIMIT']) else float('inf'))
        model.min_annual_rate = Param(model.TECHS, initialize=lambda m, t: self.tech_df.loc[t, 'TOTAL_ANNUAL_ACTIVITY_LOWER_LIMIT'] * self.year_split if pd.notna(self.tech_df.loc[t, 'TOTAL_ANNUAL_ACTIVITY_LOWER_LIMIT']) else 0)


        model.MinCapacityConstraint = Constraint(model.TECHS, rule=min_capacity_rule)
        model.MaxActivityConstraint = Constraint(model.TECHS, rule=max_activity_rule)
        model.DemandConstraint = Constraint(rule=demand_rule)

        model.MaxCapacityConstraint = Constraint(model.TECHS, rule=lambda m, k: m.capacity[k] <= m.max_capacity[k])
        model.MaxAnnualRateConstraint = Constraint(model.TECHS, rule=lambda m, k: m.rate_activity[k] <= m.max_annual_rate[k])
        model.MinAnnualRateConstraint = Constraint(model.TECHS, rule=lambda m, k: m.rate_activity[k] >= m.min_annual_rate[k])

        model.TotalCostObjective = Objective(rule=total_cost_rule, sense=minimize)

        self.model = model

    def preprocess_data(self):
        self.logger.debug(f"Preprocessing data for {self.name}")
        
        self.tech_df = self.data
        self.tech_df['AVAILABILITY_FACTOR'] = self.tech_df['AVAILABILITY_FACTOR'].fillna(0.9)  # Default to 1 if NaN
        self.tech_df['FACTOR'] = self.tech_df['CAPACITY_FACTOR'] * self.tech_df['AVAILABILITY_FACTOR'] * self.tech_df['CAPACITY_TO_ACTIVITY_UNIT'] * self.year_split
        self.tech_df['FACTOR'] = self.tech_df['FACTOR'].fillna(0)  # Replace NaN factors with zero
        self.tech_df = self.tech_df[self.tech_df['FACTOR'] > 0]
        self.tech_df = self.tech_df[
            self.tech_df['CAPITAL_COST'].notna() &
            self.tech_df['VARIABLE_COST'].notna() &
            self.tech_df['FIXED_COST'].notna()
        ]
    def process_results(self, results):
        self.logger.debug(f"Processing results for {self.name}")
        if results.solver.termination_condition == TerminationCondition.optimal:
            self.logger.info(f"Optimal solution found for {self.name}")
            solution = {
                'technology': list(self.model.TECHS),
                'capacity': [value(self.model.capacity[k]) for k in self.model.TECHS],
                'rate_activity': [value(self.model.rate_activity[k]) for k in self.model.TECHS],
                'capital_cost': [value(self.model.capital_cost[k]) for k in self.model.TECHS],
                'variable_cost': [value(self.model.variable_cost[k]) for k in self.model.TECHS],
                'fixed_cost': [value(self.model.fixed_cost[k]) for k in self.model.TECHS],
                'factor': [value(self.model.factor[k]) for k in self.model.TECHS],
                'min_capacity': [value(self.model.min_capacity[k]) for k in self.model.TECHS]
            }
            df = pd.DataFrame(solution)
            df['country'] = self.name
            return value(self.model.TotalCostObjective), self.demand, df
        else:
            self.logger.error(f"No optimal solution found for {self.name}")
            raise RuntimeError("No optimal solution found")
        
    def change_demand(self, demand_variation_percentage):
        if self.model is not None:
            self.logger.debug(f"Changing demand for {self.name} by {demand_variation_percentage}%")

            self.demand = self.demand * (1 + demand_variation_percentage)
            self.build_model()