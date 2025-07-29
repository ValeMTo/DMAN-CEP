import xml.etree.ElementTree as ET
import xml.dom.minidom
from deprecated import deprecated
import pandas as pd

class XMLGeneratorClass:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("XML generator initialized")

        self.instance = self.create_frodo2_xml_head_instance()

        self.functions = {}
        self.predicates = {}

        self.max_arity = 1

    def create_frodo2_xml_head_instance(self):
        instance = ET.Element("instance", {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "src/frodo2/algorithms/XCSPschemaJaCoP.xsd"
        })

        return instance
    
    def add_presentation(self, name, maximize):
        ET.SubElement(self.instance, "presentation", {
            "name": name,
            "maxConstraintArity": str(self.max_arity),
            "maximize": maximize,
            "format": "XCSP 2.1_FRODO"
        })
    
    def add_agents(self, agent_names):
        """Adds multiple agents inside a single <agents> element."""
        agents_element = self.instance.find("agents")
        if agents_element is None:
            agents_element = ET.SubElement(self.instance, "agents")

        for agent_name in agent_names:
            ET.SubElement(agents_element, "agent", {"name": agent_name})


    def add_domains(self, domain_values):
        """Adds multiple domains inside a single <domains> element."""
        domains_element = self.instance.find("domains")
        if domains_element is None:
            domains_element = ET.SubElement(self.instance, "domains")
              
        for name, values in domain_values.items():
            ET.SubElement(domains_element, "domain", {"name": name, "nbValues": str(len(values))}).text = " ".join(map(str, values))

    @deprecated(reason="Time resolution is not anymore fixed")
    def add_variable_from_name(self, technologies, variables, agents):

        variables_element = self.instance.find("variables")
        if variables_element is None:
            variables_element = ET.SubElement(self.instance, "variables")

        variable_list = []
        for tech_capacity in technologies:
            ET.SubElement(variables_element, "variable", {
                    "name": f"{tech_capacity}_capacity", 
                    "domain": "installable_capacity_domain", 
                    "agent": tech_capacity[:2]
                })
            variable_list.append(f"{tech_capacity}_capacity")
        for variable_rateOfCapacity in variables:
            ET.SubElement(variables_element, "variable", {
                "name": f"{variable_rateOfCapacity}_rateActivity", 
                "domain": "rate_activity_domain", 
                "agent": variable_rateOfCapacity.split('_')[1][:2]
            })
            variable_list.append(f"{variable_rateOfCapacity}_rateActivity")

        #TODO: change once the agent_names are not only neighboring
        # for fuel in fuels:
        #     for agent in agents:
        #         for agent2 in agents:
        #             if agent != agent2:
        #                 ET.SubElement(variables_element, "variable", {
        #                     "name": f"transmission_{agent}_{agent2}",
        #                     "domain": "trasferable_capacity_domain",
        #                     "agent": agent
        #                 })
        #                 variable_list.append(f"transmission_{agent}_{agent2}")

        return variable_list

    @deprecated
    def add_variables(self, technologies, agent_names):
        """Adds multiple variables inside a single <variables> element."""
        variables_element = self.instance.find("variables")
        if variables_element is None:
            variables_element = ET.SubElement(self.instance, "variables")

        variable_list = []
        for technology in technologies:
            for agent in agent_names:
                ET.SubElement(variables_element, "variable", {
                    "name": f"{agent}{technology}_capacity", 
                    "domain": "installable_capacity_domain", 
                    "agent": agent
                })
                ET.SubElement(variables_element, "variable", {
                    "name": f"{agent}{technology}_rateActivity", 
                    "domain": "rate_activity_domain", 
                    "agent": agent
                })
                variable_list.append(f"{agent}{technology}_capacity")
                variable_list.append(f"{agent}{technology}_rateActivity")

        # TODO: change once the agent_names are not only neighboring
        for agent in agent_names:
            for agent2 in agent_names:
                if agent != agent2:
                    ET.SubElement(variables_element, "variable", {
                        "name": f"transmission_{agent}_{agent2}",
                        "domain": "trasferable_capacity_domain",
                        "agent": agent
                    })
                    variable_list.append(f"transmission_{agent}_{agent2}")

        return variable_list
    
    def add_variable(self, name, domain, agent):
        """Adds a single variable element with name, domain and agent in the XML file."""

        variables_element = self.instance.find("variables")
        if variables_element is None:
            variables_element = ET.SubElement(self.instance, "variables")

        ET.SubElement(variables_element, "variable", {
            "name": name, 
            "domain": domain, 
            "agent": agent
        })
    
    def add_predicate(self, name, parameters, functional):
        """Adds a single predicate element with parameters and functional expression."""
        predicates_element = self.instance.find("predicates")
        if (predicates_element is None):
            predicates_element = ET.SubElement(self.instance, "predicates")
        
        predicate_element = ET.SubElement(predicates_element, "predicate", {"name": name})
        ET.SubElement(predicate_element, "parameters").text = parameters
        expression_element = ET.SubElement(predicate_element, "expression")
        ET.SubElement(expression_element, "functional").text = functional
        self.predicates[name] = 1

    def add_function(self, name, parameters, functional):
        """Adds a single function element with parameters and functional expression."""
        functions_element = self.instance.find("functions")
        if (functions_element is None):
            functions_element = ET.SubElement(self.instance, "functions")
        
        functions_element = ET.SubElement(functions_element, "function", {"name": name, "return": "int"})
        ET.SubElement(functions_element, "parameters").text = parameters
        expression_element = ET.SubElement(functions_element, "expression")
        ET.SubElement(expression_element, "functional").text = functional
        self.functions[name] = 1

    def add_constraint(self, name, arity, scope, reference, parameters):
        """Adds a single constraint element with arity, scope and reference."""

        constraints_element = self.instance.find("constraints")
        if (constraints_element is None):
            constraints_element = ET.SubElement(self.instance, "constraints")
        
        constraint_element = ET.SubElement(constraints_element, "constraint", {"name": name, "arity": str(arity), "scope": scope, "reference": reference})
        ET.SubElement(constraint_element, "parameters").text = parameters
        self.predicates[reference] = 1

        if arity > self.max_arity:
            self.max_arity = int(arity)

    def find_predicate(self, name):
        """Finds a predicate element by name."""
        if self.predicates.get(name):
            return True
        return False
        
    def find_function(self, name):
        """Finds a function element by name."""
        if self.functions.get(name):
            return True
        return False

    def add_minimum_capacity_constraint(self, variable_name, min_capacity):
        """Adds an hard constraint to the XML instance that enforces minimum installed capacity."""

        if not isinstance(min_capacity, int):
            raise ValueError("min_capacity must be an integer")
        
        if not self.find_predicate("alreadyInstalledCapacity"):
            self.add_predicate(
                name="alreadyInstalledCapacity", 
                parameters="int capacity int min_capacity", 
                functional=boolean_ge("capacity", "min_capacity")
            )
        
        self.add_constraint(
            name=f"alreadyInstalledCapacity_{variable_name}", 
            arity=1, 
            scope=variable_name, 
            reference="alreadyInstalledCapacity",
            parameters=f"{variable_name} {min_capacity}"
        )

    def add_maximum_capacity_constraint(self, variable_name, max_capacity):
        """Adds an hard constraint to the XML instance that enforces maximum installed capacity."""

        if not isinstance(max_capacity, int):
            raise ValueError("max_capacity must be an integer")
        
        if not self.find_predicate("withinMaxCapacity"):
            self.add_predicate(
                name="withinMaxCapacity", 
                parameters="int capacity int max_capacity", 
                functional=boolean_le("capacity", "max_capacity")
            )
        
        self.add_constraint(
            name=f"withinMaxCapacity_{variable_name}", 
            arity=1, 
            scope=variable_name, 
            reference="withinMaxCapacity",
            parameters=f"{variable_name} {str(max_capacity)}"
        )

    def add_minimum_demand_constraint(self, variables, demand, extra_name):
        """Adds an hard constraint to the XML instance that enforces minimum demand."""
        def build_recursive(variables):
            if len(variables) == 1:
                return variables[0]
            return add(variables[0], build_recursive(variables[1:]))
        
        self.logger.debug(f"Adding minimum demand constraint with extra_name '{extra_name}' and demand: {demand}")
        constraint_name = f"minimumDemand_{extra_name}"
        if not isinstance(demand, int):
            raise ValueError("demand must be an integer")
        
        variable_params = " ".join([f"{{1 {var}}}" for var in variables])
        self.add_constraint(
            name=constraint_name, 
            arity=len(variables), 
            scope=" ".join(variables),
            reference="global:weightedSum",
            parameters=f"[{variable_params}] __GE_PLACEHOLDER__ {demand}"
        )

        return constraint_name

    def remove_constraint(self, name):
        """Removes a constraint element by name."""
        constraints_element = self.instance.find("constraints")
        if constraints_element is not None:
            for constraint in constraints_element.findall("constraint"):
                if constraint.get("name") == name:
                    constraints_element.remove(constraint)
                    return

        raise ValueError(f"Constraint {name} not found")

    @deprecated
    def add_minimum_respecting_demand(self, timeslice_technologies_modes, specified_demand_profile_df, specified_annual_demand_df, year_split_df):
        def build_recursive(variables):
            if len(variables) == 1:
                return variables[0]
            return add(variables[0], build_recursive(variables[1:]))
        
        demand_df = specified_annual_demand_df.merge(specified_demand_profile_df, on=['FUEL', 'COUNTRY'])
        demand_df['DEMAND_PER_TIMESLICE'] = demand_df['SPECIFIED_ANNUAL_DEMAND'] * demand_df['SPECIFIED_DEMAND_PROFILE']
        
        agents = specified_annual_demand_df['COUNTRY'].unique()
        timeslices = specified_demand_profile_df['TIMESLICE'].unique()

        for r in agents:
            for l in timeslices:
                yearsplit_constant = year_split_df[(year_split_df['TIMESLICE'] == l)]['YEAR_SPLIT'].values[0]
                yearsplit_constant = round(1/yearsplit_constant)
                per_timeslice_country_variables = [var + "_rateActivity" for var in timeslice_technologies_modes if var.split('_')[0] == l and var.split('_')[1][:2] == r]
                specified_demand = demand_df[(demand_df['COUNTRY'] == r) & (demand_df['TIMESLICE'] == l)]['DEMAND_PER_TIMESLICE'].values[0]

                if not self.find_predicate(f"minimumRespectingDemand_{r}"):
                    self.add_predicate(
                        name=f"minimumRespectingDemand_{r}", 
                        parameters="int specified_demand int " + " int ".join([var.replace(f'{l}_', '') for var in per_timeslice_country_variables]),
                        functional=boolean_ge(build_recursive([var.replace(f'{l}_', '') for var in per_timeslice_country_variables]), "specified_demand")
                    )
                
                self.add_constraint(
                    name=f"minimumRespectingDemand_{r}_{l}", 
                    arity=len(per_timeslice_country_variables), 
                    scope=" ".join(per_timeslice_country_variables),
                    reference=f"minimumRespectingDemand_{r}",
                    parameters=f"{round(specified_demand * yearsplit_constant)} {' '.join([var for var in per_timeslice_country_variables])}"
                )

                if len(per_timeslice_country_variables) > self.max_arity:
                    self.max_arity = len(per_timeslice_country_variables)

    
    def add_minimum_rate_of_activity_constraint(self, input_output_activity_ratio_df, specified_demand_profile_df, specified_annual_demand_df, year_split_df):
        """Adds an hard constraint to the XML instance that enforces minimum rate fo activity."""
        def build_recursive(per_fuel_input_output_activity_ratio_df, year_split_weight):
            def process_row(row):
                factor = (row["OUTPUT_ACTIVITY_RATIO"] - row["INPUT_ACTIVITY_RATIO"]) * year_split_weight
                if factor >= 1:
                    return mul(f"{row['TECHNOLOGY']}_{row['MODE_OF_OPERATION']}_rateActivity", f"factor_{row['TECHNOLOGY']}_{row['MODE_OF_OPERATION']}")
                elif 0 < factor < 1:
                    return div(f"{row['TECHNOLOGY']}_{row['MODE_OF_OPERATION']}_rateActivity", f"factor_{row['TECHNOLOGY']}_{row['MODE_OF_OPERATION']}")
                elif factor < 0 or factor > -1:
                    return neg(div(f"{row['TECHNOLOGY']}_{row['MODE_OF_OPERATION']}_rateActivity", f"factor_{row['TECHNOLOGY']}_{row['MODE_OF_OPERATION']}"))
                elif factor == 0:
                    raise ValueError("The factor should not be zero")
                else:
                    return neg(mul(f"{row['TECHNOLOGY']}_{row['MODE_OF_OPERATION']}_rateActivity", f"factor_{row['TECHNOLOGY']}_{row['MODE_OF_OPERATION']}"))

            if len(per_fuel_input_output_activity_ratio_df) == 1:
                return process_row(per_fuel_input_output_activity_ratio_df.iloc[0])

            first_row_expression = process_row(per_fuel_input_output_activity_ratio_df.iloc[0])
            remaining_expression = build_recursive(per_fuel_input_output_activity_ratio_df.iloc[1:], year_split_weight)
            return add(remaining_expression, first_row_expression)

        timeslices = specified_demand_profile_df['TIMESLICE'].unique()
        fuels = pd.Series(
            list(input_output_activity_ratio_df['FUEL'].unique()) +
            list(specified_annual_demand_df['FUEL'].unique())
        ).unique()
        demand_df = specified_annual_demand_df.merge(specified_demand_profile_df, on=['FUEL', 'COUNTRY'])
        demand_df['DEMAND_PER_TIMESLICE'] = demand_df['SPECIFIED_ANNUAL_DEMAND'] * demand_df['SPECIFIED_DEMAND_PROFILE']
        for l in timeslices:
            year_split_weight = year_split_df[(year_split_df['TIMESLICE'] == l)]['YEAR_SPLIT'].values[0]
            for f in fuels:
                per_fuel_input_output_activity_ratio_df = input_output_activity_ratio_df[(input_output_activity_ratio_df['FUEL'] == f)]
                per_fuel_input_output_activity_ratio_df = per_fuel_input_output_activity_ratio_df[
                    (per_fuel_input_output_activity_ratio_df["OUTPUT_ACTIVITY_RATIO"] - per_fuel_input_output_activity_ratio_df["INPUT_ACTIVITY_RATIO"]) != 0
                ]
                if len(per_fuel_input_output_activity_ratio_df) != 0:
                    if f in specified_annual_demand_df['FUEL'].unique():
                        specified_demand = round(demand_df[(demand_df['FUEL'] == f) & (demand_df['TIMESLICE'] == l)]['DEMAND_PER_TIMESLICE'].values[0])
                    else:
                        specified_demand = 0
                    if specified_demand >0:
                        rateOfActivity_variables = [
                            f"{l}_{row['TECHNOLOGY']}_{row['MODE_OF_OPERATION']}_rateActivity" 
                            for _, row in per_fuel_input_output_activity_ratio_df.iterrows()
                        ]
                        factor_weights = [
                            f"factor_{row['TECHNOLOGY']}_{row['MODE_OF_OPERATION']}" 
                            for _, row in per_fuel_input_output_activity_ratio_df.iterrows()
                        ]
                        
                        if not self.find_predicate(f"minimumRateOfActivity_{f}"):
                            self.add_predicate(
                                name=f"minimumRateOfActivity_{f}", 
                                parameters="int " + " int ".join(map(lambda x: x[5:], rateOfActivity_variables)) +" int "+  " int ".join(factor_weights) + " int specified_demand",
                                functional=boolean_ge(build_recursive(per_fuel_input_output_activity_ratio_df, year_split_weight), "specified_demand")
                            )

                        if len(rateOfActivity_variables) > self.max_arity:
                            self.max_arity = len(rateOfActivity_variables)
                        
                        weights = []
                        for _, row in per_fuel_input_output_activity_ratio_df.iterrows():
                            factor = (row["OUTPUT_ACTIVITY_RATIO"] - row["INPUT_ACTIVITY_RATIO"]) * year_split_weight
                            if factor >= 1 or factor == 0:
                                factor = round(factor)
                            elif 0 < factor < 1:
                                factor = round(1/factor)
                            elif factor < 0 or factor > -1:
                                factor = round(-1/factor)
                            else:
                                factor = round(-factor)
                            weights.append(str(factor))
                            
                        self.add_constraint(
                            name=f"minimumRateOfActivity_{f}_{l}", 
                            arity=len(rateOfActivity_variables), 
                            scope=" ".join(rateOfActivity_variables),
                            reference=f"minimumRateOfActivity_{f}",
                            parameters=f"{' '.join(rateOfActivity_variables)} {' '.join(weights)} {specified_demand}"
                        )
    @deprecated
    def add_maximum_annual_activity_rate_per_timeslice_constraint(self, modes, factors_df):
        
        if not self.find_predicate("maximumAnnualRateActivityPerTimeslice"):
            self.add_predicate(
                name="maximumAnnualRateActivityPerTimeslice_mul", 
                parameters="int annualRatePerTimeslice int capacity int factor", 
                functional=boolean_le("annualRatePerTimeslice", mul("capacity", "factor"))
            )

        if not self.find_predicate("maximumAnnualRateActivityPerTimeslice_div"):
            self.add_predicate(
                name="maximumAnnualRateActivityPerTimeslice_div", 
                parameters="int annualRatePerTimeslice int capacity int factor", 
                functional=boolean_le("annualRatePerTimeslice", div("capacity", "factor"))
            )

        grouped_factors = factors_df.groupby(['COUNTRY', 'TECHNOLOGY'])
        for (country, technology), group in grouped_factors:
            factor = 0
            for index, row in group.iterrows():
                factor = row['CAPACITY_FACTOR'] * row['AVAILABILITY_FACTOR'] * row['CAPACITY_TO_ACTIVITY_UNIT']

                for timeslice_tech_mode in [f"{row['TIMESLICE']}_{row['TECHNOLOGY']}_{mode}" for mode in modes]:
                    if factor >= 1 or factor == 0:
                        self.add_constraint(
                            name=f"maximumAnnualRateActivityPerTimeslice_mul_{timeslice_tech_mode}",
                            arity=2,
                            scope=f"{timeslice_tech_mode}_rateActivity {row['TECHNOLOGY']}_capacity",
                            reference="maximumAnnualRateActivityPerTimeslice_mul",
                            parameters=f"{timeslice_tech_mode}_rateActivity {row['TECHNOLOGY']}_capacity {round(factor)}"
                        )
                    elif factor < 1 and factor > 0:
                        self.add_constraint(
                            name=f"maximumAnnualRateActivityPerTimeslice_div_{timeslice_tech_mode}",
                            arity=2,
                            scope=f"{timeslice_tech_mode}_rateActivity {row['TECHNOLOGY']}_capacity",
                            reference="maximumAnnualRateActivityPerTimeslice_div",
                            parameters=f"{timeslice_tech_mode}_rateActivity {row['TECHNOLOGY']}_capacity {round(1/factor)}"
                        )
                    else:
                        raise ValueError("The factor should be positive")
                    
    def add_maximum_activity_rate_constraint(self, cap_variable, capActivity_variable, factor, div_weight):
        """Adds an hard constraint to the XML instance that enforces maximum rate fo activity."""
        if factor >= 1:
            if not self.find_predicate("maximumRateActivity_mul"):
                self.add_predicate(
                    name="maximumRateActivity_mul", 
                    parameters="int rateActivity int capacity int factor int div_weight", 
                    functional=boolean_le(mul("rateActivity", 'div_weight'), mul("capacity", "factor"))
                )
        elif 0 <= factor < 1:
            if not self.find_predicate("maximumRateActivity_div"):
                self.add_predicate(
                    name="maximumRateActivity_div", 
                    parameters="int rateActivity int capacity int factor int div_weight", 
                    functional=boolean_le("rateActivity", div(div("capacity", "factor"), 'div_weight'))
                )
        else:
            raise ValueError("The factor should be positive")
        if factor >= 1 or factor == 0:
            self.add_constraint(
                name=f"maximumRateActivity_{cap_variable.replace('_capacity', '')}", 
                arity=2, 
                scope=f"{cap_variable} {capActivity_variable}", 
                reference="maximumRateActivity_mul",
                parameters=f"{capActivity_variable} {cap_variable} {round(factor)} {div_weight}"
            )
        elif factor < 1 and factor > 0: 
            self.add_constraint(
                name=f"maximumRateActivity_{cap_variable.split('_')[0]}", 
                arity=2, 
                scope=f"{cap_variable} {capActivity_variable}", 
                reference="maximumRateActivity_div",
                parameters=f"{capActivity_variable} {cap_variable} {round(1/factor)} {div_weight}"
            )
    
    def add_minimum_annual_activity_rate_per_timeslice_constraint(self, modes, factors_df, non_dispatchable_technologies):
        
        if not self.find_predicate("minimumAnnualRateActivityPerTimeslice"):
            self.add_predicate(
                name="minimumAnnualRateActivityPerTimeslice_mul", 
                parameters="int annualRatePerTimeslice int capacity int factor", 
                functional=boolean_ge("annualRatePerTimeslice", mul("capacity", "factor"))
            )

        if not self.find_predicate("minimumAnnualRateActivityPerTimeslice_div"):
            self.add_predicate(
                name="minimumAnnualRateActivityPerTimeslice_div", 
                parameters="int annualRatePerTimeslice int capacity int factor", 
                functional=boolean_ge("annualRatePerTimeslice", div("capacity", "factor"))
            )

        grouped_factors = factors_df.groupby(['COUNTRY', 'TECHNOLOGY'])
        for (country, technology), group in grouped_factors:
            if technology[2:] in non_dispatchable_technologies:
                for index, row in group.iterrows():
                    factor = 0.85 * row['CAPACITY_FACTOR'] * row['AVAILABILITY_FACTOR'] * row['CAPACITY_TO_ACTIVITY_UNIT']
                    for timeslice_tech_mode in [f"{row['TIMESLICE']}_{row['TECHNOLOGY']}_{mode}" for mode in modes]:
                        if factor >= 1 or factor == 0:
                            self.add_constraint(
                                name=f"minimumAnnualRateActivityPerTimeslice_mul_{timeslice_tech_mode}",
                                arity=2,
                                scope=f"{timeslice_tech_mode}_rateActivity {row['TECHNOLOGY']}_capacity",
                                reference="minimumAnnualRateActivityPerTimeslice_mul",
                                parameters=f"{timeslice_tech_mode}_rateActivity {row['TECHNOLOGY']}_capacity {round(factor)}"
                            )
                        elif factor < 1 and factor > 0:
                            self.add_constraint(
                                name=f"minimumAnnualRateActivityPerTimeslice_div_{timeslice_tech_mode}",
                                arity=2,
                                scope=f"{timeslice_tech_mode}_rateActivity {row['TECHNOLOGY']}_capacity",
                                reference="minimumAnnualRateActivityPerTimeslice_div",
                                parameters=f"{timeslice_tech_mode}_rateActivity {row['TECHNOLOGY']}_capacity {round(1/factor)}"
                            )
                        else:
                            raise ValueError("The factor should be positive")

    def add_maximum_rate_of_activity_per_all_technology_constraint(self, modes, factors_df):
        """Adds an hard constraint to the XML instance that enforces maximum rate fo activity."""
        def build_recursive(modes, timeslices):
            if len(modes) == 2:
                if len(timeslices) == 1:
                    return div(add(f"{timeslices[0]}_{modes[0]}", f"{timeslices[0]}_{modes[1]}"), f"yearsplit_{timeslices[0]}")
                return add(div(add(f"{timeslices[0]}_{modes[0]}", f"{timeslices[0]}_{modes[1]}"), f"yearsplit_{timeslices[0]}"), build_recursive(modes, timeslices[1:]))
            elif len(modes) == 1:
                if len(timeslices) == 1:
                    return div(f"{timeslices[0]}_{modes[0]}", f"yearsplit_{timeslices[0]}")
                return add(div(f"{timeslices[0]}_{modes[0]}", f"yearsplit_{timeslices[0]}"), build_recursive(modes, timeslices[1:]))
            else:
                raise ValueError("The number of modes should be 1 or 2")

        timeslices = factors_df['TIMESLICE'].unique()

        if not self.find_predicate("maximumRateOfActivity_mul"):
            self.add_predicate(
                name="maximumRateOfActivity_mul", 
                parameters=" ".join([f"int {l}_{m}" for m in modes for l in timeslices]) + " " + " ".join([f"int yearsplit_{l}" for l in timeslices]) + " int factor int installed_technology_capacity", 
                functional= boolean_le(build_recursive(modes, timeslices) , mul("installed_technology_capacity","factor")),
            )

        if not self.find_predicate("maximumRateOfActivity_div"):
            self.add_predicate(
                name="maximumRateOfActivity_div", 
                parameters=" ".join([f"int {l}_{m}" for m in modes for l in timeslices]) + " " + " ".join([f"int yearsplit_{l}" for l in timeslices]) + " int factor int installed_technology_capacity", 
                functional= boolean_le(build_recursive(modes, timeslices) , div("installed_technology_capacity", "factor")),
            )

        grouped_factors = factors_df.groupby(['COUNTRY', 'TECHNOLOGY'])
        for (country, technology), group in grouped_factors:
            factor = 0
            modes_variables = []
            yearsplit_constants = []
            for index, row in group.iterrows():
                factor += row['CAPACITY_FACTOR'] * row['YEAR_SPLIT'] * row['AVAILABILITY_FACTOR'] * row['CAPACITY_TO_ACTIVITY_UNIT']
                modes_variables += [f"{row['TIMESLICE']}_{row['TECHNOLOGY']}_{mode}" for mode in modes]
                yearsplit_constants += [str(round(1/row['YEAR_SPLIT']))]
            self.logger.debug(f"Factor: {factor}, Countries: {country}, Technology: {technology}")
            self.logger.debug(group)

            modes_variables_rateActivity = [f"{var}_rateActivity" for var in modes_variables]

            if factor >= 1 or factor == 0:
                self.add_constraint(
                    name=f"maximumRateOfActivity_mul_{technology}",
                    arity=len(modes_variables_rateActivity) + 1,
                    scope=" ".join(modes_variables_rateActivity) + f" {technology}_capacity",
                    reference="maximumRateOfActivity_mul",
                    parameters=f"{' '.join(modes_variables_rateActivity)} {' '.join(yearsplit_constants)} {round(factor)} {technology}_capacity"
                )
            elif factor < 1 and factor > 0:
                self.add_constraint(
                    name=f"maximumRateOfActivity_div_{technology}",
                    arity=len(modes_variables_rateActivity) + 1,
                    scope=" ".join(modes_variables_rateActivity) + f" {technology}_capacity",
                    reference="maximumRateOfActivity_div",
                    parameters=f"{' '.join(modes_variables_rateActivity)} {' '.join(yearsplit_constants)} {round(1/factor)} {technology}_capacity"
                )
            else:
                raise ValueError("The factor should be positive")

            if len(modes_variables) > self.max_arity:
                self.max_arity = len(modes_variables)


    def add_min_max_total_technology_annual_activity_constraint(self, modes, year_split_df, technology, upper_limit, lower_limit):
        def build_recursive(modes, timeslices):
            if len(modes) == 2:
                if len(timeslices) == 1:
                    return div(add(f"{timeslices[0]}_{modes[0]}", f"{timeslices[0]}_{modes[1]}"), f"yearsplit_{timeslices[0]}")
                return add(div(add(f"{timeslices[0]}_{modes[0]}", f"{timeslices[0]}_{modes[1]}"), f"yearsplit_{timeslices[0]}"), build_recursive(modes, timeslices[1:]))
            elif len(modes) == 1:
                if len(timeslices) == 1:
                    return div(f"{timeslices[0]}_{modes[0]}", f"yearsplit_{timeslices[0]}")
                return add(div(f"{timeslices[0]}_{modes[0]}", f"yearsplit_{timeslices[0]}"), build_recursive(modes, timeslices[1:]))
            else:
                raise ValueError("The number of modes should be 1 or 2")

        yearsplit_constants = [str(round(1/row['YEAR_SPLIT'])) for _, row in year_split_df.iterrows()]
        timeslices = year_split_df['TIMESLICE'].unique()
        if upper_limit is not None and not pd.isna(upper_limit):
            upper_limit = round(upper_limit)
            if not self.find_predicate("annual_technological_maximumRateOfActivity"):
                self.add_predicate(
                    name="annual_technological_maximumRateOfActivity", 
                    parameters=" ".join([f"int {l}_{m}" for m in modes for l in timeslices]) + " " + " ".join([f"int yearsplit_{l} " for l in timeslices]) + " int upper_limit", 
                    functional= boolean_le(build_recursive(modes, timeslices) , "upper_limit"),
                )
            self.add_constraint(
                name=f"annual_technological_maximumRateOfActivity_{technology}",
                arity=len(timeslices) * len(modes),
                scope=' '.join([f'{l}_{technology}_{m}_rateActivity' for m in modes for l in timeslices]),
                reference="annual_technological_maximumRateOfActivity",
                parameters=f"{' '.join([f'{l}_{technology}_{m}_rateActivity' for m in modes for l in timeslices])} {' '.join(yearsplit_constants)} {upper_limit}"
            )
            if len(timeslices) * len(modes) > self.max_arity:
                self.max_arity = len(timeslices) * len(modes)
            
        if lower_limit is not None and not pd.isna(lower_limit):
            lower_limit = round(lower_limit)
            if not self.find_predicate("annual_technological_minimumRateOfActivity"):
                self.add_predicate(
                    name="annual_technological_minimumRateOfActivity", 
                    parameters=" ".join([f"int {l}_{m}" for m in modes for l in timeslices]) + " " + " ".join([f"int yearsplit_{l}" for l in timeslices]) + " int lower_limit", 
                    functional= boolean_ge(build_recursive(modes, timeslices) , "lower_limit"),
                )
            self.add_constraint(
                name=f"annual_technological_minimumRateOfActivity_{technology}",
                arity=len(timeslices) * len(modes),
                scope=' '.join([f'{l}_{technology}_{m}_rateActivity' for m in modes for l in timeslices]),
                reference="annual_technological_minimumRateOfActivity",
                parameters=f"{' '.join([f'{l}_{technology}_{m}_rateActivity' for m in modes for l in timeslices])} {' '.join(yearsplit_constants)} {lower_limit}"
            )

            if len(timeslices) * len(modes) > self.max_arity:
                self.max_arity = len(timeslices) * len(modes)

    # def add_emission_accounting_constraint(self, emission_name, variables, modes, max_emission_limit, emission_factors_df, year_split_df):
    #     def build_recursive(timeslices, modes, technologies, emission_factor_df):
    #         if len(timeslices) == 1:

    #         expression = ""
    #         factor = emission_factor_df[(emission_factor_df['TECHNOLOGY'] == technology) & (emission_factor_df['MODE_OF_OPERATION'] == mode) & (emission_factor_df['TIMESLICE'])]['EMISSION_FACTOR'].values[0]
        



    def add_min_transmission_capacity_constraint(self, transmission_variable_name, min_transmission_capacity):
        """Adds an hard constraint to the XML instance that enforces minimum transmission capacity."""

        if not isinstance(min_transmission_capacity, int):
            raise ValueError("min_transmission_capacity must be an integer")
        
        if not self.find_predicate("minimumTransmissionCapacity"):
            self.add_predicate(
                name="minimumTransmissionCapacity", 
                parameters="int transmission int min_transmission_capacity",
                functional=boolean_ge("transmission", "min_transmission_capacity")
            )
        
        self.add_constraint(
            name=f"minimumTransmissionCapacity_{transmission_variable_name}",
            arity=1,
            scope=transmission_variable_name,
            reference="minimumTransmissionCapacity",
            parameters=f"{transmission_variable_name} {min_transmission_capacity}"
        )
    
    def add_max_transmission_capacity_constraint(self, transmission_variable_name, max_transmission_capacity):
        """Adds an hard constraint to the XML instance that enforces maximum transmission capacity."""
        
        if not isinstance(max_transmission_capacity, int):
            raise ValueError("max_transmission_capacity must be an integer")

        if not self.find_predicate("maximumTransmissionCapacity"):
            self.add_predicate(
                name="maximumTransmissionCapacity", 
                parameters="int transmission int max_transmission_capacity",
                functional=boolean_le("transmission", "max_transmission_capacity")
            )
        
        self.add_constraint(
            name=f"maximumTransmissionCapacity_{transmission_variable_name}",
            arity=1,
            scope=transmission_variable_name,
            reference="maximumTransmissionCapacity",
            parameters=f"{transmission_variable_name} {max_transmission_capacity}"
        )
    
    def add_emission_cap_constraint(self, agents, technolgies_emission_costs, max_emission):
        """Adds an hard constraint to the XML instance that enforces maximum emission."""
        def build_recursive_expression(agent_name, technolgies_emission_costs):
            technologies = list(technolgies_emission_costs.keys())
            # Base case: If only one technology remains, return its multiplication
            if len(technologies) == 1:
                random_key = technologies[0]
                return mul(f"{random_key}{agent_name}_capacity", f"{random_key}{agent_name}_rateActivity)")

            random_key = technologies[0]
            # Recursive case: Take the first technology, multiply its capacity and factor, then add to the rest
            return add(mul(mul(f"{random_key}{agent_name}_capacity", f"{random_key}{agent_name}_rateActivity"), technolgies_emission_costs[random_key]), build_recursive_expression(agent_name, {k: v for k, v in technolgies_emission_costs.items() if k != random_key}))
        
        if not isinstance(max_emission, int):
            raise ValueError("max_emission must be an integer")
        
        keys_to_remove = [tech for tech in technolgies_emission_costs.keys() if technolgies_emission_costs[tech] == 0]
        for tech in keys_to_remove:
            del technolgies_emission_costs[tech]

        technologies = technolgies_emission_costs.keys()
        variables = [f"{technology}{agent_name}_capacity" for technology in technologies for agent_name in agents]
        variables += [f"{technology}{agent_name}_rateActivity" for technology in technologies for agent_name in agents]
        variables.sort()

        functional_formula = build_recursive_expression(agents[0], technolgies_emission_costs)
        for agent in agents[1:]:
            functional_formula = add(functional_formula, build_recursive_expression(agent, technolgies_emission_costs))

        if not self.find_predicate(f"withinMaxEmission_all"):
            self.add_predicate(
                name=f"withinMaxEmission_all", 
                parameters=" ".join([f"int {variable}" for variable in variables]) + " int max_emission",
                functional=boolean_le(functional_formula, "max_emission")
            )

        self.add_constraint(
            name=f"withinMaxEmission_all_{len(agents)*len(technolgies_emission_costs)}", 
            arity=len(variables), 
            scope=" ".join(variables), 
            reference=f"withinMaxEmission_all",
            parameters=f"{' '.join(variables)} {max_emission}"
        )

        if len(variables) > self.max_arity:
            self.max_arity = len(variables)

    def add_emission_cap_constraint_per_agent(self, agent_name, technolgies_emission_costs, max_emission):
        """Adds an hard constraint to the XML instance that enforces maximum emission per agent."""
        def build_recursive_expression(agent_name, technolgies_emission_costs):
            technologies = list(technolgies_emission_costs.keys())
            # Base case: If only one technology remains, return its multiplication
            if len(technologies) == 1:
                random_key = technologies[0]
                return mul(f"{random_key}{agent_name}_capacity", f"{random_key}{agent_name}_rateActivity)")

            random_key = technologies[0]
            # Recursive case: Take the first technology, multiply its capacity and factor, then add to the rest
            return add(mul(mul(f"{random_key}{agent_name}_capacity", f"{random_key}{agent_name}_rateActivity"), technolgies_emission_costs[random_key]), build_recursive_expression(agent_name, {k: v for k, v in technolgies_emission_costs.items() if k != random_key}))
        
        if not isinstance(max_emission, int):
            raise ValueError("max_emission must be an integer")
        
        keys_to_remove = [tech for tech in technolgies_emission_costs.keys() if technolgies_emission_costs[tech] == 0]
        for tech in keys_to_remove:
            del technolgies_emission_costs[tech]

        technologies = technolgies_emission_costs.keys()
        variables = [f"{technology}{agent_name}_capacity" for technology in technologies]
        variables += [f"{technology}{agent_name}_rateActivity" for technology in technologies]
        variables.sort()

        functional_formula = build_recursive_expression(agent_name, technolgies_emission_costs)
        if not self.find_predicate(f"withinMaxEmission_{agent_name}"):
            self.add_predicate(
                name=f"withinMaxEmission_{agent_name}", 
                parameters=" ".join([f"int {variable}" for variable in variables]) + " int max_emission",
                functional=boolean_le(functional_formula, "max_emission")
            )

        self.add_constraint(
            name=f"withinMaxEmission_{agent_name}_{len(technologies)}", 
            arity=len(variables), 
            scope=" ".join(variables), 
            reference=f"withinMaxEmission_{agent_name}",
            parameters=f"{' '.join(variables)} {max_emission}"
        )

        if len(variables) > self.max_arity:
            self.max_arity = len(variables)

    def add_specified_min_demand_constraint_per_agent(
        self, 
        agent_name,
        rate_of_activity_variables,
        trade_from_country_variables,
        trade_to_country_variables,
        max_demand
        ):
        """Adds an hard constraint to the XML instance that enforces maximum demand."""
        def build_recursive_add_expression(espression, variables):
            if len(variables) == 1:
                return add(espression, variables[0])
            return build_recursive_add_expression(add(espression, variables[0]), variables[1:])
        def build_recursive_sub_expression(espression, variables):
            if len(variables) == 1:
                return sub(espression, variables[0])
            return build_recursive_sub_expression(sub(espression, variables[0]), variables[1:])
        
        if not isinstance(max_demand, int):
            raise ValueError("max_demand must be an integer")
        
        functional_expression = build_recursive_sub_expression(
            build_recursive_add_expression(
                build_recursive_add_expression(rate_of_activity_variables[0], rate_of_activity_variables[1:]), 
                trade_to_country_variables), 
                trade_from_country_variables)

        all_variables = rate_of_activity_variables + trade_from_country_variables + trade_to_country_variables
        if not self.find_predicate(f"maxSpecifiedDemandPerAgent_{agent_name}"):
            self.add_predicate(
                name=f"maxSpecifiedDemandPerAgent_{agent_name}", 
                parameters= " ".join([f"int {variable}" for variable in all_variables]) + f" int max_demand",
                functional=boolean_ge(functional_expression, "max_demand")
            )

        self.add_constraint(
            name=f"maxSpecifiedDemandPerAgent_{agent_name}_{len(all_variables)}", 
            arity=len(all_variables), 
            scope=" ".join(all_variables), 
            reference=f"maxSpecifiedDemandPerAgent_{agent_name}",
            parameters=f"{' '.join(all_variables)} {max_demand}"
        )

        if len(all_variables) > self.max_arity:
            self.max_arity = len(all_variables)

    def add_demand_constraint_per_agent(self, agent_name, min_demand, technologies, neighbor_agents):
        """Adds an hard constraint to the XML instance that enforces maximum demand."""
        def build_recursive_expression(technologies, agent_name):
            """Recursively builds the expression: 
            add(add(mul(tech1Capacity, tech1CapFactor), mul(tech2Capacity, tech2CapFactor)), mul(tech3Capacity, tech3CapFactor))"""
            agent_name = agent_name
            # Base case: If only one technology remains, return its multiplication
            if len(technologies) == 1:
                return mul(f"{technologies[0]}{agent_name}_capacity", f"{technologies[0]}{agent_name}_rateActivity")

            # Recursive case: Take the first technology, multiply its capacity and factor, then add to the rest
            return add(mul(f"{technologies[0]}{agent_name}_capacity", f"{technologies[0]}{agent_name}_rateActivity"), build_recursive_expression(technologies[1:], agent_name))
            
        if not isinstance(min_demand, int):
            raise ValueError("min_demand must be an integer")
        
        variables = [f"{technology}{agent_name}_capacity" for technology in technologies]
        variables += [f"{technology}{agent_name}_rateActivity" for technology in technologies]
        outflow_transmission_variables = [f"transmission{agent_name}{agent2}" for agent2 in neighbor_agents if agent2 != agent_name]
        inflow_transmission_variables = [f"transmission{agent2}{agent_name}" for agent2 in neighbor_agents if agent2 != agent_name]
        
        variables.sort()
        outflow_transmission_variables.sort()
        inflow_transmission_variables.sort()

        functional_formula = div(build_recursive_expression(technologies, agent_name), "100")
        for transmission_variable in inflow_transmission_variables:
            functional_formula = add(functional_formula, transmission_variable)
        for transmission_variable in outflow_transmission_variables:
            functional_formula = sub(functional_formula, transmission_variable)

        variables += outflow_transmission_variables
        variables += inflow_transmission_variables
        
        if not self.find_predicate("minDemandPerAgent"):
            self.add_predicate(
                name="minDemandPerAgent_" + agent_name, 
                parameters=" ".join([f"int {variable}" for variable in variables]) + " int min_demand",
                functional=boolean_ge(functional_formula, "min_demand")
            )
        
        self.add_constraint(
            name=f"minDemandPerAgent_{agent_name}_{len(variables)}", 
            arity=len(variables), 
            scope=" ".join(variables), 
            reference="minDemandPerAgent_" + agent_name,
            parameters=f"{' '.join(variables)} {str(min_demand)}"
        )

        if len(variables) > self.max_arity:
            self.max_arity = len(variables)

    def add_cost_constraint(self, variable_capacity_name, rateActivity_variable, previous_installed_capacity, capital_cost, variable_cost, fixed_cost):
        """Adds a soft constraint to the XML instance for cost."""
        if not self.find_function("cost_constraint"):
            self.add_function(
                name="cost_constraint", 
                parameters="int capacity int rateActivity int previous_installed_capacity int capital_cost int variable_cost int fixed_cost",
                functional=add(
                    mul(sub("capacity", "previous_installed_capacity"), "capital_cost"),
                    add(div(mul("rateActivity", "variable_cost"), 1000000), div(mul("capacity", "fixed_cost"), 1000))
                )
            )

        self.add_constraint(
            name=f"cost_constraint_{variable_capacity_name.replace('_capacity', '')}",
            arity=2,
            scope=variable_capacity_name + " " + rateActivity_variable,
            reference="cost_constraint",
            parameters=f"{variable_capacity_name} {rateActivity_variable} {previous_installed_capacity} {capital_cost} {variable_cost} {fixed_cost}"
        )


    def add_operating_cost_minimization_constraint(self, weight, variable_capacity_name, variable__rateActivity_name, cost_per_MWh):
        """Adds a soft constraint to the XML instance that enforces maximum operating cost."""

        if not isinstance(weight, int) or not isinstance(cost_per_MWh, int):
            raise ValueError("weight and cost_per_MW must be integers")

        complex_weight = 8760 * cost_per_MWh / (weight * 100)
        self.logger.info(f"Complex weight: {complex_weight}")
        if complex_weight >= 1:
            if not self.find_function(f"minimize_operatingCost_mul"):
                self.add_function(
                    name=f"minimize_operatingCost_mul", 
                    parameters="int capacity int _rateActivity int weight",
                    functional= mul(mul("capacity", "_rateActivity"), "weight")
                )

            self.add_constraint(
                name=f"minimize_operatingCost_mul_{variable_capacity_name.replace('capacity', '')}",
                arity=2,
                scope=f"{variable_capacity_name} {variable__rateActivity_name}",
                reference=f"minimize_operatingCost_mul",
                parameters=f"{variable_capacity_name} {variable__rateActivity_name} {str(round(complex_weight))}"
            )
        elif complex_weight < 1 and complex_weight >= 0:
            div_complex_weight = round(1 / complex_weight) if complex_weight != 0 else 10**6
            if complex_weight == 0:
                self.logger.warning("The complex weight is 0, the constraint will be added with a very high weight")
            if not self.find_function(f"minimize_operatingCost_div"):
                self.add_function(
                    name=f"minimize_operatingCost_div", 
                    parameters="int capacity int _rateActivity int weight",
                    functional= div(mul("capacity", "_rateActivity"), "weight")
                )
            self.add_constraint(
                name=f"minimize_operatingCost_div_{variable_capacity_name.replace('capacity', '')}",
                arity=2,
                scope=f"{variable_capacity_name} {variable__rateActivity_name}",
                reference=f"minimize_operatingCost_div",
                parameters=f"{variable_capacity_name} {variable__rateActivity_name} {str(div_complex_weight)}"
            )
        else:
            raise ValueError("the complex weight should be positive")

        if self.max_arity < 2:
            self.max_arity = 2

    def add_installing_cost_minimization_constraint(self, variable_capacity_name, previous_installed_capacity, cost_per_MW, extra_name=""):
        """Adds a soft constraint to the XML instance that enforces maximum installing cost."""

        if not self.find_function(f"minimize_installingCost"):
            self.add_function(
                name=f"minimize_installingCost", 
                parameters="int capacity int oldCapacity int cost_per_MW",
                functional= mul(sub("capacity", "oldCapacity"), "cost_per_MW")
            )

        self.add_constraint(
            name=f"minimize_installingCost_{variable_capacity_name.replace('_capacity', '')}_{extra_name}",
            arity=1,
            scope=f"{variable_capacity_name}",
            reference=f"minimize_installingCost",
            parameters=f"{variable_capacity_name} {previous_installed_capacity} {cost_per_MW}"
        )

    def add_variable_cost_minimization_constraint(self, variable_name, cost_per_TJ):
        """Adds a soft constraint to the XML instance that enforces maximum variable cost."""

        if not self.find_function(f"minimize_variableCost"):
            self.add_function(
                name=f"minimize_variableCost", 
                parameters="int variable int cost_per_TJ",
                functional= mul("variable", "cost_per_TJ")
            )
        
        if not isinstance(cost_per_TJ, int):
            raise ValueError("cost_per_TJ must be an integer")
        
        self.add_constraint(
            name=f"minimize_variableCost_{variable_name.replace('_variable', '')}",
            arity=1,
            scope=f"{variable_name}",
            reference=f"minimize_variableCost",
            parameters=f"{variable_name} {cost_per_TJ}"
        )
        
    @deprecated
    def add_minimizing_operating_cost_constraint(self, weight, rateActivity_variables, cost_per_unit_of_activity, year_split_df):
        """Adds a soft constraint to the XML instance that enforces maximum operating cost."""
        def build_recursive(factors):
            if len(factors) == 1:
                return factors[0]
            return add(factors[0], build_recursive(factors[1:]))
        
        if not isinstance(weight, int) or not isinstance(cost_per_unit_of_activity, int):
            raise ValueError("weight and cost_per_MW must be integers")
        
        factor_expressions = []
        numeric_factors = []
        for var in rateActivity_variables:
            factor = year_split_df[(year_split_df['TIMESLICE'] == var.split('_')[0])]['YEAR_SPLIT'].values[0]
            factor *= cost_per_unit_of_activity
            if factor >= 1 or factor == 0:
                numeric_factors.append(str(round(factor)))
                factor_expressions.append(mul(var.split('_')[0], f"factor_{var.split('_')[0]}"))
            elif 0 < factor < 1:
                numeric_factors.append(str(round(1/factor)))
                factor_expressions.append(div(var.split('_')[0], f"factor_{var.split('_')[0]}"))
            else:
                raise ValueError("The factor should be positive")

            # elif 0 < factor < 1:
            #     numeric_factors.append(str(round(1/factor)))
            #     factor_expressions.append(div(var.split('_')[0], f"factor_{var.split('_')[0]}"))
            # else:
            #     raise ValueError("The factor should be positive")
            
        if not self.find_function(f"minimize_operatingCost_{rateActivity_variables[0][5:].replace('_rateActivity', '')}"):
            self.add_function(
                name=f"minimize_operatingCost_{rateActivity_variables[0][5:].replace('_rateActivity', '')}", 
                parameters=" ".join([f"int {var.split('_')[0]}" for var in rateActivity_variables]) + " " + " ".join([f"int factor_{var.split('_')[0]}" for var in rateActivity_variables]) + " int weight",
                functional= div(build_recursive(factor_expressions), "weight")
            )
        if not all(factor == "0" for factor in numeric_factors):
            self.add_constraint(
                name=f"minimize_operatingCost_{rateActivity_variables[0][5:].replace('_rateActivity', '')}_{len(rateActivity_variables)}",
                arity=len(rateActivity_variables),
                scope=" ".join(rateActivity_variables),
                reference=f"minimize_operatingCost_{rateActivity_variables[0][5:].replace('_rateActivity', '')}",
                parameters=f"{' '.join(rateActivity_variables)} {' '.join(numeric_factors)} {weight}"
            )

        if len(rateActivity_variables) > self.max_arity:
            self.max_arity = len(rateActivity_variables)
    
    def add_symmetry_constraint(self, extra_name, var1, var2):
        """
        Adds a symmetry constraint to the XML instance.
        The symmetry constraint ensures that var1 = -var2.
        """
        
        if not self.find_predicate("symmetry"):
            self.add_predicate(
                name="symmetry", 
                parameters="int var1 int var2",
                functional=boolean_eq('var1', neg('var2'))
            )

        self.add_constraint(
            name=f"symmetry_{extra_name}",
            arity=2,
            scope=f"{var1} {var2}",
            reference="symmetry",
            parameters=f"{var1} {var2}"
        )

        if self.max_arity < 2:
            self.max_arity = 2

    def add_power_balance_constraint(self, extra_name, flow_variables, delta):
        """
        Adds a power balance constraint to the XML instance.
        The power balance constraint ensures that the sum of import flows minus 
        the sum of export flows equals plus delta.
        """
        def build_recursive_add_expression(vars):
            if len(vars) == 1:
                return abs_val(vars[0])
            if len(vars) == 2:
                return add(abs_val(vars[0]), abs_val(vars[1]))
            return add(abs_val(vars[0]), build_recursive_add_expression(vars[1:]))

        num = len(flow_variables)
        if num > 0:
            if not isinstance(delta, int):
                raise ValueError("delta must be an integer")

            if not self.find_predicate(f"powerBalance_{num}"):
                self.add_predicate(
                    name=f"powerBalance_{num}", 
                    parameters=" ".join([f"int var_{i}" for i in range(num)]) + " int delta",
                    functional=boolean_le(build_recursive_add_expression([f"var_{i}" for i in range(num)]), "delta")
                )
            self.add_constraint(
                name=f"powerBalance_{num}_{extra_name}",
                arity=num,
                scope=f"{' '.join(flow_variables)}",
                reference=f"powerBalance_{num}",
                parameters=f"{' '.join(flow_variables)} {delta}"
            )
            
    def add_utility_function_constaint(self, extra_name, variables, import_marginal_costs, export_marginal_costs, cost):
        def build_recursive(variables):
            if len(variables) == 1:
                return variables[0]
            return add(variables[0], build_recursive(variables[1:]))
        
        benefits = []
        for i in range(len(variables)):
            benefits.append(export_marginal_costs[i] + cost - import_marginal_costs[i])

        if not self.find_predicate(f"maximise_utilityFunction_{len(variables)}"):
            self.add_function(
                name=f"maximise_utilityFunction_{len(variables)}", 
                parameters=" ".join([f"int {var}" for var in variables]) + " " + " ".join([f"int benefit_{i}" for i in range(len(variables))]) + " int cost",
                functional=build_recursive([
                    conditional(
                        boolean_lt(var, "0"),
                        "0",
                        mul(var, f"benefit_{i}")
                    ) for i, var in enumerate(variables)
                ])
            )

        self.add_constraint(
            name=f"utilityFunction_{len(variables)}_{extra_name}",
            arity=len(variables),
            scope=f"{' '.join(variables)}",
            reference=f"maximise_utilityFunction_{len(variables)}",
            parameters=f"{' '.join(variables)} {' '.join([str(b) for b in benefits])} {cost}"
        )

    def print_xml(self, output_file = "defaultName_problem.xml"):
        """Prints the XML instance to a file."""
        self.set_max_arity_contraints()
        xml_str = ET.tostring(self.instance, encoding="unicode")

        # Replace placeholders with actual XML tags        
        xml_str = xml_str.replace("__GT_PLACEHOLDER__", "<gt/>")
        xml_str = xml_str.replace("__GE_PLACEHOLDER__", "<ge/>")

        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml()
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        self.logger.info(f"XML generated and saved to {output_file}")

    def set_max_arity_contraints(self):
        """Changes the max arity of constraints in the XML instance."""
        presentation = self.instance.find("presentation")
        if presentation is not None:    
            presentation.attrib["maxConstraintArity"] = str(self.max_arity)
        else:
            raise ValueError("Presentation element not found in XML instance")

def boolean_not(a):
    return f"not({a})"

def boolean_and(a, b):
    return f"and({a}, {b})"

def boolean_or(a, b):
    return f"or({a}, {b})"

def boolean_xor(a, b):
    return f"xor({a}, {b})"

def boolean_iff(a, b):
    return f"iff({a}, {b})"

def boolean_eq(a, b):
    return f"eq({a}, {b})"

def boolean_ne(a, b):
    return f"ne({a}, {b})"

def boolean_ge(a, b):
    return f"ge({a}, {b})"

def boolean_gt(a, b):
    return f"gt({a}, {b})"

def boolean_le(a, b):
    return f"le({a}, {b})"

def boolean_lt(a, b):
    return f"lt({a}, {b})"

# Integer Expressions
def neg(a):
    return f"neg({a})"

def abs_val(a):
    return f"abs({a})"

def add(a, b):
    return f"add({a}, {b})"

def sub(a, b):
    return f"sub({a}, {b})"

def mul(a, b):
    return f"mul({a}, {b})"

def div(a, b):
    return f"div({a}, {b})"

def mod(a, b):
    return f"mod({a}, {b})"

def pow_val(a, b):
    return f"pow({a}, {b})"

def min_val(a, b):
    return f"min({a}, {b})"

def max_val(a, b):
    return f"max({a}, {b})"

def conditional(if_expr, true_expr, false_expr):
    return f"if({if_expr}, {true_expr}, {false_expr})"

