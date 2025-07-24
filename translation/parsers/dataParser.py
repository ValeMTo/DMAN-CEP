import pandas as pd
from translation.retriever.transmissionRetriever import transmissionRetrieverClass

class dataParserClass:
    def __init__(self, logger):
        pass
    
    def get_country_data(self, country, year, timeslice):
        """
        Returns the data for a specific country, year and time slice.
        It returns a dataframe with technologies as indexes and columns as relevant data
        """
        raise NotImplementedError("This method will be implemented in child classes")
    
    def get_transmission_data(self, regions, cross_border_only=True, yearly=False):
        """
        Returns the transmission data for the specified regions.
        It returns a dataframe with the following columns:
        - start_country
        - end_country
        - tension
        - circuit
        - cables
        that are the data from Open Street Map. No assumptions are made.
        The dataframe is filtered to only include cross-border lines if cross_border_only is True.
        """
        retriever = transmissionRetrieverClass(self.logger, regions=regions)
        if cross_border_only:
            self.logger.info("Filtering cross-border lines")
            return retriever.extract_cross_border_lines(yearly=yearly)
        raise NotImplementedError("The method without cross-border lines is not implemented yet")


    