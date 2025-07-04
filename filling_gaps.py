import pandas as pd
import numpy as np
from translation.parsers.osemosysDataParser import osemosysDataParserClass
import logging
import sys

FILE = "./osemosys_data/input_data/TEMBA_SSP5-Baseline.xlsx"

if __name__ == "__main__":

    logger = logging.getLogger("OsemosysGapFiller")
    osemosysDataParser = osemosysDataParserClass(
        logger=logger,
        file_path=FILE,
    )

    technologies = osemosysDataParser.extract_technologies()
    old_technologies = []
    for idx, row in technologies.iterrows():
        if row['TECHNOLOGY'][-1] != 'N' and row['TECHNOLOGY'][:-1] + 'N' in technologies['TECHNOLOGY'].values:
            old_technologies.append(row['TECHNOLOGY'])
    residuals_df = pd.read_excel(FILE, sheet_name='ResidualCapacity', index_col=0)

    for tech, row in residuals_df.iterrows():
        if tech[-1] == 'N':
            print(f"Technology {tech} already has 'N' suffix, skipping.")
            sys.exit(0)
    print(f"No new technologies found.")

    for tech in old_technologies:
        if tech in residuals_df.index:
            new_tech = tech[:-1] + 'N'
            residuals_df = residuals_df.rename(index={tech: new_tech})
    print(f"Renamed old technologies to new ones.")

    with pd.ExcelWriter(FILE, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
        residuals_df.to_excel(writer, sheet_name='ResidualCapacity', index=True)
        print(f"Updated ResidualCapacity sheet with new technologies.")




