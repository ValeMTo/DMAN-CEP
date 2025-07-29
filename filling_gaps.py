import pandas as pd
from tqdm import tqdm
import shutil
import os

FILE = "./osemosys_data/input_data_original/TEMBA_SSP5-Baseline.xlsx"

if __name__ == "__main__":
    original_file = FILE
    backup_dir = os.path.join(os.path.dirname(FILE), "../input_data")
    backup_file = os.path.join(backup_dir, os.path.basename(FILE))
    shutil.copy2(original_file, backup_file)
    print(f"Copied {original_file} to {backup_file}")
    FILE = backup_file

    technologies_df = pd.read_excel(FILE, sheet_name="TECHNOLOGY", header=None)
    technologies_df['COUNTRY'] = technologies_df[0].map(lambda x: x[:2])
    technologies_df['TECHNOLOGY'] = technologies_df[0].map(lambda x: x[2:])

    power_plants_df = pd.read_csv("./osemosys_data/techcodes(in).csv")
    power_plants_df = power_plants_df[power_plants_df['Group'] == 'Power_plants']
    power_plants_df = power_plants_df[['code (Old)']].rename(columns={'code (Old)': 'TECHNOLOGY'})

    technologies_df = technologies_df.merge(power_plants_df, left_on='TECHNOLOGY', right_on='TECHNOLOGY', how='inner')
    technologies_df['TECHNOLOGY'] = technologies_df['COUNTRY'] + technologies_df['TECHNOLOGY']
    technologies = technologies_df[['COUNTRY', 'TECHNOLOGY']]

    old_technologies = []
    for idx, row in technologies.iterrows():
        if row['TECHNOLOGY'][-1] != 'N' and row['TECHNOLOGY'][:-1] + 'N' in technologies['TECHNOLOGY'].values:
            old_technologies.append(row['TECHNOLOGY'])

    tech_types_df = pd.read_csv("./osemosys_data/techcodes(in).csv")[['code (Old)', 'Group', 'tech_group']]
    tech_types_df = tech_types_df[tech_types_df['Group'] == 'Power_plants']

    sheet_names = pd.ExcelFile(FILE).sheet_names[8:]
    for sheet_name in tqdm(sheet_names, desc="Processing sheets"):
        country_flag = False
        df = pd.read_excel(FILE, sheet_name=sheet_name, index_col=0)

        if df.index.name != 'TECHNOLOGY':
            continue
        print(f"Processing {sheet_name} sheet with index name {df.index.name}")

        no_numeric_columns = [col for col in df.columns if not any(char.isdigit() for char in str(col))]
        for tech in old_technologies:
            if tech in df.index:
                new_tech = tech[:-1] + 'N'
                df = df.rename(index={tech: new_tech})
        print(f"Renamed old technologies to new ones.")

        df['COUNTRY'] = df.index.str[:2]
        df['TECH'] = df.index.str[2:]
        df = df.merge(tech_types_df, left_on='TECH', right_on='code (Old)', how='inner')

        df = df.sort_index()
        indexes = set(no_numeric_columns) | set(['COUNTRY', 'tech_group'])
        grouped = df.groupby(list(indexes), as_index=False)
        if sheet_name == 'ResidualCapacity':
            df_agg = grouped.sum(numeric_only=True)
        else:
            df_agg = grouped.mean(numeric_only=True)
        # Set COUNTRY and code (Old) as index
        df_agg['tech_group'] = df_agg['tech_group'].str.replace(' ', '_')
        df_agg['TECHNOLOGY'] = df_agg['COUNTRY'] + '_' + df_agg['tech_group']    
        #df_agg = df_agg.set_index(no_numeric_columns + ['TECHNOLOGY'])

        to_remove_columns = set(['COUNTRY', 'Group', 'tech_group', 'code (Old)']) - set(no_numeric_columns)
        df_agg = df_agg.drop(columns=to_remove_columns, errors='ignore')
    
        with pd.ExcelWriter(FILE, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            df_agg.to_excel(writer, sheet_name=sheet_name, index=False, header=True)
            print(f"Updated {sheet_name} sheet with new technologies.")

    capital_cost_df = pd.read_excel(FILE, sheet_name="CapitalCost", index_col=0)
    technologies_in_capital_cost = capital_cost_df['TECHNOLOGY'].unique()
    technology_sheet_df = pd.read_excel(FILE, sheet_name="TECHNOLOGY", header=None)
    technology_sheet_df = pd.DataFrame(technologies_in_capital_cost)
    with pd.ExcelWriter(FILE, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
        technology_sheet_df.to_excel(writer, sheet_name="TECHNOLOGY", index=False, header=False)
    print("TECHNOLOGY sheet updated with TECHNOLOGY values from CapitalCost sheet.")
