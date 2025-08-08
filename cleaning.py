import os
import shutil
import subprocess

def clean_folders(base_path, keep_list):
    subfolders = ['logs', 'resources', 'results', 'networks', 'benchmarks']
    for subfolder in subfolders:
        folder_path = os.path.join(base_path, subfolder)
        if not os.path.isdir(folder_path):
            continue
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path) and item not in keep_list:
                shutil.rmtree(item_path)

def clear_snakemake_logs(base_path):
    logs_path = os.path.join(base_path, '.snakemake', 'log')
    if os.path.isdir(logs_path):
        for item in os.listdir(logs_path):
            item_path = os.path.join(logs_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

# Example usage:
pypsa_earth_path = '/Users/valemto/Documents/GitHub/distributed-energy-systems/pypsa_earth'
folders_to_keep = ['2030_LS']  # Replace with your folder names to keep

# subprocess.run(
#     ["snakemake", "-j4", "--unlock"],
#     cwd=pypsa_earth_path,
#     check=True
# )

clean_folders(pypsa_earth_path, folders_to_keep)
clear_snakemake_logs(pypsa_earth_path)

