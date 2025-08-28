import os
import shutil

def clean_folders(base_path, keep_list):
    subfolders = ['logs', 'resources', 'results', 'networks', 'benchmarks']
    missing_folders = []
    for subfolder in subfolders:
        folder_path = os.path.join(base_path, subfolder)
        if not os.path.isdir(folder_path):
            missing_folders.append(subfolder)
            continue
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path) and not any(item in keep for keep in keep_list):
                shutil.rmtree(item_path)
    if missing_folders:
        raise FileNotFoundError(f"The following required folders were not found: {missing_folders}")

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
pypsa_earth_path = './pypsa_earth'
folders_to_keep = []  # Replace with your folder names to keep

clean_folders(pypsa_earth_path, folders_to_keep)
clear_snakemake_logs(pypsa_earth_path)
