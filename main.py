from translation.energySystemModel import EnergyModelClass
import os
import pickle
import time
import sys

max_retries = 10

if __name__ == "__main__":
    # Optionally pass status_file path as a command-line argument
    status_file = None
    if len(sys.argv) > 1:
        status_file = sys.argv[1]

    status_file = "./solutions/SAPP-monthly-emissions-0.2-v3+2/model_status.pkl"

    if status_file and os.path.exists(status_file):
        with open(status_file, "rb") as f:
            status = pickle.load(f)
        print(f"Loaded status from {status_file}: {status}")
        model = EnergyModelClass(reload=True)
        for attr, value in status.__dict__.items():
            setattr(model, attr, value)
    else:
        model = EnergyModelClass()

    retries = 0
    while retries < max_retries:
        try:
            model.solve()
            break
        except Exception as e:
            error_log_file = os.path.join(model.output_path, "error_log.txt")
            with open(error_log_file, "a") as log:
                log.write(f"Error during solve in the retry {retries}: {e}\n")
            status_file = os.path.join(model.output_path, "model_status.pkl")
            if os.path.exists(status_file):
                with open(status_file, "rb") as f:
                    status = pickle.load(f)
                with open(error_log_file, "a") as log:
                    log.write(f"Loaded status from {status_file}: {status}\n")
            with open(error_log_file, "a") as log:
                log.write("Reloading model and retrying...\n")
            model = EnergyModelClass(reload=True)
            for attr, value in status.__dict__.items():
                setattr(model, attr, value)
            retries += 1
    else:
        print("Max retries reached. Exiting.")

    os.system('say "The model has finished solving."')

    

    
    
