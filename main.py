from translation.energySystemModel import EnergyModelClass
import os
import pickle
import time
import sys

max_retries = 1

if __name__ == "__main__":
    # Optionally pass status_file path as a command-line argument
    status_file = None
    if len(sys.argv) > 1:
        status_file = sys.argv[1]

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
            print(f"Error during solve: {e}")
            status_file = os.path.join(model.output_path, "model_status.pkl")
            if os.path.exists(status_file):
                with open(status_file, "rb") as f:
                    status = pickle.load(f)
                print(f"Loaded status from {status_file}: {status}")
            print("Reloading model and retrying...")
            model = EnergyModelClass(reload=True)
            for attr, value in status.__dict__.items():
                setattr(model, attr, value)
            retries += 1
    else:
        print("Max retries reached. Exiting.")

    os.system('say "The model has finished solving."')

    

    
    
