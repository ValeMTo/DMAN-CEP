from translation.energySystemModel import EnergyModelClass
import os

if __name__ == "__main__":
    model = EnergyModelClass()
    model.solve()

    os.system('say "The model has finished solving."')

    

    
    
