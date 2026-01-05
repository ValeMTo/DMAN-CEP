# DMAN-CEP — Distributed Multi-Agent Negotiation for Capacity Expansion Planning

DMAN-CEP is a research prototype for distributed / multi-agent energy system planning. The workflow combines:

**Local Optimization:** A detailed energy system optimization stack for each agent (via the `pypsa_earth` submodule).

**Negotiation Layer:** A coordination mechanism that orchestrates multiple agents to negotiate cross-border electricity flows.

This approach allows for scalable, privacy-preserving, and flexible planning across interconnected energy systems, supporting collaborative decision-making without requiring centralized data sharing.

>**Reference**: This repository accompanies the approach described in “Distributed Multi-Agent Negotiation for Capacity Expansion Planning” (Amato et al.) applied to the Southern African Power Pool (SAPP).

## Key idea 

The framework models countries as autonomous agents that cooperate to minimize costs while respecting local constraints. The process follows this logic:

- **Temporal Decomposition:** The planning horizon (e.g., 2030) is split into time slices (yearly, monthly, or weekly) to manage computational complexity.
- **Local Optimization:** Each agent independently runs a local optimizer (PyPSA-Earth) to compute its optimal energy mix and system cost.
- **Marginal Valuation:** Agents quantify the value of trade by performing demand perturbations. They run additional optimizations with slightly increased/decreased demand to calculate the marginal costs of importing or exporting energy.
- **Bilateral Negotiation:** Neighboring agents exchange bids (price and quantity) based on these marginal costs. Agents accept trades that offer the highest utility (cost savings).
- **Convergence:** The system iterates—updating demands based on agreed trades and re-optimizing—until flows stabilize and no further mutually beneficial trades are possible.

## Repository requirements

- Linux (tested workflow)
- `conda` (Miniconda / Anaconda / Mambaforge)
- Git (for submodules)
- A working solver supported by your stack (depends on your PyPSA-Earth setup - Gurobi strongly suggested)

## Quickstart

1. Add `pypsa-earth` as a submodule:

    ```bash
    # Add the submodule mapped to the 'pypsa_earth' directory
    git submodule add https://github.com/pypsa-meets-earth/pypsa-earth.git pypsa_earth

    # Pin to version 0.7.0
    cd pypsa_earth
    git checkout v0.7.0
    cd ..

    # Initialize and fetch nested submodules
    git submodule update --init --recursive
    ```
    - Note: The framework has been tested with version 0.7.0 of `pypsa_earth`, it can be run also with the following versions, but you might encounter some bugs to solve.

2. Create the required Conda environments:
    - Use the YAML files in the `envs/` directory to create the required Conda environments:
        ```bash
        conda env create -f envs/dman-cep.yaml
        conda env create -f envs/pypsa-earth.yaml
        ```
    - If you encounter issues setting up the `pypsa-earth` environment, refer to the official setup instructions in the `pypsa-earth` repository.

3. Test the `pypsa-earth` module independently to ensure it is working correctly before proceeding.

4. Activate the appropriate environment `dman-cep` and run the main script:
    ```bash
    conda activate dman-cep
    python main.py
    ```
    - It will run with the configuration set in config.yaml file. 
