Model Architecture
===================

This page describes the main structure of the repository and the roles of the key classes and modules.

translation
-----------

The ``translation`` folder contains the core classes that define the model:

**energySystemModel.py**

The central orchestrator of the modeling process. It is responsible for:

- Creating the internal COP (Constraint Optimization Problem) models for each country by calling ``energyAgentModel.py``.
- Solving the internal COPs by interfacing with Frodo2 (executing the Java solver).
- Managing parallel solving of multiple COPs to improve efficiency.
- Solving the upper DCOP (Distributed Constraint Optimization Problem) for the entire system per timeslice and year.
- Gathering and processing results from the solved DCOPs.

**energyAgentModel.py**

Defines the model for a single agent (typically representing a country for a given timeslice and year). This class is called by ``energySystemModel`` and is responsible for:

- Setting up the internal COP for the specific agent.
- Encoding the agent’s variables, domains, and constraints.

parsers
^^^^^^^

The ``parsers`` subfolder contains helper modules for data ingestion:

**configParser.py**

Loads and processes configuration settings required for the model run.

**dataParser.py**

Reads and structures the input data (e.g., resource potentials, demand profiles) retrieved and preprocessed in the ``resources`` folder.

.. note::

    Different parsers have been or will be developed to handle data from various sources. 
    Depending on the origin of the data, specific parsing logic is applied. 
    The goal is to take raw data, preprocess it into a standardized format, and load it into the model through ``dataParser``.

    For example, data sources may include:

    - **Zenodo**: Publicly available datasets hosted on Zenodo.
    - **Local Storage**: Files stored locally on the user's machine.
    - **Other Sources**: Custom or external data sources specified by the user.

resources
---------

The ``resources`` folder contains all the external input data required by the model, such as:

- Preprocessed datasets
- Configuration files
- Any additional resources needed for building and solving the DCOPs.
