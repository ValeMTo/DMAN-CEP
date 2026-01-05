.. DEOpNet documentation master file, created by
   sphinx-quickstart on Thu Apr 10 17:33:53 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DEOpNet: Distributed Energy Optimization Network
=================================================

**DEOpNet** is a research prototype for distributed energy system planning based on
**Distributed Constraint Optimization Problems (DCOPs)**. Unlike traditional
centralized macro-energy models, which assume full cooperation and a single decision-maker,
DEOpNet reflects the decentralized nature of international energy planning.

Each country is modeled as an autonomous agent, independently optimizing its energy
production and investment decisions, while coordinating cross-border electricity exchanges
through structured communication. The model allows for both cooperation and partial
information sharing — without relying on a central planner.

The model enables more realistic simulations of international energy coordination and
can evaluate the feasibility of decentralized planning outcomes under varied geopolitical,
infrastructural, and environmental assumptions.

DEOpNet is solved using the Frodo2_ DCOP framework.

.. _Frodo2: https://frodo-ai.tech/

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting-started/Installation
   getting-started/Introduction

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   user-guide/Model-Architecture
   user-guide/Model-Structure
   user-guide/Model-Formulation

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   examples/Example-1
   examples/Example-2

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :hidden:

   References
   Contacts
   FAQ

**Licence**: Copyright 2025 Valeria Amato --- DEOpNet is licensed under the open source `CC0-1.0 License <https://creativecommons.org/publicdomain/zero/1.0/>`_.
