# DMAN-CEP (DEOpNet) — Distributed Multi-Agent Negotiation for Capacity Expansion Planning

Research model for **distributed macro-energy capacity expansion planning**.
Each country is modeled as an autonomous agent that solves a local planning problem (via PyPSA-Earth) and negotiates cross-border electricity exchanges through an iterative bidding mechanism.

> This repository accompanies the approach described in “Distributed Multi-Agent Negotiation for Capacity Expansion Planning” (Amato et al.).  

## Key idea (1 minute)
- Split the planning horizon into time slices (yearly/monthly/weekly).
- Each agent runs a local optimizer to compute system cost and **marginal import/export costs** via demand perturbations.
- Neighboring agents exchange bids; mutually beneficial trades are accepted iteratively until convergence (or max iterations).

## Documentation

## Quickstart

### 1) Clone (with submodules)
```bash
git clone --recurse-submodules https://github.com/ValeMTo/DMAN-CEP.git
cd DMAN-CEP
