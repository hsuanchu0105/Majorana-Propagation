# Majorana Propagation

This repository contains Python code developed for simulating fermionic dynamics using Majorana Propagation (MP) and a modified Rotated Majorana Propagation (RMP) approach.

The project represents Majorana operators using binary vectors and propagates operator expansions under sequences of fermionic gates. It also includes tools for constructing Majorana operators, generating sparse fermionic Hamiltonians, comparing with direct matrix exponentiation for small systems, and analyzing truncation errors.

## Repository structure
```text
Majorana-Propagation/
├── src/
│   ├── Op.py          # Majorana operators, Node class, matrix conversion
│   ├── MajProp.py     # Core Majorana Propagation routines
│   ├── RMP.py         # Rotated Majorana Propagation routines
│   ├── Gate.py        # Gate construction and sparse Hamiltonian helpers
│   ├── TE.py          # Direct exponential and expectation-value utilities
│   ├── Err_anlys.py   # Error-analysis helpers
│   └── Plot.py        # Plotting utilities
├── test/              # Validation and comparison scripts
└── Setting/           # Project settings or parameter files
```

## Installation

Clone the repository:
```text
git clone git@github.com:hsuanchu0105/Majorana-Propagation.git
cd Majorana-Propagation
```

Install the main dependencies:
```text
pip install numpy scipy matplotlib tqdm pytest
```
