# PML Course Assignments - Copenhagen University 2024

This repository contains assignments for the Probabilistic Machine Learning course at Copenhagen University.

## Project Structure
```
├── B1/ 
│ └── B1.ipynb # Gaussian Process Implementation and Analysis 
├── B2/ 
│ └── PMLEcode.py # PMLE Implementation 
├── diffusion_model/ # Diffusion Model Implementation 
│ ├── config.py 
│ ├── ddpm.py 
│ ├── ema.py 
│ ├── eval.py 
│ └── ... 
└── pyproject.toml # Project dependencies and metadata
```

## Installation

1. Install Poetry if you haven't already:'
```sh
pip install poetry
```
2. Install dependencies:
```sh
poetry install
```

## Usage
- B1 notebook contains Gaussian Process implementation with MAP and NUTS sampling
- B2 contains PMLE implementation
- The diffusion_model directory contains a complete implementation of diffusion models and has a `run.sh` file. 