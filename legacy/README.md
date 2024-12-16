# README

> Legacy code from my Master Thesis

# Setup

```sh
# python 3.8.10
pip install -r requirements.txt
```

Or via conda:

```sh
conda create -n ad-fidelity-legacyc
conda activate ad-fidelity-legacy
conda install python=3.8
pip install -r requirements.txt
```

# Overview

- model.py: defines the convnet
- data.py: utilities for loading and preprocessing the ADNI data
- plotutils.py: utilities for plotting
- eda
    - class-comparison.ipynb: 
    - aal-atlas-test.ipynb
- training.ipynb: train cnn's on adni data
- feature-attributions.ipynb: compute feature attributions for the attribution methods

# How to Run

- training.ipynb
- feature-attributions.ipynb
