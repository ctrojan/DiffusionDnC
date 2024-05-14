# Diffusion Generative Modelling for Divide-and-Conquer MCMC

Code to reproduce the results in the paper Diffusion Generative Modelling for
Divide-and-Conquer MCMC.

We recommend setting up a new environment to run these experiments. Dependencies can be installed by running the following command from the root directory:

`pip install -e .[all]`

Experiments can be run end to end using the `run.sh` script, e.g. the command `./run.sh all` executed from the root directory runs all of the experiments.

This repository is organised as follows:

- `/src/dnc` - package implementing the merging methods in the paper as well as the MCMC samplers and discrepancy metrics.
- `/experiments` - scripts to reproduce the results in the paper, organised by experiment. 
  - `/toy_logistic_regression` - run with `./run.sh toylogreg`
  - `/toy_logistic_regression_bigger` - run with `./run.sh bigtoylogreg`
  - `/mixture_of_gaussians` - run with `./run.sh mog`
  - `/powerplant` - run with `./run.sh powerplant`
  - `/spambase` - run with `./run.sh spambase`
  
Each experiment has a setup script that loads or generates the data and generates the MCMC samples, a script for each merging method to generate the merged samples, and a script that computes the numerical comparison and generates the plots from the paper. The merging methods that do not require optimisation or MCMC sampling are executed in the comparison script. Scripts should be run from the root directory.
    

### Requirements

The python requirements can be installed from `pyproject.toml` using the command above. The versions originally used are as follows:

Python

- `python == 3.10.13`
  - `blackjax == 1.0.0`
  - `cola-ml == 0.0.1`
  - `flax == 0.7.4`
  - `gpjax == 0.7.1`
  - `jax == 0.4.19`
  - `jaxlib == 0.4.19`
  - `matplotlib == 3.8.0`
  - `numpy == 1.26.1`
  - `optax == 0.1.7`
  - `pandas == 2.2.1`
  - `scipy == 1.11.3`
  - `tensorflow == 2.11.0`
  - `tensorflow-probability == 0.19.0`
  - `tqdm == 4.66.1`
  - `ucimlrepo == 0.0.6`

R

- `R == 4.1.2`
    - `reticulate == 1.35.0`
    - `parallelMCMCcombine == 2.0`
