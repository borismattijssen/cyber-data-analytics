# Sampling & Fingerprinting
This folder contains code and data for sampling and fingerprinting. Follow the instructions below to reproduce our obtained results.

## Downloading the data
Please note that the datafiles in the `data` folder are managed through [git lfs](https://git-lfs.github.com/). If you want to clone the full repo, you first need to set up git lfs on your system (this can be done very easily). Otherwise you can download the data files manually from [this website](https://mcfp.weebly.com/the-ctu-13-dataset-a-labeled-dataset-with-botnet-normal-and-background-traffic.html). Note, that we rename the data files to include the Malware capture number (see `data/` for examples).

## Reproducing results

### Set-up

1. Set up a virtual python environment: `virtualenv -p python3 venv`, `source venv/bin/activate`.
2. Install python dependencies: `pip install -r requirements.txt`.

### 1. Sampling
1. Start a Jupyter server: `jupyter notebook`.
2. Open `Sampling.ipynb`.

### 2. Sketching

1. Start a Jupyter server: `jupyter notebook`.
4. Open `Sketching.ipynb`.

### 3. Discretization

1. Start a Jupyter server: `jupyter notebook`.
3. Open `Discretization.ipynb`.

### 4. Profiling

1. Start a Jupyter server: `jupyter notebook`.
3. Open `Profiling.ipynb`.
