# Anomaly detection
This folder contains code and data for detecting anomalies in a SCADA system. Follow the instructions below to reproduce our obtained results.

## Reproducing results

### Set-up

1. Set up a virtual python environment: `virtualenv -p python2 venv`, `source venv/bin/activate`.
2. Install python dependencies: `pip install -r requirements.txt`.

### 1. Familiarization
1. Start a Jupyter server: `jupyter notebook`.
2. Open `DataExploration.ipynb`.

### 2. ARMA

1. Start a Jupyter server: `jupyter notebook`.
2. The script `nerdalize/arma_tuning.py` can be used to find optimal ARMA parameters for all signals(takes time).
3. Open `ARMA_all_signals.ipynb` to detect anomalies on all signals.
4. Open `ARMA.ipynb` to visualize ARMA anomaly detection on one signal.

### 3. Discrete models

1. Start a Jupyter server: `jupyter notebook`.
2. The script `nerdalize/sax_tuning.py` can be used to find optimal parameters for all signals(takes time).
3. Open `Discretization_SAX.ipynb` to detect anomalies on one signal.

### 4. PCA

1. Start a Jupyter server: `jupyter notebook`.
2. The script `nerdalize/pca_tuning.py` can be used to find optimal PCA parameters(takes time).
3. Open `PCA analysis.ipynb` to detect anomalies on all signals.

### 5. Comparison
1. Start a Jupyter server: `jupyter notebook`.
2. Open `Comparison.ipynb` to compare the methods, both visually and metric-based.
