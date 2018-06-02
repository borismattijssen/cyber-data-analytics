# Anomaly detection
This folder contains code and data for detecting anomalies in a SCADA system. Follow the instructions below to reproduce our obtained results.

## Reproducing results

### Set-up

1. Set up a virtual python environment: `virtualenv -p python2 venv`, `source venv/bin/activate`.
2. Install python dependencies: `pip install -r requirements.txt`.

### 1. Familiarization
2. Start a Jupyter server: `jupyter notebook`.
3. Open `DataExploration.ipynb`.

### 2. ARMA

2. Start a Jupyter server: `jupyter notebook`.
3. The script `nerdalize/arma_tuning.py` can be used to find optimal ARMA parameters for all signals.
4. Open `ARMA_all_signals.ipynb` to detect anomalies on all signals.
5. Open `ARMA.ipynb` to visualize ARMA anomaly detection on one signal.
