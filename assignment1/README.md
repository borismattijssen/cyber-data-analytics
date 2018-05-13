# Fraud detection
This folder contains code and data for classifying fraudulent credit card transactions. Follow the instructions below to reproduce our obtained results.

## Reproducing results

### Set-up

1. Set up a virtual python environment: `virtualenv -p python2 venv`, `source venv/bin/activate`.
2. Install python dependencies: `pip install -r requirements.txt`.
3. Pre-process data: `python fraud_detection.py`. The pre-processed data set is stored in `data/encoded_data.csv`.

### WhiteBox & BlackBox classifiers

1. Change to the notebook folder: `cd notebook`.
2. Start a Jupyter server: `jupyter notebook`.
3. Open `WhiteBox.ipynb`.
4. Open `BlackBox.ipynb`.

### Bonus

1. Pre-process data: `python bonus-preproc.py`. The augmented data set is stored in `data/augmented_data.csv`.
2. Run the bonus classifier: `python bonus.py`.
