![Tests](https://github.com/samaddmirzaa/london-property-intelligence/actions/workflows/tests.yml/badge.svg)

# London Property Intelligence 

This project uses a dataset published by the Greater London Authority that links each Land Registry transaction to its EPC certificate, which means every property comes with its floor area and its energy rating attached. 

On top of that the project merges in postcode level context such as location coordinates, travel zone, deprivation index, average income, and distance to the nearest station. The result is a model that understands both the property itself and the neighbourhood around it.

The full machine learning pipeline is finished and lives in the notebooks folder. It runs in a clear sequence from raw data to an explained, saved model.

**Notebook 01, Data Profiling.** Loads all 33 borough files, confirms they share one schema, and measures completeness and data quality. 

**Notebook 02, Data Cleaning.** Combines the 33 files into one table and applies a set of justified filters. It restricts the data to 2008 onwards because EPC certificates only became mandatory for sales in 2008, keeps prices between fifty thousand and ten million pounds, and keeps floor areas between fifteen and five hundred square metres.

**Notebook 03, Exploratory Data Analysis.** Explores what actually drives price. The headline findings are that floor area is the strongest single predictor, that prices rose strongly across the years which is why the model must be tested on the future rather than a random sample, and that energy efficiency has a surprising negative relationship with price. That last point is explained by construction age. Older period homes are expensive but energy inefficient, while newer builds are efficient but cheaper, so age is the hidden factor behind the apparent paradox.

**Notebook 04, Feature Engineering.** Merges the postcode data, creates the transformed features the model needs such as log floor area and log income, encodes the categorical columns, and prepares the postcode district column for the location encoding that happens during modelling. 

**Notebook 05, Modelling.** Trains and compares three models using a strict time based split, training on data up to 2023 and testing on 2024. Ridge regression is the simple linear baseline, Random Forest is the middle option, and XGBoost is the winner. The postcode district is turned into a number using target encoding, which replaces each district with the average price of its training properties.

**Notebook 06, SHAP Explanations.** Opens up the model to explain why it makes the predictions it does. It shows which features matter most overall, and it can break down any single prediction into the contribution of each feature.

## Key results

The model was tested on 2024 sales that it never saw during training.

* R squared on log price is approximately 0.88.
* R squared on raw price in pounds is approximately 0.89.
* The typical prediction error, measured as the median absolute error, is around forty seven thousand pounds.

The model is is trained on properties that have an EPC certificate, which biases it towards homes that have been sold or rented since 2008. Its largest errors are on very expensive properties, where price depends on condition, exact street, renovation quality, and views, none of which are in the data. Location is captured at the postcode level rather than the individual building.

## The data

The project uses three sources, all of which are external and are not committed to this repository because of their size.

* The Greater London Authority House Price per Square Metre dataset, which is HM Land Registry Price Paid Data linked to Domestic EPC certificates, split into one file per borough and covering 1995 to 2024.
* A London postcode reference file which provides coordinates, travel zone, deprivation, income, and distance to station for each postcode.

## Tech stack

* Python 3.11
* pandas and PyArrow for data handling and Parquet storage
* scikit-learn for the baseline models and pipeline tools
* XGBoost for the final model
* SHAP for model explanations
* matplotlib and seaborn for the charts

## Repository structure

```
london-property-intelligence/
├── data/
│   ├── raw/                 the 33 borough files (not committed)
│   ├── supplementary/       postcode reference and raw yearly sales (not committed)
│   └── processed/           generated Parquet files (not committed)
├── notebooks/
│   ├── 01_data_profiling.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modelling.ipynb
│   └── 06_SHAP.ipynb
├── models/
│   └── price_model.joblib   the trained model and its preprocessing
├── reports/figures/         saved charts for this README
├── requirements.txt
└── README.md
```

## Running it yourself

1. Create the environment and install the dependencies.

```
conda create -n lpi python=3.11
conda activate lpi
pip install -r requirements.txt
```

2. Download the raw data from the sources described above and place the borough files in data/raw and the postcode file in data/supplementary.

3. Run the notebooks in order from 01 to 06. Each one reads the output of the previous stage, so the order matters. The final notebook produces the saved model and the SHAP explanations.


This README will be updated as the project will be expanded further.
