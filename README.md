# London Property Intelligence Platform

A machine learning project that predicts residential property sale prices across all 33 London boroughs, built on HM Land Registry sales data linked to Domestic Energy Performance Certificates. The trained model reaches an R squared of around 0.88 on log price when tested on genuinely unseen 2024 sales, and every prediction can be explained feature by feature using SHAP.

This repository is a work in progress. The machine learning core is complete. The deployment layer, the automated retraining pipeline, and the London assistant chatbot are still to come. This README will grow as those parts are added.

## What this project is

Most house price models fall down because Land Registry data on its own tells you what a property sold for but not how big it is. Without floor area you cannot tell a studio apart from a penthouse in the same postcode, so the model has very little to work with. This project uses a richer dataset published by the Greater London Authority that links each Land Registry transaction to its EPC certificate, which means every property comes with its floor area and its energy rating already attached. Floor area turns out to be the single strongest driver of price, and having it changes everything.

On top of that the project merges in postcode level context such as location coordinates, travel zone, deprivation index, average income, and distance to the nearest station. The result is a model that understands both the property itself and the neighbourhood around it.

The longer term goal is to turn this model into a London Housing and Living Assistant. That is a chatbot which can answer questions about property prices, affordability, and what different areas of London are like, grounded in real data rather than guesswork.

## Where the project is right now

The full machine learning pipeline is finished and lives in the notebooks folder. It runs in a clear sequence from raw data to an explained, saved model.

**Notebook 01, Data Profiling.** Loads all 33 borough files, confirms they share one schema, and measures completeness and data quality. This is where the messy parts of the data get discovered before any decisions are made, for example the fact that construction age has a large chunk of invalid entries and that floor area contains some impossible zero values.

**Notebook 02, Data Cleaning.** Combines the 33 files into one table and applies a set of justified filters. It restricts the data to 2008 onwards because EPC certificates only became mandatory for sales in 2008, keeps prices between fifty thousand and ten million pounds, and keeps floor areas between fifteen and five hundred square metres to remove data errors. It also fixes a subtle date parsing problem where the dates were stored in two different formats, and it log transforms the price because prices are heavily skewed. The cleaned data comes to roughly 1.4 million transactions and is saved as a Parquet file.

**Notebook 03, Exploratory Data Analysis.** Explores what actually drives price. The headline findings are that floor area is the strongest single predictor, that prices rose strongly across the years which is why the model must be tested on the future rather than a random sample, and that energy efficiency has a surprising negative relationship with price. That last point is explained by construction age. Older period homes are expensive but energy inefficient, while newer builds are efficient but cheaper, so age is the hidden factor behind the apparent paradox.

**Notebook 04, Feature Engineering.** Merges the postcode data, creates the transformed features the model needs such as log floor area and log income, encodes the categorical columns, and prepares the postcode district column for the location encoding that happens during modelling. The output is a single feature table used by the model.

**Notebook 05, Modelling.** Trains and compares three models using a strict time based split, training on data up to 2023 and testing on 2024. Ridge regression is the simple linear baseline, Random Forest is the middle option, and XGBoost is the winner. The postcode district is turned into a number using target encoding, which replaces each district with the average price of its training properties, and this is done carefully so that no information from the test set leaks into training. After tuning, the final XGBoost model reaches an R squared of about 0.88 on log price and about 0.89 on raw pounds, with a typical error of around forty seven thousand pounds. The model is saved together with everything needed to reproduce its features at prediction time.

**Notebook 06, SHAP Explanations.** Opens up the model to explain why it makes the predictions it does. It shows which features matter most overall, and it can break down any single prediction into the contribution of each feature. Because the model predicts log price, the explanations are read as percentage effects on price rather than flat pound amounts.

## Key results

The model was tested on 2024 sales that it never saw during training.

* R squared on log price is approximately 0.88.
* R squared on raw price in pounds is approximately 0.89.
* The typical prediction error, measured as the median absolute error, is around forty seven thousand pounds.
* Accuracy is strongest through the middle of the market, with errors of roughly seven to ten percent of price, and weakest at the very top end where prices depend on things the data cannot see.

## How to read the numbers honestly

The most important choice in this project is that the model is tested on the future, not on a random shuffle of the data. Because London prices rise over time, a random split would let the model peek at future price levels during training and produce a score that looks impressive but would not hold up in the real world. Testing on 2024 after training only on earlier years is harder and more honest, and the R squared reflects real predictive ability rather than leakage.

The model is also honest about its limits. It is trained on properties that have an EPC certificate, which biases it towards homes that have been sold or rented since 2008. Its largest errors are on very expensive properties, where price depends on condition, exact street, renovation quality, and views, none of which are in the data. Location is captured at the postcode level rather than the individual building. These limits are described plainly rather than hidden, because understanding where a model is weak is as important as knowing where it is strong.

## The data

The project uses three sources, all of which are external and are not committed to this repository because of their size.

* The Greater London Authority House Price per Square Metre dataset, which is HM Land Registry Price Paid Data linked to Domestic EPC certificates, split into one file per borough and covering 1995 to 2024.
* A London postcode reference file which provides coordinates, travel zone, deprivation, income, and distance to station for each postcode.
* HM Land Registry Price Paid Data for 2022 through 2025, which is raw sales data with no floor area. This is kept for a later stage of the project to demonstrate automated drift detection on genuinely new data.

Because the raw files are large and freely available at source, they are not stored here. The notebooks regenerate every processed file from the raw data, so only the code and the small final model are version controlled.

## Tech stack

* Python 3.11 in a Conda environment
* pandas and PyArrow for data handling and Parquet storage
* scikit-learn for the baseline models and pipeline tools
* XGBoost for the final model
* SHAP for model explanations
* matplotlib and seaborn for the charts

Planned for later phases are MLflow for experiment tracking, FastAPI for serving the model, Docker and Render for deployment, Streamlit for the interactive London map, GitHub Actions for the automated retraining pipeline, and LangChain for the chatbot.

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

## What comes next

The machine learning core is done. The next stages will wrap it into a working product.

* Experiment tracking with MLflow, so every training run is recorded and the model is versioned.
* A prediction service built with FastAPI and containerised with Docker, deployed to a public URL.
* An interactive map of London built with Streamlit, where you can click an area and see predicted prices.
* An automated pipeline using GitHub Actions that watches for new sales data, checks whether the data has drifted away from what the model was trained on, and retrains and redeploys the model when the change is large enough to matter.
* A London Housing and Living Assistant, a chatbot that answers questions about prices, affordability, and neighbourhoods, grounded in the model and the data rather than made up.

This README will be updated as each of these is built.
