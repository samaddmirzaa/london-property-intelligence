# London Property Intelligence Platform

![Tests](https://github.com/samaddmirzaa/london-property-intelligence/actions/workflows/tests.yml/badge.svg)
![Drift Detection](https://github.com/samaddmirzaa/london-property-intelligence/actions/workflows/drift.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-containerised-blue)

A house price model for London, trained on 1.4 million EPC-linked Land Registry
transactions and deployed as a live API with automated testing and drift monitoring.

The model scores R² 0.89 on raw prices with a £47k median error, measured on 2024
sales that were held out of training entirely.

---

## Live demos

| Demo | Link |
|---|---|
| Interactive map | [Streamlit Map](https://samaddmirzaa-london-property-intelligen-streamlitmap-app-x55hkl.streamlit.app/) |
| Price predictor | [App](https://samaddmirzaa-london-property-intelligence-streamlitapp-xwgple.streamlit.app/) |
| API documentation | [london-property-api.onrender.com/docs](https://london-property-api.onrender.com/docs) |

The API runs on Render's free tier, which shuts the service down after 15 minutes of
inactivity. If nobody has used it recently, the first request takes 30 to 60 seconds
while the container starts up. Later requests are fast.

![London price map](reports/figures/map_ss.png)

---

## What it does

You give it a London postcode plus a few details about a property (floor area, type,
tenure, construction year) and it estimates the sale price.

The training data comes from the Greater London Authority, who matched HM Land
Registry Price Paid records against Domestic EPC certificates by address. 

Around the model sits the infrastructure you would need to actually run it: a
validated REST API, a Docker container, experiment tracking, a test suite that runs
on every push, and a pipeline that watches incoming data for signs the model is going
stale.

---

## Results

| Metric | Value |
|---|---|
| R² (log price) | 0.88 |
| R² (raw £) | 0.89 |
| Median absolute error | £46,972 |
| Mean absolute error | £91,446 |

Half of all properties are predicted within £47k of their actual sale price. The mean is nearly double that
because a handful of multi-million pound sales produce very large absolute errors and
drag the average up. Quoting only the mean would make the model look worse than it is
for typical properties, and quoting only the median would hide a real weakness.

### Accuracy across the price range

| Price decile | Median price | Median error | Error % |
|---|---|---|---|
| 1 | £245000 | £29078 | 12% |
| 2 | £330000 | £28515 | 9% |
| 3 | £390000 | £29460 | 8% |
| 4 | £440000 | £30824 | 7% |
| 5 | £495000 | £38376 | 8% |
| 6 | £555000 | £45410 | 8% |
| 7 | £625250 | £54686 | 9% |
| 8 | £750000 | £72113 | 10% |
| 9 | £935000 | £101766 | 11% |
| 10 | £1575000 | £212895 | 14% |

Accuracy holds up well through the bulk of the market and falls off at the top end,
where price depends on things the data cannot see.

### What the model actually uses

Feature importance was measured three ways, because each method gets fooled by
different things. XGBoost's built-in gain score splits credit unpredictably between
correlated features. Permutation importance is blind to any feature that is constant
in the test set, which caught out `year` here since the test set is a single year.
Grouped ablation (retrain without a whole concept, see what breaks) turned out to be
the most reliable.

| Feature group | R² lost when removed |
|---|---|
| Location and area wealth | 0.22 |
| Year of sale | 0.19 |
| Size (floor area, rooms) | 0.11 |
| Property type and tenure | 0.02 |
| Construction age | 0.01 |
| Energy efficiency | 0.00 |

Location and size do the heavy lifting, which is what you would expect for London
housing. Energy efficiency contributes nothing, for reasons covered under
Limitations.

---

## Architecture

```
              Streamlit map                Streamlit predictor
              (click a point)              (type a postcode)
                     |                              |
                     +--------------+---------------+
                                    |  HTTPS
                                    v
                       +-------------------------+
                       |   FastAPI on Render     |
                       |   POST /predict-price   |
                       |   GET  /health          |
                       |   Pydantic validation   |
                       +-----------+-------------+
                                   | loads at startup
                       +-----------v-------------+
                       |  Model bundle (joblib)  |
                       |  XGBoost + target       |
                       |  encoding + medians     |
                       +-------------------------+

  GitHub Actions
    tests.yml   runs pytest on every push
    drift.yml   runs on data commits: PSI check, then conditional retrain
```

---

## How the pipeline works

**Cleaning.** 33 borough files hold 2.81 million transactions. After filtering that
comes down to 1,411,720 rows covering 2008 to 2024. Three filters do the work.
Anything before 2008 goes, because EPCs only became mandatory for property sales that
year and the linkage before then is unreliable. Prices outside £50k to £10M go, since
those are mostly transfers between family members or data errors rather than open
market sales. Floor areas outside 15 to 500 m² go for the same reason.

**Features.** Eighteen in total. Log floor area, number of rooms, construction year,
one-hot encoded property type and tenure, year and month of sale. Then everything
derived from the postcode: latitude, longitude, travel zone, deprivation index,
average income, distance to the nearest station. Finally, a smoothed target encoding
of the postcode district.

**Validation.** Split by time, not at random. London prices roughly doubled over the
period the data covers, so a random split would let the model see 2024 price levels
while training and then reward it for "predicting" them. That inflates the score and
tells you nothing about how the model would behave on genuinely new data. Training
runs on everything up to 2023 and the model is tested on 2024.

**Leakage control.** Target encoding uses the thing we are trying to predict, so it
has to be handled carefully. The district averages are computed on training rows
only, then applied to the test set as a lookup. They are also smoothed toward the
global mean, so a district with three sales does not get an extreme value the model
would overfit to. Imputation medians are computed the same way, on training data
only.

**Model.** XGBoost, depth 10, 1000 trees, learning rate 0.04. It was picked over
Ridge and Random Forest baselines using 2023 as a validation year, then refit on all
data through 2023 before the final test.

---

## Drift detection

New data does not stay like old data. Prices move, the mix of properties being sold
changes, and a model trained on last year's market slowly gets worse without ever
throwing an error. The drift workflow watches for this.

It runs whenever something is committed to `data/drift/`. For each feature it
computes a Population Stability Index against a reference sample of the training
distribution, then decides whether the change is big enough to justify retraining.

Run against 84,310 Greater London sales from 2025:

```
price          PSI = 0.262   DRIFTED
zone           PSI = 0.004   stable
imd_index      PSI = 0.003   stable
lat / lon      PSI = 0.002   stable
avg_income     PSI = 0.001   stable
propertytype   PSI = 0.001   stable
duration       PSI = 0.000   stable

Decision: RETRAIN, target (price) drifted, PSI 0.262
```

The geography is unchanged as London is still London, the same
postcodes with the same deprivation scores and the same stations. Prices are a
different story. The 2025 median sits well above what the model was trained on, and
that is exactly the situation where predictions start drifting low.

**Thresholds.** A feature counts as drifted above PSI 0.2, following the usual
convention (below 0.1 is stable, 0.1 to 0.2 is worth watching, above 0.2 has moved).
Retraining triggers on any of three conditions: the target drifts, more than 30% of
features drift, or a single feature exceeds PSI 0.5.

---

## Tech stack

Python 3.11, pandas, scikit-learn, XGBoost, SHAP for the modelling. MLflow for
experiment tracking, using a SQLite backend and the model registry with aliases.
FastAPI, Pydantic v2 and Uvicorn for serving, packaged with Docker. Streamlit and
Folium for the front ends. GitHub Actions and pytest for CI. Hosted on Render (API)
and Streamlit Community Cloud (UI).

---

## Running it locally

```bash
git clone https://github.com/samaddmirzaa/london-property-intelligence
cd london-property-intelligence

conda create -n lpi python=3.11
conda activate lpi
pip install -r requirements-dev.txt
pip install -e .
```

Then any of:

```bash
uvicorn london_property.api:app --reload      # API at localhost:8000/docs
streamlit run streamlit/map_app.py            # map UI
pytest tests/ -v                              # test suite
python -m london_property.drift               # drift check
```

The raw data files are not in the repository, since they run to several hundred
megabytes. The GLA "House Price per Square Metre" borough files and the Doogal London
postcode dataset are both publicly available, and notebooks 01 to 04 rebuild every
processed file from them.

---

## Repository layout

```
notebooks/              01 profiling through 07 MLflow tracking
src/london_property/
    model.py            shared feature building and prediction
    api.py              FastAPI service
    drift.py            PSI calculation and retrain decision
streamlit/              map_app.py, app.py
tests/                  8 tests covering model correctness and API behaviour
models/                 trained model bundle
data/
    reference/          training distribution used for drift comparison
    drift/              new data, committing here triggers the drift workflow
.github/workflows/      tests.yml, drift.yml
Dockerfile
```
