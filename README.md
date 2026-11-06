# London Property Intelligence

![Tests](https://github.com/samaddmirzaa/london-property-intelligence/actions/workflows/tests.yml/badge.svg)
![Drift Detection](https://github.com/samaddmirzaa/london-property-intelligence/actions/workflows/drift.yml/badge.svg)
![Data refresh](https://github.com/samaddmirzaa/london-property-intelligence/actions/workflows/refresh.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-containerised-blue)

Two tools for people buying a home in London, served from one API and one web app:

- **Area Chatbot.** Type a postcode and ask what it's like to live there. It answers using public data on
  schools, transport, crime, air quality, prices, rents and council tax, and compares each figure with the rest
  of London. It also answers buying questions (leasehold, stamp duty, conveyancing) from official GOV.UK guides.
- **House Price Model.** Estimates the sale price of a London property. It was trained on 1.4 million sales and
  scores R² 0.89 with a median error of about £47k on 2024 sales it never saw during training.

**Try it:** [App](https://london-property-intelligence.streamlit.app/) · [API docs](https://london-property-api.onrender.com/docs)

Please be patient with the Streamlit app page as the demo runs on free hosting. If something still looks stuck after a few minutes,
refresh the page once. The API runs on Render's free tier, which shuts the service down after 15 minutes of inactivity,
so the first request can take 30 to 60 seconds while the container starts. The chatbot runs on
Groq's free tier, which allows roughly one detailed answer per minute.

![Area chat](reports/figures/chat_ss.png)

---

## Area Chatbot

You give it a postcode. It describes the area, what is within a 10-minute walk, nearby
schools with their latest Ofsted results, transport access, safety, environment and typical costs. It tailors the
answer to the buyer (family, commuter, first-time buyer), says where the data comes from, and lists what the data
can't tell you, such as noise.

The language model never states facts from memory. It chooses from a small set of tools
(neighbourhood overview, nearby places, council tax, stamp duty, GOV.UK guide search), my code runs them against
the data, and the model writes the answer from the results. 

Comparisons with London are calculated in code before the model sees them. Crime and census figures are ranked
across London's 704 wards, and walkable amenities are ranked against 5,000 random London postcodes. That keeps the
model from getting the maths wrong.

19 realistic buyer questions are labelled with the correct GOV.UK section and two ways tested
to find it. Searching all 76 sections directly with TF-IDF got the right section in the top two for 16 of 19.
Letting the model pick one of 12 guides first, then ranking sections inside it, got 19 of 19. 

Test questions were written on postcodes not used while writing the prompt. Each answer is
checked by code (right tool, right postcode, expected figures, every number traceable to the data) and by a second
LLM from a different model family that flags unsupported claims. I checked all of that judge's flags by hand:
19 of 29 were real problems and 10 were judge mistakes.

Results on the 27 questions run so far (the rest are pending because of free-tier token limits):

| | gpt-oss-120b (live model) | gpt-oss-20b |
|---|---|---|
| Right tool, right postcode | 27 / 27 | 25 / 27 |
| Every number traceable to data | 26 / 27 | 23 / 27 |

The main remaining weakness is that the larger model sometimes adds transport details from general knowledge,
such as which lines serve a station, which the data doesn't contain. 13 of the 27 answers had at least one small
unsupported claim. That is why 120b is the live model, and why the limitation is listed below.

**Data.** All public: ONS postcodes, wards and Census 2021, Metropolitan Police crime, Indices of Deprivation 2025,
TfL accessibility, London air emissions, UK House Price Index and ONS rents, council tax, DfE schools and Ofsted,
NHS GP practices, Geolytix supermarkets, OS Open Greenspace, Overture Maps places, planning.data.gov.uk, and
GOV.UK buying guides. A build script turns them into five small tables the API loads at startup.

**Monthly refresh.** A GitHub Action runs on the 25th of each month. It pulls the newest crime, house price and
rent releases and the GOV.UK guides, runs the tests on the new data and commits it, which redeploys the API. If a
source fails, its last good data is kept.

---

## House Price Model

**What it does.** Takes a postcode plus floor area, property type, tenure and build year, and returns an
estimated sale price.

**Results** (tested on all 2024 sales):

| Metric | Value |
|---|---|
| R² (log price / raw £) | 0.88 / 0.89 |
| Median absolute error | £46,972 |
| Mean absolute error | £91,446 |

Typical error is 7% to 14% of the price, and it is worst for the most expensive homes (about 14% in the top
decile), where price depends on things the data doesn't hold. Removing groups of features shows location and
area wealth matter most (R² drops 0.22), then year of sale (0.19) and size (0.11).

![London price map](reports/figures/map_ss.png)

**How it was built**

- **Data.** 2.81 million Land Registry sales linked to energy certificates (EPCs), cleaned down to 1,411,720
  sales from 2008 to 2024. Removed: pre-2008 sales (unreliable linkage), prices outside £50k to £10M, and floor
  areas outside 15 to 500 m².
- **Features.** 18 in total: size, rooms, build year, property type, tenure, date of sale, and postcode-based
  location, travel zone, deprivation, local income and distance to the nearest station.
- **Validation.** Trained on sales up to 2023 and tested on 2024. A random split would let the model learn
  2024 price levels in advance and overstate the score.
- **Avoiding leakage.** The average price per postcode district is calculated from training data only, and
  smoothed so districts with a few sales don't get extreme values. Missing values are filled the same way.
- **Model.** XGBoost, picked over Ridge and Random Forest baselines. Experiments are tracked in MLflow and
  explained with SHAP.

---

## How it fits together

```
        Streamlit app (area chat · price map · price estimate)
                              |  HTTPS
                              v
          FastAPI on Render (Docker)
          /chat  /neighbourhood  /nearby  /guide
          /council-tax  /stamp-duty  /predict-price
               |                        |
     Area tables (Parquet)       Price model bundle (XGBoost)

   GitHub Actions: tests on every push · drift check on new data · monthly data refresh
```

**Tech stack:** Python 3.11, pandas, scikit-learn, XGBoost, SHAP, MLflow, GeoPandas, SciPy, Groq (gpt-oss-120b),
FastAPI, Pydantic, Docker, Streamlit, Folium, pytest, GitHub Actions. Hosted on Render and Streamlit Community Cloud.

---

## Run it locally

```bash
git clone https://github.com/samaddmirzaa/london-property-intelligence
cd london-property-intelligence

conda create -n lpi python=3.11
conda activate lpi
pip install -r requirements-dev.txt -r requirements-api.txt -r requirements.txt
pip install -e .

uvicorn london_property.api:app --reload    # API at localhost:8000/docs
streamlit run streamlit/main.py             # the app (calls the live API)
pytest tests/ -v                            # tests
python -m london_property.drift             # drift check
```

The chatbot needs a free Groq API key in a `GROQ_API_KEY` environment variable, and the map pages need a CARTO
key in `.streamlit/secrets.toml`.

## Limitations

- Free hosting means slow first loads and about one chatbot answer per minute.
- The chatbot can still add details that aren't in the data, mostly transport links. The evaluation above
  measures how often.
- Prices and rents are borough averages, not street level. Crime per resident looks high in central areas full
  of visitors, and the answers say so.

## Repository layout

```
notebooks/              data cleaning, analysis, price model, SHAP, MLflow, area data
src/london_property/    model, api, agent, tools, area_data, refresh, evaluate, judge, drift
streamlit/              the web app
eval/                   the 40 evaluation questions
reports/                figures and saved evaluation runs
tests/                  automated tests
models/                 trained price model
data/                   processed area tables, drift and training reference data
.github/workflows/      tests, drift check, monthly refresh
```
