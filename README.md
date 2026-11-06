# London Property Intelligence Platform

![Tests](https://github.com/samaddmirzaa/london-property-intelligence/actions/workflows/tests.yml/badge.svg)
![Drift Detection](https://github.com/samaddmirzaa/london-property-intelligence/actions/workflows/drift.yml/badge.svg)
![Data refresh](https://github.com/samaddmirzaa/london-property-intelligence/actions/workflows/refresh.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-containerised-blue)

**An end-to-end ML and LLM project for London home buyers: a house price model and an area chatbot, served from one API and one app.**

**Try it live:** [App](https://london-property-intelligence.streamlit.app/) · [API docs](https://london-property-api.onrender.com/docs)

> The demo runs on free hosting. The API sleeps when idle, so the first request can take 30 to 60 seconds. The chatbot allows roughly one detailed answer per minute.

<p>
  <img src="reports/figures/chat_ss.png" width="49%" alt="Area chatbot">
  <img src="reports/figures/map_ss.png" width="49%" alt="London price map">
</p>

## What it does

- **House price model.** XGBoost trained on 1.4 million Land Registry sales linked to EPC records, predicting the price of a London property.
- **Area chatbot.** Ask what it is like to live at any London postcode. It covers schools and their latest Ofsted results, transport, crime, air quality, prices, rents, council tax and stamp duty, with every figure compared against the rest of London. Facts come from public data through tool calls, and the language model only writes the sentences.

## Results

| | |
|---|---|
| **Price model** | R² 0.89 and a £47k median error on 2024 sales, which were held out of training entirely |
| **Chatbot tool use** | Called the right tool for the right postcode in 27 of 27 test questions |
| **Chatbot grounding** | 26 of 27 answers had every number traceable to the data (after checking each flag by hand) |
| **Drift detection** | A Population Stability Index check flagged the 2025 price shift (PSI 0.26) and recommended retraining |
| **Tests** | 72 tests on every push, with no API key or network needed |

## Engineering decisions I'm proud of

- **Time-based validation.** Trained on data up to 2023 and tested on 2024. A random split would leak 2024 price levels into training and inflate the score.
- **Leakage control.** Target encoding and imputation values are computed on training rows only, and smoothed toward the global mean for rare postcode districts.
- **A measured, not assumed, chatbot.** A hand-written agent loop (about 20 lines, no framework) with tool calling. I wrote 40 evaluation questions, scored them with code checks plus an LLM judge from a different model family, and verified the judge's flags by hand.
- **Retrieval chosen by evidence.** On 19 labelled questions, picking a GOV.UK guide first and then ranking sections with TF-IDF scored 19/19, against 16/19 for plain TF-IDF. At this size a vector database would add cost for no measured gain, so there isn't one.
- **Automation.** GitHub Actions run the tests, check for drift, and refresh crime, prices and rents every month, then test and redeploy.

## How it fits together

```
        Streamlit app (area chat · price map · price estimate)
                              |
                              v
              FastAPI on Render (Docker)
        /chat   /neighbourhood   /nearby   /guide
        /council-tax   /stamp-duty   /predict-price
               |                        |
     Area tables (Parquet)       XGBoost model bundle
     + GOV.UK guides index

   GitHub Actions: tests · drift check · monthly data refresh
```

## Tech stack

| Area | Tools |
|---|---|
| ML | pandas, scikit-learn, XGBoost, SHAP, MLflow |
| LLM | Groq (gpt-oss-120b) with tool calling, LLM-as-judge evaluation, TF-IDF retrieval |
| Geospatial | GeoPandas, SciPy KD-trees |
| Serving | FastAPI, Pydantic v2, Uvicorn, Docker, Render |
| App | Streamlit, Folium |
| CI and testing | GitHub Actions, pytest |

## Run it locally

```bash
git clone https://github.com/samaddmirzaa/london-property-intelligence
cd london-property-intelligence
conda create -n lpi python=3.11 && conda activate lpi
pip install -r requirements-dev.txt -r requirements-api.txt -r requirements.txt
pip install -e .

uvicorn london_property.api:app --reload   # API at localhost:8000/docs
pytest tests/ -v                           # test suite
```

The chatbot needs a free Groq API key in a `GROQ_API_KEY` environment variable. The full list of commands is in the [technical details](docs/TECHNICAL_DETAILS.md#running-it-locally).

## Known limitations

- The chatbot sometimes adds general knowledge that is not in the data, mostly transport links. I measured how often (13 of 27 answers had at least one such claim, mostly small) and the details are in the write-up.
- Prices and rents are borough averages, not street level, and the answers say so.

## Read more

The full write-up is in **[docs/TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md)**: the chatbot design, the evaluation results, data sources, the price model pipeline, drift detection, the API, tests and the repository layout.
