import streamlit as st
import pandas as pd
import requests
from pathlib import Path

api_url = "https://london-property-api.onrender.com"

property_types = ['Flat', 'Terraced', 'Semi-Detached', 'Detached']
tenures = ['Freehold', 'Leasehold']


# Cached so the parquet loads once
@st.cache_data
def load_postcodes():
    root = Path(__file__).resolve().parent.parent
    return pd.read_parquet(root /'data'/'supplementary'/'postcodes_london.parquet')


postcodes = load_postcodes()

st.title('London House Price Predictor')

# Inputs in the sidebar, result in the main area
st.sidebar.header('Property')
postcode = st.sidebar.text_input('Postcode', 'E14 5AB')

# Bounds match the API's Field(ge=15, le=500) and the model's training range
floor_area = st.sidebar.number_input('Floor Area (Square Meters)', 15, 500, 70)
ptype = st.sidebar.selectbox('Type', property_types)
tenure = st.sidebar.radio('Tenure', tenures)
built = st.sidebar.slider('Construction Year', 1900, 2024, 1960)

# Field names must match the API's PropertyRequest model
payload = {
    'postcode': postcode,
    'floor_area': floor_area,
    'property_type': ptype,
    'tenure': tenure,
    'built_year': built,
}

# Spinner explains the wait since the free tier spins down after 15 min idle
with st.spinner('Predicting... (the free-tier API may take up to a minute to wake up)'):
    try:
        # timeout=120 is generous on purpose: a cold start can take 30-60s
        response = requests.post(f'{api_url}/predict-price', json=payload, timeout=120)

        # 404 = unknown postcode. A normal user mistake, so show a soft warning
        if response.status_code == 404:
            st.warning('Postcode not found in the data. Try a different London postcode.')
        else:
            # Turns 422/500 into an exception the except blocks below catch
            response.raise_for_status()
            result = response.json()
            st.metric('Predicted Price', f'£{result["predicted_price"]:,.0f}')
            # The API returns the district, so we don't re-derive it here
            st.caption(f'{floor_area} Square Meters {ptype.lower()} in {result["district"]}, built in {built}')

    except requests.exceptions.Timeout:
        st.error('The API took too long to respond. It may be waking up, please try again.')
    except requests.exceptions.RequestException as e:
        # Parent class, covers connection errors, HTTP errors, DNS failures
        st.error(f'Could not reach the prediction API: {e}')