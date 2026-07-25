import streamlit as st
import pandas as pd
import folium
import requests
from streamlit_folium import st_folium
from pathlib import Path

api_url = 'https://london-property-api.onrender.com'

property_types = ['Flat', 'Terraced', 'Semi-Detached', 'Detached']
tenures = ['Freehold', 'Leasehold']

st.set_page_config(page_title='London Property Intelligence', layout='wide')


# Loaded locally because the map must turn a click's lat/lon into a postcode
@st.cache_data
def load_postcodes():
    root = Path(__file__).resolve().parent.parent
    return pd.read_parquet(root/'data'/'supplementary'/'postcodes_london.parquet')

def nearest_postcode(postcodes, lat, lon):
    # Squared distance to every postcode, vectorised. No square root needed —
    # the closest point is the same either way, and skipping it is faster
    d = (postcodes['lat'] - lat) ** 2 + (postcodes['lon'] - lon) ** 2
    return postcodes.loc[d.idxmin()]

def get_prediction(postcode, floor_area, property_type, tenure, built_year):
    # Field names must match the API's PropertyRequest model exactly
    payload = {
        'postcode': postcode,
        'floor_area': floor_area,
        'property_type': property_type,
        'tenure': tenure,
        'built_year': built_year,
    }
    # timeout=120 covers a cold start on the free tier
    response = requests.post(f'{api_url}/predict-price', json=payload, timeout=120)
    response.raise_for_status()   # 404/422/500 become exceptions
    return response.json()


postcodes = load_postcodes()

st.title('London House Price Map')
st.caption("Click anywhere in London to estimate a property's value.")

# Property details apply to whatever point the user clicks
st.sidebar.header('Property')
floor_area = st.sidebar.number_input('Floor Area (Square Meters)', 15, 500, 70)
ptype = st.sidebar.selectbox('Type', property_types)
tenure = st.sidebar.radio('Tenure', tenures)
built = st.sidebar.slider('Construction Year', 1900, 2024, 1960)

# Base map centred on London
london_map = folium.Map(
    location=[51.5074, -0.1278],
    zoom_start=11,
    tiles='cartodbvoyager',
    min_zoom=10,
    max_zoom=15,
)

# The map is rebuilt from scratch on every rerun, so the marker must be redrawn from session state
if 'clicked' in st.session_state:
    lat, lon = st.session_state.clicked
    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color='Red',
        fill=True,
        fill_color='Red',
        fill_opacity=0.6,
        weight=2,
    ).add_to(london_map)

# Render the map and capture interaction.
map_data = st_folium(london_map, height=500, width=None,
                     returned_objects=['last_clicked'])

# A click causes a rerun store the coordinates so they persist.
if map_data and map_data.get('last_clicked'):
    st.session_state.clicked = (
        map_data['last_clicked']['lat'],
        map_data['last_clicked']['lng'],
    )

# If we have a location, snap it to the nearest postcode and ask the API
if 'clicked' in st.session_state:
    lat, lon = st.session_state.clicked
    row = nearest_postcode(postcodes, lat, lon)

    with st.spinner('Predicting... (the free-tier API may take up to a minute to wake up)'):
        try:
            result = get_prediction(row['postcode'], floor_area, ptype, tenure, built)
            col1, col2 = st.columns(2)
            col1.metric('Predicted Price', f'£{result["predicted_price"]:,.0f}')
            col2.metric('Nearest Postcode', result['postcode'])
        except requests.exceptions.Timeout:
            st.error('The API took too long to respond. It may be waking up - try again.')
        except requests.exceptions.RequestException as e:
            st.error(f'Could not reach the prediction API: {e}')
else:
    st.info('Click a point on the map to see a prediction.')