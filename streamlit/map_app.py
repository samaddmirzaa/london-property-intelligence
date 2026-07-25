import streamlit as st
import folium
from streamlit_folium import st_folium
from london_property.model import (
    property_types, tenures,
    load_bundle, load_postcodes, nearest_postcode, predict_price,
)

st.set_page_config(page_title='London Property Intelligence', layout='wide')


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_postcodes():
    return load_postcodes()


bundle = get_bundle()
postcodes = get_postcodes()

st.title('London House Price Map')
st.caption('Click anywhere on the London map to estimate the value of the property.')

# Property details apply to whatever point the user clicks
st.sidebar.header('Property')
floor_area = st.sidebar.number_input('Floor Area (Square Meters)', 15, 500, 70)
ptype = st.sidebar.selectbox('Type', property_types)
tenure = st.sidebar.radio('Tenure', tenures)
built = st.sidebar.slider('Construction Year', 1900, 2024, 1960)

# Base Map centered on London
london_map = folium.Map(
    location=[51.5074, -0.1278],
    zoom_start=11,
    tiles='cartodbvoyager',
    min_zoom=10,
    max_zoom=15,
)

if 'clicked' in st.session_state:
    lat, lon = st.session_state.clicked
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color='Red',
        fill=True,
        fill_color='Red',
        fill_opacity=0.6,
        weight=2,
    ).add_to(london_map)

map_data = st_folium(london_map, height=500, width=None,
                     returned_objects=['last_clicked'])

if map_data and map_data.get('last_clicked'):
    st.session_state.clicked = (
        map_data['last_clicked']['lat'],
        map_data['last_clicked']['lng'],
    )

if 'clicked' in st.session_state:
    lat, lon = st.session_state.clicked
    row = nearest_postcode(postcodes, lat, lon)
    price = predict_price(bundle, row, floor_area, ptype, tenure, built)

    col1, col2 = st.columns(2)
    col1.metric('Predicted Price', f'£{price:,.0f}')
    col2.metric('Nearest Postcode', row['postcode'])
else:
    st.info('Click a point on the map to see a prediction.')