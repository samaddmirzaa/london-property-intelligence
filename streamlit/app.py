import streamlit as st
from london_property.model import (
    property_types, tenures,
    load_bundle, load_postcodes, find_postcode, predict_price,
)


# Streamlit reruns this whole script on every interaction, so heavy loads must be cached.
@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_postcodes():
    return load_postcodes()


bundle = get_bundle()
postcodes = get_postcodes()

st.title('London House Price Predictor')

# Inputs live in the sidebar, results in the main area
st.sidebar.header('Property')
postcode = st.sidebar.text_input('Postcode', 'SW1A 1AA')
floor_area = st.sidebar.number_input('Floor Area (Square Meters)', 15, 500, 70)
ptype = st.sidebar.selectbox('Type', property_types)
tenure = st.sidebar.radio('Tenure', tenures)
built = st.sidebar.slider('Construction Year', 1900, 2024, 1960)

# Look up the postcode's geography (lat/lon/zone/income/etc.)
row = find_postcode(postcodes, postcode)
if row is None:
    st.warning('Postcode not found. Try a full London postcode like SW1A 1AA.')
    st.stop()

price = predict_price(bundle, row, floor_area, ptype, tenure, built)

st.metric('Predicted Price', f'£{price:,.0f}')
st.caption(f'{floor_area} Square Meters {ptype.lower()} in {row["postcode"].split(" ")[0]}, built in {built}')