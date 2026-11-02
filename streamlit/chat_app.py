import json
from pathlib import Path

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

api_url = 'https://london-property-api.onrender.com'
walk_m = 800    # a 10-minute walk, the same distance the chatbot uses

# Places pinned on the map: label -> (place type, subtype filter, colour that works in both folium and Streamlit)
pins = {'Primary school': ('school', 'primary', 'blue'), 'GP': ('gp', None, 'red'),
        'Supermarket': ('supermarket', None, 'orange'), 'Park': ('park', 'park', 'green')}

examples = ["What's it like living around E15 4EJ?", 'What is ground rent and do I have to pay it?',
            'How much stamp duty on £450,000 as a first-time buyer?']


# Loaded locally to place the map, cached so the parquet loads once
@st.cache_data
def load_postcodes():
    root = Path(__file__).resolve().parent.parent
    return pd.read_parquet(root/'data'/'supplementary'/'postcodes_london.parquet',
                           columns=['postcode', 'lat', 'lon']).set_index('postcode')


def normalise(postcode):
    # Same rule as the API: the last three characters are always the inward code
    compact = postcode.upper().replace(' ', '')
    return f'{compact[:-3]} {compact[-3:]}'


def ask(history):
    """Send the conversation to the API's /chat endpoint. Returns (reply, None) or (None, a message for the user)."""
    try:
        response = requests.post(f'{api_url}/chat', json={'messages': history}, timeout=120)
    except requests.exceptions.RequestException:
        return None, 'Could not reach the API. It may be waking up, so please try again in a minute.'
    if response.ok:
        return response.json(), None
    if response.status_code in (429, 503):
        return None, response.json()['detail']
    return None, f'Something went wrong (error {response.status_code}). Please try again.'


def mapped_postcode(tools_used):
    """The postcode in the last tool call that looked one up, or None if the answer used no postcode."""
    for call in reversed(tools_used):
        try:
            postcode = json.loads(call['arguments']).get('postcode')
        except ValueError:
            continue
        if postcode:
            return normalise(postcode)
    return None


# Cached for an hour: the same postcode's map doesn't call the API again
@st.cache_data(ttl=3600)
def nearby(postcode, place_type, subtype):
    params = {'type': place_type, 'radius_m': walk_m, 'limit': 10, 'subtype': subtype}
    response = requests.get(f'{api_url}/nearby/{postcode}', params=params, timeout=120)
    response.raise_for_status()
    return response.json()['nearest']


def walk_map(postcode, lat, lon):
    """A map of the 10-minute walk around a postcode, with the nearest everyday places pinned."""
    carto_key = st.secrets['CARTO_KEY']
    m = folium.Map(
        location=[lat, lon],
        zoom_start=15,
        tiles=f'https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}.png?key={carto_key}',
        attr='&copy; OpenStreetMap contributors &copy; CARTO',
        subdomains='abcd',
    )
    folium.Circle([lat, lon], radius=walk_m, color='gray', fill=True, fill_opacity=0.05,
                  tooltip='10-minute walk (800 m)').add_to(m)
    folium.Marker([lat, lon], tooltip=postcode, icon=folium.Icon(color='black', icon='home')).add_to(m)
    for label, (place_type, subtype, colour) in pins.items():
        for place in nearby(postcode, place_type, subtype):
            folium.CircleMarker([place['lat'], place['lon']], radius=6, color=colour, fill=True, fill_opacity=0.8,
                                tooltip=f"{label}: {place['name']} ({place['distance_m']} m)").add_to(m)
    return m


postcodes = load_postcodes()

# The conversation lives in session state, because Streamlit reruns this script on every interaction
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title('London Area Chat')
st.caption('Ask about any London postcode, or about how buying a home works. '
           'Answers come from official data and GOV.UK guides, not from the AI model\'s memory.')

chat_col, map_col = st.columns([3, 2])

with chat_col:
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if message.get('tools'):
                st.caption('Data used: ' + ', '.join(sorted({t['tool'] for t in message['tools']})))

    clicked = None
    if not st.session_state.messages:
        st.write('Try one of these:')
        for example in examples:
            if st.button(example):
                clicked = example

with map_col:
    postcode = st.session_state.get('postcode')
    if postcode in postcodes.index:
        lat, lon = postcodes.loc[postcode, ['lat', 'lon']]
        try:
            st_folium(walk_map(postcode, lat, lon), height=480, width=None, returned_objects=[])
            st.caption(' · '.join(f':{colour}[●] {label}' for label, (_, _, colour) in pins.items())
                       + ' · circle = 10-minute walk')
        except requests.exceptions.RequestException:
            st.info('The map could not load nearby places right now.')
    else:
        st.info('Ask about a postcode and a map of its 10-minute walk will appear here.')

prompt = st.chat_input('Ask about a London postcode or about buying a home', max_chars=2000) or clicked
if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    history = [{'role': m['role'], 'content': m['content']} for m in st.session_state.messages[-10:]]
    with chat_col:
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.spinner('Thinking... (up to a minute on the free tier)'):
            reply, error = ask(history)
    if error:
        st.session_state.messages.pop()
        chat_col.error(error)
    else:
        st.session_state.messages.append({'role': 'assistant', 'content': reply['answer'],
                                          'tools': reply['tools_used']})
        st.session_state.postcode = mapped_postcode(reply['tools_used']) or postcode
        st.rerun()