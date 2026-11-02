import requests
import streamlit as st

api_url = 'https://london-property-api.onrender.com'

# Set once here for every page: the browser tab title and the wide layout
st.set_page_config(page_title='London Property Intelligence', layout='wide')

# The sidebar menu: each page is its own script, run when chosen. The chat opens first.
page = st.navigation([
    st.Page('chat_app.py', title='Area chat', icon=':material/chat:', default=True),
    st.Page('map_app.py', title='Price map', icon=':material/map:'),
    st.Page('app.py', title='Price estimate', icon=':material/calculate:'),
])

# The free API sleeps after 15 minutes idle, so wake it once per visit, whichever page is opened first
if 'awake' not in st.session_state:
    with st.spinner('Waking up the API (the free server can take up to a minute)...'):
        try:
            requests.get(f'{api_url}/health', timeout=120)
        except requests.exceptions.RequestException:
            pass
    st.session_state.awake = True

page.run()