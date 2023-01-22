st.set_page_config(page_title='NUHARIPS Dashboard', page_icon=None, layout="centered", initial_sidebar_state="expanded", menu_items=None)

import streamlit as st
import streamlit_authenticator as stauth
import yaml
import os

global USERNAME

with open('/app/nuharips/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, authentication_status, USERNAME = authenticator.login('Login', 'main')

if authentication_status == True:
    authenticator.logout('Logout', 'sidebar')
    with st.sidebar:
        st.success('Login successful.', icon = "✅")
    exec(open('app.py').read())
    
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')