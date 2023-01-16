import streamlit as st
import streamlit_authenticator as stauth
import yaml
import os

with open('/app/nuharips/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status == True:
    authenticator.logout('Logout', 'main')
    st.success('Login successful.', icon = "✅")
    import app
    app.USERNAME = username
    app()
    
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')