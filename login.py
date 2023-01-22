import streamlit as st
st.set_page_config(page_title='NUHARIPS Dashboard', page_icon='https://scontent.fcai20-4.fna.fbcdn.net/v/t39.30808-6/311597083_485166766973820_8229057116837645148_n.png?_nc_cat=108&ccb=1-7&_nc_sid=09cbfe&_nc_ohc=5H1wckmc7k0AX-osCZl&_nc_ht=scontent.fcai20-4.fna&oh=00_AfCp9wwFXxqaDO7PL0PrSRlCM7a5gbPxvlzL8UhEww7xXQ&oe=63D23719', layout="centered", initial_sidebar_state="expanded", menu_items=None)

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