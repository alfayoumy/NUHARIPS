import asyncio
import json
import os
import streamlit as st
from httpx_oauth.clients.google import GoogleOAuth2

st.title("Google OAuth2 flow")

"## Configuration"
    
async def get_token():
    client_id = st.secrets['client_id']
    client_secret = st.secrets['client_secret']
    redirect_uri = st.secrets['redirect_uri']

    client = GoogleOAuth2(client_id, client_secret)
        
    "## Authorization URL"

    def write_authorization_url():
        authorization_url = client.get_authorization_url(
            redirect_uri,
            scope=['https://www.googleapis.com/auth/gmail.send'],
            extras_params={"access_type": "offline"},
        )
        st.write(authorization_url)

    write_authorization_url()

    "## Callback"

    code = st.text_input("Authorization code")


    "## Access token"

    async def write_access_token(code):
        token = await client.get_access_token(code, redirect_uri)
        st.write(token)
        token['client_id'] = st.secrets['client_id']
        token['client_secret'] = st.secrets['client_secret']
        token = json.dumps(token)

        with open('token.json', 'w') as t:
            t.write(token)

    if code:
        asyncio.run(write_access_token(code))
    
    else:
        "Waiting authorization code..."

asyncio.run(get_token())