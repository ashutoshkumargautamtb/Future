import os
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


def get_creds(user_email):
    SCOPES = ['https://mail.google.com/']
    creds = None
    token_file = f'token_{user_email}.json'
    client_secret_file = f'{user_email}.json'
    if os.path.exists(token_file):
        print(f"Token for {user_email} already exists. Skipping...")
        return
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    print(f"Token generated and saved to {token_file}")


if __name__ == "__main__":
    email = "lavonnajefferson3@gmail.com"
    get_creds(email)
