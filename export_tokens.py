"""One-off helper: log in locally (handles MFA prompt) and print a token string for Streamlit secrets."""
import getpass
from garminconnect import Garmin

email = input("Garmin email: ")
pw = getpass.getpass("Password: ")
g = Garmin(email, pw, prompt_mfa=lambda: input("MFA code: "))
g.login()
print("\nPaste this into secrets.toml as garmin.tokens:\n")
print(g.garth.dumps())
