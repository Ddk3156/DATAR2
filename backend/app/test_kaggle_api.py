import os

from backend.app.config import settings


os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
os.environ["KAGGLE_KEY"] = settings.kaggle_key

from kaggle.api.kaggle_api_extended import KaggleApi


api = KaggleApi()
api.authenticate()

print("✓ Kaggle authentication successful")