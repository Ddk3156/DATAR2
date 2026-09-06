from backend.app.config import settings


print("Kaggle username:", settings.kaggle_username)
print("Kaggle key loaded:", bool(settings.kaggle_key))