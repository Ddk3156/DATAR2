from huggingface_hub import whoami

from backend.app.config import settings


info = whoami(token=settings.hf_token)

print("✓ Hugging Face authentication successful")
print("Username:", info["name"])
