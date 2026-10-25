from pathlib import Path
import requests

MODEL_URL = "https://huggingface.co/tejas-maharana-ai/bcg-customer-churn-model/resolve/main/churn_model.joblib"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_model.joblib"

def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        print(f"Model already exists: {MODEL_PATH}")
        return

    print("Downloading trained model from Hugging Face...")

    with requests.get(MODEL_URL, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as model_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    model_file.write(chunk)

    print(f"Model downloaded successfully: {MODEL_PATH}")
    print(f"Model size: {MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
