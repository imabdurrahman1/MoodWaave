import os
from huggingface_hub import HfApi, login, create_repo

def upload_models():
    # 1. Configuration - UPDATE YOUR USERNAME HERE
    USERNAME = "imabdurrahman1"
    REPO_NAME = "MoodWaave-models"
    REPO_ID = f"{USERNAME}/{REPO_NAME}"
    
    # 2. Files to upload from the models/ directory
    model_files = [
        "genre_classifier.pkl",
        "genre_scaler.pkl",
        "genre_encoder.pkl",
        "mood_regressor.pkl",
        "mood_scaler.pkl"
    ]
    
    # 3. Authentication
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN environment variable not set.")
        return
    
    login(token=token)
    api = HfApi()
    
    # 4. Create Repo if it doesn't exist
    print(f"🚀 Checking repository: {REPO_ID}...")
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"⚠️ Note: {e}")

    # 5. Upload files
    print("📤 Starting upload...")
    for file_name in model_files:
        local_path = os.path.join("models", file_name)
        
        if os.path.exists(local_path):
            print(f"   Uploading {file_name}...")
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=file_name,
                    repo_id=REPO_ID,
                    repo_type="model"
                )
                print(f"   ✅ {file_name} uploaded successfully.")
            except Exception as e:
                print(f"   ❌ Failed to upload {file_name}: {e}")
        else:
            print(f"   ⚠️ Warning: {local_path} not found. Skipping.")

    print(f"\n✨ ALL DONE! Your models are live at:")
    print(f"🔗 https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    upload_models()
