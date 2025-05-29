
import os
import glob
import shutil

def find_best_model():
    models = glob.glob("model_v*.pkl")
    if not models:
        print("Model versiyonu bulunamadı.")
        return

    latest = max(models, key=os.path.getctime)
    shutil.copy(latest, "model.pkl")
    print(f"En iyi model yüklendi: {latest}")

if __name__ == "__main__":
    find_best_model()
