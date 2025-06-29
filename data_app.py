#This python file runs collect_data.py , prepare_data.py and train_model.py in sequence to easily add a gesture to the model with a single click
from subprocess import check_call, CalledProcessError
import sys

def run_data_app():
    try:
        print("[INFO] Starting data collection...")
        check_call([sys.executable, 'src/collect_data.py'])

        print("[INFO] Preparing dataset...")
        check_call([sys.executable, 'src/prepare_dataset.py'])

        print("[INFO] Training model...")
        check_call([sys.executable, 'src/train_model.py'])

        print("[INFO] All tasks completed successfully! ✅")

    except CalledProcessError as e:
        print(f"[ERROR] Script failed with exit code {e.returncode}. Stopping pipeline.")
        exit(1)

run_data_app()

