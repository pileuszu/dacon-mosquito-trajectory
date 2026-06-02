import subprocess
import sys
import os

sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification

URL = None

def run_command(cmd, name):
    try:
        print(f"\n==================================================")
        print(f">>> Running {name}: {' '.join(cmd)}")
        print(f"==================================================")
        subprocess.run(cmd, check=True)
        print(f"\n>>> {name} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ [Step 39 Pipeline] {name} Failed with exit code {e.returncode}."
        send_discord_notification(URL, error_msg)
        print(error_msg)
        return False

def main():
    send_discord_notification(URL, "🚀 [Step 39 Pipeline] Starting 6-Regime GMM Train & Inference Pipeline...")
    
    # 1. Train the models
    if not run_command(["venv\\Scripts\\python.exe", "step39_six_regime/train_ranker.py"], "Train Rankers"):
        return
        
    # 2. Run inference
    if not run_command(["venv\\Scripts\\python.exe", "step39_six_regime/inference.py"], "Inference"):
        return
        
    send_discord_notification(URL, "🎉 [Step 39 Pipeline] Whole GMM 6-Regime Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()
