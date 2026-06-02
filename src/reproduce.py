import subprocess
import argparse
import sys
import traceback
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.notifier import send_discord_notification

def run_script(script_name, args=[]):
    script_path = Path(__file__).parent / script_name
    print(f"\n==================================================")
    print(f"🚀 Running {script_name}...")
    print(f"==================================================")
    
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print(f"❌ Error: {script_name} failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Reproduce the best-scoring model submission from scratch.")
    parser.add_argument("--fast", action="store_true", help="Run in fast dry-run / mock mode for quick verification")
    args = parser.parse_args()
    
    mode_str = " (FAST MODE)" if args.fast else ""
    
    # 1. Send Started notification
    send_discord_notification(None, f"🚀 Started: Mosquito Trajectory Prediction Reproduction Pipeline{mode_str}")
    
    try:
        print("🦟 Mosquito Trajectory Prediction: Reproduction Pipeline 🦟")
        
        # 1. Preprocess data
        run_script("data_preprocessing.py")
        
        # 2. Train models
        train_args = ["--fast"] if args.fast else []
        run_script("train.py", train_args)
        
        # 3. Optimize Powell Weights
        run_script("powell_optimization.py")
        
        # 4. Generate final submission
        run_script("inference.py")
        
        print("\n✅ Reproduction Pipeline completed successfully!")
        print("Final submission coordinates saved to root submission.csv.")
        
        # 2. Send Finished notification
        send_discord_notification(None, f"✅ Finished: Mosquito Trajectory Prediction Reproduction Pipeline{mode_str} completed successfully!")
        
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"❌ Failed: Mosquito Trajectory Prediction Reproduction Pipeline{mode_str} failed with error:\n```\n{tb}\n```"
        print(error_msg)
        send_discord_notification(None, error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
