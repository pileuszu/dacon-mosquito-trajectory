import sys
import os
import argparse
import time
import traceback
from pathlib import Path

sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification
import step47_physics_ladder.prepare_data as prepare_data
import step47_physics_ladder.train_tcn_gru_candidate_selector as train_selector
import step47_physics_ladder.boundary_tiny_correction as train_boundary

def call_main(main_func, argv):
    old_argv = sys.argv[:]
    try:
        sys.argv = [main_func.__name__] + list(map(str, argv))
        start = time.time()
        main_func()
        print(f"[DONE] {main_func.__name__} elapsed = {round(time.time() - start, 1)}s")
    finally:
        sys.argv = old_argv

def main():
    parser = argparse.ArgumentParser(description="Master training pipeline for Step 47 Physics Ladder.")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for validation")
    parser.add_argument("--fold-limit", type=int, default=5, help="Limit fold training to this number")
    parser.add_argument("--pre-epochs", type=int, default=14, help="Pre-epochs for Selector")
    parser.add_argument("--fine-epochs", type=int, default=10, help="Fine-epochs for Selector")
    parser.add_argument("--epochs", type=int, default=50, help="Pretrain epochs for Boundary Correction Net")
    parser.add_argument("--fine-epochs-boundary", type=int, default=20, help="Finetune epochs for Boundary Correction Net")
    parser.add_argument("--skip-full", action="store_true", help="Skip full dataset training and test inference")
    parser.add_argument("--device", type=str, default="auto", help="Device to train on (auto/cuda/mps/cpu)")
    parser.add_argument("--seed", type=int, default=20260506, help="Random seed")
    args = parser.parse_args()

    try:
        msg = "🚀 Started: [Step 47 train_automl.py] Training custom PyTorch Physics Ladder selector + boundary correction pipeline..."
        send_discord_notification(None, msg)
        print(msg)

        # 1. Precompile data if not already done
        data_dir = Path("step47_physics_ladder/data")
        if not (data_dir / "train_x.npy").exists():
            print("Pre-compiled dataset not found. Generating...")
            prepare_data.main()
        else:
            print("Found pre-compiled dataset. Skipping prepare_data.")

        # 2. Train Selector
        selector_out = Path("outputs/step47_physics_ladder/selector")
        score_bank = selector_out / "oof_selector_scores.npz"
        
        if score_bank.exists() and (selector_out / "test_selector_scores.npz").exists():
            print("\nFound existing selector scores. Skipping Phase 1: Selector Training.")
        else:
            print("\n--- Phase 1: Training Attn-GRU Candidate Selector ---")
            selector_out.mkdir(parents=True, exist_ok=True)
            
            selector_args = [
                "--root", "step47_physics_ladder/data",
                "--out-dir", selector_out,
                "--models", "attn_gru",
                "--folds", args.folds,
                "--fold-limit", args.fold_limit,
                "--pre-epochs", args.pre_epochs,
                "--fine-epochs", args.fine_epochs,
                "--device", args.device,
                "--seed", args.seed,
                "--pairwise-loss-weight", "0.25",
                "--pairwise-margin", "0.12",
                "--pairwise-min-label-gap", "0.04",
                "--fine-distill-weight", "0.55",
                "--fine-distill-temp", "0.07",
                "--reverse-pretrain",
                "--norm-real-only"
            ]
            if args.skip_full:
                selector_args.append("--skip-full")
                
            call_main(train_selector.main, selector_args)

        # 3. Train Boundary Correction MLP for each fold
        print("\n--- Phase 2: Training Boundary Tiny Correction MLP ---")
        boundary_out = Path("outputs/step47_physics_ladder/boundary")
        boundary_out.mkdir(parents=True, exist_ok=True)
        
        score_bank = selector_out / "oof_selector_scores.npz"
        
        # We loop over folds trained by Selector
        num_folds_trained = min(args.folds, args.fold_limit)
        for f in range(num_folds_trained):
            print(f"\n--- Training Boundary Correction Net for Fold {f+1}/{num_folds_trained} ---")
            boundary_args = [
                "--root", "step47_physics_ladder/data",
                "--out-dir", boundary_out / f"fold_{f}",
                "--fold", f,
                "--folds", args.folds,
                "--score-bank", score_bank,
                "--epochs", args.epochs,
                "--fine-epochs", args.fine_epochs_boundary,
                "--device", args.device,
                "--seed", args.seed,
                "--save-val-pred"
            ]
            if f == 0 and not args.skip_full:
                boundary_args.append("--make-test")
                if (selector_out / "test_selector_scores.npz").exists():
                    boundary_args.extend(["--test-score-bank", selector_out / "test_selector_scores.npz"])
                    
            call_main(train_boundary.main, boundary_args)

        success_msg = "✅ Finished: [Step 47 train_automl.py] Training completed successfully!"
        send_discord_notification(None, success_msg)
        print(success_msg)

    except BaseException as e:
        error_msg = f"❌ Failed: [Step 47 train_automl.py] ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(None, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    main()
