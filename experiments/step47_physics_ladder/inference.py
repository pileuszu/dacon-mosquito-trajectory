import sys
import os
import argparse
import json
import traceback
from pathlib import Path
import numpy as np
import torch

sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification
import step47_physics_ladder.train_tcn_gru_candidate_selector as base
import step47_physics_ladder.boundary_tiny_correction as boundary

def main():
    parser = argparse.ArgumentParser(description="Step 47 custom inference script.")
    parser.add_argument("--root", type=Path, default=Path("step47_physics_ladder/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/step47_physics_ladder"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20260506)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        msg = "🚀 Started: [Step 47 inference.py] Generating final corrected test predictions..."
        send_discord_notification(None, msg)
        print(msg)

        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)

        base.set_torch_seed(args.seed)

        # 1. Load compiled data
        print("Loading pre-compiled test dataset...")
        train_x = np.load(args.root / "train_x.npy")
        train_y = np.load(args.root / "train_y.npy")
        test_x = np.load(args.root / "test_x.npy")

        with open(args.root / "test_ids.json", "r") as f:
            test_ids = json.load(f)

        # 2. Check and load selector score bank & boundary model
        selector_out = args.out_dir / "selector"
        boundary_out = args.out_dir / "boundary"
        
        test_score_bank = selector_out / "test_selector_scores.npz"
        boundary_model_path = boundary_out / "fold_0" / "boundary_model_full.pt"
        report_path = boundary_out / "fold_0" / "boundary_tiny_correction_report.json"

        if not test_score_bank.exists() or not boundary_model_path.exists():
            raise FileNotFoundError(
                f"Missing selector scores or boundary model.\n"
                f"Expected selector test scores at: {test_score_bank}\n"
                f"Expected boundary model at: {boundary_model_path}\n"
                f"Please ensure train_automl.py is run successfully first."
            )

        print("Loading selector test scores...")
        tz = np.load(test_score_bank, allow_pickle=True)
        test_scores = tz["ens_scores"].astype(np.float32)

        # 3. Rebuild normalize parameters from all training rows
        print("Fitting normalizer stats on all training candidates...")
        all_final_cf3, _, _, all_train_cands, _, _ = boundary.make_rows(
            train_x,
            train_y,
            train_x.shape[1] - 1,
            2,
            cap=0.006, # default cap
            low=0.007,
            high=0.017,
            far_weight=0.04
        )
        _, _, all_cm, all_cs = base.normalize_fit(
            np.zeros((1, 6, len(base.SEQ_FEATURE_NAMES)), dtype=np.float32),
            all_final_cf3
        )

        # 4. Prepare test candidates and features
        print("Generating test candidates and local frames...")
        test_cands = base.make_candidates(test_x, test_x.shape[1] - 1, horizon=2)
        test_cf3 = base.make_candidate_features(test_x, test_x.shape[1] - 1, test_cands, horizon=2)
        test_cf3_norm = ((test_cf3 - all_cm) / all_cs).astype(np.float32)
        tt, tn, tb, tspeed = boundary.local_frame(test_x, test_x.shape[1] - 1)
        test_scale = np.maximum(tspeed * 2.0, base.EPS)

        # 5. Initialize correction MLP and load weights
        print("Loading TinyCorrectionNet...")
        model = boundary.TinyCorrectionNet(test_cf3_norm.shape[-1], 96).to(device)
        model.load_state_dict(torch.load(boundary_model_path, map_location=device))
        model.eval()

        # 6. Predict delta and correct coordinates
        print("Predicting coordinate corrections...")
        flat_test = test_cf3_norm.reshape(-1, test_cf3_norm.shape[-1])
        
        # We dummy pass the args namespace to use default batch size
        class DummyArgs:
            batch = 4096
            cap = 0.006
            apply_scale = 0.75
        dummy_args = DummyArgs()

        delta = boundary.predict_delta(model, flat_test, dummy_args, device).reshape(test_cands.shape[0], test_cands.shape[1], 3)
        delta_vec = boundary.cap_vectors(boundary.local_to_vector(delta, (tt, tn, tb), test_scale), dummy_args.cap)
        corrected = test_cands + dummy_args.apply_scale * delta_vec

        # Get optimal temperature from fold 0 report if available
        temp = 0.03
        if report_path.exists():
            with open(report_path, "r") as f:
                rep = json.load(f)
                temp = float(rep["soft"].get("temperature", 0.03))
                print(f"Loaded optimal soft temperature from validation report: {temp}")

        # 7. Select final coordinates and write submission
        pred_soft = base.soft_select(corrected, test_scores, temp)
        pred_argmax = corrected[np.arange(len(corrected)), np.argmax(test_scores, axis=1)]

        soft_sub_path = args.out_dir / "submission_soft.csv"
        argmax_sub_path = args.out_dir / "submission_argmax.csv"
        
        base.write_submission(soft_sub_path, test_ids, pred_soft)
        base.write_submission(argmax_sub_path, test_ids, pred_argmax)

        # Compute average displacement
        p0 = test_x[:, -1]
        disp_soft_cm = np.linalg.norm(pred_soft - p0, axis=1).mean() * 100
        disp_argmax_cm = np.linalg.norm(pred_argmax - p0, axis=1).mean() * 100

        success_msg = (
            f"✅ Finished: [Step 47 inference.py] Predictions successfully generated!\n"
            f"Soft selection submission (temp={temp}): `{soft_sub_path}` (Avg Disp: {disp_soft_cm:.4f} cm)\n"
            f"Argmax selection submission: `{argmax_sub_path}` (Avg Disp: {disp_argmax_cm:.4f} cm)"
        )
        send_discord_notification(None, success_msg)
        print(success_msg)

    except BaseException as e:
        error_msg = f"❌ Failed: [Step 47 inference.py] Inference ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(None, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    main()
