import os
import sys
import json
import csv
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification

def main():
    send_discord_notification(None, "🚀 Started: [Step 66 Revised inference.py] Generating final submission coordinates with 5-Fold soft ensemble...")
    
    try:
        data_dir = Path("step66_super_feature/data")
        models_dir = Path("step66_super_feature/models")
        out_dir = Path("outputs/step66_super_feature")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading test tabular dataset...")
        test_df = pd.read_parquet(data_dir / "test_tabular.parquet")
        print(f"  Loaded dataset shape: {test_df.shape}")
        
        test_data = test_df.drop(columns=['sample_idx', 'cand_idx'])
        
        folds = 5
        pred_dists_list = []
        
        # Predict using all 5 fold models
        print("Predicting distance error using 5-Fold models...")
        for fold in range(folds):
            fold_model_path = models_dir / f"autogluon_fold_{fold}"
            print(f"  Loading predictor for Fold {fold+1} from {fold_model_path}...")
            predictor = TabularPredictor.load(fold_model_path)
            
            print(f"  Predicting with Fold {fold+1} model...")
            fold_preds = predictor.predict(test_data)
            pred_dists_list.append(fold_preds)
            
        # Average the predicted distances across folds (soft ensemble)
        print("Averaging predicted distances...")
        avg_pred_dists = np.mean(pred_dists_list, axis=0)
        test_df['pred_dist'] = avg_pred_dists
        
        # Load test targets reference
        test_candidates = np.load(data_dir / "test_candidates.npy")
        with open(data_dir / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        # Group by sample_idx to select the candidate with minimum predicted distance error
        print("Selecting coordinates with minimum predicted distance per sample...")
        best_cand_indices = test_df.groupby('sample_idx')['pred_dist'].idxmin()
        best_rows = test_df.loc[best_cand_indices]
        
        submission_coords = np.zeros((len(test_ids), 3))
        
        for _, row in best_rows.iterrows():
            s_idx = int(row['sample_idx'])
            c_idx = int(row['cand_idx'])
            submission_coords[s_idx] = test_candidates[s_idx, c_idx]
            
        # Build final submission CSV file
        sub_path = out_dir / "submission.csv"
        print(f"Writing final submission to {sub_path}...")
        with sub_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, coord in zip(test_ids, submission_coords):
                writer.writerow([sample_id, f"{coord[0]:.9f}", f"{coord[1]:.9f}", f"{coord[2]:.9f}"])
                
        print("\n=================== Inference Summary ===================")
        print(f"Submission generated successfully at: {sub_path}")
        
        # Basic sanity checks on the generated submission
        df_sub = pd.read_csv(sub_path)
        print(f"  Shape: {df_sub.shape}")
        print(f"  NaN count: {df_sub.isna().sum().to_dict()}")
        p_last_te = np.load(data_dir / "test_x.npy")[:, -1]
        diffs = np.linalg.norm(submission_coords - p_last_te, axis=1)
        disp_stats = f"Displacement stats (cm) -> Mean: {diffs.mean()*100:.3f}, Max: {diffs.max()*100:.3f}, Std: {diffs.std()*100:.3f}"
        print(f"  {disp_stats}")
        print("=========================================================")
        
        success_msg = f"✅ Finished: [Step 66 Revised] Inference completed successfully. Mean displacement: {diffs.mean()*100:.3f}cm, Max: {diffs.max()*100:.3f}cm"
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 66 Revised] Inference failed.\nError: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
