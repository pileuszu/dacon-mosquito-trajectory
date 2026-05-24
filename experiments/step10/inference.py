import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from step10.physics import make_candidates, CANDIDATES, EPS
from step10.prepare_data import extract_context_features
from step10.config import *

def inference():
    # 1. Load Predictor & Step 7 Features
    save_path = 'step10/models/ranker_v2'
    predictor = TabularPredictor.load(save_path)
    print(f"Loaded AutoML Ranker from {save_path}")
    
    s7_preds_df = pd.read_csv("step10/step7_preds_test.csv").set_index('id')
    print("Loaded Step 7 Test predictions.")
    
    # 2. Prepare Test Data
    test_files = sorted(list(TEST_DIR.glob("*.csv")))
    results = []
    
    # Batch processing for memory stability
    batch_size = 100
    for i in tqdm(range(0, len(test_files), batch_size), desc="Test Batch"):
        batch_files = test_files[i:i+batch_size]
        batch_rows = []
        batch_info = [] # (fid, candidates)
        
        for fpath in batch_files:
            fid = fpath.stem
            df = pd.read_csv(fpath)
            xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
            
            s7_pos = s7_preds_df.loc[fid].to_numpy()
            ctx = extract_context_features(xyz)
            cands = make_candidates(xyz, end_idx=-1, horizon=2)
            dists_to_s7 = np.linalg.norm(cands - s7_pos, axis=1)
            
            batch_info.append((fid, cands))
            
            for idx in range(len(CANDIDATES)):
                spec = CANDIDATES[idx]
                row = {
                    "fid": fid,
                    "cand_idx": idx,
                    "spec_d1": spec.d1,
                    "spec_par": spec.par,
                    "spec_perp": spec.perp,
                    "spec_ts": spec.ts,
                    "dist_to_s7": dists_to_s7[idx],
                    **ctx,
                    "dist_to_p0": np.linalg.norm(cands[idx] - xyz[-1])
                }
                batch_rows.append(row)
        
        # 3. Predict Probabilities
        batch_df = pd.DataFrame(batch_rows)
        pred_data = batch_df.drop(columns=['fid'])
        
        probs = predictor.predict_proba(pred_data)
        score_col = 1 if 1 in probs.columns else probs.columns[0]
        batch_df['score'] = probs[score_col].values
        
        # 4. Select best candidate for each sample
        for fid, cands in batch_info:
            sample_scores = batch_df[batch_df['fid'] == fid]
            best_idx = sample_scores['score'].idxmax()
            best_cand_idx = sample_scores.loc[best_idx, 'cand_idx']
            
            final_pos = cands[int(best_cand_idx)]
            results.append({
                "id": fid,
                "x": final_pos[0],
                "y": final_pos[1],
                "z": final_pos[2]
            })
            
    # 5. Save Submission
    submission_df = pd.DataFrame(results)
    submission_path = Path("outputs/step10/submission.csv")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"\nStep 10 Submission saved to {submission_path}")

if __name__ == "__main__":
    inference()
