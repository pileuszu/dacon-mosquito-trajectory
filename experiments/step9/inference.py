import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from step9.physics import make_candidates, CANDIDATES, EPS
from step9.prepare_data import extract_context_features
from step9.config import *

def inference():
    # 1. Load Predictor
    save_path = 'step9/models/ranker'
    predictor = TabularPredictor.load(save_path)
    print(f"Loaded AutoML Ranker from {save_path}")
    
    # 2. Prepare Test Data
    test_files = sorted(list(TEST_DIR.glob("*.csv")))
    
    results = []
    
    # We'll process in batches of 100 samples to keep memory stable
    batch_size = 100
    for i in tqdm(range(0, len(test_files), batch_size), desc="Test Batch"):
        batch_files = test_files[i:i+batch_size]
        
        batch_rows = []
        batch_info = [] # (fid, candidates)
        
        for fpath in batch_files:
            fid = fpath.stem
            df = pd.read_csv(fpath)
            xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
            
            ctx = extract_context_features(xyz)
            cands = make_candidates(xyz, end_idx=-1, horizon=2)
            
            batch_info.append((fid, cands))
            
            for idx in range(len(CANDIDATES)):
                spec = CANDIDATES[idx]
                row = {
                    "fid": fid,
                    "cand_idx": idx,
                    "spec_par": spec.par,
                    "spec_perp": spec.perp,
                    "spec_ts": spec.time_scale,
                    "spec_jerk": spec.jerk,
                    **ctx,
                    "dist_to_p0": np.linalg.norm(cands[idx] - xyz[-1])
                }
                batch_rows.append(row)
        
        # 3. Predict Probabilities for the whole batch
        batch_df = pd.DataFrame(batch_rows)
        # Drop only fid for prediction
        pred_data = batch_df.drop(columns=['fid'])
        
        # We want the probability of class 1
        probs = predictor.predict_proba(pred_data)
        if isinstance(probs, pd.DataFrame):
            # If binary, probs has columns [0, 1]
            score_col = 1 if 1 in probs.columns else probs.columns[0]
            scores = probs[score_col].values
        else:
            scores = probs
            
        batch_df['score'] = scores
        
        # 4. Pick best for each sample
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
    submission_path = OUTPUT_DIR / "submission_step9.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")

if __name__ == "__main__":
    inference()
