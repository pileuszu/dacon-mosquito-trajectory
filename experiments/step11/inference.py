import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from step11.physics import make_candidates, CANDIDATES, EPS
from step11.prepare_data import extract_context_features, calculate_advanced_features
from step6.config import *

def inference():
    # 1. Load Predictor & Step 7 Features
    save_path = 'step11/models/ranker_v3'
    predictor = TabularPredictor.load(save_path)
    print(f"Loaded Step 11 Ranker from {save_path}")
    
    s7_preds_df = pd.read_csv("step10/step7_preds_test.csv").set_index('id')
    print("Loaded Step 7 Test predictions.")
    
    # 2. Prepare Test Data
    test_files = sorted(list(TEST_DIR.glob("*.csv")))
    results = []
    
    fallback_count = 0
    ranker_count = 0
    
    # Batch processing
    batch_size = 100
    for i in tqdm(range(0, len(test_files), batch_size), desc="Test Batch"):
        batch_files = test_files[i:i+batch_size]
        batch_rows = []
        batch_info = [] # (fid, candidates, s7_pos)
        
        for fpath in batch_files:
            fid = fpath.stem
            df = pd.read_csv(fpath)
            xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
            p0 = xyz[-1]
            
            s7_pos = s7_preds_df.loc[fid].to_numpy()
            ctx = extract_context_features(xyz)
            cands = make_candidates(xyz, end_idx=-1, horizon=2)
            
            batch_info.append((fid, cands, s7_pos))
            
            for idx in range(len(CANDIDATES)):
                cand_pos = cands[idx]
                adv = calculate_advanced_features(cand_pos, s7_pos, p0)
                
                row = {
                    "fid": fid,
                    "cand_idx": idx,
                    "spec_d1": CANDIDATES[idx].d1,
                    "spec_par": CANDIDATES[idx].par,
                    "spec_perp": CANDIDATES[idx].perp,
                    "spec_ts": CANDIDATES[idx].ts,
                    **adv,
                    **ctx,
                    "dist_to_p0": np.linalg.norm(cand_pos - p0)
                }
                batch_rows.append(row)
        
        # 3. Predict
        batch_df = pd.DataFrame(batch_rows)
        pred_data = batch_df.drop(columns=['fid'])
        
        probs = predictor.predict_proba(pred_data)
        score_col = 1 if 1 in probs.columns else probs.columns[0]
        batch_df['score'] = probs[score_col].values
        
        # 4. Confidence-based Selection
        THRESHOLD = 0.4
        
        for fid, cands, s7_pos in batch_info:
            sample_scores = batch_df[batch_df['fid'] == fid]
            best_idx = sample_scores['score'].idxmax()
            max_score = sample_scores.loc[best_idx, 'score']
            
            if max_score >= THRESHOLD:
                best_cand_idx = sample_scores.loc[best_idx, 'cand_idx']
                final_pos = cands[int(best_cand_idx)]
                ranker_count += 1
            else:
                # Fallback to Step 7 Regression
                final_pos = s7_pos
                fallback_count += 1
                
            results.append({
                "id": fid,
                "x": final_pos[0],
                "y": final_pos[1],
                "z": final_pos[2]
            })
            
    # 5. Save Submission
    submission_df = pd.DataFrame(results)
    submission_path = Path("outputs/step11/submission.csv")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nInference Complete!")
    print(f"Ranker used: {ranker_count} times")
    print(f"Fallback used: {fallback_count} times")
    print(f"Step 11 Submission saved to {submission_path}")

if __name__ == "__main__":
    inference()
