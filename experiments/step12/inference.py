import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from step12.physics import make_candidates, CANDIDATES_GLOBAL, EPS
from step12.prepare_data import extract_context_features, calculate_prior_features
from step6.config import *

def inference_v12():
    # 1. Load Setup
    save_path = 'step12/models/ranker_v4'
    predictor = TabularPredictor.load(save_path)
    print(f"Loaded Step 12 Ranker from {save_path}")
    
    s7_preds_df = pd.read_csv("step10/step7_preds_test.csv").set_index('id')
    s4_preds_df = pd.read_csv("step12/step4_preds_test.csv").set_index('id')
    print("Loaded Step 7 & Step 4 Test priors.")
    
    # 2. Prepare Test Files
    test_files = sorted(list(TEST_DIR.glob("*.csv")))
    results = []
    
    ranker_count = 0
    fallback_count = 0
    
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
            s4_pos = s4_preds_df.loc[fid].to_numpy()
            
            ctx = extract_context_features(xyz)
            s7_s4_dist = np.linalg.norm(s7_pos - s4_pos)
            
            # Generate 614 candidates
            cands = make_candidates(xyz, priors=[s7_pos, s4_pos], horizon=2)
            batch_info.append((fid, cands, s7_pos))
            
            for idx in range(len(cands)):
                cand_pos = cands[idx]
                feat_s7 = calculate_prior_features(cand_pos, s7_pos, p0, "s7")
                feat_s4 = calculate_prior_features(cand_pos, s4_pos, p0, "s4")
                
                batch_rows.append({
                    "fid": fid,
                    "cand_idx": idx,
                    "is_local": 1 if idx >= len(CANDIDATES_GLOBAL) else 0,
                    **feat_s7,
                    **feat_s4,
                    **ctx,
                    "s7_s4_dist": s7_s4_dist,
                    "dist_to_p0": np.linalg.norm(cand_pos - p0)
                })
        
        # 3. Predict Probabilities
        batch_df = pd.DataFrame(batch_rows)
        pred_data = batch_df.drop(columns=['fid'])
        
        probs = predictor.predict_proba(pred_data)
        score_col = 1 if 1 in probs.columns else probs.columns[0]
        batch_df['score'] = probs[score_col].values
        
        # 4. Final Selection
        THRESHOLD = 0.4
        for fid, cands, s7_pos in batch_info:
            sample_scores = batch_df[batch_df['fid'] == fid]
            best_idx = sample_scores['score'].idxmax()
            max_score = sample_scores.loc[best_idx, 'score']
            
            if max_score >= THRESHOLD:
                best_cand_idx = int(sample_scores.loc[best_idx, 'cand_idx'])
                final_pos = cands[best_cand_idx]
                ranker_count += 1
            else:
                # Fallback to Step 7 (More robust regression)
                final_pos = s7_pos
                fallback_count += 1
                
            results.append({
                "id": fid,
                "x": final_pos[0],
                "y": final_pos[1],
                "z": final_pos[2]
            })
            
    # 5. Save Output
    submission_df = pd.DataFrame(results)
    submission_path = Path("outputs/step12/submission.csv")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nInference Complete!")
    print(f"Ranker Selected: {ranker_count} times")
    print(f"Fallback Used: {fallback_count} times")
    print(f"Step 12 Submission saved to {submission_path}")

if __name__ == "__main__":
    inference_v12()
