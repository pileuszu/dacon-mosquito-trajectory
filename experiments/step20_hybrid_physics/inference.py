import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
import sys
import os
import traceback

sys.path.append(os.getcwd())
from step16.physics import make_candidates, CANDIDATES_GLOBAL
from step20_hybrid_physics.prepare_ranker_data import extract_context_features, calculate_prior_features
from utils.notifier import send_discord_notification

URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

def run_hybrid_inference_v20():
    try:
        send_discord_notification(URL, "🚀 [Step 20] Hybrid Physics-Guided Inference Started...")
        
        data_dir = Path("data/open")
        test_dir = data_dir / "test"
        sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
        
        # Load honest Step 7 predictions (the strong physics-only baseline)
        s7_preds = pd.read_csv("outputs/step7/submission.csv").set_index('id')
        
        # Load the Hybrid Ranker V20
        model_path = 'step20_hybrid_physics/models/ranker_v20'
        predictor = TabularPredictor.load(model_path)
        
        results = []
        batch_size = 200
        ids = sample_sub['id'].tolist()
        
        print(f"Running Hybrid Physics-Guided Step 20 Inference...")
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            all_cand_rows = []
            batch_info = []
            
            for fid in tqdm(batch_ids, desc=f"Batch {i//batch_size + 1}"):
                fpath = test_dir / f"{fid}.csv"
                df = pd.read_csv(fpath)
                xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
                p0 = xyz[-1]
                
                s7_pos = s7_preds.loc[fid].to_numpy()
                
                ctx = extract_context_features(xyz)
                cands = make_candidates(xyz, priors=[s7_pos, None], horizon=2)
                
                start_idx = len(all_cand_rows)
                for idx in range(len(cands)):
                    cand_pos = cands[idx]
                    adv = calculate_prior_features(cand_pos, s7_pos, p0)
                    
                    row = {
                        "fid": fid,
                        "cand_idx": idx,
                        "is_global": 1 if idx < len(CANDIDATES_GLOBAL) else 0,
                        **ctx,
                        **adv,
                        "dist_to_p0": np.linalg.norm(cand_pos - p0)
                    }
                    
                    if idx < len(CANDIDATES_GLOBAL):
                        spec = CANDIDATES_GLOBAL[idx]
                        row.update({
                            "spec_par": spec.par, 
                            "spec_perp": spec.perp, 
                            "spec_ts": spec.ts,
                            "spec_jerk": spec.jerk
                        })
                    else:
                        row.update({
                            "spec_par": 0.0, 
                            "spec_perp": 0.0, 
                            "spec_ts": 1.0,
                            "spec_jerk": 0.0
                        })
                    all_cand_rows.append(row)
                batch_info.append({"id": fid, "start": start_idx, "end": len(all_cand_rows), "cands": cands})
                
            batch_df = pd.DataFrame(all_cand_rows)
            pred_data = batch_df.drop(columns=['fid'])
            probs = predictor.predict_proba(pred_data)
            score_col = 1 if 1 in probs.columns else probs.columns[0]
            batch_scores = probs[score_col].values
            
            for info in batch_info:
                scores = batch_scores[info['start'] : info['end']]
                best_idx = np.argmax(scores)
                final_pos = info['cands'][best_idx]
                results.append({"id": info['id'], "x": final_pos[0], "y": final_pos[1], "z": final_pos[2]})
                
        sub_df = pd.DataFrame(results)
        out_path = Path("outputs/step20_hybrid_physics/submission.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(out_path, index=False)
        
        success_msg = f"✅ [Step 20] Hybrid Physics-Guided Inference Finished Successfully!\nSaved to: {out_path}"
        send_discord_notification(URL, success_msg)
        print(success_msg)
        
    except Exception as e:
        error_msg = f"❌ [Step 20] Inference ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    run_hybrid_inference_v20()
