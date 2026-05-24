import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
import sys
import os
import traceback

sys.path.append(os.getcwd())
from step24_damped_highres.physics import make_candidates, CANDIDATES
from step24_damped_highres.prepare_data import extract_context_features
from utils.notifier import send_discord_notification

URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

def run_damped_highres_inference_v24():
    try:
        send_discord_notification(URL, "🚀 [Step 24] Damped High-Res Ranker V24 Inference Started...")
        
        data_dir = Path("data/open")
        test_dir = data_dir / "test"
        sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
        
        # Load the Damped High-Res Ranker V24
        model_path = 'step24_damped_highres/models/ranker_v24'
        predictor = TabularPredictor.load(model_path)
        
        results = []
        batch_size = 200
        ids = sample_sub['id'].tolist()
        
        print(f"Running Damped High-Res Step 24 Inference (N={len(ids)} trajectories)...")
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            all_cand_rows = []
            batch_info = []
            
            for fid in tqdm(batch_ids, desc=f"Batch {i//batch_size + 1}"):
                fpath = test_dir / f"{fid}.csv"
                df = pd.read_csv(fpath)
                xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
                p0 = xyz[-1]
                
                ctx = extract_context_features(xyz)
                cands = make_candidates(xyz, end_idx=-1, horizon=2)
                
                start_idx = len(all_cand_rows)
                for idx in range(len(cands)):
                    cand_pos = cands[idx]
                    spec = CANDIDATES[idx]
                    
                    row = {
                        "fid": fid,
                        "cand_idx": idx,
                        "spec_par": spec.par,
                        "spec_perp": spec.perp,
                        "spec_ts": spec.time_scale,
                        "spec_dmp": spec.damping,
                        "spec_jerk": spec.jerk,
                        **ctx,
                        "dist_to_p0": np.linalg.norm(cand_pos - p0)
                    }
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
        out_path = Path("outputs/step24_damped_highres/submission.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(out_path, index=False)
        
        success_msg = f"✅ [Step 24] Damped High-Res Inference Finished Successfully!\nSaved to: {out_path}"
        send_discord_notification(URL, success_msg)
        print(success_msg)
        
    except Exception as e:
        error_msg = f"❌ [Step 24] Inference ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    run_damped_highres_inference_v24()
