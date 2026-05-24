import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import TEST_DIR, SAMPLE_SUBMISSION_PATH, OUTPUT_DIR, HIDDEN_SIZE, NUM_LAYERS
from model import LSTMResidualModel

def inference():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "step1_model.pth"
    
    if not Path(model_path).exists():
        print(f"Error: Model file {model_path} not found. Please run train.py first.")
        return

    # 2. Load Model
    model = LSTMResidualModel(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Load Test Files
    test_files = sorted(TEST_DIR.glob('*.csv'))
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    
    print(f"Generating predictions for {len(test_files)} test samples...")
    
    results = []
    with torch.no_grad():
        for file_path in tqdm(test_files):
            df = pd.read_csv(file_path)
            xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
            
            last_xyz = xyz[-1]
            prev_xyz = xyz[-2]
            
            # Preprocessing
            normalized_xyz = xyz - last_xyz
            cv_prior_rel = 2.0 * (last_xyz - prev_xyz)
            
            # Convert to tensor and add batch dimension
            seq_tensor = torch.tensor(normalized_xyz).unsqueeze(0).to(device)
            cv_prior_tensor = torch.tensor(cv_prior_rel).unsqueeze(0).to(device)
            last_pos_tensor = torch.tensor(last_xyz).unsqueeze(0).to(device)
            
            # Predict
            final_pred, _ = model(seq_tensor, cv_prior_tensor, last_pos_tensor)
            
            pred_xyz = final_pred.squeeze(0).cpu().numpy()
            
            results.append({
                "id": file_path.stem,
                "x": pred_xyz[0],
                "y": pred_xyz[1],
                "z": pred_xyz[2]
            })

    # 4. Save Submission
    pred_df = pd.DataFrame(results)
    submission = sample_submission[['id']].merge(pred_df, on='id', how='left')
    
    output_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")

if __name__ == "__main__":
    inference()
