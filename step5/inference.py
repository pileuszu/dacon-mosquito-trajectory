import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MosquitoDatasetStep5
from model import SpectralEqMotion

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    test_dir = os.path.join(project_root, "data/open/test")
    model_path = os.path.join(project_root, "outputs/step5/checkpoints/best_model.pth")
    
    output_dir = os.path.join(project_root, "outputs/step5")
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")
    
    if not os.path.exists(test_dir):
        test_dir = "data/open/test"
        model_path = "outputs/step5/checkpoints/best_model.pth"
        submission_path = "outputs/step5/submission.csv"

    test_dataset = MosquitoDatasetStep5(test_dir, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = SpectralEqMotion(fft_dim=36).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded best model weights.")
    else:
        print(f"Warning: Model not found at {model_path}. Using random weights.")

    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            coords = batch['coords'].to(device)
            fft = batch['fft'].to(device)
            cv_pred = batch['cv_pred'].to(device)
            file_ids = batch['id']
            
            # Predict residual
            residual_pred = model(coords, fft)
            
            # Final Prediction = CV Baseline + Residual
            final_pred = cv_pred + residual_pred
            final_pred = final_pred.cpu().numpy()
            
            for i in range(len(file_ids)):
                results.append({
                    'id': file_ids[i],
                    'x': final_pred[i, 0],
                    'y': final_pred[i, 1],
                    'z': final_pred[i, 2]
                })

    df = pd.DataFrame(results)
    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    inference()
