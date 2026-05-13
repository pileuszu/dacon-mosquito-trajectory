import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MosquitoDataset
from model import EqMotion

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    test_dir = os.path.join(project_root, "data/open/test")
    model_path = os.path.join(project_root, "outputs/step4/checkpoints/best_model.pth")
    
    # Save output to outputs/step4
    output_dir = os.path.join(project_root, "outputs/step4")
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")
    
    if not os.path.exists(test_dir):
        # Fallback for relative paths if project_root logic fails in some envs
        test_dir = "data/open/test"
        model_path = "outputs/step4/checkpoints/best_model.pth"
        submission_path = "outputs/step4/submission.csv"
        os.makedirs("outputs/step4", exist_ok=True)

    # Dataset & Dataloader
    test_dataset = MosquitoDataset(test_dir, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model
    model = EqMotion().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded model weights.")
    else:
        print("Model weights not found! Using random weights for test run.")

    model.eval()
    results = []

    with torch.no_grad():
        for coords, file_ids in tqdm(test_loader, desc="Inference"):
            coords = coords.to(device)
            output = model(coords)
            output = output.cpu().numpy()
            
            for i in range(len(file_ids)):
                results.append({
                    'id': file_ids[i],
                    'x': output[i, 0],
                    'y': output[i, 1],
                    'z': output[i, 2]
                })

    df = pd.DataFrame(results)
    # Ensure ID order matches sample submission if necessary (though sorting is usually fine)
    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    inference()
