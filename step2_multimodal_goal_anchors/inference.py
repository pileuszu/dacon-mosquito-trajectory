import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import from Step 1 (Handling name collision)
import importlib.util

# Load Dataset module from Step 1
spec_ds = importlib.util.spec_from_file_location("step1_dataset", "step1_coordinate_invariant_baseline/dataset.py")
step1_ds_mod = importlib.util.module_from_spec(spec_ds)
spec_ds.loader.exec_module(step1_ds_mod)
get_dataloaders = step1_ds_mod.get_dataloaders

# Load Model module from Step 1
spec_md = importlib.util.spec_from_file_location("step1_model", "step1_coordinate_invariant_baseline/model.py")
step1_md_mod = importlib.util.module_from_spec(spec_md)
spec_md.loader.exec_module(step1_md_mod)
Step1Model = step1_md_mod.BaselineGRU

# Local imports
from model import MultimodalAnchorModel, load_anchors

# Configuration
DATA_DIR = 'data/open/train/'
LABEL_PATH = 'data/open/train_labels.csv'
MODEL1_PATH = 'outputs/step1/best_baseline_model.pth'
MODEL2_PATH = 'outputs/step2/best_multimodal_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_and_visualize(num_samples=3):
    anchors = load_anchors().to(DEVICE)
    _, val_loader = get_dataloaders(DATA_DIR, LABEL_PATH, batch_size=1)
    
    # 1. Load Step 1 Model
    model1 = Step1Model().to(DEVICE)
    if os.path.exists(MODEL1_PATH):
        model1.load_state_dict(torch.load(MODEL1_PATH, map_location=DEVICE))
        print(f"Loaded Step 1 model from {MODEL1_PATH}")
    
    # 2. Load Step 2 Model
    model2 = MultimodalAnchorModel(n_anchors=len(anchors)).to(DEVICE)
    if os.path.exists(MODEL2_PATH):
        model2.load_state_dict(torch.load(MODEL2_PATH, map_location=DEVICE))
        print(f"Loaded Step 2 model from {MODEL2_PATH}")
    else:
        print(f"Error: Step 2 Model not found at {MODEL2_PATH}")
        return

    model1.eval()
    model2.eval()
    
    mae_list1, mae_list2, mae_list_cv = [], [], []
    
    samples_found = 0
    with torch.no_grad():
        for hist, target, origin in val_loader:
            hist, target = hist.to(DEVICE), target.to(DEVICE)
            
            # Step 1 Prediction
            pred1 = model1(hist)
            
            # Step 2 Prediction
            logits, offsets = model2(hist)
            best_idx = torch.argmax(logits, dim=1)
            pred2 = anchors[best_idx] + offsets[0, best_idx[0]]
            
            # CV Baseline Prediction
            v_last = (hist[0, -1] - hist[0, -2])
            pred_cv = hist[0, -1] + v_last * 2
            
            # MAE Calculation
            mae_list1.append(torch.abs(pred1 - target).mean().item())
            mae_list2.append(torch.abs(pred2 - target).mean().item())
            mae_list_cv.append(torch.abs(pred_cv - target[0]).mean().item())
            
            if samples_found < num_samples:
                visualize_comparison(
                    hist[0].cpu(), target[0].cpu(), 
                    pred1[0].cpu(), pred2[0].cpu(), pred_cv.cpu(),
                    best_idx.item(), samples_found
                )
                samples_found += 1
                
    print(f"\nEvaluation Results (Mean over {len(mae_list1)} samples):")
    print(f"CV Baseline MAE: {np.mean(mae_list_cv):.6f}")
    print(f"Step 1 Model MAE: {np.mean(mae_list1):.6f}")
    print(f"Step 2 Model MAE: {np.mean(mae_list2):.6f}")
    
    imp_vs_cv = (np.mean(mae_list_cv) - np.mean(mae_list2)) / np.mean(mae_list_cv) * 100
    imp_vs_s1 = (np.mean(mae_list1) - np.mean(mae_list2)) / np.mean(mae_list1) * 100
    print(f"Step 2 Improvement vs CV: {imp_vs_cv:.2f}%")
    print(f"Step 2 Improvement vs Step 1: {imp_vs_s1:.2f}%")

def visualize_comparison(hist, target, pred1, pred2, pred_cv, anchor_idx, idx):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot History
    ax.plot(hist[:, 0], hist[:, 1], hist[:, 2], 'gray', marker='.', label='History', alpha=0.4)
    
    # Ground Truth
    ax.scatter(target[0], target[1], target[2], color='red', s=120, label='Ground Truth', marker='X')
    
    # CV Baseline
    ax.scatter(pred_cv[0], pred_cv[1], pred_cv[2], color='orange', s=80, label='CV Baseline', marker='^')
    
    # Step 1 Prediction
    ax.scatter(pred1[0], pred1[1], pred1[2], color='blue', s=80, label='Step 1 (Simple GRU)', marker='s')
    
    # Step 2 Prediction
    ax.scatter(pred2[0], pred2[1], pred2[2], color='green', s=120, label=f'Step 2 (Anchor {anchor_idx+1})', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Prediction Comparison - Sample {idx+1}')
    ax.legend()
    
    save_path = f'outputs/step2/comparison_sample_{idx+1}.png'
    plt.savefig(save_path)
    print(f"Saved comparison visualization to {save_path}")
    plt.close()

if __name__ == "__main__":
    evaluate_and_visualize()
