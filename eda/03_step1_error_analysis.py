import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step1.model import LSTMResidualModel
from step1.dataset import get_dataloaders
from step1.config import *

def analyze_errors():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model
    model = LSTMResidualModel().to(device)
    model_path = 'step1_best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Get Test/Validation Dataloader (Same split as training)
    _, test_loader = get_dataloaders(batch_size=1)
    
    results = []
    
    print(f"Analyzing {len(test_loader)} evaluation samples...")
    with torch.no_grad():
        for item in test_loader:
            seq = item['seq'].to(device)
            cv_prior = item['cv_prior'].to(device)
            last_pos = item['last_pos'].to(device)
            target = item['target'].to(device)
            
            # Model prediction (returns final_pos and residual)
            final_pred, _ = model(seq, cv_prior, last_pos)
            
            # CV-only prediction = P0 + CV_Prior
            cv_full_pred = last_pos + cv_prior
            
            # Errors (Euclidean)
            model_error = torch.norm(target - final_pred, dim=1).item()
            cv_error = torch.norm(target - cv_full_pred, dim=1).item()
            
            results.append({
                'id': item['id'][0],
                'model_error': model_error,
                'cv_error': cv_error,
                'improvement': cv_error - model_error
            })
            
    res_df = pd.DataFrame(results)
    
    # 3. Categorize
    res_df['status'] = 'Fail (>3cm)'
    res_df.loc[res_df['model_error'] < 0.03, 'status'] = 'Near Miss (1-3cm)'
    res_df.loc[res_df['model_error'] < 0.01, 'status'] = 'Success (<1cm)'
    
    # 4. Plotting
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.countplot(data=res_df, x='status', order=['Success (<1cm)', 'Near Miss (1-3cm)', 'Fail (>3cm)'], palette='viridis')
    plt.title('Prediction Success Status (Best Model)')
    
    plt.subplot(1, 2, 2)
    sns.histplot(res_df['improvement'], kde=True, color='blue')
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Model Improvement over CV Baseline (m)')
    plt.xlabel('Improvement (Positive = Model is better than CV)')
    
    os.makedirs('eda/images', exist_ok=True)
    plt.savefig('eda/images/03_error_status.png')
    
    # 5. Summary Statistics
    hit_rate_1cm = (res_df['status'] == 'Success (<1cm)').mean() * 100
    hit_rate_3cm = (res_df['model_error'] < 0.03).mean() * 100
    worsened = res_df[res_df['improvement'] < 0]
    
    print(f"\n--- Analysis Summary ---")
    print(f"Hit Rate@1cm: {hit_rate_1cm:.2f}%")
    print(f"Hit Rate@3cm: {hit_rate_3cm:.2f}%")
    print(f"Average Improvement: {res_df['improvement'].mean()*1000:.2f} mm")
    print(f"Worsened Samples: {len(worsened)} ({len(worsened)/len(res_df)*100:.2f}%)")
    
    # Write Report
    with open('eda/03_step1_error_analysis.md', 'w') as f:
        f.write("# 03. Step 1 Model Error & Failure Analysis\n\n")
        f.write("## Evaluation Metrics (Best Model)\n")
        f.write(f"- **Hit Rate@1cm**: {hit_rate_1cm:.2f}%\n")
        f.write(f"- **Hit Rate@3cm**: {hit_rate_3cm:.2f}%\n")
        f.write(f"- **Mean Distance Error**: {res_df['model_error'].mean()*100:.4f} cm\n")
        f.write(f"- **CV Baseline Error**: {res_df['cv_error'].mean()*100:.4f} cm\n\n")
        
        f.write("## Why is the performance lower than expected?\n")
        f.write("### 1. Regression to Mean / Small Improvements\n")
        f.write(f"The average improvement over CV is only **{res_df['improvement'].mean()*1000:.2f} mm**. ")
        f.write("In many cases, the model only slightly adjusts the CV prediction, but not enough to cross the 1cm threshold.\n\n")
        
        f.write("### 2. Harmful Corrections\n")
        f.write(f"In **{len(worsened)/len(res_df)*100:.2f}%** of cases, the model made the prediction **worse** than simple CV. ")
        f.write("This indicates that for some trajectories, the residual learning introduced noise or over-corrected a stable path.\n\n")
        
        f.write("## Visualizations\n")
        f.write("### Error Status Distribution\n")
        f.write("![Error Status](images/03_error_status.png)\n")

    print(f"Report generated at eda/03_step1_error_analysis.md")

if __name__ == "__main__":
    analyze_errors()
