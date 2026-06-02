import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from autogluon.tabular import TabularPredictor
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Starting Step 36 vs Step 37 OOF Error Comparison...")
    
    regimes = ["slow_straight", "fast_straight", "slow_extreme_turning", "fast_turning"]
    
    # Store error data for plotting
    data_list = []
    
    for r in regimes:
        print(f"\nProcessing regime: {r.upper()}...")
        
        # Load dataset
        train_path_36 = f"experiments/step36_four_regime/train_ranker_v36_{r}.csv"
        df = pd.read_csv(train_path_36)
        
        # 1. Step 36 Classification OOF
        model_path_36 = f"experiments/step36_four_regime/models/ranker_v36_{r}"
        print(f"  Loading Step 36 Classifier from {model_path_36}...")
        predictor_36 = TabularPredictor.load(model_path_36)
        
        print("  Predicting OOF probabilities for Step 36...")
        oof_pred_prob = predictor_36.predict_proba_oof()
        df['oof_prob_36'] = oof_pred_prob[1].values
        
        # Select best candidate per ID for Step 36 (highest probability)
        idx_best_36 = df.groupby('id')['oof_prob_36'].idxmax()
        best_36 = df.loc[idx_best_36]
        
        # 2. Step 37 Regression OOF
        model_path_37 = f"step37_turning_refinement/models/ranker_v37_{r}"
        print(f"  Loading Step 37 Regressor from {model_path_37}...")
        predictor_37 = TabularPredictor.load(model_path_37)
        
        print("  Predicting OOF distances for Step 37...")
        oof_pred_dist = predictor_37.predict_oof()
        df['oof_dist_37'] = oof_pred_dist.values
        
        # Select best candidate per ID for Step 37 (minimum predicted distance)
        idx_best_37 = df.groupby('id')['oof_dist_37'].idxmin()
        best_37 = df.loc[idx_best_37]
        
        # Merge by trajectory ID to compare errors
        merged = pd.merge(
            best_36[['id', 'reg_target']].rename(columns={'reg_target': 'err_step36'}),
            best_37[['id', 'reg_target']].rename(columns={'reg_target': 'err_step37'}),
            on='id'
        )
        merged['regime'] = r.upper()
        data_list.append(merged)
        
        # Print summary
        hit_36 = (merged['err_step36'] <= 0.01).mean()
        hit_37 = (merged['err_step37'] <= 0.01).mean()
        print(f"  {r.upper()} - Step 36 Hit@1cm: {hit_36:.2%}, Step 37 Hit@1cm: {hit_37:.2%}")

    # Combine all regimes
    full_df = pd.concat(data_list, ignore_index=True)
    
    # Create the comparison plots
    print("\nGenerating error distribution comparison plot...")
    
    # 2x2 multi-panel layout for regimes
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=False)
    axes = axes.ravel()
    
    colors = {
        'STEP 36 (CLASSIFICATION)': '#E74C3C', # Red
        'STEP 37 (REGRESSION)': '#2ECC71'      # Green
    }
    
    for idx, r in enumerate(regimes):
        ax = axes[idx]
        reg_df = full_df[full_df['regime'] == r.upper()]
        
        # Plot KDE for Step 36 and Step 37
        sns.kdeplot(reg_df['err_step36'] * 100, label='Step 36 (Classification)', 
                    color=colors['STEP 36 (CLASSIFICATION)'], shade=True, ax=ax, linewidth=2)
        sns.kdeplot(reg_df['err_step37'] * 100, label='Step 37 (Regression)', 
                    color=colors['STEP 37 (REGRESSION)'], shade=True, ax=ax, linewidth=2)
        
        # Add 1.0cm Hit threshold line
        ax.axvline(1.0, color='blue', linestyle='--', linewidth=1.5, label='1.0cm Hit Threshold')
        
        # Add median error lines
        med_36 = reg_df['err_step36'].median() * 100
        med_37 = reg_df['err_step37'].median() * 100
        ax.axvline(med_36, color=colors['STEP 36 (CLASSIFICATION)'], linestyle=':', linewidth=1.5)
        ax.axvline(med_37, color=colors['STEP 37 (REGRESSION)'], linestyle=':', linewidth=1.5)
        
        # Labels and formatting
        ax.set_title(f"Error Distribution: {r.upper()} (N={len(reg_df)})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Euclidean Distance Error (cm)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_xlim(0, 3.0) # Zoom into the relevant 0-3cm range
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Compute Hit rates for titles
        hit_36 = (reg_df['err_step36'] <= 0.01).mean() * 100
        hit_37 = (reg_df['err_step37'] <= 0.01).mean() * 100
        
        # Add text box with details
        textstr = '\n'.join((
            f"Step 36 Hit@1cm: {hit_36:.2f}% (Med: {med_36:.2f}cm)",
            f"Step 37 Hit@1cm: {hit_37:.2f}% (Med: {med_37:.2f}cm)",
            f"Gain: +{hit_37 - hit_36:.2f}%p"
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        if idx == 0:
            ax.legend(loc='upper right')
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()


    plt.suptitle("OOF Prediction Distance Error Distribution Comparison\nStep 36 (Classification) vs Step 37 (Regression)", 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Ensure folder exists
    images_dir = Path("docs/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = images_dir / "07_step37_error_comparison.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved successfully to {img_path}")
    
    # Generate overall summary plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(full_df['err_step36'] * 100, label='Step 36 (Classification)', 
                color=colors['STEP 36 (CLASSIFICATION)'], shade=True, linewidth=2.5)
    sns.kdeplot(full_df['err_step37'] * 100, label='Step 37 (Regression)', 
                color=colors['STEP 37 (REGRESSION)'], shade=True, linewidth=2.5)
    plt.axvline(1.0, color='blue', linestyle='--', linewidth=1.5, label='1.0cm Hit Threshold')
    
    med_overall_36 = full_df['err_step36'].median() * 100
    med_overall_37 = full_df['err_step37'].median() * 100
    plt.axvline(med_overall_36, color=colors['STEP 36 (CLASSIFICATION)'], linestyle=':', linewidth=1.5)
    plt.axvline(med_overall_37, color=colors['STEP 37 (REGRESSION)'], linestyle=':', linewidth=1.5)
    
    hit_overall_36 = (full_df['err_step36'] <= 0.01).mean() * 100
    hit_overall_37 = (full_df['err_step37'] <= 0.01).mean() * 100
    
    plt.title("Overall OOF Error Distribution (All 10,000 Trajectories)", fontsize=14, fontweight='bold')
    plt.xlabel("Euclidean Distance Error (cm)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.xlim(0, 2.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    textstr = '\n'.join((
        f"Overall Step 36 Hit@1cm: {hit_overall_36:.2f}% (Med: {med_overall_36:.2f}cm)",
        f"Overall Step 37 Hit@1cm: {hit_overall_37:.2f}% (Med: {med_overall_37:.2f}cm)",
        f"Overall Gain: +{hit_overall_37 - hit_overall_36:.2f}%p"
    ))
    plt.text(0.55, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    plt.legend(loc='upper right')
    
    img_path_overall = images_dir / "07_step37_error_comparison_overall.png"
    plt.savefig(img_path_overall, dpi=150, bbox_inches='tight')
    print(f"Overall summary plot saved successfully to {img_path_overall}")

if __name__ == "__main__":
    main()
