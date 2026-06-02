import pandas as pd
from autogluon.tabular import TabularPredictor
import sys
import os
import traceback
from pathlib import Path

sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification

URL = None

def train_specialized_model(model_name, train_data_path, model_path, time_limit):
    try:
        send_discord_notification(URL, f"🚀 [Step 40] Specialized Ranker ({model_name}) Training Started...")
        print(f"\n==================================================")
        print(f"Loading {model_name} data from {train_data_path}...")
        df = pd.read_csv(train_data_path)
        
        # Drop identifier and auxiliary regression target for training
        # We KEEP the 'weight' column in train_df because AutoGluon needs it for sample_weight
        train_df = df.drop(columns=['id', 'reg_target'])
        
        # Memory-safe configuration (No RF/XT to prevent bad allocation on Windows)
        hyperparameters = {
            'GBM': {},
            'CAT': {},
            'XGB': {},
            'NN_TORCH': {},
            'FASTAI': {}
        }
        
        print(f"Starting AutoGluon TabularPredictor fit ('best_quality') for {model_name}...")
        predictor = TabularPredictor(
            label='target',
            eval_metric='roc_auc',
            sample_weight='weight',  # Handle class imbalance natively in constructor
            path=model_path
        ).fit(
            train_data=train_df,
            presets='best_quality',
            hyperparameters=hyperparameters,
            num_bag_folds=5,  # 5-fold bagging for memory safety and robust OOF
            time_limit=time_limit,
            ag_args_ensemble={
                'num_folds_parallel': 1,
                'fold_fitting_strategy': 'sequential_local'
            },
            verbosity=2
        )
        
        # Calculate OOF predictions
        print("Calculating Out-Of-Fold (OOF) Hit@1cm metric...")
        oof_pred_proba = predictor.predict_proba_oof()
        score_col = 1 if 1 in oof_pred_proba.columns else oof_pred_proba.columns[0]
        df['oof_prob'] = oof_pred_proba[score_col].values
        
        # Select candidate with the highest OOF probability per trajectory
        idx_best = df.groupby('id')['oof_prob'].idxmax()
        best_cands = df.loc[idx_best]
        
        # Overall Hit@1cm for this specialized group
        overall_hit_rate = (best_cands['reg_target'] <= 0.01).mean()
        
        # Print Hit rates per original GMM regime within this group
        print(f"\n=== Hit Rates per Regime ({model_name}) ===")
        regime_stats = []
        for regime_name, group in best_cands.groupby('regime'):
            r_hit = (group['reg_target'] <= 0.01).mean()
            mean_err = group['reg_target'].mean() * 100
            median_err = group['reg_target'].median() * 100
            stat_str = f"Regime {regime_name} (N={len(group)}): Hit = {r_hit:.4%}, Mean = {mean_err:.4f} cm, Median = {median_err:.4f} cm"
            print(stat_str)
            regime_stats.append(stat_str)
            
        # Get AutoGluon Leaderboard
        # We must drop the weight column here if we want clean evaluation, but silent is fine
        leaderboard = predictor.leaderboard(train_df, silent=True)
        leaderboard_str = leaderboard.head(15).to_string()
        print(f"\n=== Model Leaderboard ({model_name}) ===")
        print(leaderboard_str)
        
        best_model = predictor.model_best
        best_score = leaderboard.loc[leaderboard['model'] == best_model, 'score_val'].values[0]
        
        success_msg = (
            f"✅ [Step 40] Specialized Ranker ({model_name}) Training Finished!\n"
            f"Group-wise **OOF Hit@1cm**: **{overall_hit_rate:.4%}** 🏆\n"
            f"Best Model: **{best_model}**\n"
            f"Best Validation AUC: **{best_score:.6f}**\n\n"
            f"**Regime Breakdown:**\n" + "\n".join(regime_stats) + "\n\n"
            f"**Leaderboard Top 10:**\n```\n{leaderboard.head(10)[['model', 'score_val', 'pred_time_val', 'fit_time']].to_string()}\n```"
        )
        send_discord_notification(URL, success_msg)
        print(success_msg)
        
        # Save OOF predictions to help verify overall OOF score later
        best_cands.to_csv(f"step40_dual_specialized/data/oof_best_{model_name}.csv", index=False)
        
    except BaseException as e:
        error_msg = f"❌ [Step 40] Training ERROR ({model_name}):\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        print(error_msg)
        raise e

def main():
    # Make sure output models directory exists
    Path("step40_dual_specialized/models").mkdir(parents=True, exist_ok=True)
    
    # Train each specialized model sequentially
    # Set time limit to 3600 seconds (1 hour) per model to allow robust fitting (total 3 hours)
    time_limit_per_model = 3600
    
    # 1. Cruising Model
    train_specialized_model(
        model_name="cruising",
        train_data_path="step40_dual_specialized/data/train_ranker_v40_cruising.csv",
        model_path="step40_dual_specialized/models/ranker_v40_cruising",
        time_limit=time_limit_per_model
    )
    
    # 2. Gliding Model
    train_specialized_model(
        model_name="gliding",
        train_data_path="step40_dual_specialized/data/train_ranker_v40_gliding.csv",
        model_path="step40_dual_specialized/models/ranker_v40_gliding",
        time_limit=time_limit_per_model
    )
    
    # 3. Steering Model
    train_specialized_model(
        model_name="steering",
        train_data_path="step40_dual_specialized/data/train_ranker_v40_steering.csv",
        model_path="step40_dual_specialized/models/ranker_v40_steering",
        time_limit=time_limit_per_model
    )
    
    # Calculate overall OOF score by combining the three OOF files
    print("\n=== Calculating Overall Step 40 OOF Score ===")
    oof_dfs = []
    for model_name in ["cruising", "gliding", "steering"]:
        fpath = Path(f"step40_dual_specialized/data/oof_best_{model_name}.csv")
        if fpath.exists():
            oof_dfs.append(pd.read_csv(fpath))
            
    if len(oof_dfs) == 3:
        combined = pd.concat(oof_dfs)
        overall_hit_rate = (combined['reg_target'] <= 0.01).mean()
        
        # Detailed print per regime
        print("\n=== Combined Regime-wise OOF Performance ===")
        for name, gp in combined.groupby('regime'):
            r_hit = (gp['reg_target'] <= 0.01).mean()
            mean_err = gp['reg_target'].mean() * 100
            median_err = gp['reg_target'].median() * 100
            print(f"Regime {name} (N={len(gp)}): Hit = {r_hit:.4%}, Mean = {mean_err:.4f} cm, Median = {median_err:.4f} cm")
            
        summary_msg = f"🏆 [Step 40] Overall Tri-Model Specialized **Combined OOF Hit@1cm**: **{overall_hit_rate:.4%}**"
        send_discord_notification(URL, summary_msg)
        print(summary_msg)

if __name__ == "__main__":
    main()
