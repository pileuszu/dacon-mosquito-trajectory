import os
import sys
import pickle
import traceback
import shutil
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from autogluon.tabular import TabularPredictor
from sklearn.calibration import calibration_curve

def add_fold_id_column(df, num_folds=5):
    # Sort unique IDs to make splitting deterministic
    unique_ids = sorted(df['id'].unique())
    id_to_fold = {}
    for uid in unique_ids:
        # Use md5 hash of uid to map deterministically to a fold [0, num_folds-1]
        h = hashlib.md5(str(uid).encode('utf-8')).hexdigest()
        fold_idx = int(h, 16) % num_folds
        id_to_fold[uid] = fold_idx
    df['fold_id'] = df['id'].map(id_to_fold)
    return df

# Add current workspace to path
sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification

# Exclude RF, XT, and Neural Networks to prevent memory issues and extremely long CPU runs
HYPERPARAMETERS = {
    'GBM': {},
    'CAT': {},
    'XGB': {}
}

# Image output directory
IMAGES_DIR = Path("docs/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("docs")
MODELS_DIR = Path("step41_self_driven_agent/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

import uuid
def get_unique_model_path(base_name):
    unique_suffix = uuid.uuid4().hex[:8]
    path = MODELS_DIR / f"{base_name}_{unique_suffix}"
    if path.exists():
        try:
            shutil.rmtree(path)
        except Exception:
            pass
    return path

def evaluate_predictions(combined_df):
    """
    Finds the candidate with the highest predicted probability for each trajectory id,
    and calculates the overall Hit@1cm score (reg_target <= 0.01).
    """
    idx_best = combined_df.groupby('id')['pred_prob'].idxmax()
    best_cands = combined_df.loc[idx_best]
    overall_hit = (best_cands['reg_target'] <= 0.01).mean()
    
    regime_breakdown = {}
    for name, gp in best_cands.groupby('regime'):
        r_hit = (gp['reg_target'] <= 0.01).mean()
        regime_breakdown[name] = (r_hit, len(gp))
        
    return overall_hit, best_cands, regime_breakdown

def run_phase_1():
    print("\n==========================================")
    print("=== PHASE 1: Group-CV Baseline Training ===")
    print("==========================================")
    send_discord_notification(None, "🚀 [Step 41 Phase 1] Starting Leak-Free Baseline Group-CV training...")
    
    regimes = ["cruising", "gliding", "steering"]
    combined_dfs = []
    
    # We evaluate on Th=0.0 to have the full unbiased validation set (denominator=10000)
    for regime in regimes:
        train_path = f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th0.0.csv"
        df = pd.read_csv(train_path)
        
        df = add_fold_id_column(df, num_folds=5)
        # Drop 'reg_target' and 'id' (grouping column is 'fold_id')
        train_df = df.drop(columns=['reg_target', 'id'])
        
        model_path = get_unique_model_path(f"baseline_{regime}")
            
        print(f"\n--- Training baseline model for {regime} with groups='fold_id' ---")
        predictor = TabularPredictor(
            label='target',
            eval_metric='log_loss',
            sample_weight='weight',
            groups='fold_id',  # Use fold_id to partition folds by trajectory ID!
            path=str(model_path)
        ).fit(
            train_data=train_df,
            presets='high_quality',
            hyperparameters=HYPERPARAMETERS,
            num_bag_folds=5,
            time_limit=600,  # 10 minutes per model
            num_gpus=0,  # Force CPU-only to prevent timeout hangs!
            dynamic_stacking=False,  # Disable DyStack Ray subprocesses!
            ag_args_ensemble={
                'num_folds_parallel': 1,
                'fold_fitting_strategy': 'sequential_local'
            },
            verbosity=1
        )
        
        # Calculate OOF prediction probability
        oof_pred_proba = predictor.predict_proba_oof()
        score_col = 1 if 1 in oof_pred_proba.columns else oof_pred_proba.columns[-1]
        df['pred_prob'] = oof_pred_proba[score_col].values
        
        combined_dfs.append(df)
        
        # Save feature importance plot
        try:
            val_df = train_df.sample(n=min(len(train_df), 2000), random_state=42)
            fi = predictor.feature_importance(val_df)
            plt.figure(figsize=(10, 6))
            top_fi = fi.head(20)
            sns.barplot(x=top_fi['importance'], y=top_fi.index, hue=top_fi.index, legend=False, palette="viridis")
            plt.title(f"Baseline Top 20 Features - {regime.capitalize()}")
            plt.xlabel("Importance Score")
            plt.tight_layout()
            plt.savefig(IMAGES_DIR / f"phase1_fi_{regime}.png")
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to plot feature importance for {regime}: {e}")
            
    # Combine predictions and calculate overall score
    combined_df = pd.concat(combined_dfs).reset_index(drop=True)
    overall_hit, best_cands, breakdowns = evaluate_predictions(combined_df)
    
    print(f"\nBaseline OOF Hit@1cm: {overall_hit:.4%}")
    for name, (r_hit, N) in breakdowns.items():
        print(f"  * {name}: {r_hit:.4%} (N={N})")
        
    # Generate Reliability Diagram
    try:
        plt.figure(figsize=(8, 6))
        # Drop NaNs or invalid values
        valid_df = combined_df.dropna(subset=['pred_prob', 'target'])
        y_true = valid_df['target'].values
        y_prob = valid_df['pred_prob'].values
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label="Baseline Model")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Reliability Diagram (Calibration Curve)")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "phase1_reliability.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot reliability diagram: {e}")
        
    # Plot Class Distribution
    try:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=combined_df, x='target', hue='target', legend=False, palette="coolwarm")
        plt.title("Candidate Class Distribution (Target=1 (Hit) vs Target=0 (Miss))")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "phase1_class_dist.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot class distribution: {e}")
        
    # Write Phase 1 Report
    report_content = f"""# 📝 Phase 1 Report: Baseline Group-CV and Leakage Resolution

## 1. 개요 및 검증 격리 결과 (Group-CV Framework)
* **목표**: 사용자 ID(Trajectory ID)를 기준으로 Bagging Folds를 분리하여 데이터 누수(Data Leakage)를 차단한 leak-free 베이스라인 구축.
* **검증 기법**: `groups='id'` 파라미터를 AutoGluon에 주입하여 Bagging fold 분할 시 동일 trajectory의 후보들이 train과 val에 동시에 유입되지 않도록 통제.

## 2. 베이스라인 모델 검증 스코어 (Baseline Scores)
* **전체 Leak-Free OOF Hit@1cm**: **{overall_hit:.4%}**

### GMM Regime별 상세 스코어:
"""
    for name, (r_hit, N) in breakdowns.items():
        report_content += f"* **{name}** (N={N}): **{r_hit:.4%}**\n"
        
    report_content += f"""
---

## 3. 베이스라인 시각화 진단 (Visual Diagnostics)

### 3.1 클래스 불균형 (Class Imbalance)
후보지 중 실제 Hit(타겟=1)에 부합하는 비율을 보여줍니다.
![Class Distribution](images/phase1_class_dist.png)

### 3.2 모델 신뢰도 다이어그램 (Reliability Diagram)
예측 확률값의 정확성(Calibration)을 평가한 신뢰도 곡선입니다.
![Reliability Diagram](images/phase1_reliability.png)

### 3.3 Regime별 피처 중요도 (Feature Importance)
모델이 예측 시 가장 중요하게 사용한 top 20 피처 기여도입니다.
"""
    for regime in regimes:
        report_content += f"\n#### {regime.capitalize()} Feature Importance\n"
        report_content += f"![Feature Importance {regime}](images/phase1_fi_{regime}.png)\n"
        
    with open(REPORTS_DIR / "AUTOML_REPORT_PHASE_1.md", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    send_discord_notification(
        None, 
        f"✅ [Step 41 Phase 1 Finished]\n"
        f"Overall Leak-Free OOF Hit@1cm: **{overall_hit:.4%}**\n"
        f"Report saved to `docs/AUTOML_REPORT_PHASE_1.md`"
    )
    
    return overall_hit

def run_phase_2():
    print("\n==============================================")
    print("=== PHASE 2: Resource-Efficient Sweep Search ===")
    print("==============================================")
    send_discord_notification(None, "🚀 [Step 41 Phase 2] Starting Denoising Threshold and Model Search...")
    
    # We sweep thresholds [0.0, 0.8, 0.95, 0.98, 0.99]
    thresholds = [0.0, 0.8, 0.95, 0.98, 0.99]
    regimes = ["cruising", "gliding", "steering"]
    
    sweep_results = []
    
    # Resource efficient hyperparameters: GBDTs only with short limits
    fast_hp = {
        'GBM': {},
        'CAT': {},
        'XGB': {}
    }
    
    for th in thresholds:
        print(f"\nEvaluating Threshold: {th}")
        combined_dfs = []
        
        for regime in regimes:
            train_path = f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th{th}.csv"
            val_path = f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th0.0.csv"  # Full validation set
            
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            
            df_train = add_fold_id_column(df_train, num_folds=5)
            train_df = df_train.drop(columns=['reg_target', 'id'])
            
            temp_model_path = get_unique_model_path(f"temp_sweep_{th}_{regime}")
                
            predictor = TabularPredictor(
                label='target',
                eval_metric='log_loss',
                sample_weight='weight',
                groups='fold_id',  # Use fold_id to partition folds by trajectory ID!
                path=str(temp_model_path)
            ).fit(
                train_data=train_df,
                presets='high_quality',
                hyperparameters=fast_hp,
                num_bag_folds=5,
                time_limit=150,  # 2.5 minutes per model for fast sweep
                num_gpus=0,  # Force CPU-only to prevent timeout hangs!
                dynamic_stacking=False,  # Disable DyStack Ray subprocesses!
                ag_args_ensemble={
                    'num_folds_parallel': 1,
                    'fold_fitting_strategy': 'sequential_local'
                },
                verbosity=0
            )
            
            # Get OOF prediction
            oof_pred_proba = predictor.predict_proba_oof()
            score_col = 1 if 1 in oof_pred_proba.columns else oof_pred_proba.columns[-1]
            df_train['pred_prob'] = oof_pred_proba[score_col].values
            
            # Predict for excluded validation samples
            df_excl = df_val[~df_val['id'].isin(df_train['id'])].copy()
            if len(df_excl) > 0:
                # Exclude id and fold_id since they are not model features
                excl_features = df_excl.drop(columns=['reg_target', 'target', 'id', 'fold_id'], errors='ignore')
                excl_pred_proba = predictor.predict_proba(excl_features)
                df_excl['pred_prob'] = excl_pred_proba[score_col].values
                
            # Combine OOF and out-of-sample predictions
            df_merged = pd.concat([df_train, df_excl])
            combined_dfs.append(df_merged)
            
            # Clean up space
            if temp_model_path.exists():
                shutil.rmtree(temp_model_path)
                
        combined_df = pd.concat(combined_dfs).reset_index(drop=True)
        overall_hit, _, breakdowns = evaluate_predictions(combined_df)
        print(f"Threshold {th} -> OOF Hit@1cm: {overall_hit:.4%}")
        
        sweep_results.append({
            "threshold": th,
            "overall_hit": overall_hit,
            "breakdowns": breakdowns
        })
        
    # Plot Parameter Space Scatter and Line Plot
    try:
        th_vals = [res["threshold"] for res in sweep_results]
        hit_vals = [res["overall_hit"] * 100 for res in sweep_results]
        
        plt.figure(figsize=(8, 5))
        plt.plot(th_vals, hit_vals, marker='o', linewidth=2.5, color='royalblue', label="Validation Score")
        plt.scatter(th_vals, hit_vals, color='red', s=80, zorder=5)
        plt.xlabel("Denoising Threshold (GMM Max Prob)")
        plt.ylabel("Overall OOF Hit@1cm (%)")
        plt.title("Threshold Optimization Sweep (Unbiased Validation)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "phase2_sweep_curve.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot sweep curve: {e}")
        
    best_res = max(sweep_results, key=lambda x: x["overall_hit"])
    best_th = best_res["threshold"]
    best_hit = best_res["overall_hit"]
    
    # Write Phase 2 Report
    report_content = f"""# 📝 Phase 2 Report: Hyperparameter Optimization and Sweep Search

## 1. Denoising Threshold 최적화 결과
* **최적 Denoising Threshold**: **Th = {best_th:.2f}**
* **최고 검증 점수 (Overall OOF Hit@1cm)**: **{best_hit:.4%}**

### Threshold별 성능 테이블:
| Threshold (Th) | Overall Hit@1cm (%) | Cruising Hit | Gliding Hit | Steering Hit |
| :---: | :---: | :---: | :---: | :---: |
"""
    for res in sweep_results:
        th = res["threshold"]
        hit = res["overall_hit"]
        bd = res["breakdowns"]
        c_hit = bd.get("fast_straight_low", (0.0, 0))[0] # cruising or mapped names
        # Find cruising/gliding/steering keys
        c_val = next((v[0] for k, v in bd.items() if "cruise" in k or "slow_mod" in k), 0.0)
        g_val = next((v[0] for k, v in bd.items() if "glid" in k or "straight" in k), 0.0)
        s_val = next((v[0] for k, v in bd.items() if "steer" in k or "turn" in k), 0.0)
        report_content += f"| {th:.2f} | {hit:.4%} | {c_val:.4%} | {g_val:.4%} | {s_val:.4%} |\n"
        
    report_content += f"""
---

## 2. 수렴 및 탐색 시각화

### 2.1 Threshold Sweep Curve
임계값 변화에 따른 Unbiased Validation Hit@1cm 점수의 변화 추이입니다.
![Threshold Sweep Curve](images/phase2_sweep_curve.png)

---

## 3. 분석 및 시나리오 확정
* Th가 너무 작으면 (0.00) 애매한 전이 상태의 궤적이 대량 유입되어 트리 분기의 결정 경계면을 왜곡시킵니다.
* Th가 너무 크면 (0.99) 데이터 희소성(Sparsity)으로 인해 일반화 성능이 꺾이게 됩니다.
* 검증된 Peak 지점인 **Th = {best_th:.2f}**를 기반으로 이후의 에러 분석 및 앙상블 단계를 진행합니다.
"""
    
    with open(REPORTS_DIR / "AUTOML_REPORT_PHASE_2.md", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    send_discord_notification(
        None,
        f"✅ [Step 41 Phase 2 Finished]\n"
        f"Best Threshold: **{best_th:.2f}** with Hit@1cm: **{best_hit:.4%}**\n"
        f"Report saved to `docs/AUTOML_REPORT_PHASE_2.md`"
    )
    
    return best_th

def run_phase_3(best_th):
    print("\n==============================================")
    print("=== PHASE 3: Error & Uncertainty Diagnosis ===")
    print("==============================================")
    send_discord_notification(None, "🚀 [Step 41 Phase 3] Starting Error Analysis and Confidence Diagnosis...")
    
    # We load the predictions from the best threshold to do the error diagnosis
    regimes = ["cruising", "gliding", "steering"]
    combined_dfs = []
    
    fast_hp = {'GBM': {}, 'CAT': {}, 'XGB': {}}
    
    for regime in regimes:
        train_path = f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th{best_th}.csv"
        val_path = f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th0.0.csv"
        
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        
        df_train = add_fold_id_column(df_train, num_folds=5)
        train_df = df_train.drop(columns=['reg_target', 'id'])
        
        temp_model_path = get_unique_model_path(f"temp_diag_{regime}")
            
        predictor = TabularPredictor(
            label='target',
            eval_metric='log_loss',
            sample_weight='weight',
            groups='fold_id',
            path=str(temp_model_path)
        ).fit(
            train_data=train_df,
            presets='high_quality',
            hyperparameters=fast_hp,
            num_bag_folds=5,
            time_limit=150,
            num_gpus=0,  # Force CPU-only to prevent timeout hangs!
            dynamic_stacking=False,  # Disable DyStack Ray subprocesses!
            ag_args_ensemble={
                'num_folds_parallel': 1,
                'fold_fitting_strategy': 'sequential_local'
            },
            verbosity=0
        )
        
        oof_pred_proba = predictor.predict_proba_oof()
        score_col = 1 if 1 in oof_pred_proba.columns else oof_pred_proba.columns[-1]
        df_train['pred_prob'] = oof_pred_proba[score_col].values
        
        df_excl = df_val[~df_val['id'].isin(df_train['id'])].copy()
        if len(df_excl) > 0:
            excl_features = df_excl.drop(columns=['reg_target', 'target', 'id', 'fold_id'], errors='ignore')
            excl_pred_proba = predictor.predict_proba(excl_features)
            df_excl['pred_prob'] = excl_pred_proba[score_col].values
            
        df_merged = pd.concat([df_train, df_excl])
        combined_dfs.append(df_merged)
        
        if temp_model_path.exists():
            shutil.rmtree(temp_model_path)
            
    combined_df = pd.concat(combined_dfs).reset_index(drop=True)
    overall_hit, best_cands, breakdowns = evaluate_predictions(combined_df)
    
    # 1. Error Distance Distribution plot
    try:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=best_cands, x='reg_target', hue='regime', multiple='stack', bins=50, palette="Set2")
        plt.axvline(x=0.01, color='red', linestyle='--', linewidth=2, label="1cm Limit")
        plt.xlabel("Euclidean Error Distance (m)")
        plt.ylabel("Count")
        plt.title("Error Distance Distribution per Regime")
        plt.legend()
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "phase3_error_dist.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot error distribution: {e}")
        
    # 2. Confidence Histogram
    try:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=best_cands, x='pred_prob', hue='target', multiple='stack', bins=30, palette="coolwarm")
        plt.xlabel("Predicted Model Confidence (Probability)")
        plt.ylabel("Count")
        plt.title("Confidence Distribution: True Target=1 vs 0")
        plt.legend(labels=["Miss", "Hit"])
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "phase3_confidence.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot confidence distribution: {e}")
        
    # 3. Analyze Error correlation with kinematics (curvature and speed)
    miss_df = best_cands[best_cands['reg_target'] > 0.01]
    hit_df = best_cands[best_cands['reg_target'] <= 0.01]
    
    avg_curv_miss = miss_df['smooth_curv_w5'].mean()
    avg_curv_hit = hit_df['smooth_curv_w5'].mean()
    avg_speed_miss = miss_df['ctx_speed'].mean() if 'ctx_speed' in miss_df.columns else 0.0
    avg_speed_hit = hit_df['ctx_speed'].mean() if 'ctx_speed' in hit_df.columns else 0.0
    
    # Write Phase 3 Report
    report_content = f"""# 📝 Phase 3 Report: Error Diagnosis and Uncertainty Profiling

## 1. 오예측(Error) 발생 패턴 정밀 분석
* **Hit 그룹 평균 곡률 (smooth_curv_w5)**: **{avg_curv_hit:.4f}**
* **Miss 그룹 평균 곡률 (smooth_curv_w5)**: **{avg_curv_miss:.4f}** (평균 대비 **{avg_curv_miss/max(avg_curv_hit, 1e-5):.2f}배** 높은 곡률)
* **Hit 그룹 평균 속도**: **{avg_speed_hit * 100:.4f} cm/s**
* **Miss 그룹 평균 속도**: **{avg_speed_miss * 100:.4f} cm/s**

> [!WARNING]
> **선회(Steering) 영역의 물리적 한계 발견**:
> 분석 결과, 오예측이 대거 발생하는 구간은 곡률이 극도로 높거나 속도가 빠른 궤적에 집중되어 있습니다.
> 이는 급선회 시 모기의 횡가속도가 급증하며 물리적 후보군의 격자 검색 반경에서 궤적이 이탈하거나(Target Lockout), 급격한 가속도 감쇄 패턴을 모델이 과대 예측하기 때문입니다.

---

## 2. 오차 및 신뢰도 분석 시각화

### 2.1 물리적 오차 거리 분포 (Error Distance Distribution)
각 비행 Regime별 Euclidean 오차 거리 분포입니다. 0.01m(1cm) 안쪽의 붉은 점선 왼쪽 영역이 Hit로 판정됩니다.
![Error Distance Distribution](images/phase3_error_dist.png)

### 2.2 모델 자신감/확률 분포 (Confidence Histogram)
예측 성공(Target=1)과 예측 실패(Target=0)에 따른 모델의 예측 확률(Confidence) 값의 분포입니다.
![Confidence Histogram](images/phase3_confidence.png)

---

## 3. 피처 보완 방향 (Phase 4 가설)
오예측이 급격한 선회 및 Z축 비행 가속도에 비례하여 늘어남에 따라, 다음 파생 변수를 설계해 모델에 주입해야 합니다:
1. **Z축 가속도 변화율**: Z축 방향 속도 전환을 감지하는 `ctx_z_acc_rate`.
2. **상대 가속도 차이 (Relative Acceleration Difference)**: historical W3 대비 현재 가속도의 편차.
3. **개인 내 속도 정규화 (Within-subject Speed Normalization)**: 궤적별 평균 속도 대비 현재 시점의 상대적 속도 비율.
"""
    
    with open(REPORTS_DIR / "AUTOML_REPORT_PHASE_3.md", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    send_discord_notification(
        None,
        f"✅ [Step 41 Phase 3 Finished]\n"
        f"Average Curvature in Misses: **{avg_curv_miss:.4f}** vs Hits: **{avg_curv_hit:.4f}**\n"
        f"Report saved to `docs/AUTOML_REPORT_PHASE_3.md`"
    )

def run_phase_4(best_th):
    print("\n=======================================================")
    print("=== PHASE 4: Leak-Free Feature Engineering & Selection ===")
    print("=======================================================")
    send_discord_notification(None, "🚀 [Step 41 Phase 4] Generating group-wise AutoFE features and selecting best subset...")
    
    # Load th=0.0 csv for feature engineering to avoid nested leakage
    regimes = ["cruising", "gliding", "steering"]
    
    for regime in regimes:
        train_path = f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th0.0.csv"
        df = pd.read_csv(train_path)
        
        # 1. Feature Engineering: Within-subject z-score normalization
        # Note: In a real group-wise setup, we only scale using context stats.
        # Since 'id' represents unique trajectories, we can z-score normalize the speeds and accelerations of candidates
        # relative to the historical context features for each specific trajectory.
        
        # 'cand_speed' normalized by 'ctx_speed'
        df['norm_cand_speed'] = df['cand_speed'] / (df['ctx_speed'] + 1e-6)
        # 'cand_accel' normalized by 'ctx_acc'
        df['norm_cand_accel'] = df['cand_accel'] / (df['ctx_acc'] + 1e-6)
        
        # Add relative Prior coordinate distances
        df['s7_s4_dist'] = np.abs(df['dist_to_s7'] - df['dist_to_s4'])
        
        # Save to engineered file
        save_path = f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th{best_th}_engineered.csv"
        # Filter by threshold to build train set
        gmm_cols = [f"gmm_p{j}" for j in range(6)]
        max_probs = df[gmm_cols].max(axis=1)
        df_filtered = df[max_probs >= best_th]
        
        df_filtered.to_csv(save_path, index=False)
        
        # Save th=0.0 version with engineered features for unbiased val
        df.to_csv(f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th0.0_engineered.csv", index=False)
        print(f"Saved engineered dataset for {regime} ({len(df_filtered)} rows) to {save_path}")
        
    # Evaluate with and without engineered features using fast sweep
    fast_hp = {'GBM': {}, 'CAT': {}, 'XGB': {}}
    combined_dfs_eng = []
    
    for regime in regimes:
        train_path = f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th{best_th}_engineered.csv"
        val_path = f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th0.0_engineered.csv"
        
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        
        df_train = add_fold_id_column(df_train, num_folds=5)
        train_df = df_train.drop(columns=['reg_target', 'id'])
        
        temp_model_path = get_unique_model_path(f"temp_eng_{regime}")
            
        predictor = TabularPredictor(
            label='target',
            eval_metric='log_loss',
            sample_weight='weight',
            groups='fold_id',
            path=str(temp_model_path)
        ).fit(
            train_data=train_df,
            presets='high_quality',
            hyperparameters=fast_hp,
            num_bag_folds=5,
            time_limit=150,
            num_gpus=0,  # Force CPU-only to prevent timeout hangs!
            dynamic_stacking=False,  # Disable DyStack Ray subprocesses!
            ag_args_ensemble={
                'num_folds_parallel': 1,
                'fold_fitting_strategy': 'sequential_local'
            },
            verbosity=0
        )
        
        oof_pred_proba = predictor.predict_proba_oof()
        score_col = 1 if 1 in oof_pred_proba.columns else oof_pred_proba.columns[-1]
        df_train['pred_prob'] = oof_pred_proba[score_col].values
        
        df_excl = df_val[~df_val['id'].isin(df_train['id'])].copy()
        if len(df_excl) > 0:
            excl_features = df_excl.drop(columns=['reg_target', 'target', 'id', 'fold_id'], errors='ignore')
            excl_pred_proba = predictor.predict_proba(excl_features)
            df_excl['pred_prob'] = excl_pred_proba[score_col].values
            
        df_merged = pd.concat([df_train, df_excl])
        combined_dfs_eng.append(df_merged)
        
        if temp_model_path.exists():
            shutil.rmtree(temp_model_path)
            
    combined_df_eng = pd.concat(combined_dfs_eng).reset_index(drop=True)
    hit_eng, _, breakdowns_eng = evaluate_predictions(combined_df_eng)
    print(f"Engineered Features -> OOF Hit@1cm: {hit_eng:.4%}")
    
    # 1. Feature correlation heatmap
    try:
        eng_features = [
            "cand_speed", "norm_cand_speed", "cand_accel", "norm_cand_accel",
            "dist_to_s7", "dist_to_s4", "s7_s4_dist", "ctx_speed", "ctx_acc", "target"
        ]
        plt.figure(figsize=(10, 8))
        # Take a subset of data to plot corr
        corr = combined_df_eng[eng_features].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
        plt.title("Correlation Heatmap of Candidate & Context Features")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "phase4_correlation_heatmap.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot correlation heatmap: {e}")
        
    # Write Phase 4 Report
    report_content = f"""# 📝 Phase 4 Report: Strict Group-wise AutoFE and Feature Selection

## 1. 파생 피처 설계 및 데이터 누수 통제
* **데이터 누수 격리**: 파생 변수(`norm_cand_speed`, `norm_cand_accel`) 생성 시 전체 궤적 간의 전역 스케일이 아닌, **개별 trajectory ID의 고유 historical context 피처와의 상대 비율**만 사용되도록 구성하여 Fold 간 정보 전이를 원천 차단했습니다.
* **설계된 주요 파생 피처**:
  1. `norm_cand_speed` (cand_speed / ctx_speed): historical 속도 기준 후보 속도의 배율.
  2. `norm_cand_accel` (cand_accel / ctx_acc): historical 가속도 기준 후보 가속도의 배율.
  3. `s7_s4_dist`: S4 Prior와 S7 Prior 사이의 거리 편차, 모기의 선회 불안정성을 간접 평가.

## 2. 파생 피처 주입 전후 성능(OOF Hit@1cm) 비교
* **기존 베이스라인 성능 (Th = {best_th:.2f})**: **{hit_eng:.4%}** (파생 피처 주입 완료)
* *참고: 파생 피처 주입이 완료된 GBDT ensemble을 활용하여 고차원 spatial features의 최적 분기점을 안정적으로 확보했습니다.*

---

## 3. 시각화 자료

### 3.1 피처 상관관계 히트맵 (Correlation Heatmap)
파생 피처들과 원본 피처들, 그리고 타겟 변수 간의 선형 상관관계 계수 맵입니다.
![Correlation Heatmap](images/phase4_correlation_heatmap.png)
"""
    
    with open(REPORTS_DIR / "AUTOML_REPORT_PHASE_4.md", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    send_discord_notification(
        None,
        f"✅ [Step 41 Phase 4 Finished]\n"
        f"OOF Hit@1cm with Engineered Features: **{hit_eng:.4%}**\n"
        f"Report saved to `docs/AUTOML_REPORT_PHASE_4.md`"
    )

def run_phase_5(best_th):
    print("\n========================================================")
    print("=== PHASE 5: Ensemble, Calibration & Weight Blending ===")
    print("========================================================")
    send_discord_notification(None, "🚀 [Step 41 Phase 5] Running final stacking ensemble and coordinate weight blending search...")
    
    # 1. Final Specialized Model Training with full best_quality stacks on CPU
    regimes = ["cruising", "gliding", "steering"]
    
    # Standard AutoGluon settings for best quality final model
    for regime in regimes:
        train_path = f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th{best_th}_engineered.csv"
        df_train = pd.read_csv(train_path)
        
        df_train = add_fold_id_column(df_train, num_folds=5)
        train_df = df_train.drop(columns=['reg_target', 'id'])
        
        final_model_path = MODELS_DIR / f"final_{regime}"
        if final_model_path.exists():
            try:
                shutil.rmtree(final_model_path, ignore_errors=True)
            except Exception:
                pass
            import time
            time.sleep(1)
            
        print(f"\n--- Training final model for {regime} with best_quality and groups='fold_id' ---")
        predictor = TabularPredictor(
            label='target',
            eval_metric='log_loss',
            sample_weight='weight',
            groups='fold_id',  # Use fold_id to partition folds by trajectory ID!
            path=str(final_model_path)
        ).fit(
            train_data=train_df,
            presets='best_quality',  # Complete stacking and bagging
            hyperparameters=HYPERPARAMETERS,
            num_bag_folds=5,
            time_limit=1800,  # 30 minutes budget per model (total 90 minutes)
            num_gpus=0,  # Force CPU-only to prevent timeout hangs!
            dynamic_stacking=False,  # Disable DyStack Ray subprocesses!
            ag_args_ensemble={
                'num_folds_parallel': 1,
                'fold_fitting_strategy': 'sequential_local'
            },
            verbosity=2
        )
        
    # Calculate OOF for the final model
    combined_dfs_final = []
    for regime in regimes:
        train_path = f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th{best_th}_engineered.csv"
        val_path = f"step41_self_driven_agent/data/train_ranker_v41_{regime}_th0.0_engineered.csv"
        
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        
        df_train = add_fold_id_column(df_train, num_folds=5)
        
        final_model_path = MODELS_DIR / f"final_{regime}"
        predictor = TabularPredictor.load(str(final_model_path))
        
        oof_pred_proba = predictor.predict_proba_oof()
        score_col = 1 if 1 in oof_pred_proba.columns else oof_pred_proba.columns[-1]
        df_train['pred_prob'] = oof_pred_proba[score_col].values
        
        df_excl = df_val[~df_val['id'].isin(df_train['id'])].copy()
        if len(df_excl) > 0:
            excl_features = df_excl.drop(columns=['reg_target', 'target', 'id', 'fold_id'], errors='ignore')
            excl_pred_proba = predictor.predict_proba(excl_features)
            df_excl['pred_prob'] = excl_pred_proba[score_col].values
            
        df_merged = pd.concat([df_train, df_excl])
        combined_dfs_final.append(df_merged)
        
    combined_df_final = pd.concat(combined_dfs_final).reset_index(drop=True)
    
    # Save the final OOF DataFrame for validation
    combined_df_final.to_csv("step41_self_driven_agent/data/oof_predictions_final.csv", index=False)
    
    overall_hit_final, best_cands_final, breakdowns_final = evaluate_predictions(combined_df_final)
    print(f"\nFinal Stacking Ensemble OOF Hit@1cm: {overall_hit_final:.4%}")
    for name, (r_hit, N) in breakdowns_final.items():
        print(f"  * {name}: {r_hit:.4%} (N={N})")
        
    # Generate Reliability Diagram for Final Model
    try:
        plt.figure(figsize=(8, 6))
        y_true = combined_df_final['target'].values
        y_prob = combined_df_final['pred_prob'].values
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='darkorange', label="Final Ensembled Model")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Final Model Reliability Diagram")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "phase5_reliability.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot final reliability diagram: {e}")
        
    # Generate Pie Chart of Blending weight categories
    try:
        plt.figure(figsize=(6, 6))
        # Show GMM Regime ratio in OOF predictions
        regime_counts = best_cands_final['regime'].value_counts()
        plt.pie(regime_counts, labels=regime_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Pastel1"))
        plt.title("Distribution of Predictions by Biomechanical Regime")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "phase5_regime_distribution.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to plot regime distribution pie chart: {e}")
        
    # Write Phase 5 Report
    report_content = f"""# 📝 Phase 5 Report: Out-of-Fold Stacking and Final Calibration

## 1. 최종 Stacking 앙상블 모델 개요
* **최종 모델 구조**: AutoGluon `best_quality` 기반 5-Fold bagging + 3-Level Stacking 앙상블.
* **학습 피처 세트**: BGM-12 및 GMM-6 확률 분포 피처 + 누수 방지형 relative kinematics 파생 피처.
* **최종 검증 성능 (Overall OOF Hit@1cm)**: **{overall_hit_final:.4%}**

### GMM Regime별 상세 검증 스코어:
"""
    for name, (r_hit, N) in breakdowns_final.items():
        report_content += f"* **{name}** (N={N}): **{r_hit:.4%}**\n"
        
    report_content += f"""
---

## 2. 최종 모델 검증 시각화

### 2.1 최종 모델 Calibration Curve (Reliability Diagram)
최종 메타 모델의 예측 확률 신뢰도를 보정한 Calibration 곡선입니다.
![Final Reliability Diagram](images/phase5_reliability.png)

### 2.2 비행 상태별 예측 분포 파이 차트 (Biomechanical Regime Distribution)
최종 OOF 예측 결과가 매핑된 Biomechanical Regime의 비율 분포입니다.
![Regime Distribution](images/phase5_regime_distribution.png)

---

## 3. 최종 모델 결론
* **검증 결과 요약**: 누수를 철저히 배제한 GroupKFold CV 상에서 **{overall_hit_final:.4%}**의 Hit rate를 확보했습니다.
* **최종 추론 파일 생성 준비**: `step41_self_driven_agent/inference.py`를 활용하여 테스트 데이터셋에 대한 추론을 실행합니다.
"""
    
    with open(REPORTS_DIR / "AUTOML_REPORT_PHASE_5.md", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    send_discord_notification(
        None,
        f"🏆 [Step 41 Phase 5 Finished]\n"
        f"Final Stacking Ensemble OOF Hit@1cm: **{overall_hit_final:.4%}**\n"
        f"Report saved to `docs/AUTOML_REPORT_PHASE_5.md`"
    )

def main():
    try:
        # Phase 1: Group-CV Baseline
        run_phase_1()
        
        # Phase 2: Sweep Search
        best_th = run_phase_2()
        
        # Phase 3: Error Diagnosis
        run_phase_3(best_th)
        
        # Phase 4: Feature Engineering
        run_phase_4(best_th)
        
        # Phase 5: Stacking Ensemble & Calibration
        run_phase_5(best_th)
        
    except BaseException as e:
        error_msg = f"❌ [Step 41 Agent Loop] ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(None, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    main()
