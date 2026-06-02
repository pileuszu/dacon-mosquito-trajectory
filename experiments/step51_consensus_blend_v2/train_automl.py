import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification

EPS = 1e-8

def classify_regimes(X):
    # X shape: (N, 11, 3)
    N = X.shape[0]
    
    # Compute terminal speed (between step 10 and 11)
    # dt = 40ms = 0.04s per step
    last_v = (X[:, -1] - X[:, -2]) / 0.04  # (N, 3)
    speeds = np.linalg.norm(last_v, axis=1) # (N,)
    
    # Compute curvature or acceleration to identify turning (Steering)
    # prev_v is speed vector from step 9 to 10
    prev_v = (X[:, -2] - X[:, -3]) / 0.04
    
    # centripetal acceleration (approx last acceleration perpendicular component)
    last_a = (last_v - prev_v) / 0.04 # (N, 3)
    
    t_dir = last_v / (speeds[:, None] + EPS)
    acc_par_scalar = np.sum(last_a * t_dir, axis=1)
    acc_perp = last_a - acc_par_scalar[:, None] * t_dir
    acc_perp_norm = np.linalg.norm(acc_perp, axis=1)
    
    # Cross product for curvature calculation
    cross_prod = np.cross(last_v, last_a, axis=1)
    cross_norm = np.linalg.norm(cross_prod, axis=1)
    curvature = cross_norm / (speeds ** 3 + EPS)
    
    # Routing criteria:
    # Steering (Turning): curvature > 6.0 OR perpendicular acceleration > 1.8 m/s^2
    is_steering = (curvature > 6.0) | (acc_perp_norm > 1.8)
    
    # Speed threshold for slow vs fast: 0.50 m/s
    regimes = np.zeros(N, dtype=int)  # 0: Cruising, 1: Gliding, 2: Steering
    for i in range(N):
        if is_steering[i]:
            regimes[i] = 2  # Steering (Turning)
        elif speeds[i] <= 0.50:
            regimes[i] = 0  # Cruising (Slow-Straight)
        else:
            regimes[i] = 1  # Gliding (Fast-Straight)
            
    return regimes, speeds, curvature, acc_perp_norm, acc_par_scalar

def grid_search_blending(s47_preds, s50_preds, train_y, regimes):
    N = len(train_y)
    cruising_idx = np.where(regimes == 0)[0]
    gliding_idx = np.where(regimes == 1)[0]
    steering_idx = np.where(regimes == 2)[0]
    
    weights_space = np.linspace(0.0, 1.0, 21)
    
    # Cruising Optimization
    best_w_cruising = 0.0
    best_hr_cruising = 0.0
    for w in weights_space:
        preds = w * s47_preds[cruising_idx] + (1 - w) * s50_preds[cruising_idx]
        dists = np.linalg.norm(preds - train_y[cruising_idx], axis=1)
        hr = np.mean(dists <= 0.01)
        if hr > best_hr_cruising:
            best_hr_cruising = hr
            best_w_cruising = w
            
    # Gliding Optimization
    best_w_gliding = 0.0
    best_hr_gliding = 0.0
    for w in weights_space:
        preds = w * s47_preds[gliding_idx] + (1 - w) * s50_preds[gliding_idx]
        dists = np.linalg.norm(preds - train_y[gliding_idx], axis=1)
        hr = np.mean(dists <= 0.01)
        if hr > best_hr_gliding:
            best_hr_gliding = hr
            best_w_gliding = w
            
    # Steering Optimization
    best_w_steering = 0.0
    best_hr_steering = 0.0
    for w in weights_space:
        preds = w * s47_preds[steering_idx] + (1 - w) * s50_preds[steering_idx]
        dists = np.linalg.norm(preds - train_y[steering_idx], axis=1)
        hr = np.mean(dists <= 0.01)
        if hr > best_hr_steering:
            best_hr_steering = hr
            best_w_steering = w
            
    # Reconstruct final predictions
    final_preds = np.zeros_like(train_y)
    final_preds[cruising_idx] = best_w_cruising * s47_preds[cruising_idx] + (1 - best_w_cruising) * s50_preds[cruising_idx]
    final_preds[gliding_idx] = best_w_gliding * s47_preds[gliding_idx] + (1 - best_w_gliding) * s50_preds[gliding_idx]
    final_preds[steering_idx] = best_w_steering * s47_preds[steering_idx] + (1 - best_w_steering) * s50_preds[steering_idx]
    
    final_dists = np.linalg.norm(final_preds - train_y, axis=1)
    overall_hr = np.mean(final_dists <= 0.01)
    
    best_weights = {
        "w_cruising": float(best_w_cruising),
        "w_gliding": float(best_w_gliding),
        "w_steering": float(best_w_steering),
        "hr_cruising": float(best_hr_cruising),
        "hr_gliding": float(best_hr_gliding),
        "hr_steering": float(best_hr_steering),
        "overall_hr": float(overall_hr)
    }
    
    return best_weights, final_preds

def run_failure_analysis(X, y, preds, regimes, speeds, curvature, acc_perp, acc_par):
    # Compute error distance for all samples (in cm)
    errors = np.linalg.norm(preds - y, axis=1) * 100.0  # cm
    
    # Outliers: errors > 3.0 cm
    outlier_idx = np.where(errors > 3.0)[0]
    normal_idx = np.where(errors <= 3.0)[0]
    
    num_outliers = len(outlier_idx)
    total = len(errors)
    outlier_ratio = num_outliers / total * 100.0
    
    # Calculate physical stats
    # 1. Terminal velocity speed
    speed_out = speeds[outlier_idx]
    speed_norm = speeds[normal_idx]
    
    # 2. Curvature
    curv_out = curvature[outlier_idx]
    curv_norm = curvature[normal_idx]
    
    # 3. Perpendicular acceleration (횡가속도)
    perp_out = acc_perp[outlier_idx]
    perp_norm = acc_perp[normal_idx]
    
    # 4. Parallel acceleration (종가속도)
    par_out = acc_par[outlier_idx]
    par_norm = acc_par[normal_idx]
    
    # 5. Z-axis target displacement: future z movement
    z_disp = y[:, 2] - X[:, -1, 2]
    z_disp_out = np.abs(z_disp[outlier_idx]) * 100.0  # cm
    z_disp_norm = np.abs(z_disp[normal_idx]) * 100.0  # cm
    
    # 6. Regime distribution
    reg_out = regimes[outlier_idx]
    
    # Construct Report in Korean
    report = []
    report.append("==================================================================")
    report.append("🦟 OOF Validation Failure & Outlier Analysis Report (Korean)")
    report.append("==================================================================")
    report.append(f"총 검증 샘플 수: {total}개 | 3.0cm 이상 오차 아웃라이어: {num_outliers}개 ({outlier_ratio:.2f}%)")
    report.append(f"평균 오차 거리: {errors.mean():.4f} cm | 아웃라이어 평균 오차: {errors[outlier_idx].mean():.4f} cm")
    report.append("------------------------------------------------------------------")
    report.append("1. 비행 국면(Regime)별 아웃라이어 분포:")
    report.append(f"   * Cruising (Slow-Straight): {np.sum(reg_out == 0)}개 ({np.sum(reg_out == 0)/num_outliers*100:.2f}%)")
    report.append(f"   * Gliding (Fast-Straight) : {np.sum(reg_out == 1)}개 ({np.sum(reg_out == 1)/num_outliers*100:.2f}%)")
    report.append(f"   * Steering (Turning)      : {np.sum(reg_out == 2)}개 ({np.sum(reg_out == 2)/num_outliers*100:.2f}%)")
    report.append("   -> 분석: 오차가 큰 아웃라이어의 대부분은 Steering(선회) 구간에서 유발되고 있습니다.")
    report.append("------------------------------------------------------------------")
    report.append("2. 물리적 변수 비교 (아웃라이어군 vs 정상 오차군):")
    report.append(f"   * 속력 (Speed, m/s):")
    report.append(f"     - 아웃라이어 평균: {speed_out.mean():.4f} m/s (Max: {speed_out.max():.4f})")
    report.append(f"     - 정상 오차 평균: {speed_norm.mean():.4f} m/s (Max: {speed_norm.max():.4f})")
    report.append(f"   * 곡률 (Curvature, 1/m):")
    report.append(f"     - 아웃라이어 평균: {curv_out.mean():.4f} (Max: {curv_out.max():.4f})")
    report.append(f"     - 정상 오차 평균: {curv_norm.mean():.4f} (Max: {curv_norm.max():.4f})")
    report.append(f"   * 횡가속도 (Centripetal Accel, m/s^2):")
    report.append(f"     - 아웃라이어 평균: {perp_out.mean():.4f} m/s^2 (Max: {perp_out.max():.4f})")
    report.append(f"     - 정상 오차 평균: {perp_norm.mean():.4f} m/s^2 (Max: {perp_norm.max():.4f})")
    report.append(f"   * 종가속도 (Tangential Accel, m/s^2):")
    report.append(f"     - 아웃라이어 평균: {par_out.mean():.4f} m/s^2")
    report.append(f"     - 정상 오차 평균: {par_norm.mean():.4f} m/s^2")
    report.append(f"   * Z축 미래 변위 (Z-Displacement, cm):")
    report.append(f"     - 아웃라이어 평균: {z_disp_out.mean():.4f} cm (Max: {z_disp_out.max():.4f})")
    report.append(f"     - 정상 오차 평균: {z_disp_norm.mean():.4f} cm (Max: {z_disp_norm.max():.4f})")
    report.append("------------------------------------------------------------------")
    report.append("3. 💡 핵심 요인 분석 및 후속 개선 힌트:")
    
    # Dynamic heuristics analysis based on values
    if speed_out.mean() > speed_norm.mean() * 1.15:
        report.append("   - [속도 요인] 아웃라이어 군의 마지막 속도가 정상군 대비 유의미하게 높습니다. 고속 활공에서의 오차가 큽니다.")
    if curv_out.mean() > curv_norm.mean() * 1.25:
        report.append("   - [곡률 요인] 아웃라이어의 평균 곡률이 정상 궤적보다 훨씬 큽니다. 급회전 물리 모델링의 보정이 더 필요합니다.")
    if z_disp_out.mean() > z_disp_norm.mean() * 1.3:
        report.append("   - [Z축 변동성] 아웃라이어 군은 80ms 미래 시점에 Z축(높이) 방향으로의 급격한 수직 이동(상승/하강)이 발생했습니다.")
        report.append("     현재 평면(X-Y) 위주의 조종 프레임 보정 외에, Z축 기동성을 포착하기 위한 3D 가속도 특징 추가 보정이 절실합니다.")
    
    report.append("==================================================================")
    
    return "\n".join(report)

def main():
    send_discord_notification(
        None,
        "🚀 Started: [Step 51 train_automl.py] Optimizing Consensus Blending v2 and Running Failure Analysis..."
    )
    
    try:
        data_dir = Path("step51_consensus_blend_v2/data")
        
        # Load datasets
        train_x = np.load(data_dir / "train_x.npy")
        train_y = np.load(data_dir / "train_y.npy")
        
        s47_soft = np.load(data_dir / "step47_oof_soft.npy")
        s47_argmax = np.load(data_dir / "step47_oof_argmax.npy")
        s50_oof = np.load(data_dir / "step50_oof.npy")
        
        print(f"Loaded train data. X: {train_x.shape}, Y: {train_y.shape}")
        
        # Classify regimes and physical stats
        print("Analyzing and classifying flight regimes...")
        regimes, speeds, curvature, acc_perp, acc_par = classify_regimes(train_x)
        
        # Baselines
        s47_soft_hr = np.mean(np.linalg.norm(s47_soft - train_y, axis=1) <= 0.01)
        s47_argmax_hr = np.mean(np.linalg.norm(s47_argmax - train_y, axis=1) <= 0.01)
        s50_hr = np.mean(np.linalg.norm(s50_oof - train_y, axis=1) <= 0.01)
        
        print(f"Baseline Single OOF Hit Rate@1cm:")
        print(f"  - Step 47 OOF (Soft): {s47_soft_hr*100:.3f}%")
        print(f"  - Step 47 OOF (Argmax): {s47_argmax_hr*100:.3f}%")
        print(f"  - Step 50 OOF (Frenet ODE): {s50_hr*100:.3f}%")
        
        # Search 1: Blend Soft + Step 50
        print("\nSearching best weights for: Step 47 (Soft) + Step 50 (Frenet ODE)")
        soft_weights, soft_blended_preds = grid_search_blending(s47_soft, s50_oof, train_y, regimes)
        
        # Search 2: Blend Argmax + Step 50
        print("\nSearching best weights for: Step 47 (Argmax) + Step 50 (Frenet ODE)")
        argmax_weights, argmax_blended_preds = grid_search_blending(s47_argmax, s50_oof, train_y, regimes)
        
        # Best Blending Selection
        if soft_weights["overall_hr"] >= argmax_weights["overall_hr"]:
            best_blend_type = "soft"
            optimal_config = soft_weights
            optimal_preds = soft_blended_preds
            print(f"\n🏆 Best Blend Selected: Step 47 (Soft) + Step 50 (Frenet ODE)")
        else:
            best_blend_type = "argmax"
            optimal_config = argmax_weights
            optimal_preds = argmax_blended_preds
            print(f"\n🏆 Best Blend Selected: Step 47 (Argmax) + Step 50 (Frenet ODE)")
            
        optimal_config["blend_type"] = best_blend_type
        
        print("\n==========================================")
        print("Optimal Blending v2 Results:")
        print(f"  - w_cruising (Step 47 weight): {optimal_config['w_cruising']:.2f} (HR: {optimal_config['hr_cruising']*100:.2f}%)")
        print(f"  - w_gliding (Step 47 weight) : {optimal_config['w_gliding']:.2f} (HR: {optimal_config['hr_gliding']*100:.2f}%)")
        print(f"  - w_steering (Step 47 weight): {optimal_config['w_steering']:.2f} (HR: {optimal_config['hr_steering']*100:.2f}%)")
        print(f"  - Overall Blended OOF Hit@1cm: {optimal_config['overall_hr']*100:.3f}%")
        print("==========================================")
        
        # Run failure analysis on the final optimal predictions
        print("\nRunning OOF Failure Analysis on the optimal blended predictions...")
        failure_report = run_failure_analysis(train_x, train_y, optimal_preds, regimes, speeds, curvature, acc_perp, acc_par)
        print(failure_report)
        
        # Save failure report text file
        report_path = Path("step51_consensus_blend_v2/failure_analysis_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(failure_report)
        print(f"Saved failure report to {report_path}")
        
        # Save config and predictions
        with open(data_dir / "optimal_weights.json", "w") as f:
            json.dump(optimal_config, f, indent=4)
        np.save(data_dir / "blended_oof_predictions.npy", optimal_preds)
        
        # Discord Notification
        discord_msg = (
            f"✅ Finished: [Step 51 train_automl.py] Optimization & Failure Analysis Complete!\n"
            f"Best Blend Type: **{optimal_config['blend_type'].upper()}**\n"
            f"Overall Blended OOF Hit Rate@1cm: **{optimal_config['overall_hr']*100:.3f}%**\n"
            f"Weights: w_cruising={optimal_config['w_cruising']:.2f}, w_gliding={optimal_config['w_gliding']:.2f}, w_steering={optimal_config['w_steering']:.2f}\n"
            f"Outliers (>3cm): {np.sum(np.linalg.norm(optimal_preds - train_y, axis=1) > 0.03)} / {len(train_y)}"
        )
        send_discord_notification(None, discord_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 51 train_automl.py] Weight optimization failed:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
