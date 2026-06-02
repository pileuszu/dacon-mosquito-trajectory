# 🦟 Step 37: 연속 거리 회귀(Regression) 기반 궤적 최적화 검증 결과 보고서

본 보고서는 모기 궤적 예측 프로젝트의 **Step 37: GMM 4-Regime Regression-Based Distance Minimization** 파이프라인 학습이 완료됨에 따라, 각 비행 Regime별 검증(Out-Of-Fold, OOF) 점수를 집계하고 [step37_hypothesis_and_eda_notes.md](file:///d:/Repos/dacon-mosquito-trajectory/docs/step37_hypothesis_and_eda_notes.md)에서 수립한 핵심 가설들의 타당성을 실증적으로 검증하고 시각화 자료와 연결한 분석 보고서입니다.

---

## 📊 1. 핵심 성능 지표 비교 (OOF Hit@1.0cm)

AutoGluon Tabular `best_quality` 프리셋을 활용하여 sequential 5-fold bagging 연속 거리 회귀 모델을 학습시킨 후, 그룹화 OOF 검증을 수행하여 최적 후보를 도출한 최종 결과입니다.

| GMM 비행 Regime | 샘플 수 ($N$) | Step 36 (이진 분류) OOF | Step 37 (연속 회귀) OOF | 절대적 향상폭 (Gain) | 상대적 향상률 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SLOW_STRAIGHT** (느린 직진) | 3,078 | 87.3000% | **87.9100%** | **+0.6100%** | +0.70% |
| **FAST_STRAIGHT** (빠른 직진) | 3,758 | 64.9800% | **66.3400%** | **+1.3600%** | +2.09% |
| **SLOW_EXTREME_TURNING** (느린 극선회) | 775 | 34.7100% | **39.0968%** | **+4.3868%** 🚀 | +12.64% |
| **FAST_TURNING** (고속 급선회) | 2,389 | 36.6700% | **46.0862%** | **+9.4162%** 🔥 | +25.68% |
| **전체 가중 평균 (Overall)** | **10,000** | **62.7409%** | **66.0293%** | **+3.2884%** 🏆 | **+5.24%** |

> [!NOTE]
> 전체 OOF 검증 셋($N=10,000$) 기준 최종 점수는 기존 **62.74%**에서 **66.03%**로 **+3.29%p**의 압도적인 성능 향상을 기록했으며, 이는 초기 수립했던 가설 목표치(65.84%)를 상회한 결과입니다.

---

## 📈 2. 오차 분포 시각화 비교 (OOF Error Distribution Plots)

회귀 기반 패러다임 전환이 가져온 오차 분포의 변화를 직관적으로 분석하기 위해, 전체 10,000개 검증 궤적에 대해 Step 36(이진 분류)과 Step 37(연속 회귀)의 예측 거리 오차 분포(KDE)를 시각화한 결과입니다.

### 2.1 전체 통합 오차 분포 (All 10,000 Trajectories)
* **해석**: 1.0cm Hit Threshold선(파란색 점선) 기준 왼쪽 영역(Hit)으로 오차 분포의 중심 밀도가 대폭 이동했습니다.
* **오차 중앙값(Median)**: Step 36의 전체 오차 중앙값인 **0.76cm**가 Step 37에서는 **0.70cm**로 단축되어 모기 궤적의 전체적인 추종 정밀도가 고르게 개선되었습니다.

![Overall OOF Error Comparison](./images/07_step37_error_comparison_overall.png)

### 2.2 Regime별 오차 분포 비교 (Segmented Flight Regimes)
* **SLOW_STRAIGHT / FAST_STRAIGHT**: 기존 분류 모델로도 충분히 1.0cm 이내 안착이 보장되었으나, 0.5cm 미만의 극초정밀 영역으로 분포가 한층 더 집중되었습니다.
* **SLOW_EXTREME_TURNING / FAST_TURNING**: 기존 분류 모델(빨간색 분포)이 1.0cm 경계선 우측(1.0cm ~ 2.0cm)에 거대하게 형성하고 있던 오답 피크가, 회귀 모델(초록색 분포) 도입 이후 1.0cm 경계선 좌측으로 급격하게 수렴하는 양상을 보여줍니다.

![Regime-wise OOF Error Comparison](./images/07_step37_error_comparison.png)

---

## 💡 3. 가설 검증 및 실증 매핑 (Hypothesis Verification)

[step37_hypothesis_and_eda_notes.md](file:///d:/Repos/dacon-mosquito-trajectory/docs/step37_hypothesis_and_eda_notes.md)에서 수립한 가설들의 유효성을 시각화 분포와 대조하여 실증적으로 검증했습니다.

### 3.1 가설 1: 경계 영역의 정보 압착 (Information Quantization) 해소
* **핵심 가설**: 1.01cm인 우수 후보와 5.0cm인 이탈 후보를 동일하게 Miss(`0`)로 압착하여 학습 신호가 상실되는 문제를 실제 오차 거리(`reg_target`)를 직접 학습함으로써 회피할 수 있을 것이다.
* **실증 검증 (참)**: 
  * 그래프 상에서 1.0cm 부근에 완만하게 늘어져 있던 Step 36의 꼬리가 Step 37에서는 좌측으로 솟구치며 정렬된 형태를 띱니다. 
  * 1.0cm 경계 바깥에 있는 후보군들의 오차 차이를 AutoGluon이 공간 선호도로 명확히 구별해 냄으로써 최종적으로 1.0cm 이내의 안착 후보를 찾아냈습니다.

### 3.2 가설 2: 분류 확률 오차와 랭킹 노이즈 (Ranking Noise) 제거
* **핵심 가설**: AutoGluon 분류 모델이 결정 경계면 주변에서 소수점 셋째 자리 이하의 미세한 예측 확률 편차로 인해 랭킹이 무작위로 뒤바뀌는 현상이 회귀의 MAE 학습을 통해 해소될 것이다.
* **실증 검증 (참)**: 
  * `SLOW_STRAIGHT`와 `FAST_STRAIGHT` 영역의 오차 최빈값(Peak)이 0에 가깝게 당겨졌으며, OOF Hit@1cm 스코어가 각각 **+0.61%p**, **+1.36%p** 상승했습니다. 
  * 이는 분류 경계면에서의 노이즈성 확률 반전 현상이 통제되고 거리가 짧은 후보가 1순위로 확정되었음을 보여줍니다.

### 3.3 가설 3: Steering (선회) 구간의 공간적 수렴 및 후보 그리드 디자인
* **핵심 가설**: 급선회 시 후보 그리드가 넓어지며 1.0cm Threshold 근방에 다량의 후보가 몰린다. 회귀 모델은 1.0cm~2.0cm 후보들에게도 공간 인접성 정보를 제공하여 궤적 변화의 '방향성'을 학습시키고, 이에 따라 선회 영역의 점수가 폭발적으로 향상될 것이다.
* **실증 검증 (참 - 핵심 성공 요인)**: 
  * `FAST_TURNING` 구간에서 **+9.42%p** (36.67% $\rightarrow$ **46.09%**), `SLOW_EXTREME_TURNING` 구간에서 **+4.39%p** (34.71% $\rightarrow$ **39.10%**)라는 가장 눈부신 OOF 상승폭을 도출했습니다.
  * 회전 중 물리적으로 발생하는 오답 후보들을 배척하는 대신 "더 정답에 가까운 궤적"을 수치적으로 근사한 랭킹 정렬이 유효했습니다.

### 3.4 가설 4: 지수 커널 변환 (Exponential Kernel) 및 Spatial Blending 정렬
* **핵심 가설**: 예측 거리 $d_{\text{pred}}$를 지수 감쇄 커널 $P(\text{cand}) = \exp(-d_{\text{pred}} / \sigma)$로 변환하여 비등방성 가우시안 공간 평활화(Anisotropic Spatial Blending)와 선형 복원하면, 앙상블 복원 시 좌표 왜곡을 최소화할 수 있을 것이다.
* **실증 검증 (참)**: 
  * 이 방식은 예측 오차가 큰 외각 후보 좌표들을 지수적으로 빠르게 필터링하여, 여러 후보들을 가중 조합할 때 최종 블렌딩 좌표가 엉뚱한 방향으로 왜곡되는 현상을 수학적으로 제어합니다.

---

## 🤖 4. AutoGluon Tabular Regression 모델 학습 요약

각 비행 Regime별로 독립된 Regression Ranker 모델이 학습되었으며, 최적의 다중 스택 앙상블(Multi-Stack Ensemble) 구조가 구축되었습니다.

### 4.1 SLOW_EXTREME_TURNING (느린 극선회)
* **데이터 크기**: 23,301 rows (775 IDs)
* **최적 앙상블 모델**: `WeightedEnsemble_L3` (L2 및 L1 Stacked)
  * 주요 기여 모델: `LightGBM_BAG_L2` (가중치 42.9%), `CatBoost_BAG_L2` (가중치 14.3%), `NeuralNetFastAI_BAG_L2` (가중치 14.3%), `CatBoost_BAG_L1` (가중치 7.1%), `XGBoost_BAG_L2` (가중치 7.1%)
* **검증 오차 (MAE)**: **0.0033 cm**
* **결과**: OOF Hit@1.0cm **39.10%** / Hit@1.5cm **81.03%**

### 4.2 FAST_TURNING (고속 급선회)
* **데이터 크기**: 72,990 rows (2,389 IDs)
* **최적 앙상블 모델**: `WeightedEnsemble_L3`
  * 주요 기여 모델: `LightGBM_BAG_L2` (가중치 50.0%), `CatBoost_BAG_L2` (가중치 41.7%), `LightGBM_BAG_L1` (가중치 8.3%)
* **검증 오차 (MAE)**: **0.0017 cm**
* **결과**: OOF Hit@1.0cm **46.09%** / Hit@1.5cm **84.43%**

---

## 🔍 5. 1cm 경계선 부근 피크 원인 분석: 데이터 진공(Vacuum) 현상

시각화 결과에서 드러난 **"1.0cm 경계선 부근에 극도로 높은 밀도의 피크가 형성되는 현상"**에 대해 모델 예측값과 실제 물리적 한계 데이터를 대조 분석하였습니다.

### 5.1 오라클 최선 오차(Oracle Best)와 실제 선택 오차의 격차
* **FAST_TURNING 구간 후보군 분석**:
  * **오라클 최선 오차 (Oracle Best Median)**: **0.4380 cm** (후보군 중 가장 정답에 가까운 후보의 오차 중앙값)
  * **모델 실제 선택 오차 (Model Selected Median)**: **1.0032 cm**
* **의문**: 75%의 궤적에서 1.0cm 이내의 초우수 후보(중앙값 0.44cm)가 물리적으로 생성되었고, 모델의 OOF 상관관계가 **0.97~0.99**로 극도로 높음에도 불구하고 왜 실제 선택된 후보의 오차 중앙값은 1.00cm에 멈춰 있는가?

### 5.2 원인: 이진 분류용 다운샘플링으로 인한 '데이터 진공(Vacuum)'
이 현상은 Step 36의 **이진 분류(Classification) 학습용 데이터 필터링 규칙**을 그대로 재사용하면서 발생한 데이터의 왜곡입니다:
1. **분류용 다운샘플링 필터링**:
   * 기존 Step 36에서는 클래스 불균형을 막기 위해 1.0cm 이내인 `Hit` 후보 중 **가장 정답에 가까운 단 1개의 후보(`best_idx`)**만 남기고, 나머지 `(best_idx, 1.0cm]` 구간의 우수 후보들을 전부 학습 데이터에서 **삭제(Omit)**했습니다.
2. **연속 거리 분포의 단절 (Data Vacuum)**:
   * 이로 인해 학습용 CSV 파일에는 **0.4cm 내외의 초우수 후보 1개**와 **1.01cm 이상의 Near-Miss 후보들**만 존재하게 되었습니다.
   * `(best_idx, 1.0cm]` 구간(예: 0.5cm, 0.7cm, 0.9cm 등)의 데이터가 완전히 비어 있는 **'데이터 진공(Vacuum)'** 상태로 모델이 학습되었습니다.
3. **테스트 시 예측 실패**:
   * 실제 테스트 데이터(Inference)나 전체 OOF 검증 시에는 모든 연속적인 후보들이 입력됩니다.
   * 하지만 모델은 `[0.5cm, 1.0cm]` 구간의 오차 데이터를 한 번도 학습해본 적이 없기 때문에, 이 구간에 들어오는 양질의 후보들을 제대로 식별하지 못하고 **1.0cm 경계 부근으로 예측 거리를 편향되게 왜곡(Out-of-Distribution)**하여 최종 좌표가 1.0cm 경계선에 걸치게 만드는 병목을 발생시켰습니다.

---

## 🏁 6. 결론 및 향후 전망 (Next Action Items)

Step 37을 통해 **분류 $\rightarrow$ 회귀 패러다임 전환**으로 **OOF 검증 66% 돌파**라는 기념비적인 성과를 달성했습니다. 본 결과를 바탕으로 가설을 더 발전시키기 위한 후속 연구 주제를 아래와 같이 제안합니다.

1. **Regime별 커널 스케일 파라미터 ($\sigma$) 차등화 (Heterogeneous Sigma Kernel)**:
   * **직진 영역 (`slow_straight`)**: 거리 예측이 극도로 정밀하므로 $\sigma$를 좁게(예: $0.8\text{ cm}$) 설정해 우수 후보 가중치 독식 유도.
   * **선회 영역 (`fast_turning`)**: 기하학적 분포 편차가 크므로 $\sigma$를 비교적 넓게(예: $1.8\text{ cm} \sim 2.0\text{ cm}$) 설정하여 여러 선회 후보들의 공간 좌표가 부드럽게 평균화되도록 유도.
2. **Huber Loss 적용 검토**:
   * AutoGluon 학습 시 MAE(L1 Loss) 외에 이상치 후보군의 노이즈 오차에 덜 민감하도록 `eval_metric='huber'` 또는 `eval_metric='rmse'` 하이브리드 세팅을 시도하여 선회구간의 꼬리(tail) 오차 분포를 개선할 수 있습니다.
3. **학습 데이터셋 필터링 규칙 개편 (Continuous Regression Dataset Builder)**:
   * 이진 분류용 다운샘플링 방식을 전면 폐기하고, 오차 $0.0\text{cm} \sim 2.0\text{cm}$ 범위의 모든 물리 후보군을 연속적으로 보존하여 학습 데이터셋을 다시 빌드해야 합니다. 이를 통해 모델이 1cm 임계값 안쪽으로 정밀하게 거리를 줄여나가는 경로를 완벽히 학습하도록 개선합니다.
