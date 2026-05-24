# 🦟 Step 35 Report: GMM 4-Regime Specialized Ranker

본 보고서는 GMM(Gaussian Mixture Model) 군집 분류 기반의 4개 비행 모드(Regime)별 독립 Ranker 학습 실험(**Step 35**)의 최종 모델링 결과, 오차 분석 및 향후 개선 방향을 기록한 문서입니다.

---

## 📊 1. 실험 요약 및 목적
* **핵심 가설**: 모기의 비행 상태를 속도와 곡률(회전 여부)에 따라 4가지 Regime으로 세분화하고, 각 영역에 특화된 전용 AutoML Ranker 모델을 학습시키면 예측 오차(Target Calibration Mismatch)를 보정하고 리더보드 성능을 극대화할 수 있을 것이다.
* **군집 분류 방법론**: 비행 속도, 가속도, 곡률, 횡가속도, Z축 속도/가속도, Saccade 확률 등 8차원 물리 피처 공간에서 GMM 소프트 군집화 수행.
* **데이터 파티셔닝**:
  * **SLOW_STRAIGHT (Cruising)**: 3,315개 궤적 (33.15%)
  * **FAST_STRAIGHT (Gliding)**: 3,697개 궤적 (36.97%)
  * **SLOW_EXTREME_TURNING (Extreme Turning)**: 854개 궤적 (8.54%)
  * **FAST_TURNING (Saccadic Maneuver)**: 2,134개 궤적 (21.34%)

---

## 📈 2. 4개 비행 모드별 학습 결과 (AutoML Leaderboard)

### ① SLOW_STRAIGHT (느린 직진 / Hovering)
* **데이터 수**: 3,315개 궤적
* **Group-wise OOF Hit@1cm**: **86.1598%** 🏆
* **Best Model**: `WeightedEnsemble_L3` (Validation AUC: **0.999657**)
* **학습 시간**: 2,218.60초

| 순위 | 모델명 | Validation AUC | 추론 시간 (val, sec) | 학습 시간 (sec) |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **WeightedEnsemble_L3** | **0.999657** | **4.123516** | **2218.598568** |
| 2 | CatBoost_BAG_L2 | 0.999586 | 2.803834 | 1818.074545 |
| 3 | XGBoost_BAG_L2 | 0.999546 | 3.029408 | 1804.451752 |
| 4 | LightGBM_BAG_L2 | 0.999415 | 2.828270 | 1797.074466 |
| 5 | WeightedEnsemble_L2 | 0.999224 | 2.342898 | 1746.827859 |

---

### ② FAST_STRAIGHT (빠른 직진 / Cruise)
* **데이터 수**: 3,697개 궤적
* **Group-wise OOF Hit@1cm**: **64.5822%**
* **Best Model**: `WeightedEnsemble_L3` (Validation AUC: **0.980013**)
* **학습 시간**: 2,636.76초

| 순位 | 모델명 | Validation AUC | 추론 시간 (val, sec) | 학습 시간 (sec) |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **WeightedEnsemble_L3** | **0.980013** | **7.023643** | **2636.761846** |
| 2 | XGBoost_BAG_L2 | 0.979427 | 5.123742 | 1795.461764 |
| 3 | CatBoost_BAG_L2 | 0.979333 | 4.651864 | 1848.561070 |
| 4 | LightGBM_BAG_L2 | 0.979160 | 4.725932 | 1787.624894 |
| 5 | NeuralNetFastAI_BAG_L2 | 0.977877 | 6.306367 | 2530.344014 |

---

### ③ SLOW_EXTREME_TURNING (느린 제자리 극선회)
* **데이터 수**: 854개 궤적 (데이터 희소 영역)
* **Group-wise OOF Hit@1cm**: **32.6452%**
* **Best Model**: `WeightedEnsemble_L3` (Validation AUC: **0.900498**)
* **학습 시간**: 615.36초

| 순위 | 모델명 | Validation AUC | 추론 시간 (val, sec) | 학습 시간 (sec) |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **WeightedEnsemble_L3** | **0.900498** | **1.030372** | **615.359198** |
| 2 | CatBoost_BAG_L2 | 0.899167 | 0.996637 | 605.190906 |
| 3 | WeightedEnsemble_L2 | 0.897951 | 0.487827 | 384.161613 |
| 4 | CatBoost_BAG_L1 | 0.895838 | 0.044714 | 104.548999 |
| 5 | LightGBM_BAG_L2 | 0.893982 | 0.988411 | 570.316032 |

---

### ④ FAST_TURNING (빠른 고속 급선회 / Saccade)
* **데이터 수**: 2,134개 궤적
* **Group-wise OOF Hit@1cm**: **38.0913%**
* **Best Model**: `WeightedEnsemble_L3` (Validation AUC: **0.924919**)
* **학습 시간**: 2,016.32초

| 순위 | 모델명 | Validation AUC | 추론 시간 (val, sec) | 학습 시간 (sec) |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **WeightedEnsemble_L3** | **0.924919** | **4.337742** | **2016.324939** |
| 2 | CatBoost_BAG_L2 | 0.924031 | 2.866730 | 1548.972383 |
| 3 | XGBoost_BAG_L2 | 0.923171 | 3.060172 | 1509.043470 |
| 4 | LightGBM_BAG_L2 | 0.922533 | 2.909556 | 1507.124780 |
| 5 | NeuralNetFastAI_BAG_L2 | 0.917646 | 3.906234 | 1942.660973 |

---

## 🔍 3. 핵심 결과 및 오차 분석 (Key Findings)

### 1) 느린 직진 비행(SLOW_STRAIGHT) 성능 극대화
* SLOW_STRAIGHT 데이터셋은 비행 상태가 단순하고 정답 분산이 작아 **86.1598%**의 높은 Hit@1cm 성능을 기록했습니다. 
* 이는 단일 모델 학습 시절의 Cruising OOF 점수(79.07%) 대비 **+7.08%** 상승한 수치로, 직진 노이즈가 제거된 상태에서 전용 Ranker가 최적의 물리 격자(damping factor, par=0, perp=0)를 일관되게 매핑하는 능력이 강화되었음을 증명합니다.

### 2) 선회 비행(Turning) 구간의 학습 난이도 및 데이터 희소성(Sparsity)
* **느린 극선회(32.65%)** 및 **고속 급선회(38.09%)** 구간은 모델의 예측 Hit rate가 현저히 낮았습니다.
* **분석 결과**:
  1. **데이터 파편화로 인한 학습 불안정**: 특히 `slow_extreme_turning`은 전체 10,000개 궤적 중 단 **854개(8.54%)**만 존재하여 AutoGluon이 다양한 물리 변동성을 배깅 학습하기에 표본수가 너무 부족했습니다.
  2. **가속도 산출 지연(Acceleration Lag)**: 회전 구간에서는 비행 속도와 가속도의 방향이 매 10ms 단위로 격렬하게 진동합니다. 기존의 W5-Quad 평활화 필터는 50ms 윈도우를 사용하기 때문에 급격한 턴이 개시되는 극초반부의 횡력 방향을 포착하는 데 시간적 지연(Lag)이 발생하여 격자가 표적을 비껴갔습니다.

---

## 💡 4. 자가 진화 및 Step 36 설계 방향성 (Self-Evolution)

Step 35의 교훈을 바탕으로, **Step 36**에서는 회전 예측 병목을 해소하기 위한 3분할 구조(Three-Regime)로 파이프라인을 고도화합니다.

```mermaid
graph TD
    A[Step 35: 4-Regime Split] -->|Turning Sparsity Bottleneck| B[Step 36: 3-Regime Split]
    B --> C[Cruising (Slow-Straight, 3,078)]
    B --> D[Gliding (Fast-Straight, 3,758)]
    B --> E[Steering (Unified Turning, 3,164)]
    E --> F[Adaptive W3 Acceleration Smooth to solve lag]
```

1. **Turning 데이터 병합 (Steering Regime 신설)**:
   * `slow_extreme_turning`과 `fast_turning`을 하나의 **Steering** 모드로 병합하여 **3,164개**의 대규모 선회 데이터셋을 구축합니다. 이를 통해 오버피팅을 방지하고 일반화 성능을 유도합니다.
2. **어댑티브 가속도 필터(W3-Quad) 연동**:
   * Steering 구간에서는 윈도우를 30ms(`w3`)로 단축하여 가속도 지연 현상을 원천 차단하고 즉각적인 횡가속도 벡터 변화를 물리 격자에 투영합니다.
3. **격자 동적 확장 배율 상향**:
   * Steering regime의 격자 스케일 $S_{\text{grid}}$ 계수를 최대 $3.5\times$까지 확장 적용하여 Target Lockout(격자 밖 탈출)을 근본적으로 방지합니다.
