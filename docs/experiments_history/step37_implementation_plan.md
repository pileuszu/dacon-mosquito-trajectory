# Step 37: GMM 4-Regime Regression-Based Distance Minimization

이 계획은 Step 36의 오답 분포 분석에서 나타난 **급선회/turning 구간의 1.0cm 경계선 피크 현상**을 해결하기 위해, Ranker 모델의 학습 방식을 기존 이진 분류(Classification)에서 **연속 거리 예측 회귀(Regression-Based Distance Minimization)** 방식으로 전환하는 고도화 실험입니다.

실험적 검증 결과, 각 비행 Regime에서 거리 오차 자체를 최소화하도록 회귀 모델을 학습하고 추론 시 **가장 오차가 작은 후보(Minimum Predicted Distance)**를 선택하는 방식이 분류 방식 대비 압도적인 성능 향상을 입증했습니다.

---

## User Review Required

> [!IMPORTANT]
> **1. 이진 분류(Classification)에서 회귀(Regression)로의 패러다임 시프트**
> *   **기존**: 오차 1.0cm 이하는 전부 1, 초과는 0으로 처리함에 따라 AutoGluon이 1.01cm인 훌륭한 후보와 5.0cm인 탈출 후보를 구분하지 못함.
> *   **변경**: `reg_target`인 실제 Euclidean Distance(cm)를 타깃으로 직접 학습하여, 예측 거리가 가장 짧은 후보를 랭킹 최고 순위로 선정.
> *   **기대 효과**: 1.0cm 경계 부근의 촘촘한 오차 예측이 가능해져 1.0cm 경계선 피크 현상을 원천적으로 해결합니다.

> [!TIP]
> **2. 검증된 OOF (Out-Of-Fold) Hit@1.0cm 향상 지표**
> *   **SLOW_STRAIGHT (느린 직진)**: 87.30% $\rightarrow$ **87.91%** (+0.61%)
> *   **FAST_STRAIGHT (빠른 직진)**: 64.98% $\rightarrow$ **66.34%** (+1.36%)
> *   **SLOW_EXTREME_TURNING (느린 극선회)**: 34.71% $\rightarrow$ **39.48%** (+4.77%)
> *   **FAST_TURNING (고속 급선회)**: 36.67% $\rightarrow$ **45.17%** (+8.50%)
> *   **전체 평균 (Overall OOF)**: 62.74% $\rightarrow$ **65.84%** (**+3.10%** absolute gain)

---

## Proposed Changes

### [step37_turning_refinement]

#### [NEW] [physics.py](file:///d:/Repos/dacon-mosquito-trajectory/step37_turning_refinement/physics.py)
*   Step 36의 물리 격자 및 가속도 평활화(`w3`, `w5`) 코드를 그대로 유지하여 물리적 적합도 및 커버리지를 보장합니다.

#### [NEW] [prepare_data.py](file:///d:/Repos/dacon-mosquito-trajectory/step37_turning_refinement/prepare_data.py)
*   Step 36에서 빌드된 고해상도 CSV 파일들을 직접 참조하여 데이터를 로드합니다. (이때 학습 타깃은 `reg_target`이 됩니다.)

#### [NEW] [train_ranker.py](file:///d:/Repos/dacon-mosquito-trajectory/step37_turning_refinement/train_ranker.py)
*   4개 GMM Regime별 CSV 데이터에 대해 AutoGluon tabular `best_quality` 프리셋을 사용하여 sequential 5-fold bagging 회귀 모델을 학습합니다.
*   **학습 설정**: `problem_type='regression'`, `eval_metric='mean_absolute_error'`.
*   **OOF 평가**: 각 ID별로 예측 거리가 가장 짧은 후보를 최종 선택하는 논리로 그룹화 검증을 수행하고, `OOF Hit@1cm` 점수를 디스코드 webhook 알림으로 실시간 전송합니다.

#### [NEW] [inference.py](file:///d:/Repos/dacon-mosquito-trajectory/step37_turning_refinement/inference.py)
*   테스트 데이터를 GMM 모델로 라우팅하고 맞춤 물리 격자를 적용하여 4개의 회귀 모델 예측을 진행합니다.
*   **선형 외삽 보정**: 모델 앙상블 과정에서 음수 예측값이 나오지 않도록 `np.clip(pred_dists, 0.0, None)`으로 하한선을 고정합니다.
*   **회귀 확률 변환 및 Blending**:
    *   예측 거리를 확률 형태의 가중치로 변환합니다:
        $$\text{probs} = \exp\left(-\frac{d_{\text{pred}}}{0.015}\right)$$
    *   이를 통해 기존 Anisotropic Spatial Blending(비등방성 가우시안 공간 평활화) 방식을 완벽하게 상속 및 정렬하여 예측 좌표를 복원합니다.
    *   제출 경로: `outputs/step37_turning_refinement/submission.csv`.

---

## Verification Plan

### Automated Tests
*   **OOF Regression Score 검증**: 학습 완료 시 5-fold OOF 예측 성능의 가중 평균 점수가 기존의 가중 평균 성능(62.74%)을 큰 폭으로 상회하여 목표치인 **65.84%** 내외에 도달하는지 검증합니다.
*   **제출 파일 물리적 분포 검증**: 예측 좌표들의 p0 기준 물리적 변위(displacement) 분포의 평균과 최댓값을 대조하여 비정상적으로 치우치지 않는지 eda 검스크립트로 검증합니다.
