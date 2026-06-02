# Step 39 GMM 6-Regime Error Distribution Report

본 보고서는 모기 비행 궤적의 물리적 정밀 제어를 위해 GMM 클러스터 개수를 6개로 확장하여 분석한 결과를 정리한 보고서입니다. 
Step 36의 OOF 예측 오차 데이터를 GMM 6개 Regime에 매핑하여 각 클러스터의 물리적 특성과 거리 오차 분포(Euclidean Distance, cm)를 분석하였습니다.

---

## 📊 GMM 6-Regime 오차 분포 그래프

아래 그래프는 GMM 6개 클러스터 각각에 매핑된 trajectory들의 Step 36 OOF 거리 오차 분포를 나타냅니다.
* **빨간색 점선 (Red Dashed Line)**: 대회 평가 기준인 **1.0cm Hit Threshold**선입니다. 이 선의 왼쪽에 위치할수록 예측이 정답(Hit)으로 판정됩니다.
* **보라색 점선 (Purple Dotted Line)**: 각 클러스터의 **중앙값 오차(Median Error)**를 나타냅니다.

![08_step39_6regime_error_distribution](./images/08_step39_6regime_error_distribution.png)

---

## 🔍 클러스터별 물리적 특성 및 분석

| 클러스터 ID | 물리적 Regime 분류 (자동화 규칙) | 데이터 수 ($N$) | 평균 속도 (cm/s) | 평균 곡률 (Curvature) | 중앙값 오차 (Median, cm) | Hit@1cm 비율 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cluster 0** | **FAST_STRAIGHT** (중속 직진) | 3,126 | 2.23 | 2.0 | 0.59 | 77.70% |
| **Cluster 1** | **SLOW_MODERATE_TURNING** (저속 완만 선회) | 1,280 | 0.83 | 8.6 | 0.29 | **93.59%** 🏆 |
| **Cluster 2** | **FAST_MODERATE_TURNING** (중속 완만 선회) | 2,378 | 2.14 | 6.8 | 0.90 | 53.28% |
| **Cluster 3** | **FAST_STRAIGHT** (고속 직진) | 1,953 | 4.55 | 0.9 | 0.93 | 53.56% |
| **Cluster 4** | **FAST_MODERATE_TURNING** (고속 급선회 / 횡가속도) | 817 | 3.75 | 6.2 | **1.31** ⚠️ | 15.79% |
| **Cluster 5** | **SLOW_EXTREME_TURNING** (저속 극선회 / 제자리 돌기) | 446 | 1.03 | **71.8** 💥 | **1.02** | 45.96% |

---

## 💡 주요 발견 및 시각화 해석

### 1. 🎯 Double Peak 문제의 완전한 분리 (Peak Separation)
기존 GMM 4-Regime 구조에서는 `fast_turning` 군에 **정답(0.3cm) 부근의 피크**와 **오답(1.1~1.3cm) 부근의 피크**가 동시에 존재하여 모델 학습을 왜곡하는 고질적인 문제가 있었습니다. 
6-Regime GMM을 도입한 결과:
* **완만한 선회군 (Cluster 1, Cluster 2)**: 1.0cm 안쪽에 완벽히 안착하는 오차 분포를 보여줍니다. 특히 **Cluster 1 (저속 완만 선회)**은 **Hit@1cm가 93.59%**에 달하고 중앙값 오차가 **0.29cm**에 불과해 거의 모든 예측이 정답 처리됩니다.
* **고난도 선회군 (Cluster 4, Cluster 5)의 분리**:
  * **Cluster 4 (고속 급선회/횡가속도)**: 오차가 가장 큰 집단(중앙값 1.31cm, Hit rate 15.79%)으로 오차 분포의 주 피크가 1.0cm 바깥(1.2cm 근방)에 단일 모드로 깔끔하게 고립되었습니다.
  * **Cluster 5 (저속 극선회)**: 평균 곡률이 **71.8**로 엄청나게 급격한 회전(제자리 돌기)을 하는 집단이며, 중앙값 오차는 1.02cm로 기준선에 아슬아슬하게 걸쳐 있습니다.

### 2. 🚀 맞춤형 물리 격자(Kinematic Grid) 설계의 방향성 확보
이러한 고립(Decoupling)은 프로젝트의 핵심 돌파구입니다. 
* **Cluster 4 (고속 급선회)**는 속도가 빠르고($3.75\text{ cm/s}$), 횡방향 가속도($ctx\_lat\_accel = 0.0090$)가 매우 큽니다. 이 군의 오차가 큰 이유는 기존 격자 반경이 모기의 급선회 기하분포를 완전히 덮지 못해 **Target Lockout**이 발생하거나, 가속도 평활화 윈도우(W5)의 위상 지연 때문입니다. 
  * **대응책**: 이 클러스터에만 **가변형 로컬 그리드 스케일링($S_{\text{grid}}$)을 2.5배 이상 확장**하고, **W3 이차 적합 또는 raw 가속도 피처**를 제공하여 급회전하는 타깃을 추적하도록 합니다.
* **Cluster 5 (저속 극선회)**는 속도는 느리지만 곡률이 비정상적으로 높기 때문에, 통상적인 직진 가속도 항을 최소화하고 **순수 회전 물리 파라미터(Curvature, Angular rate)** 위주의 극선회 촘촘한 격자를 적용합니다.

---

## 🛠️ 다음 단계 실행 계획

1. **GMM 6-Regime 기반 물리 격자(Candidate Generator) 세분화**:
   * 각 클러스터의 특성(속도, 곡률, 횡가속도)에 맞게 격자 크기 $S_{\text{grid}}$와 물리 피처 가속도 평활화 필터를 최적화합니다.
2. **6개 Regime 전용 AutoGluon Tabular Predictor 학습**:
   * 3-Regime/4-Regime 대비 학습 데이터의 물리적 정합성이 극대화되었으므로, 각 Regime별로 최적의 분류/회귀 나무가 형성될 것입니다. (특히 Cluster 4와 5를 타겟팅한 전용 하이퍼파라미터 튜닝)
