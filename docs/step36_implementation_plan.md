# Step 36: GMM 4-Regime Tailored Physics-Explicit Divide-and-Conquer

이 계획은 **명확한 분류(Clear Trajectories) 데이터의 오답 분석 결과**를 바탕으로, GMM으로 정의된 4개 비행 Regime 각각의 독특한 물리적 거동 오류와 한계를 극복하기 위해 **Regime별 맞춤형 물리 격자(physics candidate grid) 생성기 및 피처 엔지니어링**을 구축하는 고도화 실험입니다.

## User Review Required

> [!IMPORTANT]
> **1. 4개 Regime별 맞춤형 물리 격자(Custom Candidates) 설계**
> *   **SLOW_STRAIGHT (느린 직진)**: 오답의 **70.34%**가 갑작스러운 회전(평균 39.5°)으로 이탈함.
>     *   *개선*: 역사적 context 상의 곡률(`ctx_curv`)에 따라 격자 스케일 $S_{\text{grid}}$를 동적으로 팽창($1.0 \rightarrow 1.8$)시켜 횡방향 탐색 면적을 확보합니다.
> *   **FAST_STRAIGHT (빠른 직진)**: 오답의 **62.56%**가 ML 선택 오류임.
>     *   *개선*: 상대 속도 비율(`cand_speed_ratio = cand_speed / roll_speed_mean_10`) 등 후보군 선택을 직접 보조하는 상대적 물리 피처를 추가합니다.
> *   **SLOW_EXTREME_TURNING (느린 극선회)**: 오답의 **37.58%**가 평균 56.1°의 극단적 턴을 수반하며, **34.45%**는 가속하여 빠른 회전으로 전이됨.
>     *   *개선*: **3프레임 평활화(w3)** 가속도를 도입하여 회전 위상 지연(Phase Lag)을 없애고, 격자 스케일을 최대 **$3.5\times$**로 대폭 확장하며 회전각 범위를 최대 70°로 넓힙니다.
> *   **FAST_TURNING (고속 급선회)**: 오답의 **41.40%**가 회전할 듯하다가 갑자기 턴을 멈추고 직선 비행(평균 8.6°)으로 복귀함 (Saccade Termination).
>     *   *개선*: `w3` 평활화를 적용하고, 선회 도중 직선으로 뻗어나가는 궤적을 조준하기 위해 감쇄 계수가 큰 **직선 복귀 후보군(perp=0.0, damping=[0.2, 0.5, 0.8])**을 항상 풀에 상주시켜 탈출을 방어합니다.
>
> **2. 3분할 대신 4분할 독립 모델을 통한 정밀 매칭**
> *   각 Regime의 물리 오답 원인이 완전히 다름(느린 직진은 기습적 턴, 빠른 직진은 선택 오차, 고속 급선회는 직선 복귀 등)이 확인되었습니다.
> *   따라서 이들을 하나의 Steering 모델로 합치는 대신, 각 영역의 맞춤형 격자를 기반으로 **4개의 고도화된 전용 Predictor를 독립적으로 구축**하는 것이 오차의 기하학적 근본 원인을 해결하는 최적의 방법입니다.

---

## Proposed Changes

### [step36_four_regime]

#### [NEW] [physics.py](file:///d:/Repos/dacon-mosquito-trajectory/step36_four_regime/physics.py)
*   각 GMM Regime에 최적화된 격자 생성 및 윈도우 조절을 지원하는 물리 엔진을 설계합니다:
    *   `make_candidates` 함수에 `regime` 인자를 전달받아 처리합니다:
        *   `slow_straight`: `CANDIDATES_SLOW` 기본 격자에 대해 곡률 기반으로 $S_{\text{grid}} = \text{clip}(1.0 + 0.15 \cdot \text{smooth\_curv\_w5}, 1.0, 1.8)$ 스케일링을 적용합니다.
        *   `fast_straight`: `CANDIDATES_FAST` 기반, $S_{\text{grid}} = \text{clip}(1.0 + 0.6 \cdot P_{\text{saccade}}, 1.0, 2.5)$ 및 선택 피처 강화를 적용합니다.
        *   `slow_extreme_turning`: **`w3` 가속도/속도 벡터** 사용, $S_{\text{grid}} = \text{clip}(1.5 + 0.1 \cdot \text{smooth\_curv\_w3}, 1.5, 3.5)$로 격자 팽창.
        *   `fast_turning`: **`w3` 가속도** 사용, $S_{\text{grid}} = \text{clip}(1.2 + 0.6 \cdot P_{\text{saccade}}, 1.2, 3.0)$ 스케일링. 후보 격자에 감쇄가 강한 **직선 전진 후보들(damp=[0.2, 0.5, 0.8], perp=0.0)**을 추가하여 회전 조기 종료 방어.
    *   후보군 피처 추출기(`c_features`)에 relative feature인 `cand_speed_ratio`를 추가하여 속도 스케일에 무관하게 최적의 비율을 나무 모델이 선택할 수 있도록 보조합니다.

#### [NEW] [prepare_data.py](file:///d:/Repos/dacon-mosquito-trajectory/step36_four_regime/prepare_data.py)
1.  기존 `step35_four_regime/models/` 경로의 `gmm_model.pkl`과 `scaler.pkl`을 로드하여 10,000개 트랙의 GMM 군집을 매핑합니다.
2.  각 궤적에 대해 `regime` 값을 할당하고, 위에 정의된 맞춤형 `physics.py` 코드를 호출하여 후보 격자 및 피처 공간을 생성합니다.
3.  생성된 학습 데이터를 4개의 독립된 csv 파일로 분할 저장합니다:
    *   `step36_four_regime/train_ranker_v36_slow_straight.csv`
    *   `step36_four_regime/train_ranker_v36_fast_straight.csv`
    *   `step36_four_regime/train_ranker_v36_slow_extreme_turning.csv`
    *   `step36_four_regime/train_ranker_v36_fast_turning.csv`

#### [NEW] [train_ranker.py](file:///d:/Repos/dacon-mosquito-trajectory/step36_four_regime/train_ranker.py)
1.  4개의 CSV 데이터셋에 대해 AutoGluon Tabular `best_quality` 프리셋을 사용하여 5-fold bagging 모델 학습을 sequential하게 실행합니다.
2.  Windows OS 환경의 Out-Of-Memory/할당 에러를 방지하기 위해 `RF`, `XT` 트리 모델은 제외하고 학습을 실행합니다.
3.  각 학습 단계의 OOF Hit@1cm 점수를 디스코드 webhook 알림으로 즉각 전송합니다.

#### [NEW] [inference.py](file:///d:/Repos/dacon-mosquito-trajectory/step36_four_regime/inference.py)
1.  테스트 궤적에 대해 GMM 모델로 라우팅을 진행하고, 각 궤적에 맞는 맞춤형 물리 격자(Steering 계열에는 `w3` 평활화, Saccade Termination 방어 격자 등)를 동적으로 생성합니다.
2.  4개의 전용 AutoGluon 모델을 분기 적용하여 적합도 확률을 도출합니다.
3.  속도 스케일에 정렬된 Anisotropic Spatial Blending을 수행하여 최종 예측 좌표를 도출하고 `outputs/step36_four_regime/submission.csv` 파일을 빌드합니다.

---

## Verification Plan

### Automated Tests
1.  **후보군 타겟 커버리지(Target Coverage) 평가**: `prepare_data.py` 실행 시 각 regime별로 생성된 후보 격자가 target 좌표를 1cm 이내로 커버하는 비율(Generator Hit Rate)이 Step 35 대비 향상되는지 모니터링합니다.
2.  **OOF Hit@1cm 성능 검증**: 5-fold OOF 예측 성능의 가중 평균 점수가 기존 Step 35의 가중 평균 성능(63.36%)을 상회하는지 점검합니다.
3.  **Submission 물리적 분포 검증**: 예측 좌표들의 변위 및 변동폭 분포가 물리적으로 타당한지 eda 스크립트를 통해 확인합니다.
