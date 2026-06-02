# Walkthrough: Step 36 GMM 4-Regime Tailored Physics-Explicit Divide-and-Conquer

GMM 4-Regime 비행 분류 체계하에서 각 영역별 물리적 오답 패턴(선회 지연, Saccade 종료 직선 복귀, 극단적 회전각 등)을 방어하기 위해 **맞춤형 물리 격자(custom candidate generators) 및 relative feature engineering**을 구축하고, 4개의 전용 AutoGluon Tabular Predictor 학습을 개시했습니다.

## 1. 물리 격자 커버리지 검증 결과

`prepare_data.py`를 통해 10,000개 트랙 전체에 대해 맞춤형 격자를 적용한 결과, **Target Lockout(정답 격자 탈출) 비율이 대폭 감소**하여 이론적 상한선인 Generator Hit@1cm 성능이 아래와 같이 향상되었습니다:

*   **SLOW_STRAIGHT (느린 직진)**: **88.40%** coverage (2721 / 3078)
*   **SLOW_EXTREME_TURNING (느린 극선회)**: **72.90%** coverage (565 / 775)  (★ 30%대 수준의 오차를 `w3` 평활화와 $3.5\times$ 격자로 극복)
*   **FAST_STRAIGHT (빠른 직진)**: **81.80%** coverage (3074 / 3758)
*   **FAST_TURNING (고속 급선회)**: **76.18%** coverage (1820 / 2389) (★ 직선 복귀 fallback 격자로 Saccade Termination 방어)
*   **전체 평균 커버리지 (Overall Coverage)**: **81.80%** (8180 / 10000)

## 2. 작업 내역 및 수정된 파일

1.  **[physics.py](file:///d:/Repos/dacon-mosquito-trajectory/step36_four_regime/physics.py)**: GMM 4개 Regime별 격자 팽창 및 Smoothing 윈도우 분기(`w3`/`w5`) 처리, `cand_speed_ratio` 상대 속도 피처 추가, `fast_turning` 직선 fallback spec 추가.
2.  **[prepare_data.py](file:///d:/Repos/dacon-mosquito-trajectory/step36_four_regime/prepare_data.py)**: GMM으로 트랙별 비행 Regime 분류, 맞춤형 격자 생성 호출, 각 영역별 독립 csv 학습 데이터 추출 및 커버리지 계산.
3.  **[train_ranker.py](file:///d:/Repos/dacon-mosquito-trajectory/step36_four_regime/train_ranker.py)**: 4개 CSV 데이터셋에 대해 AutoGluon tabular `best_quality` (5-fold bagging, sequential, RF/XT 제외) 모델을 순차 학습하는 스크립트.
4.  **[inference.py](file:///d:/Repos/dacon-mosquito-trajectory/step36_four_regime/inference.py)**: 테스트 데이터를 GMM으로 라우팅하고, 각 궤적에 최적화된 맞춤 격자와 AutoGluon Predictor를 적용한 후 Anisotropic spatial blending으로 최종 submission 좌표를 복원하는 추론 코드.
5.  **[MOSQUITO_TRAJECTORY_STUDY_NOTES.md](file:///d:/Repos/dacon-mosquito-trajectory/docs/MOSQUITO_TRAJECTORY_STUDY_NOTES.md)**: Step 36의 물리 격자 디자인 및 커버리지 검증 결과를 영구 기록으로 추가.

## 3. 추론 최적화 및 결과 검증 (Optimized Inference & Results)

기존 추론 스크립트가 소형 배치 크기(`batch_size=20`) 및 AutoGluon의 L3 스택 앙상블 신경망 평가 비용으로 인해 약 4시간 이상 소요되는 병목 현상이 발생했습니다. 이를 방어하기 위해 다음과 같은 고속 배치 추론 기법을 적용했습니다:

*   **배치 크기 확대 (`batch_size=250` 기본값)**: 파일 시스템 및 파이썬 루프 오버헤드를 약 12배 절감했습니다.
*   **모델 스피드 분기 파라미터 (`--model_type`) 추가**:
    *   `l3`: 최상 품질의 WeightedEnsemble_L3 적용 (~1시간)
    *   `l2`: WeightedEnsemble_L2 적용 (~45분)
    *   `fast_tree`: CatBoost (60%) + XGBoost (40%) L1 앙상블 적용 (~10분)
    *   `ultra_fast`: CatBoost_BAG_L1 단일 예측 적용 (~3~4분 내외)

### 4. 1차 검증용 추론 수행 결과
안정성 및 파이프라인 검증을 위해 `ultra_fast` 예측 옵션으로 10,000개 전수 추론을 수행하여 성공적으로 제출 파일을 추출했습니다.
*   **실행 명령어**: `venv\Scripts\python.exe step36_four_regime/inference.py --batch_size 250 --model_type ultra_fast`
*   **물리 변위 분포 검증 (Displacement from p0)**:
    *   **평균 이동 거리 (Mean)**: **4.8148 cm**
    *   **최대 이동 거리 (Max)**: **10.9534 cm**
    *   **제출 파일 위치**: `outputs/step36_four_regime/submission.csv` (10,000개 행 완벽 복원 확인)

물리적 제한 범위 내에서 매우 합리적인 수준의 변위 분포를 보이고 있어 파이프라인이 정상적으로 관측되었음을 입증했습니다. 최종 성능 극대화를 원할 경우 `--model_type l3` 또는 `--model_type fast_tree`를 선택적으로 실행할 수 있습니다.

