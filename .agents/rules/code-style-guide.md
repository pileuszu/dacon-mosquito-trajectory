---
trigger: always_on
---

# 🦟 Mosquito Trajectory Prediction: Code & Pipeline Style Guide

이 문서는 모기 궤적 예측 프로젝트의 일관성과 성능을 유지하기 위해 AI 어시스턴트가 반드시 준수해야 할 개발 규칙입니다.

## 1. 🚫 Data Leakage Prevention (Critical)
*   **OOF (Out-of-Fold) Priors**: Ranker 모델을 학습시킬 때 사용하는 이전 단계의 예측값(Step 7, Step 12 등)은 반드시 OOF 방식(5-Fold 등)으로 생성된 데이터여야 합니다.
    *   학습 데이터에 대해 단순 'Train Prediction'을 사용하면 모델이 정답을 외우게 되어 검증 점수와 리더보드 점수 사이의 거대한 격차를 유발합니다.
*   **Validation Integrity**: 검증 데이터가 어떤 방식으로든 후보 생성(Candidate Generation)이나 피처 엔지니어링 과정에서 정답 정보를 미리 보지 않도록 엄격히 분리해야 합니다.

## 2. 📢 Robust Notifications & Monitoring
*   **Discord Webhook**: 학습(`train_automl.py`)이나 추론(`inference.py`)처럼 10분 이상 소요되는 모든 작업은 반드시 `utils.notifier`를 사용하여 디스코드 알림을 보냅니다.
*   **Error Handling**: 스크립트 실행부는 반드시 `try-except` 블록으로 감싸야 하며, 에러 발생 시 상세한 Traceback을 디스코드로 전송하여 즉각적인 대응이 가능케 합니다.
*   **Start & End**: 작업 시작(`🚀 Started`)과 종료(`✅ Finished` 또는 `❌ Failed`) 알림을 모두 포함해야 합니다.
*   **Agent Turn Ending (No Blocking)**: AI 어시스턴트는 10분 이상 소요되는 작업(학습, 추론 등)을 백그라운드 프로세스로 실행한 뒤, 굳이 완료될 때까지 루프를 돌며 기다리지 않고 사용자에게 진행 상황을 요약하여 보고한 후 즉시 턴을 마칩니다. 이후의 검증 및 후속 작업은 디스코드 알림을 확인한 사용자의 피드백을 받아 진행합니다.

## 3. 🧪 Physics-Guided Candidate Engineering
*   **Hybrid Grid Search**: 후보군 생성 시 Step 9의 '촘촘한 글로벌 그리드'와 Step 15의 '가변형 로컬 그리드'를 결합한 하이브리드 방식을 우선 고려합니다.
*   **Adaptive Search Range**: 모기의 비행 속도(`speed`)가 높을수록 후보군 검색 반경을 동적으로 확장하여 고속 비행 구간의 커버리지를 확보합니다.
*   **Simplified Features**: Ranker 학습 시 복잡한 계산 피처보다 물리 파라미터(`par`, `perp`, `ts`)를 직접 피처로 사용하여 AutoML의 학습 안정성을 높입니다.
*   **GMM-Guided 3-Regime Flight Routing**: GMM 클러스터링 기반의 궤적 분류 시 4분할로 지나치게 세분화하면 학습 데이터 희소성(Data Sparsity)이 발생하므로, Cruising(Slow-Straight), Gliding(Fast-Straight), Steering(모든 Turning)의 **3분할 전용 Predictor** 구조를 취하여 모델당 2,300개 이상의 충분한 포지티브 샘플을 확보합니다.
*   **Adaptive Acceleration Smoothing**: 모기가 급격하게 회전할 때, 긴 윈도우의 가속도 평활화(w5)는 위상 지연(Phase Lag)과 진폭 감쇄를 일으킵니다. 선회(Steering) 영역 모델의 후보 격자 생성 시에는 3프레임 이차 적합(w3) 또는 원시 가속도(raw)를 활용하여 급선회 구간의 Target Lockout을 37.33% 이하로 낮춥니다.
*   **Feature Space Decoupling**: GBDT 모델의 학습 피처(예: `spec_par`, `spec_perp` 등)는 스케일링되지 않은 원래의 고정된 이산 값을 유지하여 의사결정 나무의 분할 왜곡을 방지해야 합니다. 물리적 후보군 좌표 변환에만 kinematic grid scaling ($S_{\text{grid}}$)을 적용하여 기하학적 분포와 피처 분포를 분리합니다.

## 4. 🤖 AutoML Best Practices
*   **Model Quality**: 최종 제출용 모델은 항상 `presets='best_quality'`를 사용하여 다중 스택 앙상블을 구축합니다.
*   **Time Management**: `time_limit`을 명시적으로 설정하여 무한 루프나 시스템 리소스 고갈을 방지합니다.
*   **Metadata Fix**: AutoGluon 로드 시 발생하는 `AttributeError`와 같은 라이브러리 내부 버그는 발견 즉시 `venv` 내부를 패치하여 대응합니다.
*   **Memory Safety**: Windows OS 환경에서는 메모리 오버플로우나 할당 오류(`_ArrayMemoryError`)를 방지하기 위해 랜덤 포레스트(`RF`) 및 엑스트라 트리(`XT`) 모델을 학습 하이퍼파라미터에서 제외하고 fit을 진행합니다.
*   **Consensus Coordinate Ensembling**: unscaled 그리드로 학습된 모델과 adaptive scaled 그리드로 학습된 모델의 예측 좌표를 가중 평균하여 앙상블하면 각 모델의 공간 오차가 상쇄되어 검증 점수가 대폭 향상됩니다. 앙상블 가중치는 OOF validation grid search를 통해 산출합니다.

## 5. 📂 Project Structure & Documentation
*   **Step-wise Organization**: 각 실험은 `stepX` 폴더에 독립적으로 관리하며, 이전 스텝의 결과를 참조할 때는 명시적인 경로를 사용합니다.
*   **Study Notes**: 주요 실험이 완료될 때마다 `MOSQUITO_TRAJECTORY_STUDY_NOTES.md`를 업데이트하여 성공/실패 원인을 기록합니다.
*   **Clean Scripts**: `prepare_data.py`, `train_automl.py`, `inference.py`의 3단 구조를 유지하여 파이프라인 가독성을 높입니다.

## 6. 🔄 Continuous Improvement (Self-Evolution)
*   **Feedback Loop**: 새로운 실험 결과(리더보드 점수 등)가 나오면, 그 원인을 분석하여 이 가이드라인이 현재 프로젝트에 최적인지 끊임없이 의심해야 합니다.
*   **Rule Update**: 실험을 통해 더 나은 방법론(예: 특정 피처의 유용성, 새로운 후보 생성 로직 등)이 증명되면, 지체 없이 이 `code-style-guide.md`를 업데이트하여 지식을 자산화합니다.
