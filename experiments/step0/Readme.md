# Step 0: Constant Velocity Baseline

이 단계는 물리 기반의 가장 단순한 예측 모델인 **등속도 외삽(Constant Velocity Extrapolation)** 베이스라인을 구현합니다.

## 핵심 개념
- **물리적 가정**: 모기가 매우 짧은 시간(80ms) 동안은 직전의 속도와 방향을 유지하며 비행한다고 가정합니다.
- **예측 방식**: 
    - 마지막 두 지점(0ms, -40ms) 사이의 변위 벡터를 구합니다: $V = P_{0} - P_{-40}$
    - 80ms 뒤의 위치는 마지막 지점에서 해당 벡터의 2배만큼 더 이동한 지점으로 예측합니다: $P_{80} = P_{0} + 2 \times V$

## 특징
- **강력한 Baseline**: 80ms라는 짧은 예측 시간 내에서는 복잡한 딥러닝 모델보다도 강력한 성능(Hit Rate 0.5536)을 보여주는 경우가 많습니다.
- **물리적 Prior**: 이후 단계인 Step 5에서는 이 모델의 예측값을 기준(Prior)으로 삼고, 그 오차(Residual)만을 딥러닝으로 보정하는 하이브리드 방식을 사용합니다.

## 실행 방법

1. **데이터 분할**: Train 데이터를 Train/Test 셋으로 나눕니다. (최초 1회 실행)
    ```bash
    # 별도의 스크립트 대신 train.py나 test.py 실행 시 자동 생성됩니다.
    ```

2. **학습(평가)**: Train 셋에 대한 성능을 측정합니다.
    ```bash
    python step0/train.py
    ```

3. **테스트(검증)**: Held-out Test 셋에 대한 성능을 측정하고 **WandB**에 기록합니다.
    ```bash
    python step0/test.py
    ```

4. **추론(Inference)**: 실제 대회용 Test 셋에 대한 예측 결과(`submission.csv`)를 생성합니다.
    ```bash
    python step0/inference.py
    ```

- 결과물은 `outputs/step0/submission.csv`에 저장됩니다.
- WandB 로그는 설정된 프로젝트(`mosquito-trajectory`)로 전송됩니다.
