# Step 4: EqMotion 기반 모기 궤적 예측

이 프로젝트는 400ms 동안 관측된 11개의 3차원 좌표(x, y, z)를 바탕으로 80ms 이후의 모기 위치를 예측하는 모델을 구현한 것입니다. 특히 최신 연구 트렌드인 **SE(3)-Equivariant Neural Networks (EqMotion)** 구조를 도입하여 3차원 회전과 평행 이동에 강건한 예측 성능을 확보하고자 했습니다.

## 주요 특징

- **기하학적 등변성 (Geometric Equivariance)**: EqMotion 아키텍처를 통해 모기가 어떤 방향으로 날아가든 동일한 운동 패턴을 포착할 수 있습니다. 별도의 기준축 정렬 전처리 없이도 회전 불변성을 수학적으로 유지합니다.
- **이상치 강건성 (Robustness to Outliers)**: 평가 지표인 R-Hit@1cm에 최적화하기 위해, 오차가 큰 샘플에 민감하게 반응하지 않는 **Huber Loss (Smooth L1 Loss)**를 손실 함수로 사용했습니다.
- **안정적인 수렴**: **AdamW** 옵티마이저와 **Cosine Annealing Scheduler**를 결합하여 학습 후반부의 미세한 보정 및 안정적인 수렴을 유도했습니다.

## 파일 구성

1. **`dataset.py`**:
   - `data/open/train` 및 `test` 폴더의 CSV 파일을 로드합니다.
   - `train_labels.csv`에서 타겟 좌표를 가져옵니다.
   - 11개의 과거 시점 좌표를 `(11, 3)` 텐서로 변환하여 공급합니다.

2. **`model.py`**:
   - **EqMotion Layer**: 노드 특징(Node Features)과 좌표(Coordinates)를 동시에 업데이트하는 레이어입니다.
   - 거리 정보를 불변 특징(Invariant Feature)으로 활용하며, 좌표 업데이트 시 벡터 차이를 이용해 등변성을 유지합니다.
   - 최종적으로 80ms 뒤의 상대적 변위를 예측합니다.

3. **`train.py`**:
   - 모델 학습을 위한 메인 스크립트입니다.
   - 데이터셋을 8:2 비율로 Train/Validation으로 나누어 학습합니다.
   - 매 에폭마다 Validation Loss를 체크하여 최적의 모델 가중치를 `outputs/step4/checkpoints/best_model.pth`에 저장합니다.

4. **`inference.py`**:
   - 학습된 최적 모델을 불러와 테스트 데이터에 대한 예측을 수행합니다.
   - 대회 제출 양식에 맞춘 `submission.csv` 파일을 `outputs/step4/` 폴더에 생성합니다.

## 실행 방법

### 1. 가상환경 활성화 및 패키지 설치
(이미 완료된 경우 생략 가능)
```bash
python -m venv venv
source venv/Scripts/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 모델 학습
```bash
python step4/train.py --epochs 50 --lr 1e-3
```
- `--epochs`: 학습 횟수 (기본값: 50)
- `--lr`: 학습률 (기본값: 0.001)

### 3. 결과 추론
```bash
python step4/inference.py
```
- `outputs/step4/submission.csv` 파일이 생성됩니다.

## 기대 효과

모기와 같은 작은 곤충은 불규칙한 지그재그 비행과 급격한 방향 전환을 수행합니다. EqMotion은 이러한 본질적인 '운동 역학'을 3차원 공간의 방향성과 독립적으로 학습하므로, 일반적인 LSTM이나 Transformer보다 적은 데이터로도 높은 일반화 성능을 보여줄 것으로 기대됩니다.
