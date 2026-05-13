# Project Modeling Keywords

This document outlines the core technical keywords and research areas for the DACON Mosquito Trajectory Prediction competition, categorized by their specific modeling objectives.

## 1. 3차원 시계열 궤적 예측 (3D Time-series Trajectory Prediction)
과거 400ms 동안 11개의 시점 데이터를 통해 미래를 예측하는 시계열 문제를 해결하기 위한 모델링 키워드입니다.

- **3D Trajectory Prediction Deep Learning** (3차원 궤적 예측 딥러닝)
- **Time-series Coordinate Forecasting Transformer** (시계열 좌표 예측 트랜스포머)
- **Sequence-to-Sequence Motion Prediction** (Seq2Seq 모션 예측)

---

## 2. 상대 운동 패턴 일반화 (Relative Motion Pattern Generalization)
속도, 가속도, 공간 구조 등 보조 데이터 없이 오직 좌표의 변화만으로 장소에 구애받지 않는 일반화된 패턴을 학습해야 하는 제약에 맞춘 키워드입니다.

- **Translation-Invariant Trajectory Prediction** (이동 불변성을 가진 궤적 예측 - 절대 좌표가 아닌 상대적 움직임 학습)
- **Coordinate-only Motion Forecasting** (좌표 데이터만을 활용한 모션 예측)
- **Data-driven Relative Motion Generalization** (데이터 기반 상대 운동 일반화)

---

## 3. 센서 지연 보정 및 곤충 비행 특성 (Sensor Latency & Insect Flight)
LiDAR의 80ms 시스템 지연을 보정하고, 비행 방향을 급격하게 바꾸는 모기 등 곤충의 미세 비행 특성을 반영하기 위한 키워드입니다.

- **LiDAR Latency Compensation Motion Prediction** (LiDAR 지연 시간 보정 모션 예측)
- **Insect/Mosquito Flight Trajectory Modeling** (곤충/모기 비행 궤적 모델링)
- **Short-term Micro-movement Forecasting** (단기 미세 움직임 예측)

---

## 4. 물리 법칙 기반 AI (Physics-Informed AI)
현재의 등속도 예측(Constant Velocity)을 개선하여 가속도나 관성 등 물리적 패턴을 신경망이 스스로 추론하게 만드는 방법론입니다.

- **Physics-Informed Neural Networks (PINN) for Trajectory** (물리 정보 신경망 기반 궤적 예측)
- **Beyond Constant Velocity Motion Prediction** (등속도 모델을 넘어서는 모션 예측)