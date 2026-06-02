# 🦟 3D 모기 비행 궤적 예측 AI 솔루션

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-green?style=for-the-badge)](https://github.com/microsoft/LightGBM)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-orange?style=for-the-badge)](https://autogluon.mxnet.io/)
[![SOTA Score](https://img.shields.io/badge/SOTA_Score-0.6868-gold?style=for-the-badge)](https://dacon.io/)

본 저장소는 **DACON 모기 궤적 예측 AI 경진대회**의 최종 솔루션 및 재현 파이프라인 코드를 포함하고 있습니다. 모기의 3차원 비행에 따른 연속적 공기역학 모델링과 제약 조건을 설계하여 미래 비행 좌표를 정밀하게 예측합니다.

---

## 🏆 주요 성능 성과

> [!TIP]
> ### 📊 검증 및 리더보드 결과
> * **리더보드 최종 예측 적중률 (Hit@1cm)**: **`0.6868`** 🏆 *(SOTA)*
> * **검증 데이터셋 적중률 (OOF Hit@1cm)**: **`67.87%`**
> * **평균 오차 변위 (Average Displacement)**: **`4.79 cm`**
> * **최대 오차 변위 (Maximum Displacement)**: **`11.11 cm`** *(실험 챔버 이탈 한계 조건인 12.0cm 완벽 준수)*

---

## 💡 핵심 아키텍처 특징

본 솔루션은 3D 궤적 문제를 단순 3차원 좌표 외삽이 아닌 **연속시간 물리-기하학 하이브리드 프레임워크**로 정의합니다:

1. **Frenet-Serret Neural ODE 시뮬레이터**: 시간에 따라 변하는 Frenet 국소 좌표계(접선, 법선, 종법선) 내에서 모기의 비행 가속도 벡터장을 수치화하고, 4차 Runge-Kutta(RK4)를 이용하여 80ms 동안의 연속시간 궤적을 미분 연산합니다. 학습 시 미분 가능성을 확보하기 위해 **Focal Soft-Hit Loss**를 직접 설계했습니다.
2. **Clifford Geometric Algebra $Cl(3,0)$ CFM**: 시계열 비행 궤적의 장기 의존성 학습을 위한 Mamba SSM 블록에 3D 공간 회전 공변성(Rotational Covariance)을 보장하는 Clifford 대수 선형 계층을 결합하고, 연속 흐름 매칭(CFM)으로 훈련시켰습니다.
3. **표 형식 랭킹 모델 (Feature Decoupling)**: 43개 최종 확장 격자 후보군에 대해 오차 거리를 산출하는 고정밀 LightGBM 랭커를 학습하고, 결정 경계면 왜곡을 방지하기 위해 기하 좌표와 랭커 특징 피처 공간을 완전히 분리했습니다.
4. **Powell 앙상블 가중치 최적화**: 4가지 GMM 비행 모드(저속/고속 직진, 저속/고속 선회)별 검증 손실을 최소화하기 위해 21개 모델 앙상블의 블렌딩 가중치를 실수 공간에서 직접 탐색하는 Nelder-Mead 및 Powell 알고리즘을 사용했습니다.
5. **이상치 댐핑(Outlier Damping) 판별 제어**: 선회 구간에서 비 물리적 이탈(1.5cm 초과 오차)을 감지하기 위해 LightGBM 분류기를 학습시켜, 이상치로 감지된 트랙에 대해 마지막 위치($p_{\text{last}}$) 수축 보정 및 Clifford-Mamba CFM 물리 가이드 융합을 적용했습니다.

---

## 📂 저장소 구조

```
dacon-mosquito-trajectory/
├── .env.example                # 환경 변수 설정 템플릿 파일
├── README.md                   # 영문 메인 랜딩 페이지
├── README_KR.md                # 한글 메인 랜딩 페이지
├── requirements.txt            # 파이썬 의존성 패키지 목록
├── src/                        # 최종 재현용 단일 소스코드 폴더
│   ├── models/
│   │   ├── cfm_model.py        # Clifford-Mamba CFM 모델 정의
│   │   ├── neural_ode.py       # Frenet Neural ODE 모델 정의
│   │   └── outlier_classifier.py # Outlier 이상치 판별기 정의
│   ├── data_preprocessing.py   # 특징량 추출 및 GMM 비행 모드 분류
│   ├── candidate_generator.py  # 물리/ODE/CFM 기반 예측 후보 격자 생성
│   ├── train.py                # 5-Fold 전체 모델 훈련 스크립트
│   ├── powell_optimization.py  # Powell 최적화 기반 앙상블 블렌딩 가중치 산출
│   ├── inference.py            # 최종 앙상블, 스냅 매핑 및 이상치 제어 추론
│   └── reproduce.py            # 전체 프로세스 통합 재현 마스터 스크립트
├── docs/                       # 기술 보고서 및 아카이브 문서
│   ├── wrapup_report.md        # 영문 최종 기술 보고서 (Wrap-Up Report)
│   ├── wrapup_report_kr.md     # 국문 최종 기술 보고서 (Wrap-Up Report)
│   ├── analysis/               # 일반 EDA 및 분석 리포트
│   ├── biomechanics/           # 생체역학 및 관련 학술 문헌 정리
│   └── experiments_history/    # 단계별 실험 설계 및 히스토리 분석 보고서
├── eda/                        # 탐색적 데이터 분석 (EDA)
│   ├── reports/                # 분석 결과 마크다운 리포트
│   ├── images/                 # 시각화 이미지 출력 폴더
│   └── *.py                    # EDA 연산 및 시각화용 개별 파이썬 스크립트군
├── experiments/                # 단계별 실험 이력 및 백업 코드 아카이브
├── models_trained/             # 학습 완료된 모델 체크포인트 (Git 제외)
└── outputs_reproduced/         # 모델 재현 파이프라인 실행 결과물 (Git 제외)
```

---

## 📄 기술 보고서 (Technical Wrap-Up Reports)

물리 기하학적 시스템 설계, 가설 검증 과정 및 수식적 상세 명세는 기술 보고서를 참조하시기 바랍니다:
* **[한국어 기술 보고서 (Korean Wrap-Up Report)](docs/wrapup_report_kr.md)**
* **[영어 기술 보고서 (English Wrap-Up Report)](docs/wrapup_report.md)**

---

## 🛠️ 파이프라인 재현 방법

### 1. 패키지 설치 및 환경 설정
필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```
환경 변수 템플릿 파일을 복사하여 크리덴셜(WandB API Key, optional Discord Webhook URL)을 입력합니다:
```bash
cp .env.example .env
```
> [!IMPORTANT]
> 원본 대회 데이터셋(`train/`, `test/`, `train_labels.csv`, `sample_submission.csv`)이 `data/open/` 폴더 내에 배치되어 있어야 합니다.

### 2. 전체 파이프라인 재현 실행 (Full Run)
모든 주요 모델(Neural ODE, Clifford-Mamba CFM, LightGBM 랭커 및 이상치 분류기)을 훈련하고 Powell 블렌딩 가중치를 최적화하여 제출용 좌표 파일(`submission.csv`)을 생성합니다:
```bash
python src/reproduce.py
```

### 3. 고속 검증 모드 실행 (Fast Verification)
에폭 수를 줄이고 surrogate 모델을 활용하여, 전체 학습 및 추론 프로세스의 입출력 정합성을 3분 이내에 빠르게 점검합니다:
```bash
python src/reproduce.py --fast
```

#### 고속 검증 실행 통계량 결과:
* **평균 이동 변위**: **`4.7973 cm`**
* **최대 이동 변위**: **`11.1151 cm`** *(챔버 범위 이탈 없음)*
* **결측 검증**: 10,000행의 예측 데이터 중 NaN/Inf 값 `0`개 확인.
