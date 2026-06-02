# 🦟 모기 궤적 예측 프로젝트: 모델링 및 파이프라인 총정리 (Methodology Map)

본 문서는 프로젝트 시작부터 현재 **Step 40**까지 진행된 핵심 가설, 수학적/물리적 구현 기법, 성공 및 실패 원인을 총망라하여 **NotebookLM** 등 AI 분석 도구가 프로젝트의 현황을 파악하고 최적의 차기 방향을 도출할 수 있도록 작성된 종합 요약 가이드라인입니다.

---

## 📌 1. 프로젝트 목표 및 평가 메트릭

* **목표**: 모기 비행 궤적(과거 400ms)을 기반으로 미래 80ms 시점의 3D 좌표 $(x, y, z)$ 예측
* **평가 메트릭**: **Hit@1.0cm** (예측 좌표와 실제 좌표 간의 Euclidean Distance가 1.0cm 이하인 비율)
* **리더보드 타깃**: Public Leaderboard Hit@1.0cm **0.700+** 돌파 (현재 최고점: **0.675**)

---

## 📈 2. 프로젝트 연대기 및 핵심 이정표 (Chronology)

### 1단계: 베이스라인 구축 (Step 4 ~ Step 12)
* **Step 4**: 그래프 뉴럴 네트워크 기반 궤적 예측 모델인 **EqMotion** 학습. (Public Score: **0.5608**)
* **Step 7**: 단순 물리 법칙인 **Constant Velocity (CV, 등속도)** 물리 베이스라인 적용. (Public Score: **0.6234**)
* **Step 8**: CV와 EqMotion 예측 좌표를 가중 평균 앙상블했으나, 비행 궤적 왜곡 발생. (Public Score: **0.5950**로 하락)
* **Step 9**: 442개 격자 후보군을 생성해 평가하는 **첫 물리 격자 탐색 (Physics-Guided Grid Search)** 도입. (Public Score: **0.6434**로 반등)

### 2단계: 물리 격자 엔지니어링 및 의존성 탈피 (Step 15 ~ Step 30)
* **Step 22 (Prior Copycat 루프 탈피)**: 
  * *발견*: 이전 단계의 모델 예측값(EqMotion, CV 등)을 Ranker의 Tabular 피처로 직접 주입하자, GBDT 트리 모델들이 이를 정답으로 과적합하여 스스로 물리 경로를 탐색하지 않고 prior 근처 격자만 베끼는 **Feature Dominance (우세 피처 왜곡)** 현상 발생.
  * *해결*: 모델 Prior 피처를 Tabular에서 완전히 삭제하고, 격자 후보군 좌표 변환에만 기하학적으로 활용하여 독자적인 물리 보정을 복원. (Public Score: **0.6516** 달성)
* **Step 26 (Prior Candidate Injection & Spatial Blending)**:
  * *개선 1*: Prior 좌표를 피처가 아닌 **격자 후보군 풀(Candidate Pool)에 직접 점으로 주입**하여 물리 후보들과 동등하게 경쟁하도록 유도.
  * *개선 2 (W5 Smooth)*: 5프레임(200ms) Polynomial 이차 피팅을 적용해 모기 트랙 고주파 노이즈 제거 및 안정된 물리 파라미터 계산.
  * *개선 3 (Spatial Blending)*: 단일 최고점(`argmax`) 대신 3D Gaussian Kernel을 통한 **비등방성 공간 확률 평활화(Anisotropic Spatial Blending)** 도입하여 타깃 인근 예측 적중 극대화. (Public Score: **0.6582** 달성)
* **Step 30 (Feature Decoupling)**:
  * *발견*: 격자 크기를 모기 속도와 비례하여 동적으로 조절하는 Kinematic Grid Scaling ($S_{\text{grid}}$) 적용 시, 학습 데이터의 스케일링된 피처가 트리의 의사결정 경계를 왜곡함.
  * *해결*: GBDT 입력 피처는 가공하지 않은 이산값(unscaled raw specs)으로 유지하고, 기하학적 후보 좌표 변환에만 $S_{\text{grid}}$를 적용하여 피처 공간과 기하 공간을 물리 분리. (OOF Hit@1cm: **69.70%** 달성)

### 3단계: 멀티 Regime 분할 및 앙상블 고도화 (Step 32 ~ Step 39)
* **Step 32-33 (Curvature-Adaptive Dynamic Scaling)**:
  * 느린 비행이라도 회전 지표(Curvature > 12.0, Lateral Acceleration > 0.0020)가 감지되면 넓은 격자와 고속 선회용 모델로 동적 라우팅하여 급선회 구간의 Target Lockout 차단. (Public Score: **0.6672** 달성)
* **Step 35-36 (GMM 4-Regime Divide-and-Conquer)**:
  * 8개 비행 역학 지표 기반 GMM-4 분할. 각 비행 Regime(Slow-St, Fast-St, Slow-Turn, Fast-Turn)에 최적화된 맞춤형 격자(회전 구간에는 W3 슬라이딩 가속도 투영, 직진 소실 대비 straight fallback 후보군 추가) 설계. (Public Score: **0.6728** 달성)
* **Step 37-38 (Regression-Based Distance Ranking)**:
  * Hit(1) / Miss(0) 분류가 초래하는 경계선 정보 손실을 막기 위해 **실제 오차 거리(cm)를 예측하는 Regression 패러다임** 도입. (OOF Hit@1cm: **65.84%**로 절대 성능 향상)
* **Step 39 (GMM 6-Regime Decoupling & Consensus Coordinate Blending)**:
  * 회전 구간의 이중 에러 피크를 해소하기 위해 GMM 컴포넌트를 6개로 확장.
  * 추론 시 모델의 예측 좌표와 Prior 모델(EqMotion S4, CV S7)의 가중 합을 Regime별로 최적 그리드 서치하여 최종 보정.
  * **Consensus 앙상블(Step 36 GMM-4 + Step 39 GMM-6 50:50 blend) 결과 최고점 달성 (Public Score: 0.6750, Soft Blended 0.6680)**.

### 4단계: Tri-Model 및 Denoising 시도 (Step 40)
* **Step 40 (Tri-Model & 95% Outlier Exclusion)**:
  * *의도*: GMM-6 세분화 시 발생한 학습 데이터 희소성(Sparsity) 문제를 막기 위해 **Cruising, Gliding, Steering**의 3대 Regime으로 모델을 풀링(Pooling).
  * *이상치 처리*: GMM 사후 소속 확률이 95% 미만인 모호한 경계선 데이터(33.06%)를 학습 데이터에서 전면 배제하여 결정 경계 청소.
  * *클래스 가중치*: 1.0cm 이내의 Positive 후보에 `weight=15.0` 부여.
  * *결과*: **Public Score 0.6634로 소폭 regression (0.6680/0.6750 대비 하락)**.

---

## 🎨 3. 물리 격자 생성 및 앙상블 기하 아키텍처

### 1) 격자 스케일링 및 가속도 평활화
* **직진 구간 (Cruising/Gliding)**: 5프레임 이차 적합 평활화 가속도 적용. 위상 지연을 억제하며 관성 직선 격자 유지.
* **선회 구간 (Steering)**: 3프레임 이차 적합(W3) 가속도 또는 원시 가속도(Raw)를 직접 사용하여 급격한 회전 시 발생하는 위상 지연(Phase Lag)을 막아 타깃 락아웃을 방지.

### 2) 비등방성 공간 확률 블렌딩 (Anisotropic Spatial Blending)
예측 좌표를 단일 최고 확률점으로 찍는 대신, 비행 속도와 방향 탄젠트 벡터를 기반으로 오차 타원(Error Ellipsoid)을 형성:
* **진행 방향(Tangential $\sigma$)**: 속도와 Saccade 확률에 따라 가변 확장 ($3.0\text{mm} \sim 11.0\text{mm}$)
* **수직 방향(Normal $\sigma$)**: 횡력을 방어하기 위해 조밀하게 조절 ($3.0\text{mm} \sim 6.0\text{mm}$)
* 후보군들 간 가중 투표를 통해 최종 공간 무게중심 산출.

---

## ⚠️ 4. 현재 당면한 과제 및 딜레마 (Dilemma for NotebookLM)

### Q1. Step 40 (Tri-Model + Outlier Exclude)의 성능 하락 원인은 무엇인가?
* **가설 A (데이터 결핍)**: GMM 확률 95% 미만인 경계선 궤적(33.06%)을 제거하면서, 모델이 다차원 전이 상태의 궤적을 평가하는 보간 능력을 잃어버렸을 가능성. (즉, 노이즈 제거의 부작용)
* **가설 B (과분할 풀링의 정보 손실)**: 6개 컴포넌트를 3개(Cruising, Gliding, Steering)로 단순화하면서, Fast-Extreme Turning(난이도 최상)과 Slow-Extreme Turning의 고유 물리 차이가 섞여버린 것일 수 있음.

### Q2. Dirichlet Process GMM (BGM) 도입 시 12개 군집을 어떻게 요리할 것인가?
* BGM 자동 군집화 결과 **12개의 활성 미세 비행 상태**가 도출되었으며, 8차원 주성분(PCA) 공간에서 기하학적으로 완벽히 격리됨을 확인 (보고서 및 시각 자료 [bgm_auto_clustering_distribution.png](file:///d:/Repos/dacon-mosquito-trajectory/docs/images/bgm_auto_clustering_distribution.png) 참고).
* **Option 1**: 12개 전문 미세 모델을 개별 학습시킬 것인가(Data Sparsity 극대화 우려), 단일 대용량 모델에 12-BGM 라벨을 카테고리 피처로 넘길 것인가?
* **Denoising Threshold 임계값**: 95%로 데이터를 과도하게 쳐내는 대신 80%, 50%, 혹은 0%(전체 학습 데이터 보존 + 확률 가중치 소프트 피처링) 중 어떤 방향이 최적일까?

---

## 🚀 5. AI 어시스턴트용 핵심 파일 참조 링크

* **BGM 군집화 분석 보고서**: [trajectory_clustering_analysis.md](file:///d:/Repos/dacon-mosquito-trajectory/docs/analysis/trajectory_clustering_analysis.md)
* **Step 40 워크스루**: [step40_walkthrough.md](file:///d:/Repos/dacon-mosquito-trajectory/docs/experiments_history/step40_walkthrough.md)
* **BGM 시각화 이미지**: [bgm_auto_clustering_distribution.png](file:///d:/Repos/dacon-mosquito-trajectory/docs/images/bgm_auto_clustering_distribution.png)
* **비행 상태 학습 데이터 준비 스크립트**: [prepare_data.py](file:///d:/Repos/dacon-mosquito-trajectory/experiments/step40_dual_specialized/prepare_data.py)
* **학습 스크립트**: [train_ranker.py](file:///d:/Repos/dacon-mosquito-trajectory/experiments/step40_dual_specialized/train_ranker.py)
* **추론 및 블렌딩 스크립트**: [inference.py](file:///d:/Repos/dacon-mosquito-trajectory/experiments/step40_dual_specialized/inference.py)
