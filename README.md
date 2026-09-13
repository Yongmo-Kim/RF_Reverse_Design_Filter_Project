<div align="center">
  <h1>⚡ CorRaL V4 : 딥러닝 기반 픽셀형 RF 대역통과 필터(BPF) 역설계 및 전자기 해석 완전 자동화 플랫폼</h1>
  <h3>금오공과대학교 캡스톤 디자인 종합설계 프로젝트 (한국전자파학회 2026 하계종합학술대회 발표 논문)</h3>
  <p>
    <img src="https://img.shields.io/badge/Language-Python_3.9+-blue?style=for-the-badge&logo=python" alt="Python"/>
    <img src="https://img.shields.io/badge/Framework-PyTorch_&_PyQt5-orange?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/HPC-PBS_Pro_Cluster-red?style=for-the-badge" alt="HPC"/>
    <img src="https://img.shields.io/badge/Simulation-3D_FEM_EMerge-green?style=for-the-badge" alt="EMerge"/>
    <img src="https://img.shields.io/badge/Dataset-100%2C000%2B_Simulations-purple?style=for-the-badge" alt="Dataset"/>
  </p>
</div>

<br/>

## 📖 1. 프로젝트 총괄 개요 (Executive Summary)

본 프로젝트는 **UWB(Ultra-Wideband) 레이더 및 차세대 통신 시스템**에 적용 가능한 고성능 평면형 마이크로스트립 대역통과 필터(BPF)를 **초고속으로 자동 설계하는 AI-Assisted Inverse Design 플랫폼**입니다.

전통적인 RF 회로 설계 방식은 고주파 엔지니어의 경험과 직관에 의존하여 기본 구조를 잡은 후, 상용 전자계 시뮬레이터(Keysight ADS, Ansys HFSS 등)를 이용해 파라미터를 수동으로 하나하나 튜닝하는 '반복적 시행착오'를 거칩니다. 이는 1회의 3D EM 시뮬레이션에만 수십 분이 소요되어 설계 기간과 비용이 천문학적으로 증가하는 치명적 한계가 있습니다.

이를 혁신하기 위해 본 프로젝트는 다음과 같은 **완전 자동화 엔드투엔드(End-to-End) 파이프라인**을 완성했습니다:
1. **오픈소스 3D FEM 솔버(EMerge)**를 금오공대 슈퍼컴퓨터(HPC) 클러스터에 이식하여 **100,000건 이상의 대규모 고품질 S-파라미터 데이터셋**을 완전 무인 자동화로 구축.
2. 25×25 이진 픽셀 격자 구조를 273차원 복소 S-파라미터(크기·위상)로 매핑하는 **ResNet18 / ResNet50 / DenseNet121 기반 전이학습(Transfer Learning) 순방향 대리 모델(Forward Surrogate)** 학습.
3. 사용자가 원하는 필터 사양(차단 주파수, 통과대역, 삽입 손실, 반사 손실, 감쇠도)을 입력하면 유전 알고리즘(GA) 및 앙상블 탐색을 통해 최적의 25×25 픽셀 레이아웃을 역으로 찾아내는 **헤드리스 배치 역설계 엔진 및 PyQt5 GUI 플랫폼** 구축.

---

## 🏗️ 2. 전체 시스템 아키텍처 및 워크플로우 (System Workflow)

전체 시스템은 **[데이터 생성 (HPC)] → [데이터 전처리 및 압축] → [AI 대리 모델 학습] → [역설계 최적화 알고리즘] → [GUI 시각화 및 EM 재검증]**의 유기적인 5단계로 연결됩니다.

```mermaid
graph TD
    subgraph Step1["1. HPC 기반 대규모 전자기 시뮬레이션"]
        A[25x25 Random Pixel Generator] -->|DRC 검증: 고립패턴 제거| B[기하학적 2D 레이아웃]
        B -->|GMSH 자동 메쉬 분할| C[3D FEM EMerge 수치해석]
        C -->|PARDISO 16코어 병렬 연산| D[원시 결과: S2P 파일 추출]
    end

    subgraph Step2["2. 고속 전처리 파이프라인"]
        D --> E[S-Parameter 크기/위상 273차원 분리]
        E --> F[NumPy 압축 포맷 NPZ 변환]
        F --> G[10만 건 고품질 데이터베이스 구축]
    end

    subgraph Step3["3. 심층 신경망 대리 모델 학습"]
        G --> H[ResNet18 / ResNet50 / DenseNet121]
        H -->|Kaiming Normal & Dropout 0.3->0.2| I[순방향 S-Parameter 예측 앙상블 모델]
    end

    subgraph Step4["4. 역설계 탐색 엔진 (Batch Inverse)"]
        J[사용자 목표 BPF 사양 입력<br>f_low, f_high, IL, RL, Attenuation] --> K[유전 알고리즘 & 앙상블 목적함수 최적화]
        I -.->|초고속 대리 평가 0.001초| K
        K --> L[최적의 25x25 바이너리 픽셀 레이아웃 도출]
    end

    subgraph Step5["5. GUI 시각화 및 최종 검증"]
        L --> M[PyQt5 Interactive GUI 표출]
        M --> N[스미스 차트 & S-파라미터 실시간 비교]
        M --> O[EMerge 3D FEM 사후 검증 수행]
    end

    style Step1 fill:#e3f2fd,stroke:#1565c0
    style Step2 fill:#e8f5e9,stroke:#2e7d32
    style Step3 fill:#fff3e0,stroke:#e65100
    style Step4 fill:#f3e5f5,stroke:#7b1fa2
    style Step5 fill:#fbe9e7,stroke:#d84315
```

---

## 📐 3. EMerge 3D FEM 시뮬레이션 물리적 기판 및 에어박스 규격

상용 툴(ADS)과 동일한 정확도를 보장하면서 슈퍼컴퓨터의 메모리 폭발(Memory Kill)을 방지하기 위해 정밀 튜닝된 3차원 물리 구조입니다.

```
       ▲ Z축 (전체 높이: 10.592 mm)
       │
═══════╪══════════════════════════════════════════════════ 2차 Bayliss-Turkel ABC 흡수 경계면 (상단)
       │  ▲
       │  │
       │  │  에어박스 공간 (Air Box)
       │  │  높이: 9.375 mm (λ/4 메쉬 폭발 억제 최적점)
       │  │  X-Y 단면: 40.0 mm x 40.0 mm (기판 크기의 2배)
       │  │  (외곽 메쉬: 2.5 mm, Growth Rate: 5)
       │  ▼
───────┼──────────────────────────────────────────────────
┌──────┴────────────────────────────────────────────────┐ 
│   [Top Metal Layer] 금(Gold) 픽셀 패턴 (두께: 0.017 mm)  │ ◄── 25x25 격자 (셀당 0.4x0.4 mm, 활성영역 10x10 mm)
├───────────────────────────────────────────────────────┤     (가장자리 밀집 메쉬: 0.333 mm, Skin Effect 포착)
│                                                       │
│   [Dielectric Substrate] FR-4 유사 유전체 기판          │
│   두께(H): 1.2 mm, 유전율(εr): 4.0, 손실 탄젠트: 0.013     │
│   기판 크기: 20.0 mm x 20.0 mm                         │
│   (기판 영역 메쉬: 1.0 mm)                              │
│                                                       │
├───────────────────────────────────────────────────────┤
│   [Ground Plane] 하부 접지면 도체 (두께: 0.017 mm)         │
└───────────────────────────────────────────────────────┘
══════════════════════════════════════════════════════════ PEC 바닥 접지 (흡수 경계 미적용)
```

### 🔬 전자기 물리 현상을 고려한 그라데이션 메쉬(Gradient Mesh) 전략
* **Skin Effect(표피 효과) 대응:** 고주파(최대 30GHz)에서 전류가 도체 표면에 집중되는 현상을 잡기 위해 금속 가장자리 메쉬는 유전체 파장의 1/15 수준인 **0.333mm**로 극도로 조밀하게 분할.
* **Fringing Field(프린징 필드) 대응:** 도체 모서리에서 공기 중으로 방사되는 전자기파를 모사하기 위해 도체 부근 에어 메쉬는 촘촘하게, 외곽으로 갈수록 거칠어지는 **Growth Rate 5**의 완화 메쉬 적용.
* **에어박스 Z축 최적화:** 이론적 $\lambda/4$ 설정(18.75mm) 시 발생하는 수백만 개 노드의 메모리 오버플로우를 차단하고, 수많은 반복 실험을 통해 정확도를 완벽히 보존하는 **9.375mm 최적 높이** 도출.

---

## 🧠 4. 딥러닝 순방향 대리 모델 아키텍처 (`models.py`)

기존 ImageNet 기반 거대 모델들을 **25×25 1채널 소형 바이너리 입력**에 맞게 수학적/구조적으로 전면 개조했습니다.

### 1) 입력 어댑테이션 및 가중치 전이 (Weight Interpolation)
* 원본 ResNet의 `conv1` (7×7, Stride 2, 3채널)은 25×25 크기의 픽셀 정보를 단번에 잃어버리는 문제가 있습니다.
* 이를 해결하기 위해 첫 단을 **3×3 Kernel, Stride 1, Padding 1, 1채널 Conv 레이어로 개조**하고, ImageNet 사전학습 3채널 가중치를 채널 평균 후 Bilinear Interpolation(3×3)하여 이식하는 **커스텀 전이학습**을 구현했습니다.
* Max Pooling 레이어는 `nn.Identity()`로 치환하여 소형 픽셀의 공간 해상도 손실을 방지했습니다.

### 2) 다층 회귀 헤드 (Multi-Stage Regression Head)
* **ResNet50:** 2048 차원 특징 벡터 $\rightarrow$ `Linear(2048, 1024)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(1024, 512)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(512, 273)`
* **출력 273차원 구성:** 0.1GHz ~ 30GHz 구간의 총 91개 주파수 샘플 포인트에 대해 4개 S-파라미터 크기 및 위상 완벽 예측.
  * Magnitude: $S_{11}(dB), S_{21}(dB)$ (주요 대역)
  * Phase: $S_{11}(\theta), S_{21}(\theta)$

---

## ⚙️ 5. 헤드리스 배치 역설계 엔진 (`run_batch_inverse.py`)

슈퍼컴퓨터 및 서버 환경에서 GUI 없이 대량의 필터 후보군을 고속 탐색하는 최적화 엔진입니다.

1. **타깃 스펙 정의 (Target Spec):**
   * 통과대역 시작 주파수($f_{low}$) 및 종료 주파수($f_{high}$)
   * 삽입 손실(Insertion Loss), 반사 손실(Return Loss), 저지대역 감쇠량(Stopband Loss)
2. **복합 손실 함수 (Composite Cost Function):**
   $$\text{Cost} = w_1 \cdot \text{Loss}_{\text{Passband}} + w_2 \cdot \text{Loss}_{\text{Stopband}} + w_3 \cdot \text{Loss}_{\text{ReturnLoss}} + w_{\text{DRC}} \cdot \text{Penalty}_{\text{DRC}}$$
   * 실제 제작이 불가능한 고립 패턴(Floating Metal) 및 에칭 한계 미달 패턴에 강력한 페널티 부여 (DRC 필터링).
3. **앙상블 대리 평가:** ResNet18, ResNet50, DenseNet 복수 모델의 추론 결과를 앙상블하여 예측 분산을 줄이고 전자기적 신뢰성을 극대화.

---

## 🖥️ 6. PyQt5 기반 대화형 GUI 플랫폼 (`inverse_gui_fixed.py`)

엔지니어가 복잡한 코드 없이 클릭 몇 번으로 필터를 역설계하고 검증할 수 있는 통합 워크스테이션입니다.

* **Spec Controller:** $f_{low}, f_{high}$, 대역폭, 손실 허용치를 슬라이더 및 입력창으로 실시간 세팅.
* **Layout Grid View:** AI가 추천한 25×25 픽셀 레이아웃을 직관적인 2D 그래픽 뷰어로 표출하며, 엔지니어가 직접 마우스 클릭으로 패턴을 수정/추가 가능.
* **S-Parameter & Smith Chart Display:** 
  * 목표 스펙 라인 vs AI 예측치 vs EM 실측치를 동일 그래프에 오버레이 표출.
  * 복소 임피던스 궤적을 확인할 수 있는 스미스 차트(Smith Chart) 뷰어 내장.
* **원클릭 검증 파이프라인:** 도출된 최적 픽셀 패턴을 즉시 `EMerge` 시뮬레이션 입력 파일로 변환하여 실제 3D 전자기 해석을 백그라운드로 트리거.

---

## 📁 디렉토리 구조 (Repository Layout)

```
RF_Reverse_Design_Filter_Project/
├── 25x25_pixel_circuit/           # 25x25 랜덤 픽셀 생성기 및 EMerge 연동 스크립트
│   ├── generator_new.py          # DRC 규칙 기반 25x25 바이너리 패턴 생성기
│   ├── simulation.py             # EMerge 전자기 수치해석 실행 스크립트
│   ├── simulation_gmsh.py        # GMSH 메쉬 생성 및 경계조건 설정
│   └── simulation_structure.py   # 기판 및 포트 형상 정의
├── 25x25_custome_pixel_circuit/  # 특정 패턴 분석용 커스텀 파이프라인
├── KIT_HPC/                      # 금오공대 슈퍼컴퓨터(HPC) PBS 배치 스크립트
│   ├── 8CPU_CORE/                # 8코어 단일 노드 최적화 실행 스크립트 (run_batch.sh)
│   └── 16CPU_CORE/               # 16코어 고속 분산 병렬 스크립트 (PARDISO 솔버)
├── Deeplearning&GUI/CorRaL_V4/   # 메인 딥러닝 역설계 플랫폼
│   ├── training/                 # 딥러닝 모델 학습 소스코드
│   │   ├── models.py             # ResNet18/50, DenseNet121 대리 모델 정의
│   │   └── train_25x25.py        # 10만 건 데이터셋 학습 및 검증 루프
│   ├── batch_inverse/            # 슈퍼컴퓨터용 헤드리스 배치 역설계 엔진
│   │   ├── run_batch_inverse.py  # 목적함수 기반 픽셀 탐색 알고리즘
│   │   └── run_batch_inverse.pbs # PBS 작업 제출 스크립트
│   └── gui_resnet/               # 엔드유저용 PyQt5 GUI 통합 애플리케이션
│       ├── inverse_gui_fixed.py  # 메인 GUI 구동 파일
│       ├── Sparam_compare_gui.py # S-파라미터 비교 분석 도구
│       └── Renorm_ops.py         # 임피던스 재정규화 연산 모듈
└── README.md                     # 프로젝트 총괄 기술 문서
```

---

## 🏆 핵심 성과 및 의의 (Key Achievements)

1. **압도적인 설계 시간 단축 (TAT 혁신):**
   * 기존 3D EM 시뮬레이션 기반 반복 설계(수일~수주일 소요) $\rightarrow$ **딥러닝 대리 모델 역설계로 단 1분 이내에 최적 레이아웃 도출**.
2. **100,000건 무인 데이터 파이프라인 완성:**
   * 리눅스 PBS 큐와 EMerge를 결합하여 서버 다운 없는 100% 무인 자동화 전자기 해석 인프라를 구축.
3. **물리 기반 인공지능(Physics-Aware AI):**
   * 블랙박스 형태의 AI를 지양하고, Skin Effect, Fringing Field, DRC 최소 선폭 등 전자기학적·제조 공학적 물리 제약을 모델과 손실 함수에 직접 반영하여 **'실제 제작 가능한(Manufacturable)' 무결점 RF 필터 구조 설계 실현**.
4. **학술적 성과 인정:**
   * 본 연구의 성과를 인정받아 **2026년도 한국전자파학회(KIEES) 하계종합학술대회 논문 발표** 완료.