# RF_Reverse_Design_Filter_Project[README.md](https://github.com/user-attachments/files/32098765/README.md)
# AI-Assisted RF Filter Inverse Design Platform

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-GUI-green.svg)](https://riverbankcomputing.com/software/pyqt/)

본 프로젝트는 **"AI 기반 UWB 레이더 시스템을 위한 픽셀 타입 RF 필터 회로 역설계(Inverse Design) 자동화 플랫폼"**입니다. 전통적인 3D EM(전자기) 시뮬레이션(FDTD/FEM 등)의 막대한 컴퓨팅 비용과 시간 비효율성을 극복하기 위해 딥러닝(Deep Learning)을 활용하여 직관적이지 않은 임의의 RF 대역통과 필터(BPF) 설계를 자동화하는 전문가 AI 회로 설계 플랫폼입니다.

---

## 🎯 주요 목표 (Core Goals)
사용자가 원하는 목표 S-parameter 사양(Spec)을 입력하면, 이를 만족하는 **25x25 픽셀 형태의 RF 필터 레이아웃을 단 몇 초 만에 자동으로 생성 및 렌더링**합니다. 생성된 모든 레이아웃은 공정 설계 규칙(DRC, Design Rule Check)을 엄격하게 준수합니다.

## 🧠 시스템 아키텍처 (Architecture)

본 시스템은 크게 5가지 핵심 파이프라인으로 구성됩니다.

1. **Data Generation & Preprocessing (`preprocess_25.py`)**
   - 존(Zone) 기반 가중치 확률 분포를 사용하여 무작위 25x25 이진 픽셀 레이아웃을 생성합니다.
   - S-parameter(`.s2p`) 및 레이아웃(`.npz`) 데이터를 파싱하여 모델 학습을 위한 균일한 데이터셋으로 전처리합니다. 기하학적 대칭성을 활용하여 데이터를 4배로 증강(Augmentation)합니다.
2. **Forward Modeling (Surrogate Model) (`models.py`, `train_25x25.py`)**
   - 주어진 25x25 픽셀 레이아웃의 주파수 응답(S-parameter)을 예측하는 딥러닝 모델(대체 모델, Surrogate Model)을 학습합니다.
   - **Physics-Informed Asymmetric Hinge Loss**를 적용하여 통과 대역(Passband)과 저지 대역(Stopband)의 민감도에 따라 다르게 학습되도록 최적화합니다.
3. **Hybrid Inverse Design (`run_batch_inverse.py`)**
   - $2^{625}$에 달하는 방대한 설계 공간을 탐색하기 위한 하이브리드 최적화 기법을 사용합니다.
   - 글로벌 메타 휴리스틱 알고리즘(유전 알고리즘 GA, BPSO)과 로컬 연속 튜닝 알고리즘(경사 하강법 GD, DBS)을 결합하여 최적의 픽셀 레이아웃을 역추적합니다.
4. **HPC EM Simulation Validation (`simulation.py`)**
   - AI가 생성한 레이아웃을 Full-wave EM 시뮬레이터와 연동하여 실제 S-parameter를 추출하고 검증합니다. (Gmsh 메싱 및 PCB 지오메트리 생성 지원)
5. **Integrated GUI Dashboard (`inverse_gui_fixed.py`)**
   - 사용자가 대역파괴 주파수, 삽입 손실, 반사 손실 등의 타겟을 직접 설정할 수 있는 대시보드를 제공합니다.
   - PyQt5와 QThread를 이용해 비동기 AI 추론, 실시간 레이아웃 변경, S-parameter 커브 및 스미스 차트(Smith Chart)를 시각적으로 렌더링합니다.

---

## 🎛️ 기판 및 시뮬레이션 환경 (Substrate & Simulation Environment)
- **기판 크기 (Board Size)**: 20 x 20 mm
- **기판 두께 (Thickness)**: 1.2 mm
- **유전율 ($\epsilon_r$, Dielectric Constant)**: 4.0
- **손실 탄젠트 ($\tan \delta$, Loss Tangent)**: 0.013
- **도체 두께 (Trace Thickness)**: 0.017 mm ($17\mu m$, Half-oz)
- **도체 전도도 (Conductivity)**: $4.1 \times 10^7$ S/m (Gold)
- **에어박스 (Airbox Size)**: 40 x 40 mm

---

## 🔄 전체 실행 흐름 (Execution Flow)
프로젝트는 아래와 같은 순서로 데이터를 생성하고 AI 모델을 학습시켜 최종 역설계를 수행합니다.

1. **데이터 생성 및 시뮬레이션 (`simulation.py`)**
   - 무작위 25x25 픽셀 레이아웃 패턴을 생성합니다 (DRC 준수).
   - 생성된 패턴을 HPC EM 시뮬레이터에 입력하여 원시 S-parameter 결과(`.s2p`)와 레이아웃 정보(`.npz`)를 추출합니다.
2. **데이터 전처리 (`preprocess_25.py`)**
   - 생성된 대량의 시뮬레이션 데이터 파일들을 취합하여 정렬합니다.
   - 주파수 포인트를 통일하고 모델 학습에 적합한 형태의 텐서 데이터셋으로 일괄 변환 및 증강(Augmentation)합니다.
3. **AI 대체 모델 학습 (`train_25x25.py`)**
   - 전처리된 데이터셋을 이용하여 딥러닝 모델(ResNet, DenseNet 기반 Surrogate 모델)을 학습시킵니다.
   - 학습이 완료된 모델은 무거운 3D 시뮬레이션을 대체하여 레이아웃의 전자기적 응답을 밀리초(ms) 단위로 추론할 수 있게 됩니다.
4. **역설계 및 최적화 (`inverse_gui_fixed.py` 또는 `run_batch_inverse.py`)**
   - 사용자가 목표로 하는 S-parameter 타겟 사양을 입력합니다 (GUI 또는 CLI).
   - 메타 휴리스틱 최적화 알고리즘(GA, BPSO 등)이 수백만 개의 후보 레이아웃을 생성하고, 학습된 AI 모델이 이를 초고속으로 평가하여 최적의 레이아웃을 역추적합니다.
5. **최종 검증 (Final Validation)**
   - AI 기반 역설계로 도출된 최종 최적의 레이아웃을 다시 `simulation.py`에 통과시켜, 실제 Full-wave EM 환경에서의 물리적 응답을 검증하고 최종 결과를 확정합니다.

---

## 📂 파일 구조 및 역할 (Project Structure)

| 파일명 | 설명 |
|---|---|
| `models.py` | PyTorch 기반 신경망 아키텍처 정의. 1채널 25x25 이진 입력을 다채널 회귀 출력으로 변환하도록 최적화된 **ResNet18, ResNet50, DenseNet**을 포함합니다. |
| `train_25x25.py` | Forward Surrogate 모델 학습 스크립트. K-Fold 교차 검증, EMA, TF32/Autocast(H100 최적화), Channel-Weighted SmoothL1Loss 등을 지원합니다. |
| `preprocess_25.py` | S-parameter와 레이아웃 원시 데이터를 모델 학습이 가능한 형태로 병합하고 정렬하는 전처리 스크립트입니다. |
| `run_batch_inverse.py` | HPC 클러스터 등에서 Headless 방식으로 역설계 배치 작업을 수행합니다. GA, BPSO, DBS 등의 최적화 알고리즘이 구현되어 있습니다. |
| `simulation.py` | `emerge` HPC EM 시뮬레이터와 인터페이싱하여 예측된 레이아웃의 실제 EM 반응을 시뮬레이션하고 검증합니다. |
| `inverse_gui_fixed.py` | 대화형 역설계를 위한 PyQt5 기반 GUI 애플리케이션입니다. |
| `run_batch.sh` / `submit_batch.sh` | HPC 환경에서의 배치 실행 및 작업 스케줄링을 위한 쉘 스크립트입니다. |
| `2026_1학기_캡스톤_계획서.pdf` | 본 프로젝트의 캡스톤 디자인 기획서 |
| `전자파학회 하계 논문.pdf` | 본 연구의 전자파학회 하계 학술대회 발표 논문 |

---

## 🤖 사용된 머신러닝 모델 (Machine Learning Models)

물리적 EM 시뮬레이션을 근사(Approximate)하기 위해 2D CNN을 Surrogate 모델로 활용하여 역설계 평가 속도를 획기적으로 높였습니다.
- **ResNet18_25x25**: 25x25 그리드에 맞춘 경량화된 ResNet-18 + 커스텀 Regression Head
- **ResNet50_25x25**: 더 높은 표현 능력(Representational Capacity)을 갖춘 깊은 ResNet-50 변형 모델
- **DenseNet25x25**: Dense Feature Block 연결을 활용한 DenseNet-121 변형 모델
*(모든 모델은 25x25 이진 이미지 레이아웃을 입력으로 받아 연속된 주파수 포인트에 대한 $S_{11}, S_{12}, S_{21}, S_{22}$ (dB 및 phase)를 출력합니다.)*

---

## 📊 주요 성능 및 결과 (Key Results)

- **설계 성능 검증**: 타겟 주파수 대역(5.6 ~ 7.7 GHz)에서 Return Loss $\le -10$ dB, Insertion Loss $\le -1.5$ dB 스펙을 만족하는 픽셀형 RF 필터 레이아웃을 성공적으로 생성했습니다.
- **높은 정확도**: 실제 EM 시뮬레이션으로 검증한 결과, 타겟 통과 대역(Passband) 내에서 **오차 약 0.3 dB 수준, 최소 상대 오차 불과 0.93%** 라는 매우 뛰어난 정확도를 달성했습니다.
- **혁신적인 효율성**: 수작업에 의존하던 파라미터 스위핑과 장시간이 소요되는 3D EM 시뮬레이션 사이클에서 벗어나, AI Surrogate 모델 기반 자동화를 통해 필터 설계의 패러다임을 바꿨습니다.

---

## 🚀 시작하기 (Getting Started)

### 1. GUI 실행 (인터랙티브 모드)
타겟 S-parameter를 설정하고 실시간으로 최적화 과정을 시각적으로 확인하려면 GUI를 실행합니다.
```bash
python inverse_gui_fixed.py
```

### 2. 모델 학습
새로운 데이터셋으로 모델을 처음부터 학습시키려면 다음 명령어를 사용합니다.
```bash
python train_25x25.py
```

### 3. CLI 기반 배치 역설계
백그라운드에서 하이브리드 최적화를 통해 새로운 레이아웃을 탐색하려면 다음 스크립트를 실행합니다.
```bash
python run_batch_inverse.py
```
*(HPC 클러스터를 사용하는 경우 `submit_batch.sh` 스크립트를 활용하세요.)*
