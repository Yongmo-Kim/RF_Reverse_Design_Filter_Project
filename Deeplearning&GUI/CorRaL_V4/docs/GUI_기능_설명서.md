# Pixelated RF Filter Wizard — GUI 기능 설명서

> 25×25 픽셀 RF 필터 역설계 GUI의 **모든 기능**을 위젯 단위로 정리한 문서입니다.
> 코드 위치: `gui_resnet/inverse_gui_fixed.py`, `impedance_test_gui.py`, `Sparam_compare_gui.py`, `inverse_core.py`

---

## 0. 전체 구조

GUI는 **상단의 3개 메인 탭**으로 구성됩니다.

| 메인 탭 | 역할 | 소스 파일 |
|---|---|---|
| **Inverse Design** | ML 모델로 25×25 픽셀 레이아웃 역설계 | `inverse_gui_fixed.py` |
| **Impedance Renormalization** | 단일 s2p 파일을 다른 Z₀ 기준으로 재정규화 후 시각화 | `impedance_test_gui.py` |
| **S-Param Compare** | 역설계 / 시뮬레이션 / 측정 3종 s2p 비교 | `Sparam_compare_gui.py` |

창 제목: *"Pixelated RF Filter Wizard"*, 최소 크기 1600×1000.

---

# 탭 1 : Inverse Design

가장 핵심 탭. 화면이 **좌측 패널(파라미터 입력) / 우측 패널(시각화)** 로 나뉘고, 좌측 패널은 다시 **3개의 소탭(Model / Target / Log·Convergence)** 으로 구성됩니다.

## 1.1 좌측 — 소탭 ① **Model**

### Model 그룹

| 위젯 | 기능 | 입력 예 |
|---|---|---|
| **Forward Model** 콤보 | 사용할 forward 신경망 패밀리 선택 | `DenseNet (Friend)` / `ResNet (Mine: 18/50)` |
| **Status 라벨** | 현재 로드된 모델 개수·타입·디바이스 표시 | "resnet Forward 6개 로드" |
| **모델 폴더 직접 선택** 버튼 | 저장된 `.pth` 폴더를 수동으로 지정 | 폴더 다이얼로그 |

내부에서는 `load_models()` → `_GUIEnsembleSurrogate` 가 ensemble 평균/표준편차를 계산해 forward 예측을 수행합니다. ResNet 선택 시 18·50을 자동으로 모두 로드해서 6개 ensemble을 구성합니다.

### Optimization 그룹

| 위젯 | 기능 | 기본값 |
|---|---|---|
| **Pop Size** | GA/BPSO 한 세대당 후보 수 (population) | 4096 |
| **Chunk Size** | 신경망에 한 번에 통과시킬 batch 크기 (GPU VRAM 한도) | 512 |

`Pop Size`가 클수록 탐색 다양성↑·GPU 메모리 부담↑.
`Chunk Size`는 OOM 방지용 분할 크기로 결과 품질에는 영향 없음.

### Algorithm 그룹

**Algorithm 콤보** (10개 옵션)

| 항목 | 의미 |
|---|---|
| `GD Only` | Gumbel-Softmax 연속 완화 + 그라디언트 디센트 (논리량 → 0/1) |
| `GA Only` | 유전 알고리즘 단독 |
| `BPSO Only` | 이진 PSO 단독 |
| `BO Only` | Bayesian Optimization (가우시안 프로세스 기반) |
| `DE Only` | Differential Evolution |
| `GA->GD Only` | GA로 거친 탐색 → 상위 시드를 GD로 미세조정 (하이브리드) |
| `BPSO->GD Only` | BPSO → GD 하이브리드 |
| `GA->DBS Only` | GA → DBS(Direct Binary Search, 픽셀 단위 단방향 그리디) |
| `BPSO->DBS Only` | BPSO → DBS |
| `All (9 Algorithms)` | 위 9개 모두 동시 실행 후 비교 |

콤보 팝업은 `setMaxVisibleItems(12)` + 커스텀 stylesheet로 가독성 강화.

### Result 그룹 (스크롤)

`Show:` 버튼에서 켜진 알고리즘만 행을 표시. 각 행은:

| 컬럼 | 내용 |
|---|---|
| Algorithm | 알고리즘 이름 (예: GA_DBS) — 13px bold |
| **S21 dB** | 통과대역 평균 S21 (dB), 18px font·높이 50px 박스 |
| **S11 dB** | 통과대역 평균 S11 (dB), 18px font·높이 50px 박스 |

하단 `Best:` 라벨은 가장 낮은 에러를 낸 알고리즘과 에러값(`<algo> (Error <err>)` 형식)을 표시.

---

## 1.2 좌측 — 소탭 ② **Target**

### Parameter 그룹

| 위젯 | 의미 | 비고 |
|---|---|---|
| **F<sub>L</sub> (GHz)** | 통과대역 시작 주파수 | 사용자 입력 |
| **F<sub>H</sub> (GHz)** | 통과대역 끝 주파수 | 사용자 입력 |
| **F<sub>c</sub> (GHz)** | 통과대역 중심 = (F_L+F_H)/2 | **읽기 전용 (자동 계산)** |
| **BW (%)** | 부분 대역폭 = (F_H-F_L)/F_c × 100 | **읽기 전용** |
| **L-PB off (GHz)** | 좌측 통과대역 오프셋 | S21/S12 타겟선 배경 폭만 결정 |
| **L-PB (GHz)** | F_L − offset (자동 계산) | 읽기 전용 |
| **R-PB off (GHz)** | 우측 통과대역 오프셋 | 동일 |
| **R-PB (GHz)** | F_H + offset (자동 계산) | 읽기 전용 |
| **Insertion Loss (dB)** | 통과대역 S21 목표 수준 (음수, 예: −1.5) | 통과대역 손실 hinge에 사용 |
| **Return Loss (dB)** | 통과대역 S11/S22 목표 수준 (음수, 예: −20) | hinge `max(0, |Γ|−Γ_limit)` 의 한계값 결정 |
| **Stopband Loss (dB)** | 저지대역 감쇠 추가 마진 (음수) | **실제 stopband 타겟 = IL − 3 dB + SL**. 예: IL=−1, SL=−6 → −10 dB |

`F_L`/`F_H` 또는 `*-PB off`가 바뀔 때마다 `_update_fc_bw()` / `_update_pass_band()` 가 자동 호출되어 읽기 전용 셀들이 동기화.
`_check_spec_feasibility()` 가 IL/RL/SL 값이 비물리적이면 **노란 경고 박스**를 띄워줍니다 (예: IL이 SL보다 낮을 때).

### Port Impedance (재정규화) 그룹

| 위젯 | 의미 |
|---|---|
| **P1 Z0 (Ω)** | 포트 1의 목표 임피던스. `50+0j`, `100+0j`, `75-5j` 등 복소수 입력 가능 |
| **P2 Z0 (Ω)** | 포트 2의 목표 임피던스 |
| 안내 라벨 | "50Ω 기준 모델을 사용합니다…" |
| **▶ Apply Renorm** 버튼 | 현재 모델 결과를 P1/P2 Z₀로 재정규화 → 스미스차트와 S-파라미터 그래프 즉시 갱신 |

내부 동작: `_renormalize_complex_eval()` (inverse_core.py L524) 이 power-wave S-parameter 변환을 수행하여 50Ω 기준 예측을 사용자 Z₀ 기준으로 변환. 역설계 score 계산도 이 새 Z₀ 기준에서 수행됩니다(통과대역 |Γ|² 원점-당김 항이 새 원점을 따라감).

### Plot X-Axis 그룹

| 위젯 | 의미 | 기본값 |
|---|---|---|
| **Plot FL (GHz)** | 우측 그래프들의 X축 시작 | 0.1 |
| **Plot FH (GHz)** | 우측 그래프들의 X축 끝 | 10.0 |
| **X-Axis Ticks** | 눈금 개수 | 11 |
| **Apply Plot X-Axis** 버튼 | `_viz_compare()` 호출 → S-파라미터 / 스미스차트 다시 그림 |

스미스차트의 `_smith_display_mask()` 는 이 FL/FH 범위 안의 점만 표시합니다. **통과대역만 보고 싶으면 FL=F_L, FH=F_H 로 좁히면** 점이 통과대역만 남습니다.

---

## 1.3 좌측 — 소탭 ③ **Log / Convergence** (수직 분할)

| 패널 | 내용 |
|---|---|
| **상단: Log** | 역설계 진행 텍스트 (스크롤). 알고리즘 시작/종료, best score 갱신, 워닝 등 |
| **하단: Convergence** | matplotlib 그래프. X=Step, Y=Error. 알고리즘별 색으로 수렴 곡선 표시 |

분할 비율은 3:2 (Splitter로 사용자가 드래그해 조정 가능).

---

## 1.4 좌측 하단 — 실행 컨트롤

| 위젯 | 기능 |
|---|---|
| **▶ Start Inverse Design** | 14pt bold·높이 70px. `run_inverse()` 호출 — `InverseDesignWorker(QThread)` 가 백그라운드에서 알고리즘 실행 |
| **■ Stop** | 12pt bold·높이 50px. 워커에 stop 시그널 송신 (현 step 종료 후 안전 종료) |
| **Progress Bar** | 28px 두께·% 텍스트 표시. 알고리즘 진행률 |

---

## 1.5 우측 — 시각화 영역

### 상단 컨트롤 바

| 위젯 | 기능 |
|---|---|
| **Target Line: Show Target** 토글 | S11/S22/S21/S12 비교 그래프의 점선 타겟 표시·숨김 (`_on_target_line_toggle()`) |

> *Low-pass / Band-pass 분리선 프리셋 버튼은 제거되었으나 내부 호환을 위해 hidden widget으로 유지됨.*

### S-Parameter Response 그룹 (좌측 큰 영역)

상하로 4개의 plot:

1. **S11 (Return Loss)** — 통과대역(녹색) / 전이대역(흰색) / 저지대역(분홍) 배경. 점선 타겟선.
2. **S22 (Return Loss)** — 동일 배경 / 타겟선
3. **S21 (Insertion Loss)** — 트라페조이드(사다리꼴) 타겟선. Stopband 타겟 = IL − 3dB + SL
4. **S12 (Insertion Loss)** — 동일

배경 의미:
- 분홍 = stopband
- 흰색 = transition band (offset 영역)
- 녹색 = passband

#### Show 버튼 행 (9개)

| 버튼 | 색 | 용도 |
|---|---|---|
| GD | 초록 | GD 알고리즘 곡선 토글 |
| GA | 파랑 | … |
| BPSO | 빨강 | … |
| BO | 보라 | … |
| DE | 주황 | … |
| GA_GD | 진청 | GA→GD 하이브리드 |
| BPSO_GD | 진적 | BPSO→GD |
| GA_DBS | 청록 | GA→DBS |
| BPSO_DBS | 진오렌지 | BPSO→DBS |

체크 상태에 따라 곡선이 표시/숨김되며, **Result 그룹의 행 표시도 동기화**됩니다 (`_filter_result_rows_by_algo()`).

### 우측 컬럼

#### Export s2p / npz 버튼
현재 best 알고리즘의 결과를 `.s2p`(Touchstone) 와 `.npz`(numpy archive) 두 형식으로 동시 저장.

#### Generated Layout (25×25)
matplotlib heatmap. 0=substrate(흰색), 1=metal(검은색). 역설계가 찾아낸 픽셀 패턴 시각화.

#### Smith Chart 그룹

| 위젯 | 기능 |
|---|---|
| **Smith** 캔버스 | matplotlib + skrf로 그린 스미스차트. 파랑=S11 trace, 빨강=S22 trace |
| **S11** 토글 | S11 곡선 표시/숨김 |
| **S22** 토글 | S22 곡선 표시/숨김 |
| **All** 토글 | ON: 모든 주파수 점을 한꺼번에 표시 (`[ALL pts]` 모드). OFF: 슬라이더로 단일 주파수 점 |
| **슬라이더** | (All 모드 OFF일 때만) 주파수 인덱스 이동. 값 변경 시 `_on_smith_slider()` 가 단일 점 강조 |
| **Freq 라벨** | 현재 주파수 (예: `9.800 GHz`) 또는 `ALL` |
| **Info 라벨** | 현재 점의 `S11 |Γ|=… ∠=…° Z=…+…jΩ` 형식 정보 |
| **Z₀ 표시 박스** | `Port1 Z0 = 50Ω | Port2 Z0 = 50Ω` (P1/P2 Z₀ 입력에서 자동 동기화) |

마우스 hover 시 `_on_smith_hover()` 가 가장 가까운 점의 |Γ|, 각도, Z, S-dB를 툴팁으로 띄워줍니다. 클릭 시 `_on_smith_click()` 으로 해당 주파수 인덱스로 슬라이더 점프.

워터마크: 캔버스 위에 wCoRaL 로고 PNG가 30% 투명도로 오버레이. 데이터가 없으면 중앙 큰 로고, 있으면 우하단 작은 로고로 자동 전환 (`_sync_external_smith_overlays()` 0.4초 주기).

---

# 탭 2 : Impedance Renormalization

단일 s2p 파일을 사용자 지정 P1/P2 Z₀ 기준으로 변환해 **임피던스 변경의 효과를 시각적으로 검증**하는 탭.

## 2.1 좌측 패널 (폭 360px 고정)

### File 그룹
| 위젯 | 기능 |
|---|---|
| Loaded s2p path 라벨 | 현재 파일 경로 표시 |
| **Select s2p File** | s2p 파일 다이얼로그 |

### Reference Impedance 그룹
| 위젯 | 의미 |
|---|---|
| **P1 Z0 (Ω)** / **P2 Z0 (Ω)** | 변환할 목표 Z₀. 복소수 가능 (예: `50-25j`) |

### Apply And Update Smith Chart 버튼
`_apply_update()` 호출 — 50Ω 기준 s2p를 새 Z₀로 변환 후 Smith·S-Parameter 모두 갱신.

### Plot X-Axis 그룹
탭 1과 동일 (Plot FL / FH / Ticks / Apply).

### Info 그룹 (자동 갱신)
| 항목 | 의미 |
|---|---|
| Status | "Loaded" / "Loading..." 등 |
| File range | s2p 주파수 범위 |
| Total points | s2p 총 주파수 점 개수 |
| Band points | Plot 범위 내 점 개수 |
| **S11 / S22 @ start** | 시작 주파수에서의 S11·S22 값 |
| **S11 / S22 @ end** | 끝 주파수에서의 S11·S22 값 |

## 2.2 우측 패널

| 영역 | 내용 |
|---|---|
| **S-Parameter Response** | 4-plot 그리드 (S11/S12/S21/S22 dB) |
| **Smith Chart** | 변환된 S11/S12/S21/S22 trace |
| **Show: S11/S12/S21/S22** 토글 | 스미스차트의 4 trace 개별 ON/OFF |
| **Z₀ 라벨** | `Port1 Z0 = ___ | Port2 Z0 = ___` |

탭 1과 같은 wCoRaL 워터마크 오버레이 적용.

---

# 탭 3 : S-Param Compare

3개 s2p 파일을 한 화면에서 비교하는 탭 (역설계 vs 시뮬레이션 vs 측정).

## 3.1 좌측 패널 (폭 260px)

### S2P Files 그룹 — 3슬롯
각 슬롯은 라벨(색 코딩) + 경로 입력 + `...` 브라우저 버튼.

| 슬롯 | 색 | 용도 |
|---|---|---|
| **Inverse Design** | 색1 | 역설계 결과 s2p |
| **Simulation** | 색2 | EM 시뮬레이션(HFSS/CST) 결과 s2p |
| **Measurement** | 색3 | VNA 측정 s2p |

### Reference Impedance (Renorm) 그룹
P1/P2 Z₀ 입력 (복소수 OK). 안내: "Loaded s2p files are treated as 50 ohm data and renormalized for comparison."

### Plot X-Axis 그룹
탭 1·2와 동일.

### View Mode 그룹

| 위젯 | 의미 |
|---|---|
| **Center Panel** 콤보 | 가운데 큰 패널을 무엇으로 채울지 선택: `Cartesian (dB) x 4` / `Smith Chart` / `Smith + Cartesian` |
| **Resample** 체크박스 | inverse·measurement 데이터를 simulation 주파수에 자동 리샘플링 |
| **Error Y-limit (%)** | 우측 에러 그래프의 ±Y축 범위 (기본 50) |
| **Smith Data Toggle** | Inverse/Simulation/Measurement 중 어느 데이터를 스미스에 그릴지 선택 |

## 3.2 중앙 — 비교 패널 (Center Panel 모드별)

- **Cartesian × 4**: S11/S12/S21/S22 4-plot. inv/sim/meas가 다른 색으로 겹쳐 그려짐.
- **Smith Chart**: 단일 큰 스미스차트.
- **Smith + Cartesian**: 위 둘을 분할 표시.

## 3.3 우측 — Error 패널
inv/meas vs sim의 % 오차 그래프 (4-plot). `_err_hover_*` 핸들러로 점 위 hover 시 정보 표시.

워터마크 오버레이는 탭 1·2와 동일 자동 동기화 규칙.

---

# 부록 A : 역설계 알고리즘 흐름

`run_inverse()` → `InverseDesignWorker(QThread).run()` → 알고리즘별 백엔드 호출.

```
사용자 입력 (Target/Z0/Algorithm)
       ↓
_build_args_namespace()
       ↓
_GUIEnsembleSurrogate (forward 모델 ensemble)
       ↓
알고리즘 (GA/BPSO/BO/DE/GD/GA_GD/...)
       ↓ population
_predict_full_stats() : 신경망 forward (mean, std)
       ↓ predicted S-params (50Ω 기준)
_renormalize_complex_eval() : P1/P2 Z0 으로 변환
       ↓
_constraint_first_score() : score 계산
   ├─ refl11/refl22 hinge   : max(0, |Γ|−Γ_limit) 평균
   ├─ origin_pull11/22      : |Γ|² 평균  ← 새 Z0 기준
   ├─ s21/s12 통과대역 hinge : max(0, IL−|S21|_dB)
   └─ s21/s12 저지대역 hinge : max(0, |S21|_dB−stopband_target)
       ↓ score
다음 세대 생성 (변이/교차)
```

`_constraint_first_score()` 내부 가중치 (현재 값):
- `_S11_HINGE_WEIGHT` (선형 hinge)
- `_S11_ORIGIN_PULL_WEIGHT = 2500.0`  ← Smith 우측 쏠림 방지용
- `_S22_ORIGIN_PULL_WEIGHT = 1750.0`
- S21/S12 passband·stopband 가중치는 기존 값 유지

---

# 부록 B : 자주 묻는 질문

**Q1. 스미스차트가 우측에 몰려 보여요.**
A. `[ALL pts]` 모드에서는 통과대역(BW≈14%)보다 저지대역 점이 훨씬 많고, 저지대역은 |Γ|≈1 (open-like) 이 정상입니다. **Plot FL/FH 를 통과대역으로 좁히면** 원점 근처로 모이는 것이 보입니다.

**Q2. Return Loss를 −20dB로 했는데 통과대역 S11이 −15dB 정도예요.**
A. 25×25 픽셀의 자유도 한계 + S21과의 trade-off 때문입니다. 알고리즘을 **GA→DBS** 또는 **GA→GD** 하이브리드로 바꾸고 Pop Size를 늘리면 개선됩니다.

**Q3. P1 Z0를 100+0j로 바꾸고 Apply Renorm을 누르면 무엇이 바뀌나요?**
A. (1) 스미스차트 각 점의 |Γ|, 위상이 새 Z₀ 기준으로 재계산됩니다. (2) S-파라미터 dB 값도 갱신됩니다. (3) 다음에 Start Inverse Design을 누르면 score 함수의 원점-당김 항이 100Ω을 새 원점으로 잡고 최적화합니다.

**Q4. ALL 모드에서 슬라이더가 안 보여요.**
A. ALL 모드에서는 슬라이더가 의미 없으므로 자동 숨김됩니다. **All 토글을 OFF**하면 단일 주파수 모드로 전환되며 슬라이더가 다시 활성화됩니다.

**Q5. 좌측 워터마크/wCoRaL 로고가 갑자기 코너로 이동했어요.**
A. 데이터가 들어오면 (탭 2/3에서 s2p 로드 또는 탭 1에서 역설계 완료) 자동으로 중앙 큰 로고 → 우하단 작은 로고로 전환되도록 설계되어 있습니다.

---

# 부록 C : 키 위젯 ↔ 코드 위치 매핑

| 위젯 | 정의 라인 (`inverse_gui_fixed.py`) |
|---|---|
| Forward Model 콤보 | L167 |
| Pop/Chunk Size | L183-184 |
| Algorithm 콤보 | L189 |
| Result 그룹 (스크롤) | L259-314 |
| F_L / F_H / IL / RL / SL | L345-404 |
| Port Impedance | L411-441 |
| Plot X-Axis (좌측) | L445-461 |
| Log / Convergence 분할 | L512-543 |
| Start / Stop / Progress | L555-572 |
| Target Line 토글 | L598-605 |
| S-Parameter Response | L614-647 |
| Show 알고리즘 버튼 | L626-643 |
| Export 버튼 | L655-657 |
| Generated Layout | L661-668 |
| Smith Chart | L670-792 |
| Smith S11/S22/All 토글 | L722-746 |
| Smith 슬라이더 | L749-761 |

---

*Generated for the wCoRaL pixelated RF filter inverse-design GUI project.*
