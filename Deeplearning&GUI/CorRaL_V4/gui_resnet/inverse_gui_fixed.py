#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inverse_gui.py: GUI layer split from inverse_add_m (1).py
"""
import os  # Windows OpenMP 중복 로딩 회피
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # GUI 실행 시 libiomp 중복 초기화 오류 회피
from datetime import datetime
try:
    from .inverse_core import *
    from .inverse_core import _FALLBACK_FREQS
except ImportError:
    from inverse_core import *
    from inverse_core import _FALLBACK_FREQS

# 변경: import * 가 underscore prefix 이름을 자동 제외하므로 _db_to_complex_np 를 명시적으로 import.
#        Renorm_ops 는 inverse_core 가 sys.path 에 등록한 _base_gui 폴더에서 찾아짐.
try:
    from Renorm_ops import _db_to_complex_np  # 변경: viz / smith chart 그릴 때 dB → complex |Γ| 변환
except ImportError:
    try:
        from _base_gui.Renorm_ops import _db_to_complex_np  # 변경: 패키지 경로 fallback
    except ImportError:
        def _db_to_complex_np(mag_db):  # 변경: 최후 fallback - 위상 0 으로 가정
            import numpy as _np
            mag = _np.power(10.0, _np.asarray(mag_db, dtype=_np.float32) / 20.0)
            return mag.astype(_np.complex64)

# 변경: 왼쪽 탭 재구성 — Result 스크롤, Convergence+Log 분할
from PyQt5.QtWidgets import QScrollArea  # 변경: Model 탭 내부 Result 스크롤 영역용
# 변경: 알고리즘 드롭다운 팝업 행 높이를 Windows 스타일에서도 확실히 키우기 위한 Delegate
from PyQt5.QtWidgets import QStyledItemDelegate as _AlgoItemDelegate
from PyQt5.QtCore import QSize as _AlgoQSize


class _AlgoBigRowDelegate(_AlgoItemDelegate):
    """알고리즘 드롭다운 각 항목의 행 높이/내부 폰트를 강제해주는 delegate."""
    def __init__(self, parent=None, row_height=32, font_point=10):
        super().__init__(parent)
        self._row_h = int(row_height)
        self._font_pt = int(font_point)

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        s.setHeight(self._row_h)
        return s

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        f = option.font
        f.setPointSize(self._font_pt)
        option.font = f

try:
    from .Sparam_compare_gui import SParamCompareGUI
except ImportError:
    try:
        from Sparam_compare_gui import SParamCompareGUI
    except ImportError:
        SParamCompareGUI = None

class InverseDesignGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixelated RF Filter Wizard")
        self.setMinimumSize(1600, 1000)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_family   = 'densenet'
        self.forward_models = []
        self.config         = None
        self.frequencies    = None
        self.worker         = None
        self.all_results    = {}
        self.pending_algos  = []
        self.targets_norm   = None
        self._pb_mask       = None
        # ?? ?명꽣?숉떚釉??ㅻ??ㅼ감???곹깭 ??????????????????????????????????????
        self._smith_Z         = None
        self._smith_freqs     = None
        self._smith_pb_mask   = None
        self._smith_best_algo = None
        self._smith_has_smith = False
        self._smith_color     = '#3498db'
        self._smith_phase     = None
        self._smith_mag       = None
        self._smith_s11_db    = None
        self._smith_s21_db    = None
        self._smith_s22_db    = None
        self._smith_gamma_s11 = None
        self._smith_gamma_s22 = None
        self._smith_Z_s11     = None
        self._smith_Z_s22     = None
        self._smith_show_s11  = True
        self._smith_show_s22  = True
        self._show_vswr_circle = True   # 변경: RL 기준 VSWR 원 표시 토글
        self._show_sp_bg      = True    # 변경: S-Param 그래프 배경색 표시 토글
        self._ax_smith        = None
        self._sp_gridspec     = None
        self._smith_all_mode  = False
        self._split_preset    = "lpf"  # 변경: GUI에서는 LPF/BPF 프리셋만 노출하고 내부 분리선 설정은 자동화
        self._extra_smith_overlays = {}  # 변경: 탭 2/3 스미스차트 워터마크 오버레이를 별도로 관리
        # ?? ?섎졃 洹몃옒???곹깭 ????????????????????????????????????????????????
        self._conv_data       = {}
        self._conv_ax         = None
        self._base_algo_options = [
            "GD Only", "GA Only", "BPSO Only", "BO Only", "DE Only",
            "GA->GD Only", "BPSO->GD Only",
            "GA->DBS Only", "BPSO->DBS Only",
            "All (9 Algorithms)"]  # 변경: 긴 이름 대신 짧게 — 팝업 가로폭 축소
        self.init_ui()
        self.load_models()

    # ??????????????????????????????????????????????????????????
    @staticmethod
    def _plbl(text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size:10px;")
        return lbl

    def _set_split_preset(self, preset):
        self._split_preset = preset  # 변경: 현재 선택된 필터 프리셋을 저장
        is_bpf = (preset == "bpf")
        self.lpf_btn.setChecked(not is_bpf)  # 변경: LPF/BPF 버튼을 상호 배타적으로 동기화
        self.bpf_btn.setChecked(is_bpf)  # 변경: LPF/BPF 버튼을 상호 배타적으로 동기화
        self.split_line_enabled.setChecked(is_bpf)  # 변경: 코어에는 기존 분리선 플래그를 그대로 전달
        if is_bpf:
            self.split_line_mode.setCurrentText("Vertical")  # 변경: BPF 기본 프리셋은 중앙 세로 분리선 사용
            self.split_line_index.setText(str(GRID_SIZE // 2))  # 변경: 기본 표시값은 중앙으로 두되 실제 실행 시 랜덤 index를 다시 뽑음
            self.split_line_width.setText("1")  # 변경: BPF 기본 프리셋은 폭 1픽셀 사용

    # ??????????????????????????????????????????????????????????
    def init_ui(self):
        # ?? [?곷떒 ??쑝濡?援ъ꽦] ???????????????????????????????????????????
        self.main_tabs = QTabWidget()
        self.setCentralWidget(self.main_tabs)

        # 기존 Inverse Design GUI
        inverse_tab = QWidget()
        ml = QHBoxLayout(inverse_tab)
        self.main_tabs.addTab(inverse_tab, "Inverse Design")

        # Impedance Renormalization 탭
        try:
            import sys, os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from impedance_test_gui import ImpedanceTestGUI
            self.impedance_test_tab = ImpedanceTestGUI()
            self.main_tabs.addTab(self.impedance_test_tab, "Impedance Renormalization")
        except Exception as e:
            err_lbl = QLabel(f"Failed to load ImpedanceTestGUI:\n{e}")
            self.main_tabs.addTab(err_lbl, "Impedance Test (Error)")

        if SParamCompareGUI is not None:
            try:
                self.sparam_compare_tab = SParamCompareGUI(self)
                self.main_tabs.addTab(self.sparam_compare_tab, "S-Param Compare")
            except Exception as e:
                err_lbl = QLabel(f"Failed to load SParamCompareGUI:\n{e}")
                self.main_tabs.addTab(err_lbl, "S-Param Compare (Error)")
        else:
            err_lbl = QLabel("Failed to import SParamCompareGUI.")
            self.main_tabs.addTab(err_lbl, "S-Param Compare (Error)")

        # 변경: 왼쪽 패널을 5개 탭으로 재구성 (Model / Target / Result / Convergence / Log)
        left = QWidget(); ll = QVBoxLayout(left); left.setMaximumWidth(350)
        ll.setContentsMargins(4, 4, 4, 4); ll.setSpacing(4)

        self.left_tabs = QTabWidget()  # 변경: 왼쪽 패널 최상단 탭 위젯

        # =========================================================
        # Tab 1: Model  (Model + Optimization + Algorithm)
        # =========================================================
        tab_model = QWidget(); tab_model_lay = QVBoxLayout(tab_model)
        tab_model_lay.setContentsMargins(6, 6, 6, 6); tab_model_lay.setSpacing(6)

        mg = QGroupBox("Model"); mgl = QVBoxLayout()
        mgl.addWidget(QLabel("Forward Model"))
        sleek_combo_style = (
            "QComboBox {"
            "  font-size: 11px;"
            "  font-weight: 600;"
            "  border: 1px solid #ced4da;"
            "  border-radius: 4px;"
            "  padding: 4px 10px;"
            "  background-color: white;"
            "}"
            "QComboBox:hover {"
            "  border: 1px solid #adb5bd;"
            "}"
            "QComboBox::drop-down {"
            "  subcontrol-origin: padding;"
            "  subcontrol-position: top right;"
            "  width: 20px;"
            "  border-left: none;"
            "}"
            "QComboBox::down-arrow {"
            "  image: none;"
            "  border-left: 4px solid transparent;"
            "  border-right: 4px solid transparent;"
            "  border-top: 5px solid #666;"
            "  margin-top: 2px;"
            "}"
            "QComboBox QAbstractItemView {"
            "  outline: 0;"
            "  border: 1px solid #ced4da;"
            "  background-color: white;"
            "  selection-background-color: #0078D7;"
            "  selection-color: white;"
            "}"
            "QComboBox QListView {"
            "  border: none;"
            "  padding: 0px;"
            "  margin: 0px;"
            "}"
        )
        self.model_family_combo = QComboBox()
        self.model_family_combo.setMinimumHeight(32)
        self.model_family_combo.setStyleSheet(sleek_combo_style)
        self.model_family_combo.addItem("DenseNet", "densenet")
        self.model_family_combo.addItem("ResNet", "resnet")
        self.model_family_combo.addItem("EfficientNet", "efficientnet")
        self.model_family_combo.currentIndexChanged.connect(self._on_model_family_changed)
        
        # [FIX] 모델 선택창도 동일하게 Delegate 적용하여 아이템 가독성/가시성 확보
        _model_delegate = _AlgoBigRowDelegate(self.model_family_combo, row_height=30, font_point=10)
        self.model_family_combo.setItemDelegate(_model_delegate)
        self._model_delegate = _model_delegate
        
        mgl.addWidget(self.model_family_combo)
        self.model_status = QLabel("모델 로딩 중...")
        self.model_status.setWordWrap(True)
        self.model_status.setStyleSheet("font-size: 10px;")
        mgl.addWidget(self.model_status)
        self.reload_btn = QPushButton("📁 모델 폴더 직접 선택")
        self.reload_btn.clicked.connect(self._manual_load)
        mgl.addWidget(self.reload_btn)
        mg.setLayout(mgl)
        tab_model_lay.addWidget(mg)

        og = QGroupBox("Optimization"); ol = QGridLayout()
        ol.addWidget(QLabel("Pop Size"),   0, 0); self.pop_input   = QLineEdit("4096"); ol.addWidget(self.pop_input,   0, 1)
        ol.addWidget(QLabel("Chunk Size"), 1, 0); self.chunk_input = QLineEdit("512");  ol.addWidget(self.chunk_input, 1, 1)
        og.setLayout(ol)
        tab_model_lay.addWidget(og)

        ag = QGroupBox("Algorithm"); al = QVBoxLayout()
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(self._base_algo_options)
        # 닫혀 있을 때 (헤더) 높이/패딩
        self.algo_combo.setMinimumHeight(34)
        self.algo_combo.setStyleSheet(sleek_combo_style)
        self.algo_combo.setMaxVisibleItems(12)

        # 변경: 드롭다운 팝업의 각 항목 행 높이/폰트를 Delegate로 강제
        #   스타일시트의 QComboBox::item { min-height } 은 Windows 스타일에서 무시되는 경우가 있어 사용 X
        _algo_delegate = _AlgoBigRowDelegate(self.algo_combo, row_height=36, font_point=11)
        self.algo_combo.setItemDelegate(_algo_delegate)
        self._algo_delegate = _algo_delegate  # 레퍼런스 유지

        try:
            _view = self.algo_combo.view()
            _pf = QFont(_view.font())
            _pf.setPointSize(10)
            _view.setFont(_pf)
            _view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            _view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            _view.setTextElideMode(Qt.ElideNone)
        except Exception:
            pass

        al.addWidget(self.algo_combo); ag.setLayout(al)
        tab_model_lay.addWidget(ag)

        # 변경: 팝업 가로폭을 가장 긴 항목 기준으로 타이트하게 맞춤 (오른쪽 빈 공간/좌우 길이 축소)
        def _fit_algo_popup_width():
            try:
                _v = self.algo_combo.view()
                fm = _v.fontMetrics()
                longest = 0
                for i in range(self.algo_combo.count()):
                    w = fm.horizontalAdvance(self.algo_combo.itemText(i))
                    if w > longest:
                        longest = w
                # 컨텐츠 + 좌우 패딩(약 24) + 테두리 여유
                target_w = longest + 28
                # 콤보박스 자체 폭 이상만 보장 (너무 좁지 않게)
                min_w = max(target_w, self.algo_combo.width())
                _v.setMinimumWidth(min_w)
                _v.setMaximumWidth(min_w + 20)
            except Exception:
                pass
        _orig_show_popup = self.algo_combo.showPopup
        def _patched_show_popup():
            _fit_algo_popup_width()
            _orig_show_popup()
        self.algo_combo.showPopup = _patched_show_popup

        # 변경: Model 탭의 빈 하단 공간에 Result 섹션 — 스크롤 영역으로 내려볼 수 있게 구성
        # 변경: 글자 크기를 가독성 있는 수준(12~13px)으로 복구, 행 높이도 넉넉히
        result_group = QGroupBox("Result")
        result_outer_lay = QVBoxLayout(result_group)
        result_outer_lay.setContentsMargins(4, 6, 4, 4)
        result_outer_lay.setSpacing(4)

        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setFrameShape(QScrollArea.NoFrame)
        result_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        result_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        pf_grp = QWidget()
        pf_lay = QGridLayout(pf_grp)
        # 변경: 오른쪽 셀이 스크롤바 폭에 잘리지 않도록 우측 여백/간격 축소
        pf_lay.setContentsMargins(8, 8, 4, 8); pf_lay.setHorizontalSpacing(6); pf_lay.setVerticalSpacing(8)
        # 변경: 값 박스가 두 배 가량 커지도록 컬럼 비율 재조정 (3:2:2 → 2:3:3)
        pf_lay.setColumnStretch(0, 2)
        pf_lay.setColumnStretch(1, 3)
        pf_lay.setColumnStretch(2, 3)
        for col, h in enumerate(["Algorithm", "S21 dB", "S11 dB"]):
            h_lbl = QLabel(h)
            h_lbl.setAlignment(Qt.AlignCenter)
            # 변경: 값 박스가 커진 만큼 헤더 폰트도 키움 (12 → 14px)
            h_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #333; padding: 4px 0;")
            pf_lay.addWidget(h_lbl, 0, col)

        self._result_rows = {}
        # 변경: 각 알고리즘 행을 QWidget 컨테이너로 감싸 visibility 토글을 한 번에 제어
        self._result_row_widgets = {}
        for idx, name in enumerate(
                ['GD', 'GA', 'BPSO', 'BO', 'DE', 'GA_GD', 'BPSO_GD', 'GA_DBS', 'BPSO_DBS'],
                start=1):
            n_lbl = QLabel(name)
            n_lbl.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px 2px;")
            n_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            s21_v = QLabel("-"); s11_v = QLabel("-")
            for l in [s21_v, s11_v]:
                l.setAlignment(Qt.AlignCenter)
                # 변경: 값 박스 높이 두 배(26 → 50px), 폭/폰트도 함께 확대
                l.setMinimumHeight(50)
                # 변경: min-width 110→90으로 살짝 축소(스크롤바 폭 잘림 방지), 패딩도 좌우 축소
                l.setStyleSheet(
                    "font-size: 18px; font-weight: 600; background: #f8f9fa; "
                    "border-radius: 4px; min-width: 90px; padding: 6px 2px;")
            pf_lay.addWidget(n_lbl, idx, 0)
            pf_lay.addWidget(s21_v, idx, 1)
            pf_lay.addWidget(s11_v, idx, 2)
            self._result_rows[name] = (s21_v, s11_v)
            self._result_row_widgets[name] = (n_lbl, s21_v, s11_v)

        self._pf_detail_lbl = QLabel("Best:")
        self._pf_detail_lbl.setStyleSheet("font-size: 11px; color: #6c757d; padding-top: 4px;")
        pf_lay.addWidget(self._pf_detail_lbl, 10, 0, 1, 3)
        pf_lay.setRowStretch(11, 1)

        result_scroll.setWidget(pf_grp)
        result_outer_lay.addWidget(result_scroll)
        # 변경: 탭 2/3과 같은 스크롤 느낌이 나도록 최소 높이 설정
        result_group.setMinimumHeight(200)
        tab_model_lay.addWidget(result_group, stretch=1)

        # 변경: 알고리즘 콤보 선택이 바뀌면 Result 행 표시 필터링
        self.algo_combo.currentIndexChanged.connect(self._filter_result_rows_by_algo)
        # 초기 표시 필터 적용
        self._filter_result_rows_by_algo()

        self.left_tabs.addTab(tab_model, "Model")

        # =========================================================
        # Tab 2: Target  (Parameter + Port Impedance + Plot X-Axis)
        # =========================================================
        tab_target = QWidget(); tab_target_lay = QVBoxLayout(tab_target)
        tab_target_lay.setContentsMargins(6, 6, 6, 6); tab_target_lay.setSpacing(6)

        pg = QGroupBox("Parameter"); pl = QGridLayout()
        pl.setVerticalSpacing(5); pl.setHorizontalSpacing(4)

        ro_style = "background:#f0f0f0; color:#555; font-size:10px;"

        def _rlbl(html):
            """아래첨자 라벨 생성."""
            lbl = QLabel(html)
            lbl.setTextFormat(Qt.RichText)
            lbl.setStyleSheet("font-size:10px;")
            return lbl

        pl.addWidget(_rlbl("F<sub>L</sub> (GHz)"),  0, 0)
        self.f_low = QLineEdit("6.0"); pl.addWidget(self.f_low, 0, 1)
        pl.addWidget(_rlbl("F<sub>c</sub> (GHz)"),  0, 2)
        self.fc_input = QLineEdit("7.00")
        self.fc_input.setReadOnly(True); self.fc_input.setStyleSheet(ro_style)
        pl.addWidget(self.fc_input, 0, 3)

        pl.addWidget(_rlbl("F<sub>H</sub> (GHz)"), 1, 0)
        self.f_high = QLineEdit("8.0"); pl.addWidget(self.f_high, 1, 1)
        pl.addWidget(_rlbl("BW (%)"),    1, 2)
        self.bw_input = QLineEdit("28.57")
        self.bw_input.setReadOnly(True); self.bw_input.setStyleSheet(ro_style)
        pl.addWidget(self.bw_input, 1, 3)

        # 변경: Pass Band 입력을 GHz 오프셋으로 변경 (S21/S12 타겟선에만 적용)
        # L-PB(GHz) = F_L - offset, R-PB(GHz) = F_H + offset
        pl.addWidget(_rlbl("L-PB off (GHz)"), 2, 0)
        self.l_pb_offset_input = QLineEdit("2.0")
        self.l_pb_offset_input.setToolTip(
            "왼쪽 Pass Band 오프셋 (GHz)\n"
            "L-PB (GHz) = F_L − offset\n"
            "예: F_L=6GHz, 2 → L-PB=4GHz\n"
            "※ S21/S12 타겟선 배경에만 적용됩니다."
        )
        pl.addWidget(self.l_pb_offset_input, 2, 1)
        pl.addWidget(_rlbl("L-PB (GHz)"), 2, 2)
        self.l_pb_freq = QLineEdit("4")
        self.l_pb_freq.setReadOnly(True); self.l_pb_freq.setStyleSheet(ro_style)
        pl.addWidget(self.l_pb_freq, 2, 3)

        pl.addWidget(_rlbl("R-PB off (GHz)"), 3, 0)
        self.r_pb_offset_input = QLineEdit("2.0")
        self.r_pb_offset_input.setToolTip(
            "오른쪽 Pass Band 오프셋 (GHz)\n"
            "R-PB (GHz) = F_H + offset\n"
            "예: F_H=8GHz, 2 → R-PB=10GHz\n"
            "※ S21/S12 타겟선 배경에만 적용됩니다."
        )
        pl.addWidget(self.r_pb_offset_input, 3, 1)
        pl.addWidget(_rlbl("R-PB (GHz)"), 3, 2)
        self.r_pb_freq = QLineEdit("10")
        self.r_pb_freq.setReadOnly(True); self.r_pb_freq.setStyleSheet(ro_style)
        pl.addWidget(self.r_pb_freq, 3, 3)

        pl.addWidget(QLabel("Insertion Loss (dB)"), 4, 0, 1, 2)
        self.il_input = QLineEdit("-1.0"); pl.addWidget(self.il_input, 4, 2, 1, 2)
        pl.addWidget(QLabel("Return Loss (dB)"),    5, 0, 1, 2)
        self.rl_input = QLineEdit("-10.0"); pl.addWidget(self.rl_input, 5, 2, 1, 2)
        # 변경: Stopband Loss 추가 - S21/S12 저지대역 목표 레벨.
        # 실제 stopband 타겟 dB = IL - 3 + SL (3dB는 전이 마진 고정값)
        # 예: IL=-1, SL=-11 → stopband target = -1 - 3 + (-11) = -15 dB
        pl.addWidget(QLabel("Stopband Loss (dB)"),  6, 0, 1, 2)
        self.sl_input = QLineEdit("-11.0")
        self.sl_input.setToolTip(
            "저지대역(Stopband) 감쇠 목표 (dB, 음수)\n"
            "실제 stopband 타겟 = Insertion Loss - 3 dB + Stopband Loss\n"
            "예) IL=-1 dB, SL=-6 dB → Stopband 타겟 = -10 dB\n"
            "※ 3 dB는 전이 구간 고정 마진"
        )
        pl.addWidget(self.sl_input, 6, 2, 1, 2)

        pg.setLayout(pl)
        tab_target_lay.addWidget(pg)

        # Port Impedance
        from PyQt5.QtWidgets import QFormLayout as _QFL
        z0_group = QGroupBox("Port Impedance (재정규화)")
        z0_layout = _QFL()
        self.p1_z0_edit = QLineEdit("50+0j")
        self.p1_z0_edit.setToolTip(
            "Port 1 목표 임피던스 (Ω)\n"
            "예: 50+0j (기본), 100+0j, 75-5j\n"
            "50Ω 기준 예측 결과를 이 임피던스 기준으로 다시 변환합니다."
        )
        self.p2_z0_edit = QLineEdit("50+0j")
        self.p2_z0_edit.setToolTip(
            "Port 2 목표 임피던스 (Ω)\n"
            "예: 50+0j (기본), 100+0j, 75-5j"
        )
        z0_layout.addRow("P1 Z0 (Ω)", self.p1_z0_edit)
        z0_layout.addRow("P2 Z0 (Ω)", self.p2_z0_edit)
        z0_note = QLabel(
            "50Ω 기준 모델을 사용합니다.\n"
            "임피던스를 조정하면 스미스차트와\n"
            "S-파라미터가 재정규화된 기준으로 갱신됩니다."
        )
        z0_note.setWordWrap(True)
        z0_note.setStyleSheet("color: #888; font-size: 10px;")
        z0_layout.addRow(z0_note)
        self.z0_apply_btn = QPushButton("▶ Apply Renorm")
        self.z0_apply_btn.setToolTip(
            "포트 1/2의 Z0 값을 바꾼 뒤 이 버튼을 누르면\n"
            "스미스차트와 S-파라미터 그래프가 재정규화된 결과로 갱신됩니다."
        )
        self.z0_apply_btn.clicked.connect(self._apply_renorm)
        z0_layout.addRow(self.z0_apply_btn)
        z0_group.setLayout(z0_layout)
        tab_target_lay.addWidget(z0_group)

        # 변경: 상단 중앙의 Plot X-Axis 컨트롤을 Target 탭 안으로 이동
        plot_xaxis_group = QGroupBox("Plot X-Axis")
        plot_xaxis_lay = QGridLayout()
        plot_xaxis_lay.setVerticalSpacing(5); plot_xaxis_lay.setHorizontalSpacing(4)
        plot_xaxis_lay.addWidget(QLabel("Plot FL (GHz):"), 0, 0)
        self.plot_fl_input = QLineEdit("3.0")
        plot_xaxis_lay.addWidget(self.plot_fl_input, 0, 1)
        plot_xaxis_lay.addWidget(QLabel("Plot FH (GHz):"), 1, 0)
        self.plot_fh_input = QLineEdit("11.0")
        plot_xaxis_lay.addWidget(self.plot_fh_input, 1, 1)
        plot_xaxis_lay.addWidget(QLabel("X-Axis Ticks:"), 2, 0)
        self.plot_pts_input = QLineEdit("11")
        plot_xaxis_lay.addWidget(self.plot_pts_input, 2, 1)
        self.plot_update_btn = QPushButton("Apply Plot X-Axis")
        self.plot_update_btn.clicked.connect(self._viz_compare)
        plot_xaxis_lay.addWidget(self.plot_update_btn, 3, 0, 1, 2)
        plot_xaxis_group.setLayout(plot_xaxis_lay)
        tab_target_lay.addWidget(plot_xaxis_group)

        # 변경: S11/S22 와 S21/S12 Y축 박스를 좌우로 나란히 배치 (가로 절약)
        sp_y_pair_widget = QWidget()
        sp_y_pair_lay = QHBoxLayout(sp_y_pair_widget)
        sp_y_pair_lay.setContentsMargins(0, 0, 0, 0); sp_y_pair_lay.setSpacing(4)

        # S11/S22 Y축
        sp_refl_group = QGroupBox("Y-Axis (S11/S22)")
        sp_refl_lay = QGridLayout()
        sp_refl_lay.setContentsMargins(6, 6, 6, 6)
        sp_refl_lay.setVerticalSpacing(4); sp_refl_lay.setHorizontalSpacing(2)
        sp_refl_lay.setColumnStretch(0, 0); sp_refl_lay.setColumnStretch(1, 1)
        sp_refl_lay.addWidget(QLabel("min:"), 0, 0)
        self.plot_y_lo_refl = QLineEdit("-30"); self.plot_y_lo_refl.setMinimumWidth(40)
        sp_refl_lay.addWidget(self.plot_y_lo_refl, 0, 1)
        sp_refl_lay.addWidget(QLabel("max:"), 1, 0)
        self.plot_y_hi_refl = QLineEdit("0.5"); self.plot_y_hi_refl.setMinimumWidth(40)
        sp_refl_lay.addWidget(self.plot_y_hi_refl, 1, 1)
        sp_refl_lay.addWidget(QLabel("Ticks:"), 2, 0)
        self.plot_y_ticks_refl = QLineEdit("7"); self.plot_y_ticks_refl.setMinimumWidth(40)
        sp_refl_lay.addWidget(self.plot_y_ticks_refl, 2, 1)
        self.plot_refl_apply_btn = QPushButton("Apply")
        self.plot_refl_apply_btn.clicked.connect(self._viz_compare)
        sp_refl_lay.addWidget(self.plot_refl_apply_btn, 3, 0, 1, 2)
        sp_refl_group.setLayout(sp_refl_lay)
        sp_y_pair_lay.addWidget(sp_refl_group)

        # S21/S12 Y축
        sp_trans_group = QGroupBox("Y-Axis (S21/S12)")
        sp_trans_lay = QGridLayout()
        sp_trans_lay.setContentsMargins(6, 6, 6, 6)
        sp_trans_lay.setVerticalSpacing(4); sp_trans_lay.setHorizontalSpacing(2)
        sp_trans_lay.setColumnStretch(0, 0); sp_trans_lay.setColumnStretch(1, 1)
        sp_trans_lay.addWidget(QLabel("min:"), 0, 0)
        self.plot_y_lo_trans = QLineEdit("-14"); self.plot_y_lo_trans.setMinimumWidth(40)
        sp_trans_lay.addWidget(self.plot_y_lo_trans, 0, 1)
        sp_trans_lay.addWidget(QLabel("max:"), 1, 0)
        self.plot_y_hi_trans = QLineEdit("1"); self.plot_y_hi_trans.setMinimumWidth(40)
        sp_trans_lay.addWidget(self.plot_y_hi_trans, 1, 1)
        sp_trans_lay.addWidget(QLabel("Ticks:"), 2, 0)
        self.plot_y_ticks_trans = QLineEdit("8"); self.plot_y_ticks_trans.setMinimumWidth(40)
        sp_trans_lay.addWidget(self.plot_y_ticks_trans, 2, 1)
        self.plot_trans_apply_btn = QPushButton("Apply")
        self.plot_trans_apply_btn.clicked.connect(self._viz_compare)
        sp_trans_lay.addWidget(self.plot_trans_apply_btn, 3, 0, 1, 2)
        sp_trans_group.setLayout(sp_trans_lay)
        sp_y_pair_lay.addWidget(sp_trans_group)

        tab_target_lay.addWidget(sp_y_pair_widget)

        self._dist_warn_lbl = QLabel("")
        self._dist_warn_lbl.setWordWrap(True)
        self._dist_warn_lbl.setStyleSheet("""
            QLabel { background: #fff3cd; color: #856404; border: 1px solid #ffc107;
                     border-radius: 4px; padding: 5px 7px; font-size: 10px; }
        """)
        self._dist_warn_lbl.setVisible(False)
        tab_target_lay.addWidget(self._dist_warn_lbl)

        tab_target_lay.addStretch()
        self.left_tabs.addTab(tab_target, "Target")

        # Fc/BW + Pass Band auto-update connections
        self.f_low.textChanged.connect(self._update_fc_bw)
        self.f_high.textChanged.connect(self._update_fc_bw)
        self._update_fc_bw()

        # 변경: Pass Band 자동 업데이트 연결
        self.f_low.textChanged.connect(self._update_pass_band)
        self.f_high.textChanged.connect(self._update_pass_band)
        self.l_pb_offset_input.textChanged.connect(self._update_pass_band)
        self.r_pb_offset_input.textChanged.connect(self._update_pass_band)
        self._update_pass_band()

        for w in [self.f_low, self.f_high, self.il_input, self.rl_input, self.sl_input]:
            w.textChanged.connect(self._check_spec_feasibility)

        # Hidden split line widgets (필요 위젯은 유지하되 UI에는 노출하지 않음)
        self.split_line_enabled = QCheckBox("Use Split Line")
        self.split_line_enabled.setToolTip("분리선을 강제로 삽입해 포트 간 DC 연결을 끊습니다.")
        self.split_line_mode = QComboBox()
        self.split_line_mode.addItems(["Vertical", "Horizontal", "Cross"])
        self.split_line_mode.setToolTip("Vertical=세로 절개, Horizontal=가로 절개, Cross=십자 절개")
        self.split_line_index = QLineEdit(str(GRID_SIZE // 2))
        self.split_line_index.setToolTip("0~24 범위의 중심 index")
        self.split_line_width = QLineEdit("1")
        self.split_line_width.setToolTip("분리선 폭(픽셀 수)")
        self.split_line_enabled.hide()
        self.split_line_mode.hide()
        self.split_line_index.hide()
        self.split_line_width.hide()

        # =========================================================
        # 변경: 기존 Tab 3 Result는 Model 탭 내부로 이동 — 여기서는 생성하지 않음
        # =========================================================

        # =========================================================
        # 변경: Tab 3 Log + Convergence — Log(상단, 스크롤) / Convergence 그래프(하단) 통합
        # =========================================================
        log_conv_grp = QWidget()
        log_conv_outer_lay = QVBoxLayout(log_conv_grp)
        log_conv_outer_lay.setContentsMargins(2, 2, 2, 2); log_conv_outer_lay.setSpacing(2)
        log_conv_split = QSplitter(Qt.Vertical)

        # 상단: Log
        log_panel = QWidget(); log_panel_lay = QVBoxLayout(log_panel)
        log_panel_lay.setContentsMargins(2, 2, 2, 2); log_panel_lay.setSpacing(2)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        self.log_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        log_panel_lay.addWidget(self.log_text)
        log_conv_split.addWidget(log_panel)

        # 하단: Convergence 그래프
        conv_panel = QWidget(); conv_panel_lay = QVBoxLayout(conv_panel)
        conv_panel_lay.setContentsMargins(2, 2, 2, 2); conv_panel_lay.setSpacing(2)
        self._conv_fig    = Figure(figsize=(3, 1.6))
        self._conv_canvas = FigureCanvas(self._conv_fig)
        self._conv_canvas.setMinimumHeight(110)
        self._conv_ax = self._conv_fig.add_subplot(111)
        self._conv_ax.set_xlabel('Step', fontsize=7)
        self._conv_ax.set_ylabel('Error', fontsize=7)
        self._conv_ax.tick_params(labelsize=6)
        self._conv_ax.grid(True, alpha=0.3)
        self._conv_ax.set_title('Waiting...', fontsize=7)
        self._conv_fig.tight_layout(pad=0.5)
        conv_panel_lay.addWidget(self._conv_canvas)
        log_conv_split.addWidget(conv_panel)

        log_conv_split.setStretchFactor(0, 3)
        log_conv_split.setStretchFactor(1, 2)
        log_conv_outer_lay.addWidget(log_conv_split)
        self.left_tabs.addTab(log_conv_grp, "Log / Convergence")

        # 변경: 기존 코드와의 호환성을 위해 bottom_tabs alias 유지
        self.bottom_tabs = self.left_tabs

        ll.addWidget(self.left_tabs, stretch=1)

        # =========================================================
        # 변경: 하단 Start/Stop/Progress 컨트롤 (왼쪽 패널 최하단)
        # 변경: 버튼/프로그레스바 크기를 키워 가시성 향상
        # =========================================================
        self.run_btn = QPushButton("▶ Start Inverse Design")
        self.run_btn.setMinimumHeight(70)  # 변경: 50 → 70
        self.run_btn.setFont(QFont("Arial", 14, QFont.Bold))  # 변경: 12 → 14
        self.run_btn.clicked.connect(self.run_inverse)
        ll.addWidget(self.run_btn)

        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setMinimumHeight(50)  # 변경: 기본 높이 → 50
        self.stop_btn.setFont(QFont("Arial", 12, QFont.Bold))  # 변경: Stop 버튼도 굵게 강조
        self.stop_btn.clicked.connect(self.stop_inverse)
        self.stop_btn.setEnabled(False)
        ll.addWidget(self.stop_btn)

        self.progress = QProgressBar()
        self.progress.setMinimumHeight(28)  # 변경: 진행률 바 두께 증가
        self.progress.setFont(QFont("Arial", 10, QFont.Bold))  # 변경: % 텍스트 가독성 향상
        self.progress.setTextVisible(True)
        ll.addWidget(self.progress)

        ml.addWidget(left)

        # ?? ?ㅻⅨ履??곸뿭 ????????????????????????????????????????????????????
        right = QWidget(); right_main_lay = QVBoxLayout(right)
        right_main_lay.setContentsMargins(0, 0, 0, 0); right_main_lay.setSpacing(4)

        xaxis_group = QWidget()
        xaxis_lay = QHBoxLayout(xaxis_group)
        xaxis_lay.setContentsMargins(4, 2, 4, 2)
        # 변경: LPF/BPF 분리선 기능 제거 - 타겟선 On/Off 토글로 교체
        # 주석 처리된 이전 코드 (Band-pass 분리선 프리셋): 호환성을 위해 숨김 위젯으로 유지
        # --------------------------------------------------------
        # self.lpf_btn = QPushButton("Low-pass")  ...  (분리선 없음 프리셋)
        # self.bpf_btn = QPushButton("Band-pass") ...  (랜덤 분리선 프리셋)
        # --------------------------------------------------------
        self.lpf_btn = QPushButton("Low-pass")  # 변경: 내부 로직 유지용 숨김 위젯 (GUI 노출 X)
        self.lpf_btn.setCheckable(True); self.lpf_btn.setChecked(True); self.lpf_btn.hide()
        self.lpf_btn.clicked.connect(lambda: self._set_split_preset("lpf"))
        self.bpf_btn = QPushButton("Band-pass")  # 변경: 내부 로직 유지용 숨김 위젯 (GUI 노출 X)
        self.bpf_btn.setCheckable(True); self.bpf_btn.setChecked(False); self.bpf_btn.hide()
        self.bpf_btn.clicked.connect(lambda: self._set_split_preset("bpf"))
        self._set_split_preset("lpf")  # 항상 LPF(분리선 없음) 모드 고정

        # 변경: 타겟선 On/Off 토글 버튼 추가 - 빈 자리에 배치
        xaxis_lay.addWidget(QLabel("Target Line:"))
        self.show_target_btn = QPushButton("Show Target")
        self.show_target_btn.setCheckable(True)
        self.show_target_btn.setChecked(True)  # 기본값: 타겟선 표시
        self.show_target_btn.setFixedWidth(110)
        self.show_target_btn.setToolTip("S11/S22/S21/S12 비교 그래프의 점선 타겟을 보이거나 숨깁니다.")
        self.show_target_btn.clicked.connect(self._on_target_line_toggle)
        xaxis_lay.addWidget(self.show_target_btn)

        # 변경: S-Param 그래프 배경 색깔 On/Off 토글 버튼
        self.show_bg_btn = QPushButton("Show BG")
        self.show_bg_btn.setCheckable(True)
        self.show_bg_btn.setChecked(True)
        self.show_bg_btn.setFixedWidth(100)
        self.show_bg_btn.setToolTip("S-Parameter 그래프의 배경(연두/흰/핑크)을 켜거나 끕니다.")
        self.show_bg_btn.clicked.connect(self._on_sp_bg_toggle)
        xaxis_lay.addWidget(self.show_bg_btn)

        # 변경: 스미스차트 VSWR 원 On/Off 토글 버튼
        self.show_vswr_btn = QPushButton("Show VSWR")
        self.show_vswr_btn.setCheckable(True)
        self.show_vswr_btn.setChecked(True)
        self.show_vswr_btn.setFixedWidth(110)
        self.show_vswr_btn.setToolTip("스미스차트의 Return Loss 기준 VSWR 원(연두색)을 켜거나 끕니다.")
        self.show_vswr_btn.clicked.connect(self._on_vswr_toggle)
        xaxis_lay.addWidget(self.show_vswr_btn)

        xaxis_lay.addStretch()
        right_main_lay.addWidget(xaxis_group)

        rl_widget = QWidget()
        rl = QHBoxLayout(rl_widget)
        rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(8)
        right_main_lay.addWidget(rl_widget, stretch=1)

        spg = QGroupBox("S-Parameter Response")
        spg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scl = QVBoxLayout(); scl.setContentsMargins(4,4,4,4)
        self.sp_fig    = Figure()
        self.sp_canvas = FigureCanvas(self.sp_fig)
        self.sp_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scl.addWidget(self.sp_canvas, stretch=1)

        sp_toggle_widget = QWidget()
        sp_toggle = QHBoxLayout(sp_toggle_widget)
        sp_toggle.setContentsMargins(6, 2, 6, 2); sp_toggle.setSpacing(6)
        sp_toggle.addWidget(QLabel("Show:"))
        self._sp_algo_btns = {}
        for name, color in [('GD',       '#2ecc71'), ('GA',       '#3498db'),
                             ('BPSO',     '#e74c3c'), ('BO',       '#9b59b6'),
                             ('DE',       '#f39c12'), ('GA_GD',    '#1a5276'),
                             ('BPSO_GD',  '#922b21'), ('GA_DBS',   '#16a085'),
                             ('BPSO_DBS', '#d35400')]:
            btn = QPushButton(name)
            btn.setCheckable(True); btn.setChecked(True); btn.setFixedWidth(52)
            btn.setToolTip(f"Toggle {name} curve")
            btn.setStyleSheet(f"""
                QPushButton {{ border: 1px solid {color}; border-radius: 3px;
                    background: #ffffff; font-size: 10px; padding: 2px 6px; }}
                QPushButton:checked {{ background: {color}; color: #ffffff; }}
                QPushButton:hover {{ background: #f3f6f9; }}
            """)
            btn.toggled.connect(self._on_sp_algo_toggle)
            self._sp_algo_btns[name] = btn
            sp_toggle.addWidget(btn)
        sp_toggle.addStretch()
        scl.addWidget(sp_toggle_widget)
        spg.setLayout(scl)
        rl.addWidget(spg, stretch=7)

        right_col_widget = QWidget()
        right_col_lay = QVBoxLayout(right_col_widget)
        right_col_lay.setContentsMargins(0, 0, 0, 0); right_col_lay.setSpacing(4)
        right_top_bar = QHBoxLayout()
        right_top_bar.addStretch()
        
        self.export_btn = QPushButton("Export s2p / npz")
        self.export_btn.setToolTip("현재 최적 역설계 결과를 s2p와 npz로 저장")  # 변경: 깨진 툴팁 문자열을 정상 문구로 복원
        self.export_btn.clicked.connect(self._export_current_best_result)
        right_top_bar.addWidget(self.export_btn)
        right_col_lay.addLayout(right_top_bar)

        # 상단 - Layout 25x25
        lg2 = QGroupBox(f"Generated Layout ({GRID_SIZE}x{GRID_SIZE})")
        lcl = QVBoxLayout(); lcl.setContentsMargins(4,4,4,4)
        self.layout_fig    = Figure(figsize=(4, 2.0))
        self.layout_canvas = FigureCanvas(self.layout_fig)
        lcl.addWidget(self.layout_canvas)
        lg2.setLayout(lcl)
        right_col_lay.addWidget(lg2, stretch=2)

        # [?섎떒 - Smith Chart]
        smith_grp = QGroupBox("Smith Chart")
        smith_lay = QVBoxLayout(); smith_lay.setContentsMargins(4,4,4,4)
        self.smith_fig = Figure(constrained_layout=True)
        self.smith_canvas = FigureCanvas(self.smith_fig)
        self.smith_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        smith_lay.addWidget(self.smith_canvas, stretch=1)

        # Smith chart 위 배경 오버레이 이미지
        self._overlay_img_lbl  = None
        self._overlay_pix_center = None
        self._overlay_pix_corner = None
        self._overlay_at_corner = False   # ?꾩옱 ?꾩튂 ?곹깭 (False=以묒븰, True=援ъ꽍)
        try:
            import os as _os
            from PyQt5.QtGui import QPixmap as _QPixmap
            from PyQt5.QtWidgets import QGraphicsOpacityEffect as _QGOEff
            _picture_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "picture")  # 변경: 현재 프로젝트 내부 picture 폴더 경로 계산
            _img_center = r"E:\백업파일\학업자료\4학년\1학기\창의설계\HAN\coral_v1\code\dense_code\CorRaL_V2\gui_resnet\picture\KakaoTalk_20260325_165731666_01.png"
            _img_corner = r"E:\백업파일\학업자료\4학년\1학기\창의설계\HAN\coral_v1\code\dense_code\CorRaL_V2\gui_resnet\picture\KakaoTalk_20260325_165731666.png"
            _img_center = _os.path.join(_picture_dir, "KakaoTalk_20260325_165731666_01.png")  # 변경: 절대경로 대신 현재 프로젝트 기준 중앙 이미지 사용
            _img_corner = _os.path.join(_picture_dir, "KakaoTalk_20260325_165731666.png")  # 변경: 절대경로 대신 현재 프로젝트 기준 코너 이미지 사용
            if _os.path.exists(_img_center) and _os.path.exists(_img_corner):
                self._overlay_pix_center = _QPixmap(_img_center)
                self._overlay_pix_corner = _QPixmap(_img_corner)
                self._overlay_img_lbl  = QLabel(self.smith_canvas)
                _init_pix = self._overlay_pix_center.scaled(
                    200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._overlay_img_lbl.setPixmap(_init_pix)
                self._overlay_img_lbl.setAlignment(Qt.AlignCenter)
                _eff = _QGOEff(); _eff.setOpacity(0.30)
                self._overlay_img_lbl.setGraphicsEffect(_eff)
                self._overlay_img_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
                self._overlay_img_lbl.raise_()
        except Exception:
            pass

        smith_ctrl_widget = QWidget()
        smith_ctrl_widget.setStyleSheet(
            "background: #f0f4f8; border-radius: 4px; padding: 2px;")
        smith_ctrl = QVBoxLayout(smith_ctrl_widget)
        smith_ctrl.setContentsMargins(6, 4, 6, 4)
        smith_ctrl.setSpacing(4)
        smith_ctrl_top = QHBoxLayout()
        smith_ctrl_top.setSpacing(6)
        smith_ctrl_bottom = QHBoxLayout()
        smith_ctrl_bottom.setSpacing(6)

        lbl_sm = QLabel("Smith")
        lbl_sm.setStyleSheet("font-weight:bold; font-size:10px; color:#2c3e50;")
        smith_ctrl_top.addWidget(lbl_sm)

        self._smith_s11_btn = QPushButton("S11")
        self._smith_s11_btn.setCheckable(True)
        self._smith_s11_btn.setChecked(True)
        self._smith_s11_btn.setFixedWidth(38)
        self._smith_s11_btn.toggled.connect(self._on_smith_trace_toggle)
        smith_ctrl_top.addWidget(self._smith_s11_btn)

        self._smith_s22_btn = QPushButton("S22")
        self._smith_s22_btn.setCheckable(True)
        self._smith_s22_btn.setChecked(True)
        self._smith_s22_btn.setFixedWidth(38)
        self._smith_s22_btn.toggled.connect(self._on_smith_trace_toggle)
        smith_ctrl_top.addWidget(self._smith_s22_btn)

        self._smith_all_btn = QPushButton("All")
        self._smith_all_btn.setCheckable(True); self._smith_all_btn.setFixedWidth(40)
        self._smith_all_btn.setToolTip("모든 주파수 포인트를 스미스차트에 함께 표시")
        self._smith_all_btn.setStyleSheet("""
            QPushButton { border:1px solid #bdc3c7; border-radius:3px;
                          background:#fff; font-size:10px; padding:2px 4px; }
            QPushButton:checked { background:#e74c3c; color:#fff; border-color:#c0392b; }
            QPushButton:hover { background:#ecf0f1; }
        """)
        self._smith_all_btn.toggled.connect(self._on_smith_all_toggle)
        smith_ctrl_top.addWidget(self._smith_all_btn)
        smith_ctrl_top.addStretch(1)

        self._smith_slider = QSlider(Qt.Horizontal)
        self._smith_slider.setMinimum(0); self._smith_slider.setMaximum(100)
        self._smith_slider.setValue(0)
        self._smith_slider.setTickPosition(QSlider.TicksBelow)
        self._smith_slider.setTickInterval(10)
        self._smith_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 5px; background: #bdc3c7; border-radius: 2px; }
            QSlider::handle:horizontal { background: #e74c3c; border: 2px solid #c0392b;
                width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #e74c3c; border-radius: 2px; }
        """)
        self._smith_slider.valueChanged.connect(self._on_smith_slider)
        smith_ctrl_bottom.addWidget(self._smith_slider, stretch=1)

        self._smith_freq_label = QLabel("-- GHz")  # 변경: 초기 주파수 라벨을 정상 문자열로 복원
        self._smith_freq_label.setStyleSheet(
            "font-weight:bold; font-size:11px; color:#e74c3c; min-width:75px;")
        smith_ctrl_bottom.addWidget(self._smith_freq_label)

        self._smith_info_label = QLabel("|Γ|=--  ∠=--")  # 변경: 초기 Smith 정보 라벨을 정상 문자열로 복원
        self._smith_info_label.setStyleSheet(
            "font-size:9px; color:#7f8c8d; min-width:220px;")
        smith_ctrl_bottom.addWidget(self._smith_info_label)
        smith_ctrl.addLayout(smith_ctrl_top)
        smith_ctrl.addLayout(smith_ctrl_bottom)

        self._smith_trace_guide_lbl = QLabel(
            "파랑=S11, 빨강=S22 | 실선: 주파수 경로 | 점선: 현재 선택 포인트 기준선"
        )
        self._smith_trace_guide_lbl.setWordWrap(True)
        self._smith_trace_guide_lbl.setStyleSheet("font-size:9px; color:#6c757d; padding:1px 2px;")
        smith_ctrl.addWidget(self._smith_trace_guide_lbl)

        # Smith 차트 개별 주파수 제어 UI 활성화
        for _w in [self._smith_all_btn]:
            _w.setVisible(True)
        self._smith_slider.setVisible(True)
        self._smith_all_btn.setChecked(True)
        self._smith_all_mode = True
        self._smith_slider.setVisible(False)

        smith_lay.addWidget(smith_ctrl_widget)
        smith_grp.setLayout(smith_lay)
        right_col_lay.addWidget(smith_grp, stretch=5)

        rl.addWidget(right_col_widget, stretch=13)

        self._smith_hover_lbl = QLabel("", self.smith_canvas)
        self._smith_hover_lbl.setStyleSheet("""
            background: rgba(30,39,46,225); color: #ecf0f1;
            font-size: 10px; padding: 5px 8px;
            border-radius: 5px; border: 1px solid #636e72;
        """)
        self._smith_hover_lbl.setVisible(False)
        self._smith_hover_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)

        self._smith_hide_timer = QTimer(self)
        self._smith_hide_timer.setSingleShot(True)
        self._smith_hide_timer.timeout.connect(
            lambda: self._smith_hover_lbl.setVisible(False))

        self.smith_canvas.mpl_connect('motion_notify_event', self._on_smith_hover)
        self.smith_canvas.mpl_connect('button_press_event',  self._on_smith_click)
        self.sp_canvas.mpl_connect('resize_event', self._on_sp_resize)
        self.smith_canvas.mpl_connect('resize_event', self._on_sp_resize)

        ml.addWidget(right, stretch=1)
        if hasattr(self, "impedance_test_tab"):
            self._attach_external_smith_overlay("impedance", getattr(self.impedance_test_tab, "canvas", None) or getattr(self.impedance_test_tab, "smith_canvas", None))  # 변경: 탭 2의 실제 Smith 캔버스(canvas)에 먼저 연결
        if hasattr(self, "sparam_compare_tab"):
            self._attach_external_smith_overlay("compare", getattr(self.sparam_compare_tab, "_smith_canvas", None))  # 변경: 탭 3 스미스차트에도 사진 마크 적용
        self._overlay_state_timer = QTimer(self)  # 변경: 탭 2/3 로고 상태를 데이터 유무에 따라 자동 동기화하는 타이머 추가
        self._overlay_state_timer.timeout.connect(self._sync_external_smith_overlays)  # 변경: 비어 있으면 중앙, 데이터가 있으면 우하단으로 자동 전환
        self._overlay_state_timer.start(400)  # 변경: 사용자 조작 직후에도 빠르게 로고 상태가 따라오도록 짧은 주기로 갱신
        self._sync_external_smith_overlays()  # 변경: 시작 시점에도 탭 2/3 로고 상태를 즉시 맞춤

    # ??????????????????????????????????????????????????????????
    def _picture_overlay_paths(self):
        """현재 프로젝트의 워터마크 이미지 경로를 돌려준다."""  # 변경: 모든 탭이 같은 사진 자산을 재사용
        picture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picture")
        return (
            os.path.join(picture_dir, "KakaoTalk_20260325_165731666_01.png"),
            os.path.join(picture_dir, "KakaoTalk_20260325_165731666.png"),
        )

    def _attach_external_smith_overlay(self, key, canvas):
        """탭 2/3의 스미스차트 캔버스 위에 사진 마크를 붙인다."""  # 변경: 외부 탭 스미스차트에도 동일한 워터마크 적용
        if canvas is None:
            return
        try:
            from PyQt5.QtGui import QPixmap
            from PyQt5.QtWidgets import QGraphicsOpacityEffect

            img_center, img_corner = self._picture_overlay_paths()
            if not (os.path.exists(img_center) and os.path.exists(img_corner)):
                return

            overlay = QLabel(canvas)
            overlay.setAlignment(Qt.AlignCenter)
            overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
            effect = QGraphicsOpacityEffect()
            effect.setOpacity(0.20)
            overlay.setGraphicsEffect(effect)
            overlay.raise_()
            canvas.installEventFilter(self)
            self._extra_smith_overlays[key] = {
                "canvas": canvas,
                "label": overlay,
                "pix_center": QPixmap(img_center),
                "pix_corner": QPixmap(img_corner),
                "corner": False,
            }
            self._reposition_external_smith_overlay(key)
        except Exception:
            pass

    def _external_smith_overlay_has_data(self, key):
        """탭 2/3 스미스차트에 실제 데이터가 있는지 판별한다."""  # 변경: 데이터 없을 때는 큰 중앙 로고, 있으면 우하단 로고로 상태를 분기
        if key == "impedance":
            tab = getattr(self, "impedance_test_tab", None)
            return bool(tab is not None and getattr(tab, "current_network", None) is not None)
        if key == "compare":
            tab = getattr(self, "sparam_compare_tab", None)
            return bool(
                tab is not None and (
                    getattr(tab, "_inv_data", None) is not None or
                    getattr(tab, "_sim_data", None) is not None or
                    getattr(tab, "_meas_data", None) is not None
                )
            )
        return False

    def _sync_external_smith_overlays(self):
        """탭 2/3 스미스차트 로고를 첫 번째 탭과 같은 규칙으로 동기화한다."""  # 변경: 데이터 없으면 중앙 대형 로고, 데이터 있으면 우하단 소형 로고 유지
        for key, item in self._extra_smith_overlays.items():
            corner = self._external_smith_overlay_has_data(key)
            if item.get("corner") != corner:
                item["corner"] = corner
            self._reposition_external_smith_overlay(key)

    def _reposition_external_smith_overlay(self, key):
        """탭 2/3 워터마크 위치와 크기를 캔버스 기준으로 다시 계산한다."""  # 변경: 리사이즈 후에도 사진 마크가 정상 위치 유지
        item = self._extra_smith_overlays.get(key)
        if not item:
            return
        canvas = item["canvas"]
        label = item["label"]
        cw = canvas.width()
        ch = canvas.height()
        if cw < 20 or ch < 20:
            return
        corner = bool(item.get("corner", False))
        pix = item["pix_corner"] if corner else item["pix_center"]
        size = max(16, int(min(cw, ch) * (0.22 if corner else 0.72)))  # 변경: 탭 2/3 중앙 로고가 잘리지 않도록 첫 탭보다 더 보수적인 비율 사용
        scaled = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lw = scaled.width()
        lh = scaled.height()
        label.setPixmap(scaled)
        if corner:
            x = cw - lw - 6
            y = ch - lh - 6
        else:
            x = (cw - lw) // 2
            y = (ch - lh) // 2
        label.setGeometry(x, y, lw, lh)
        label.raise_()

    def eventFilter(self, watched, event):
        """외부 탭 스미스차트 캔버스 리사이즈 시 워터마크 위치를 갱신한다."""  # 변경: 탭 2/3 마크가 창 크기 변경에도 따라가도록 이벤트 필터 추가
        try:
            from PyQt5.QtCore import QEvent
            etype = event.type()
            if etype in (QEvent.Resize, QEvent.Show):
                for key, item in self._extra_smith_overlays.items():
                    if watched is item["canvas"]:
                        self._reposition_external_smith_overlay(key)
                        break
            self._sync_external_smith_overlays()
        except Exception:
            pass
        return super().eventFilter(watched, event)

    def log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())

    def _set_status(self, msg, ok=True):
        self.model_status.setText(msg)
        self.model_status.setStyleSheet(
            f"font-size: 10px; color: {'#2ecc71' if ok else '#e74c3c'};")

    def _manual_load(self):
        sel = QFileDialog.getExistingDirectory(self, "모델 폴더 선택")
        if not sel: return
        self.forward_models = []; self.inverse_model = None
        self.config = None; self.frequencies = None
        self.load_models(force_dir=Path(sel))

    # ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€

    def _on_model_family_changed(self):
        self.model_family = self.model_family_combo.currentData() or 'densenet'
        self.forward_models = []
        self.config = None
        self.frequencies = None
        self.load_models()

    def load_models(self, force_dir: Path = None):
        """forward_fold*.pt를 찾아 순차적으로 로드."""
        self.log("=" * 50)
        requested_family = self.model_family_combo.currentData() or self.model_family
        self.model_family = requested_family
        self.log(f"Model search start  family={requested_family}")
        try:
            script_dir = Path(__file__).parent.resolve()

            if force_dir is not None:
                search_dirs = [force_dir]
                self.log(f"  지정 경로: {force_dir}")
            else:
                search_dirs = default_model_search_dirs(script_dir, requested_family)

            fold_files = []; fold_dir = None; base_dir = None

            for d in search_dirs:
                self.log(f"  탐색: {d}  존재={d.exists()}")
                if not d.exists(): continue
                ff, fd = find_fold_files(d)
                if ff:
                    fold_files, fold_dir, base_dir = ff, fd, d
                    break

            if not fold_files:
                self.log("  forward_fold*.pt를 찾지 못했습니다.")
                self.log("  '모델 폴더 직접 선택' 버튼으로 경로를 지정해 주세요.")
                self._set_status("모델을 찾지 못했습니다\n모델 폴더를 직접 선택해 주세요.", ok=False)  # 변경: 깨진 상태 메시지 문자열 복원
                self._setup_demo(); return

            self.log(f"  fold 파일 {len(fold_files)}개 발견 ({fold_dir})")
            for f in fold_files: self.log(f"     {f.name}")

            # config 로드
            cfg_file = find_config(base_dir, fold_dir)
            if cfg_file:
                with open(cfg_file) as f: self.config = json.load(f)
                self.config = normalize_forward_config(self.config, base_dir)
                self.log(f"  Config: {cfg_file.name}")
            else:
                # 변경: config 없을 때 첫 fold 체크포인트의 마지막 fc/classifier Linear weight
                #        shape 을 보고 output_dim 자동 감지 (3채널=273 vs 8채널=728 구분).
                detected_dim = None
                try:
                    _ck0 = torch.load(fold_files[0], map_location='cpu', weights_only=False)
                    _sd0 = _ck0.get('model', _ck0.get('state_dict', _ck0))
                    last_lin_out = None
                    for _k, _v in _sd0.items():
                        if _k.endswith('.weight') and hasattr(_v, 'ndim') and _v.ndim == 2:
                            if any(p in _k for p in ('fc.', 'classifier.', 'head.', 'output.')):
                                last_lin_out = int(_v.shape[0])
                    if last_lin_out and last_lin_out > 0:
                        detected_dim = last_lin_out
                except Exception:
                    pass
                if detected_dim:
                    self.log(f"  config.json 없음 → 체크포인트에서 output_dim={detected_dim} 자동 감지")
                    self.config = {'output_dim': detected_dim, 'normalization': {}}
                else:
                    self.log(f"  config.json 없음, 기본값 사용 (output_dim={OUTPUT_DIM_DEFAULT})")
                    self.config = {'output_dim': OUTPUT_DIM_DEFAULT,
                                   'normalization': {}}

            resolved_family = infer_model_family(self.config, requested_family)
            self.model_family = resolved_family
            idx_family = self.model_family_combo.findData(resolved_family)
            if idx_family >= 0 and self.model_family_combo.currentIndex() != idx_family:
                self.model_family_combo.blockSignals(True)
                self.model_family_combo.setCurrentIndex(idx_family)
                self.model_family_combo.blockSignals(False)
            output_dim = self.config.get('output_dim', OUTPUT_DIM_DEFAULT)
            # stem_type 추론: 루트에 없으면 fold_results[0]에서 탐색
            stem_type = self.config.get('stem_type')
            if not stem_type and 'fold_results' in self.config and len(self.config['fold_results']) > 0:
                stem_type = self.config['fold_results'][0].get('stem_type')
            if not stem_type:
                stem_type = 'standard'
            self.config['stem_type'] = stem_type  # build_forward_model에서 참조할 수 있게 저장

            backbone = self.config.get('backbone', resolved_family)
            display_name = "ResNet" if resolved_family == "resnet" else ("DenseNet" if resolved_family == "densenet" else resolved_family.capitalize())
            self.log(f"  family = '{display_name}'  backbone = '{backbone}'")
            self.log(f"  stem_type = '{stem_type}'  output_dim = {output_dim}")

            # forward ?? ??
            for ff in fold_files:
                try:
                    m = build_forward_model(resolved_family, self.config)
                    # [FIX 4] weights_only=True ??PyTorch 踰꾩쟾???곕씪
                    # ?ㅽ뙣?????덉쑝誘濡?False ?대갚 ?ы븿
                    try:
                        ckpt = torch.load(ff, map_location=self.device,
                                          weights_only=True)
                    except Exception:
                        ckpt = torch.load(ff, map_location=self.device,
                                          weights_only=False)
                    # [FIX] state_dict 키 매핑 보정 (densenet. 접두어 유무 대응)
                    sd = ckpt.get('model', ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt)))
                    new_sd = OrderedDict()
                    model_keys = m.state_dict().keys()
                    has_densenet_prefix = any(k.startswith('densenet.') for k in model_keys)

                    for k, v in sd.items():
                        if k.startswith('module.'):
                            k = k[7:]
                        if has_densenet_prefix and not k.startswith('densenet.'):
                            new_sd['densenet.' + k] = v
                        elif not has_densenet_prefix and k.startswith('densenet.'):
                            new_sd[k[9:]] = v
                        elif resolved_family == 'efficientnet' and not k.startswith('model.'):
                            new_sd['model.' + k] = v
                        else:
                            new_sd[k] = v

                    m.load_state_dict(new_sd, strict=False)
                    m.to(self.device).eval()
                    for p in m.parameters(): p.requires_grad = False
                    self.forward_models.append(m)
                    self.log(f"  로드 성공: {ff.name}")
                except Exception as e:
                    self.log(f"  로드 실패: {ff.name} / {e}")

            if not self.forward_models:
                self.log("  모든 모델 로드 실패")
                self._set_status("모델 로드 실패", ok=False)
                self._setup_demo(); return

            # 주파수 로드
            freq_file = find_frequencies(base_dir, fold_dir)
            if freq_file:
                self.frequencies = np.load(freq_file)
                self.log(f"  Freq: {self.frequencies[0]:.2f}~{self.frequencies[-1]:.2f} GHz "
                         f"({len(self.frequencies)} pts)")
            else:
                self.log(f"  frequencies.npy 없음, 내부 기본값 사용 ({N_FREQS_DEFAULT} pts)")
                self.frequencies = _FALLBACK_FREQS.copy()

            # algo combo ?낅뜲?댄듃
            self.algo_combo.clear()
            self.algo_combo.addItems(self._base_algo_options)

            family_display = "ResNet" if self.model_family == "resnet" else ("DenseNet" if self.model_family == "densenet" else self.model_family.capitalize())
            self._set_status(
                f"✅ {family_display} Forward {len(self.forward_models)}개 로드", ok=True)
            self.log("=" * 50)
            self._verify_forward()
            self._check_spec_feasibility()

        except Exception as e:
            self.log(f"ERROR: {e}\n{traceback.format_exc()}")
            self._set_status("모델 로드 중 오류", ok=False)
            self._setup_demo()

    def _verify_forward(self):
        """랜덤 25x25 레이아웃으로 Forward 모델 동작 검증."""
        if not self.forward_models: return
        self.log(f"Forward 검증({len(self.forward_models)}개 모델, {GRID_SIZE}x{GRID_SIZE})")
        norm = self.config.get('normalization', {})
        # nested dict ?щ㎎: {'s11_db': {'min':..,'max':..}}
        if 's11_db' in norm:
            s_min = norm['s11_db'].get('min', norm['s11_db'].get('raw_min', -140.0))
            s_max = norm['s11_db'].get('max', norm['s11_db'].get('raw_max', 0.0))
        else:
            s_min = norm.get('s_min', -140.0)
            s_max = norm.get('s_max',    0.0)
        with torch.no_grad():
            for i in range(2):
                np.random.seed(i * 42)
                l  = np.random.randint(0, 2, (GRID_SIZE, GRID_SIZE)).astype(np.float32)
                t  = torch.FloatTensor(l).unsqueeze(0).unsqueeze(0).to(self.device)
                p  = ensemble_predict(self.forward_models, t, 512).cpu().numpy()[0]
                try:
                    raw = self.denorm_clamp(p)
                    view = self._extract_plot_channels(raw, 50.0 + 0j, 50.0 + 0j)
                    db = np.concatenate([view[k] for k in ('s11_db', 's21_db', 's22_db')])
                except Exception:
                    db = p * (s_max - s_min) + s_min
                self.log(f"  Test{i+1}: norm=[{p.min():.3f},{p.max():.3f}] "
                         f"dB=[{db.min():.1f},{db.max():.1f}]")
        self.log("")

    def _setup_demo(self):
        if self.frequencies is None:
            self.frequencies = _FALLBACK_FREQS.copy()
        if self.config is None:
            self.config = {'output_dim': OUTPUT_DIM_DEFAULT,
                           'normalization': {
                               's11_db': {'min': -140.0, 'max': 0.0},
                               's21_db': {'min': -140.0, 'max': 0.0},
                               's22_db': {'min': -140.0, 'max': 0.0},
                           }}

    # ??????????????????????????????????????????????????????????
    def _apply_renorm(self):
        """Apply Renorm 버튼: 현재 Z0 값으로 스미스 차트 + S-param 렌더링 + 타겟 밴드 업데이트."""
        if not self.all_results:
            QMessageBox.information(self, "Info",
                "역설계 결과가 없습니다.\n먼저 역설계를 실행해 주세요.")
            return
        try:
            p1 = parse_z0_input(self.p1_z0_edit.text())
            p2 = parse_z0_input(self.p2_z0_edit.text())
        except ValueError as e:
            QMessageBox.warning(self, "Z0 입력 오류", str(e))
            return
            
        try:
            params = self.validate()
            if params:
                self.config['f_low'] = params['f_low']
                self.config['f_high'] = params['f_high']
        except Exception:
            pass
            
        self.log(f"Renorm 및 Target 갱신: P1={p1}, P2={p2}")
        self._viz_compare()   # _viz_compare 안에서 Z0 읽어 재정규화 적용

    # ??????????????????????????????????????????????????????????
    def _update_fc_bw(self):
        try:
            fl = float(self.f_low.text())
            fh = float(self.f_high.text())
            if fh > fl > 0:
                fc = (fl + fh) / 2.0
                bw = (fh - fl) / fc * 100.0
                self.fc_input.setText(f"{fc:.4g}")
                self.bw_input.setText(f"{bw:.4g}")
        except ValueError:
            pass

    # 변경: Pass Band 경계(GHz) 자동 계산
    # L-PB (GHz) = F_L * (1 - pct/100)
    # R-PB (GHz) = F_H * (1 + pct/100)
    def _update_pass_band(self):
        # 변경: GHz 오프셋 기반. L-PB = F_L - offset, R-PB = F_H + offset. S21/S12 타겟선 전용.
        try:
            fl = float(self.f_low.text())
            l_off = float(self.l_pb_offset_input.text())
            l_pb = fl - l_off
            self.l_pb_freq.setText(f"{l_pb:.4g}")
        except (ValueError, AttributeError):
            try:
                self.l_pb_freq.setText("")
            except AttributeError:
                pass
        try:
            fh = float(self.f_high.text())
            r_off = float(self.r_pb_offset_input.text())
            r_pb = fh + r_off
            self.r_pb_freq.setText(f"{r_pb:.4g}")
        except (ValueError, AttributeError):
            try:
                self.r_pb_freq.setText("")
            except AttributeError:
                pass
        # 변경: Pass Band가 바뀌면 S21/S12 플롯 배경도 다시 그려야 함
        try:
            self._refresh_plot_backgrounds()
        except AttributeError:
            pass


    # ??????????????????????????????????????????????????????????
    def validate(self):
        try:
            fl = float(self.f_low.text());  fh = float(self.f_high.text())
            il = float(self.il_input.text()); rl = float(self.rl_input.text())
            sl = float(self.sl_input.text())  # 변경: Stopband Loss (dB, 음수)
            ps = int(self.pop_input.text());  cs = int(self.chunk_input.text())
            split_index = int(self.split_line_index.text())  # 변경: LPF 모드에선 유지하고 BPF 모드에선 아래에서 랜덤 index로 대체 가능
            split_width = int(self.split_line_width.text())  # 변경: 분리선 폭을 입력값에서 읽음
            # ?? ?꾪뵾?섏뒪 ?뚯떛 ??????????????????????????????????????
            p1_z0 = parse_z0_input(self.p1_z0_edit.text())
            p2_z0 = parse_z0_input(self.p2_z0_edit.text())
            # ?????????????????????????????????????????????????????
            if fl >= fh:       QMessageBox.warning(self, "Error", "Low < High 이어야 합니다."); return None  # 변경: 깨진 범위 검증 문구 복원
            if il > 0 or rl > 0 or sl > 0: QMessageBox.warning(self, "Error", "dB 값은 음수여야 합니다."); return None  # 변경: 깨진 dB 검증 문구 복원
            if split_width < 1: QMessageBox.warning(self, "Error", "Split width는 1 이상이어야 합니다."); return None  # 변경: 분리선 폭 검증
            if self._split_preset == "bpf":
                lo = max(1, split_width)
                hi = max(lo + 1, GRID_SIZE - split_width)
                split_index = int(np.random.randint(lo, hi))  # 변경: Band-pass는 실행할 때마다 랜덤 분리선 위치를 뽑아 탐색 다양성 확보
                self.split_line_index.setText(str(split_index))  # 변경: 실제 사용된 랜덤 위치를 GUI에도 반영
                split_seed = int(np.random.randint(0, 1_000_000_000))  # 변경: 실행마다 새로운 지그재그/대각선 분리선 경로를 생성할 seed 저장
                split_pattern = "jagged"  # 변경: BPF 프리셋은 직선 대신 대각선이 섞인 지그재그 분리선을 사용
            else:
                split_seed = 0  # 변경: LPF는 분리선이 없으므로 seed 불필요
                split_pattern = "straight"  # 변경: LPF는 직선 패턴 기본값만 유지
            if not (0 <= split_index < GRID_SIZE): QMessageBox.warning(self, "Error", f"Split index는 0~{GRID_SIZE - 1} 범위여야 합니다."); return None  # 변경: 최종 분리선 위치 범위 검증
            return dict(f_low=fl, f_high=fh, il=il, rl=rl, sl=sl,
                        pop_size=ps, chunk_size=cs,
                        p1_z0=p1_z0, p2_z0=p2_z0,
                        split_line_preset=self._split_preset,
                        split_line_pattern=split_pattern,
                        split_line_seed=split_seed,
                        split_line_enabled=self.split_line_enabled.isChecked(),
                        split_line_mode=self.split_line_mode.currentText().strip().lower(),
                        split_line_index=split_index,
                        split_line_width=split_width)
        except ValueError as e:
            QMessageBox.warning(self, "Error", f"입력 오류: {e}"); return None  # 변경: 깨진 입력 오류 문구 복원

    def gen_targets(self, freqs, fl, fh, il, rl, sl=-6.0):
        # 변경: Stopband Loss를 반영한 S21/S12 stopband 목표 계산
        # 실제 stopband 타겟 dB = IL - 3 + SL (3 dB 고정 전이 마진, SL/IL 모두 음수)
        pb = (freqs >= fl) & (freqs <= fh)
        stopband_level = float(il) - 3.0 + float(sl)
        t21 = np.where(pb, float(il), stopband_level)
        return (gaussian_filter1d(np.where(pb, rl, -3.0), 2),
                t21,
                gaussian_filter1d(np.where(pb, rl, -3.0), 2))



    def _parse_algo(self, text):
        if "All (" in text:       return ['GD', 'GA', 'BPSO', 'BO', 'DE',
                                          'GA_GD', 'BPSO_GD', 'GA_DBS', 'BPSO_DBS']
        if "GA->GD" in text:      return ['GA_GD']
        if "BPSO->GD" in text:    return ['BPSO_GD']
        if "GA->DBS" in text:     return ['GA_DBS']
        if "BPSO->DBS" in text:   return ['BPSO_DBS']
        if "Only"  in text:       return [text.split()[0]]
        return [a.strip() for a in text.replace("(","").replace(")","").split("+")]

    def _run_next(self):
        if not self.pending_algos: self._on_all_done(); return
        algo = self.pending_algos.pop(0)
        self.log(f"→ {algo} 시작...")
        if not self._conv_data:
            self._conv_data = {}
        self._conv_data[algo] = []
        self._redraw_conv()
        init = None
        self.worker = InverseDesignWorker(
            self.forward_models, self.targets_norm, self.config, algo,
            self.device, self.frequencies, self._pb_mask, self._sb_mask, init,
            self._pop_size, self._chunk_size,
            p1_z0=self._p1_z0, p2_z0=self._p2_z0)
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._on_single_done)
        self.worker.error.connect(self._on_error)
        self.worker.conv_update.connect(self._on_conv_update)
        self.worker.layout_update.connect(self._on_layout_update)
        self.worker.start()

    def _on_single_done(self, result):
        self.all_results[result['algorithm']] = result
        self.log(f"  {result['algorithm']}: Error {result['error']:.4f}")
        self._run_next()

    def _sp_algo_visible(self, algo):
        if not hasattr(self, '_sp_algo_btns'): return True
        btn = self._sp_algo_btns.get(algo)
        return True if btn is None else btn.isChecked()

    def _on_sp_algo_toggle(self, _checked):
        if self.all_results: self._viz_compare()

    def _add_band_shading(self, ax, fl, fh, fmin, fmax, mode='refl'):
        # 변경: mode별로 분기 — refl(S11/S22)는 pink/green/pink, trans(S21/S12)는 pink/white/green/white/pink
        GREEN = '#2ECC71'   # Pass band
        RED   = '#FF6B6B'   # Stop band (연한 빨강/핑크)
        WHITE = '#FFFFFF'   # Transition band

        if mode == 'trans':
            # 변경: 배경 토글이 OFF 면 아예 axvspan 안 그림 (모두 흰배경)
            if not getattr(self, '_show_sp_bg', True):
                return
            # L-PB / R-PB를 GUI 입력값에서 읽어옴 (없거나 잘못되면 FL/FH 사용)
            try:
                l_pb = float(self.l_pb_freq.text())
            except (ValueError, AttributeError):
                l_pb = fl
            try:
                r_pb = float(self.r_pb_freq.text())
            except (ValueError, AttributeError):
                r_pb = fh
            l_pb = min(l_pb, fl)
            r_pb = max(r_pb, fh)

            if l_pb > fmin:
                ax.axvspan(fmin, l_pb, alpha=0.18, color=RED, zorder=0)
            if fl > l_pb:
                ax.axvspan(l_pb, fl, alpha=1.0, color=WHITE, zorder=0)
            ax.axvspan(fl, fh, alpha=0.15, color=GREEN, zorder=0)
            if r_pb > fh:
                ax.axvspan(fh, r_pb, alpha=1.0, color=WHITE, zorder=0)
            if fmax > r_pb:
                ax.axvspan(r_pb, fmax, alpha=0.18, color=RED, zorder=0)
            # 경계선 (중복 제거)
            seen = set()
            for fv in [l_pb, fl, fh, r_pb]:
                key = round(float(fv), 6)
                if key in seen:
                    continue
                seen.add(key)
                ax.axvline(fv, color='gray', ls=':', lw=0.7, alpha=0.55)
        else:
            # 변경: refl(S11/S22) 는 핑크 stopband / 흰색 transition 모두 제거.
            #   FL~FH 초록 (passband) 만 표시. 나머지는 흰색 (axvspan 없음 = 기본 흰배경).
            #   배경 토글이 OFF 면 초록도 그리지 않음.
            if getattr(self, '_show_sp_bg', True):
                ax.axvspan(fl, fh, alpha=0.15, color=GREEN, zorder=0)
                ax.axvline(fl, color='gray', ls=':', lw=0.7, alpha=0.55)
                ax.axvline(fh, color='gray', ls=':', lw=0.7, alpha=0.55)

    # 변경: 타겟선 On/Off 토글 핸들러
    def _on_target_line_toggle(self):
        visible = bool(self.show_target_btn.isChecked())
        self.show_target_btn.setText("Show Target" if visible else "Hide Target")
        lines = getattr(self, '_target_lines', None)
        if not lines:
            # 아직 그려진 그래프가 없으면 현재 결과로 한 번 다시 그려둠
            if getattr(self, 'all_results', None):
                try:
                    self._viz_compare()
                except Exception:
                    pass
            return
        for ln in lines:
            try:
                ln.set_visible(visible)
            except Exception:
                pass
        try:
            self.sp_canvas.draw_idle()
        except Exception:
            pass

    # 변경: S-Param 배경 토글 핸들러 - 그래프를 다시 그려서 axvspan 적용
    def _on_sp_bg_toggle(self):
        visible = bool(self.show_bg_btn.isChecked())
        self.show_bg_btn.setText("Show BG" if visible else "Hide BG")
        self._show_sp_bg = visible
        if getattr(self, 'all_results', None):
            try:
                self._viz_compare()
            except Exception:
                pass

    # 변경: VSWR 원 토글 핸들러 - 스미스차트만 다시 그림
    def _on_vswr_toggle(self):
        visible = bool(self.show_vswr_btn.isChecked())
        self.show_vswr_btn.setText("Show VSWR" if visible else "Hide VSWR")
        self._show_vswr_circle = visible
        # 스미스차트 다시 그리기
        try:
            slider = getattr(self, '_smith_slider', None)
            slider_val = slider.value() if slider is not None else 0
            self._redraw_smith(slider_val)
        except Exception:
            pass

    # 변경: Pass Band(L-PB/R-PB) 값이 바뀌면 배경/타겟선을 다시 그리는 헬퍼
    def _refresh_plot_backgrounds(self):
        if not getattr(self, 'all_results', None):
            return
        try:
            self._viz_compare()
        except Exception:
            pass

    def _update_progress(self, val, msg):
        self.progress.setValue(val); self.log(msg)

    def _on_error(self, msg):
        self.log(f"ERROR: {msg}")
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)

    def stop_inverse(self):
        if self.worker: self.worker.stop(); self.log("以묒? 以?..")
        self.stop_btn.setEnabled(False); self.pending_algos = []

    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    # ?명꽣?숉떚釉??ㅻ??ㅼ감??
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    def _rebuild_smith_slider(self):
        if self._smith_freqs is None: return
        freq_mask = self._smith_display_mask()
        self._smith_indices = np.where(freq_mask)[0]
        n = len(self._smith_indices)
        self._smith_slider.blockSignals(True)
        self._smith_slider.setMinimum(0)
        self._smith_slider.setMaximum(max(0, n - 1))
        self._smith_slider.setValue(0)
        self._smith_slider.setTickInterval(max(1, n // 10))
        self._smith_slider.blockSignals(False)

    def _on_smith_pb_toggle(self):
        self._rebuild_smith_slider(); self._redraw_smith(0)

    def _on_smith_trace_toggle(self):
        self._smith_show_s11 = self._smith_s11_btn.isChecked()
        self._smith_show_s22 = self._smith_s22_btn.isChecked()
        if self._smith_Z is not None:
            self._redraw_smith(self._smith_slider.value())

    def _smith_display_mask(self):
        if self._smith_freqs is None:
            return np.array([], dtype=bool)
        mask = np.ones_like(self._smith_freqs, dtype=bool)
        try:
            fl = float(self.plot_fl_input.text())
            fh = float(self.plot_fh_input.text())
            if fh > fl:
                mask &= (self._smith_freqs >= fl) & (self._smith_freqs <= fh)
        except Exception:
            pass
        return mask



    def _on_smith_slider(self, slider_val):
        self._redraw_smith(slider_val)

    def _draw_smith_grid(self, ax):
        import skrf as rf
        MARGIN = 1.30
        ax.set_xlim(-MARGIN, MARGIN); ax.set_ylim(-MARGIN, MARGIN)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_facecolor('#FFFFFF')
        rf.plotting.smith(ax=ax, draw_labels=True)
        try:
            z0_s11 = parse_z0_input(self.p1_z0_edit.text())  # 변경: 표시 라벨은 항상 현재 Inverse Design 입력칸 값을 직접 사용
            z0_s22 = parse_z0_input(self.p2_z0_edit.text())  # 변경: 상태 캐시가 꼬여도 화면에는 현재 입력값을 그대로 반영
        except Exception:
            z0_s11 = getattr(self, '_smith_z0_ref_s11', getattr(self, '_smith_z0_ref', 50.0))
            z0_s22 = getattr(self, '_smith_z0_ref_s22', getattr(self, '_smith_z0_ref', 50.0))
        def _fmt_z0(value):
            z = complex(value)
            if abs(z.imag) < 1e-9:
                return f"{z.real:.0f}Ω"
            return f"{z.real:.0f}{z.imag:+.0f}jΩ"
        ax.text(0.015, 0.015, f'Port1 $Z_0$ = {_fmt_z0(z0_s11)}   |   Port2 $Z_0$ = {_fmt_z0(z0_s22)}',
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=7.5, color='#555', zorder=8,
                bbox=dict(boxstyle='round,pad=0.22', fc='#f8f8f8',
                          ec='#ccc', alpha=0.92))

    def _redraw_smith(self, slider_val):
        if self._smith_Z is None or not hasattr(self, '_smith_indices'): return
        if self._ax_smith is None: return
        indices = self._smith_indices
        if len(indices) == 0: return

        slider_val = int(np.clip(slider_val, 0, len(indices) - 1))
        freq_idx   = indices[slider_val]
        cur_freq   = self._smith_freqs[freq_idx]
        cur_mag    = self._smith_mag[freq_idx]
        cur_phase  = np.degrees(self._smith_phase[freq_idx])
        cur_Z      = self._smith_Z[freq_idx]
        gamma_s11_all = self._smith_gamma_s11 if self._smith_gamma_s11 is not None else getattr(self, '_smith_gamma_all', None)
        gamma_s22_all = self._smith_gamma_s22
        Z_s11_all = self._smith_Z_s11 if self._smith_Z_s11 is not None else self._smith_Z
        Z_s22_all = self._smith_Z_s22 if self._smith_Z_s22 is not None else self._smith_Z

        if not self._smith_all_mode:
            self._smith_freq_label.setText(f"{cur_freq:.3f} GHz")
            info_parts = []
            if self._smith_show_s11 and gamma_s11_all is not None and Z_s11_all is not None:
                g11 = gamma_s11_all[freq_idx]
                z11 = Z_s11_all[freq_idx]
                info_parts.append(
                    f"S11 |Γ|={abs(g11):.3f} ∠={np.degrees(np.angle(g11)):.1f}° "
                    f"Z={z11.real:.1f}{z11.imag:+.1f}jΩ"
                )
            if self._smith_show_s22 and gamma_s22_all is not None and Z_s22_all is not None:
                g22 = gamma_s22_all[freq_idx]
                z22 = Z_s22_all[freq_idx]
                info_parts.append(
                    f"S22 |Γ|={abs(g22):.3f} ∠={np.degrees(np.angle(g22)):.1f}° "
                    f"Z={z22.real:.1f}{z22.imag:+.1f}jΩ"
                )
            self._smith_info_label.setText("   ".join(info_parts) if info_parts else f"|Γ|={cur_mag:.3f}  ∠={cur_phase:.1f}°")
        else:
            self._smith_freq_label.setText("ALL")
            self._smith_info_label.setText("마우스를 올려 값을 확인")

        try:
            self.smith_fig.delaxes(self._ax_smith)
        except Exception:
            pass
        self._ax_smith = self.smith_fig.add_subplot(111, aspect='equal')
        self._ax_smith.set_xlim(-1.30, 1.30); self._ax_smith.set_ylim(-1.30, 1.30)
        self._ax_smith.axis('off')

        ax    = self._ax_smith
        Z     = self._smith_Z
        color = self._smith_color
        self._draw_smith_grid(ax)

        # 변경: VSWR(=RL 기준) 원 그리기. RL[dB] -> |Γ| = 10^(RL/20).
        #        중심 (0,0), 반지름 |Γ|, 연두색 fill (passband 배경과 동일).
        # ── VSWR(RL) 기반 가이드 원 및 목표 영역 시각화 (최종 추천안) ─────
        if getattr(self, '_show_vswr_circle', True):
            try:
                from matplotlib.patches import Circle as _MCircle
                
                # 1. 실제 사용자 Target RL 구역 (초록색 반투명 강조)
                rl_db = float(self.rl_input.text())
                gamma_target = 10.0 ** (rl_db / 20.0)
                if 0.0 < gamma_target < 1.0:
                    target_circle = _MCircle(
                        (0.0, 0.0), gamma_target,
                        fill=True, color='green', alpha=0.08, zorder=2.3,
                    )
                    ax.add_patch(target_circle)
                    
                    # 타겟 경계선 및 라벨
                    edge_circle = _MCircle(
                        (0.0, 0.0), gamma_target,
                        fill=False, edgecolor='#27AE60', ls='-', lw=1.8, alpha=0.6, zorder=2.5
                    )
                    ax.add_patch(edge_circle)
                    
                    vswr_target = (1 + gamma_target) / (1 - gamma_target)
                    ax.text(0.0, -gamma_target,
                            f'TARGET RL {rl_db:.1f}dB (VSWR {vswr_target:.2f})',
                            ha='center', va='top', fontsize=7.5, color='#1E8449',
                            fontweight='bold', zorder=2.7,
                            bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF',
                                      ec='#27AE60', alpha=0.9))
            except (ValueError, AttributeError):
                pass

        gamma_all = gamma_s11_all if gamma_s11_all is not None else getattr(self, '_smith_gamma_all', (Z - 50.0) / (Z + 50.0))
        display_mask = self._smith_display_mask()
        if display_mask.size and np.any(display_mask):
            display_indices = np.where(display_mask)[0]
        else:
            display_indices = np.arange(len(gamma_all))

        def _draw_trace(gamma_src, trace_color, title, current_idx, zorder_base=3):
            if gamma_src is None:
                return
            gx_display = gamma_src[display_indices].real
            gy_display = gamma_src[display_indices].imag
            if self._smith_all_mode:
                n = len(gx_display)
                for i in range(n - 1):
                    ax.plot([gx_display[i], gx_display[i+1]], [gy_display[i], gy_display[i+1]],
                            color=trace_color, lw=1.2, alpha=0.8, zorder=zorder_base,
                            solid_capstyle='round')
                ax.scatter(gx_display, gy_display, c=trace_color, s=22, alpha=0.85, zorder=zorder_base + 1,
                           linewidths=0.3, edgecolors='white')
            else:
                ax.plot(gx_display, gy_display, color='#cccccc', lw=1.0, alpha=0.5,
                        zorder=zorder_base, solid_capstyle='round')
                gx_sel = gamma_src[indices].real
                gy_sel = gamma_src[indices].imag
                ni = len(indices)
                for i in range(ni - 1):
                    ax.plot([gx_sel[i], gx_sel[i+1]], [gy_sel[i], gy_sel[i+1]],
                            color=trace_color, lw=2.0, alpha=0.95, zorder=zorder_base + 1,
                            solid_capstyle='round')
                g_cur = gamma_src[current_idx]
                ax.plot(g_cur.real, g_cur.imag, color=trace_color,
                        marker='o', markersize=8, linestyle='none',
                        markeredgecolor='white', markeredgewidth=1.5, zorder=zorder_base + 2)
                ax.plot([0, g_cur.real], [0, g_cur.imag],
                        color=trace_color, lw=0.8, alpha=0.45, linestyle='--', zorder=zorder_base)

        shown = []
        if self._smith_show_s11:
            _draw_trace(gamma_s11_all, '#3498db', 'S11', freq_idx, 3)
            shown.append("S11")
        if self._smith_show_s22:
            _draw_trace(gamma_s22_all, '#e74c3c', 'S22', freq_idx, 6)
            shown.append("S22")

        shown_text = " / ".join(shown) if shown else "No Trace"
        suffix = " [ALL pts]" if self._smith_all_mode else ""
        ax.set_title(f"{shown_text} Smith Chart - {self._smith_best_algo}{suffix}",
                     fontsize=9, fontweight='bold', pad=6)



        self.smith_canvas.draw()
        # ?ㅻ??ㅼ감?멸? 洹몃젮議뚯쑝誘濡??ㅻ쾭?덉씠瑜??ㅻⅨ履??섎떒 援ъ꽍?쇰줈 ?대룞
        self._reposition_overlay(corner=True)

    def _on_smith_all_toggle(self, checked):
        self._smith_all_mode = checked
        self._smith_slider.setEnabled(not checked)
        self._smith_slider.setVisible(not checked) # ALL 모드일 때는 슬라이더 숨김
        if self._smith_Z is not None:
            self._redraw_smith(self._smith_slider.value())

    def _reposition_overlay(self, corner: bool):
        """?ㅻ쾭?덉씠 ?대?吏 ?꾩튂/?ш린 議곗젙.
        corner=False ??罹붾쾭???뺤쨷??(?ㅻ??ㅼ감???놁쓣 ??
        corner=True  ???ㅻⅨ履??섎떒 援ъ꽍 (?ㅻ??ㅼ감???덉쓣 ??
        """
        if not (hasattr(self, '_overlay_img_lbl') and self._overlay_img_lbl is not None):
            return
        try:
            self._overlay_at_corner = corner
            cw = self.smith_canvas.width()
            ch = self.smith_canvas.height()
            if cw < 20 or ch < 20:
                return
            if corner:
                if not hasattr(self, '_overlay_pix_corner') or self._overlay_pix_corner is None: return
                pix_target = self._overlay_pix_corner
                # 援ъ꽍: ?⑤???22% ?ш린, ?ㅻⅨ履??섎떒?먯꽌 6px ?щ갚
                size = int(min(cw, ch) * 0.22)
                margin = 6
            else:
                if not hasattr(self, '_overlay_pix_center') or self._overlay_pix_center is None: return
                pix_target = self._overlay_pix_center
                # 以묒븰: ?⑤???72% ?ш린
                size = int(min(cw, ch) * 1.3)
                margin = 0
            size = max(size, 16)
            scaled = pix_target.scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._overlay_img_lbl.setPixmap(scaled)
            lw, lh = scaled.width(), scaled.height()
            if corner:
                x = cw - lw - margin
                y = ch - lh - margin
            else:
                x = (cw - lw) // 2
                y = (ch - lh) // 2
            self._overlay_img_lbl.setGeometry(x, y, lw, lh)
            self._overlay_img_lbl.raise_()
        except Exception:
            pass

    def _on_sp_resize(self, event):
        try:
            self.sp_fig.canvas.draw_idle()
            self.smith_fig.canvas.draw_idle()
        except Exception:
            pass
        if hasattr(self, '_gui_logo'):
            p = self._gui_logo.parentWidget()
            if p:
                self._gui_logo.move(p.width() - self._gui_logo.width() - 10,
                                    p.height() - self._gui_logo.height() - 10)
        # ?꾩옱 corner ?곹깭瑜??좎??섎ŉ ?ш린/?꾩튂留?媛깆떊
        self._reposition_overlay(getattr(self, '_overlay_at_corner', False))

    def _on_smith_hover(self, event):
        self._smith_hide_timer.stop()
        if self._smith_Z is None or not self.smith_fig.axes:
            self._smith_hover_lbl.setVisible(False); return
        if event.xdata is None or event.ydata is None:
            self._smith_hide_timer.start(300); return
        if self._ax_smith is None or event.inaxes is not self._ax_smith:
            self._smith_hide_timer.start(150); return
        hit = self._smith_nearest_idx(event.xdata, event.ydata)
        if hit is None:
            self._smith_hide_timer.start(150); return
        trace_name, idx = hit
        self._smith_show_tooltip(trace_name, idx, event.xdata, event.ydata)

    def _on_smith_click(self, event):
        if self._smith_Z is None: return
        if event.inaxes is None or event.xdata is None or event.ydata is None: return
        if self._ax_smith is None or event.inaxes is not self._ax_smith: return
        if event.button != 1: return
        hit = self._smith_nearest_idx(event.xdata, event.ydata, threshold_factor=3.0)
        if hit is None: return
        trace_name, idx = hit
        if hasattr(self, '_smith_indices') and len(self._smith_indices) > 0:
            slider_pos = int(np.argmin(np.abs(self._smith_indices - idx)))
            self._smith_slider.blockSignals(True)
            self._smith_slider.setValue(slider_pos)
            self._smith_slider.blockSignals(False)
        self._smith_show_tooltip(trace_name, idx, event.xdata, event.ydata, pinned=True)

    def _smith_nearest_idx(self, xdata, ydata, threshold_factor=1.0):
        display_mask = self._smith_display_mask()
        if display_mask.size and np.any(display_mask):
            candidate_idx = np.where(display_mask)[0]
        else:
            candidate_idx = np.arange(len(self._smith_freqs))
        threshold = (0.04 * threshold_factor) ** 2
        best = None
        if self._smith_show_s11 and self._smith_gamma_s11 is not None:
            gamma = self._smith_gamma_s11[candidate_idx]
            dist = (gamma.real - xdata) ** 2 + (gamma.imag - ydata) ** 2
            pos = int(np.argmin(dist))
            best = ("S11", int(candidate_idx[pos]), float(dist[pos]))
        if self._smith_show_s22 and self._smith_gamma_s22 is not None:
            gamma = self._smith_gamma_s22[candidate_idx]
            dist = (gamma.real - xdata) ** 2 + (gamma.imag - ydata) ** 2
            pos = int(np.argmin(dist))
            cand = ("S22", int(candidate_idx[pos]), float(dist[pos]))
            if best is None or cand[2] < best[2]:
                best = cand
        if best is None or best[2] > threshold:
            return None
        return best[0], best[1]

    def _smith_show_tooltip(self, trace_name, idx, xdata, ydata, pinned=False):
        f = self._smith_freqs[idx]
        if trace_name == "S22" and self._smith_gamma_s22 is not None:
            gamma = self._smith_gamma_s22[idx]
            z = self._smith_Z_s22[idx] if self._smith_Z_s22 is not None else self._smith_Z[idx]
        else:
            trace_name = "S11"
            gamma = self._smith_gamma_s11[idx] if self._smith_gamma_s11 is not None else self._smith_gamma_all[idx]
            z = self._smith_Z_s11[idx] if self._smith_Z_s11 is not None else self._smith_Z[idx]
        pin = "  [pin]" if pinned else ""
        txt = (
            f"freq = {f:.3f} GHz{pin}\n"
            f"{trace_name} = {gamma.real:.4f}{gamma.imag:+.4f}j\n"
            f"impedance = {z.real:.2f}{z.imag:+.2f}j Ω"
        )
        self._smith_freq_label.setText(f"{f:.3f} GHz")
        self._smith_info_label.setText(
            f"{trace_name} = {gamma.real:.3f}{gamma.imag:+.3f}j   Z = {z.real:.1f}{z.imag:+.1f}j Ω")
        self._smith_hover_lbl.setText(txt)
        self._smith_hover_lbl.adjustSize()
        try:
            ax    = self._ax_smith
            disp  = ax.transData.transform((gamma.real, gamma.imag))
            fig_w = self.smith_fig.get_size_inches()[0] * self.smith_fig.dpi
            fig_h = self.smith_fig.get_size_inches()[1] * self.smith_fig.dpi
            cw    = self.smith_canvas.width()
            ch    = self.smith_canvas.height()
            cx    = int(disp[0] * cw / max(fig_w, 1)) + 14
            cy    = int((fig_h - disp[1]) * ch / max(fig_h, 1)) + 14
        except Exception:
            cx, cy = 20, 20
        lw = self._smith_hover_lbl.width()
        lh = self._smith_hover_lbl.height()
        cw2 = self.smith_canvas.width()
        ch2 = self.smith_canvas.height()
        cx  = int(min(max(0, cx), cw2 - lw - 4))
        cy  = int(min(max(0, cy), ch2 - lh - 4))
        self._smith_hover_lbl.move(cx, cy)
        self._smith_hover_lbl.setVisible(True)
        self._smith_hover_lbl.raise_()

    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    # 데이터 분포 경고
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    def _check_spec_feasibility(self):
        if self.config is None:
            self._dist_warn_lbl.setVisible(False); return
        try:
            il = float(self.il_input.text())
        except ValueError:
            self._dist_warn_lbl.setVisible(False); return

        norm = self.config.get('normalization', {})
        # nested dict ?щ㎎ 泥섎━
        if 's21_db' in norm:
            p = norm['s21_db']
            s21_max = p.get('max', p.get('s_max', 0.0))
        else:
            s21_max = norm.get('s21_max', norm.get('s_max', None))
        if s21_max is None:
            self._dist_warn_lbl.setVisible(False); return

        warnings = []
        if il > s21_max:
            warnings.append(
                f"S21 목표 {il:.1f} dB > 데이터 최대 {s21_max:.1f} dB\n"
                f"사양이 너무 높습니다. {s21_max:.1f} dB 이하로 낮춰 주세요.")
        if warnings:
            self._dist_warn_lbl.setText("⚠ " + "\n⚠ ".join(warnings))
            self._dist_warn_lbl.setVisible(True)
        else:
            self._dist_warn_lbl.setVisible(False)

    # Pass / Fail 표시 업데이트
    def _on_layout_update(self, algo, hard_layout, soft_layout):
        # 레이아웃 현재 상태 시각화
        try:
            fig = self.layout_fig; fig.clear()
            ALGO_COLOR_L = {'GD': '#2ecc71', 'GA': '#3498db',
                            'BPSO': '#e74c3c', 'BO': '#9b59b6', 'DE': '#f39c12',
                            'GA_GD': '#1a5276', 'BPSO_GD': '#922b21',
                            'GA_DBS': '#16a085', 'BPSO_DBS': '#d35400'}
            color = ALGO_COLOR_L.get(algo, '#888888')

            if soft_layout is not None:
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)
                im1 = ax1.imshow(soft_layout, cmap='RdYlGn', vmin=0, vmax=1,
                                 interpolation='nearest', aspect='equal')
                ax1.set_title(f"{algo} soft\n(prob)", fontsize=8,
                              color=color, fontweight='bold')
                ax1.set_xticks([]); ax1.set_yticks([])
                fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                ax2.imshow(hard_layout, cmap='binary', interpolation='nearest',
                           aspect='equal')
                ax2.set_title(f"{algo} hard\n(binary)", fontsize=8,
                              color=color, fontweight='bold')
                ax2.set_xticks([]); ax2.set_yticks([])
                ax1.contour(soft_layout, levels=[0.5],
                            colors=[color], linewidths=[0.8], alpha=0.7)
            else:
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(hard_layout, cmap='binary', interpolation='nearest',
                          aspect='equal')
                fill = float(np.mean(hard_layout)) * 100
                ax.set_title(f"{algo}  fill={fill:.1f}%", fontsize=9,
                             color=color, fontweight='bold')
                ax.set_xticks([]); ax.set_yticks([])
                for v in range(0, GRID_SIZE + 1, 5):
                    ax.axhline(v - 0.5, color='#aaaaaa', lw=0.3, alpha=0.5)
                    ax.axvline(v - 0.5, color='#aaaaaa', lw=0.3, alpha=0.5)

            fig.tight_layout(pad=0.5)
            self.layout_canvas.draw_idle()
        except Exception:
            pass

    # 실시간 수렴 그래프
    def _on_conv_update(self, algo, step, error):
        if algo not in self._conv_data: self._conv_data[algo] = []
        self._conv_data[algo].append((step, error))
        self._redraw_conv()

    def _redraw_conv(self):
        ax = self._conv_ax
        if ax is None: return
        ax.clear()
        ax.set_xlabel('Step', fontsize=7); ax.set_ylabel('Error', fontsize=7)
        ax.tick_params(labelsize=6); ax.grid(True, alpha=0.3, lw=0.5)
        colors = {'GD': '#2ecc71', 'GA': '#3498db',
                  'BPSO': '#e74c3c', 'BO': '#9b59b6', 'DE': '#f39c12',
                  'GA_GD': '#1a5276', 'BPSO_GD': '#922b21',
                  'GA_DBS': '#16a085', 'BPSO_DBS': '#d35400'}
        has_data = False
        for algo, pts in self._conv_data.items():
            if not pts: continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=colors.get(algo, '#888'),
                    lw=1.5, label=algo, alpha=0.9)
            min_i = int(np.argmin(ys))
            ax.plot(xs[min_i], ys[min_i], 'o',
                    color=colors.get(algo, '#888'), markersize=5, zorder=5)
            has_data = True
        if has_data:
            ax.legend(fontsize=6, loc='upper right')
            ax.set_title('Optimization progress', fontsize=7)
        else:
            ax.set_title('Waiting...', fontsize=7)
        try:
            self._conv_fig.tight_layout(pad=0.8)
        except Exception:
            pass
        self._conv_canvas.draw_idle()

    def denorm_clamp(self, pred_norm):
        """3채널/8채널 모델 모두 지원하는 key 기반 역정규화."""
        norm = self.config['normalization']
        target_keys = get_target_keys(self.config)
        n = len(pred_norm)
        nf = n // max(len(target_keys), 1)

        def _inv_ch(arr, p, clip_db=True):
            t = p.get('type', 'minmax')
            if t == 'standard_direct':
                return arr * p['std'] + p['mean']
            if t == 'standard':
                clip = p.get('clip', 3.0)
                z = np.clip(arr, 0.0, 1.0) * (2.0 * clip) - clip
                raw = z * p['std'] + p['mean']
                return np.clip(raw, p.get('raw_min', -200), 0) if clip_db else raw
            lo = p.get('min', p.get('s_min', -140.0))
            hi = p.get('max', p.get('s_max', 0.0))
            raw = np.clip(arr, 0, 1) * (hi - lo) + lo
            return np.clip(raw, lo, 0) if clip_db else raw

        if any(key in norm for key in target_keys):
            restored = []
            for key in target_keys:
                sl = channel_slice(target_keys, key, nf)
                restored.append(_inv_ch(pred_norm[sl], norm.get(key, {}), clip_db=key.endswith('_db')))
            return np.concatenate(restored)

        if 's11_min' in norm:
            restored = []
            for key in target_keys:
                sl = channel_slice(target_keys, key, nf)
                if key.endswith('_db'):
                    lo = norm[key.replace('_db', '_min')]
                    hi = norm[key.replace('_db', '_max')]
                else:
                    lo = norm[key.replace('_deg', '_min')]
                    hi = norm[key.replace('_deg', '_max')]
                raw = np.clip(pred_norm[sl], 0, 1) * (hi - lo) + lo
                restored.append(np.clip(raw, lo, 0) if key.endswith('_db') else raw)
            return np.concatenate(restored)

        s_min = norm.get('s_min', -140.0)
        s_max = norm.get('s_max', 0.0)
        return np.clip(np.clip(pred_norm, 0, 1) * (s_max - s_min) + s_min, s_min, 0)

    def _renormalize_complex_channels(self, s_complex, z01_new, z02_new, z0_old=50.0):
        """scikit-rf power-wave renormalization으로 2-port complex S를 재정규화."""
        import skrf as rf

        n_f = len(next(iter(s_complex.values())))
        s12_src = s_complex.get('s12', s_complex.get('s21'))
        s_mat = np.zeros((n_f, 2, 2), dtype=complex)
        s_mat[:, 0, 0] = s_complex['s11']
        s_mat[:, 0, 1] = s12_src
        s_mat[:, 1, 0] = s_complex['s21']
        s_mat[:, 1, 1] = s_complex['s22']
        z0_old_arr = np.tile(np.array([complex(z0_old), complex(z0_old)], dtype=complex), (n_f, 1))
        z0_new_arr = np.tile(np.array([complex(z01_new), complex(z02_new)], dtype=complex), (n_f, 1))
        s_new = rf.network.renormalize_s(s_mat, z0_old_arr, z0_new_arr, s_def='power')
        s11_new = s_new[:, 0, 0]
        s12_new = s_new[:, 0, 1]
        s21_new = s_new[:, 1, 0]
        s22_new = s_new[:, 1, 1]

        to_db = lambda arr: 20.0 * np.log10(np.maximum(np.abs(arr), 1e-15))
        return {
            's11_db': np.clip(to_db(s11_new), a_min=None, a_max=0.0),
            's12_db': np.clip(to_db(s12_new), a_min=None, a_max=0.0),
            's21_db': np.clip(to_db(s21_new), a_min=None, a_max=0.0),
            's22_db': np.clip(to_db(s22_new), a_min=None, a_max=0.0),
            's11_complex': s11_new,
            's12_complex': s12_new,
            's21_complex': s21_new,
            's22_complex': s22_new,
        }

    def _extract_plot_channels(self, pred_raw, z01_new, z02_new):
        """dB-only/8채널 모델을 공통 경로에서 해석."""
        target_keys = get_target_keys(self.config)
        n_f = len(self.frequencies)
        s_complex = complex_sparams_from_db_deg(pred_raw, target_keys, n_f)
        if {'s11', 's21', 's22'}.issubset(s_complex.keys()):
            return self._renormalize_complex_channels(s_complex, z01_new, z02_new)

        ch = split_prediction_channels(pred_raw, target_keys, n_f)
        real_imag_complex = {}
        for port in ('s11', 's12', 's21', 's22'):
            rk = f'{port}_real'
            ik = f'{port}_imag'
            if rk in ch and ik in ch:
                real_imag_complex[port] = ch[rk] + 1j * ch[ik]
        if {'s11', 's21', 's22'}.issubset(real_imag_complex.keys()):
            return self._renormalize_complex_channels(real_imag_complex, z01_new, z02_new)

        r_s11, r_s21, r_s22 = numpy_renormalize_s(
            ch['s11_db'], ch['s21_db'], ch['s22_db'],
            z0_old=50.0, z01_new=z01_new, z02_new=z02_new)
        return {
            's11_db': np.clip(r_s11, a_min=None, a_max=0.0),
            's12_db': np.clip(r_s21, a_min=None, a_max=0.0),
            's21_db': np.clip(r_s21, a_min=None, a_max=0.0),
            's22_db': np.clip(r_s22, a_min=None, a_max=0.0),
            's11_complex': _db_to_complex_np(np.clip(r_s11, a_min=None, a_max=0.0)),
            's12_complex': _db_to_complex_np(np.clip(r_s21, a_min=None, a_max=0.0)),
            's21_complex': _db_to_complex_np(np.clip(r_s21, a_min=None, a_max=0.0)),
            's22_complex': _db_to_complex_np(np.clip(r_s22, a_min=None, a_max=0.0)),
        }

    def _build_export_freqs(self):
        """추출 저장 전용 91포인트 주파수 축."""
        freq_tenths = np.concatenate([
            np.array([1], dtype=np.int32),
            np.arange(5, 20, 5, dtype=np.int32),
            np.arange(20, 121, 2, dtype=np.int32),
            np.arange(125, 301, 5, dtype=np.int32),
        ])
        return freq_tenths.astype(np.float64) / 10.0

    def _resample_complex_trace(self, src_freqs, trace_complex, dst_freqs):
        """복소 S를 실수/허수 보간으로 새 주파수 축에 맞춤."""
        real_interp = np.interp(dst_freqs, src_freqs, trace_complex.real)
        imag_interp = np.interp(dst_freqs, src_freqs, trace_complex.imag)
        return real_interp + 1j * imag_interp

    @staticmethod
    def _format_touchstone_freq(freq_ghz):
        return f"{round(float(freq_ghz), 5):.10g}"

    @staticmethod
    def _is_nearly_real(value, tol=1e-9):
        return abs(complex(value).imag) <= tol

    def _write_touchstone_s2p(self, out_path, freqs_ghz, s11, s12, s21, s22, p1_z0, p2_z0):
        """변경: 현재 포트 기준으로 renorm된 2-port S-parameter를 Touchstone 형식으로 저장."""
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("! Exported from CorRaL inverse GUI\n")  # 변경: 파일 출처 기록
            f.write(f"! Port1 Z0 = {p1_z0}\n")  # 변경: 실제 포트1 기준 임피던스 주석 저장
            f.write(f"! Port2 Z0 = {p2_z0}\n")  # 변경: 실제 포트2 기준 임피던스 주석 저장
            same_real_z0 = (
                self._is_nearly_real(p1_z0)
                and self._is_nearly_real(p2_z0)
                and abs(complex(p1_z0).real - complex(p2_z0).real) <= 1e-9
            )
            if same_real_z0:
                f.write(f"# GHZ S RI R {complex(p1_z0).real:.12g}\n")
            else:
                f.write("! WARNING: Unequal or complex per-port Z0 cannot be represented exactly by the Touchstone R field.\n")
                f.write("! Data below are already renormalized to the Port1/Port2 Z0 values written above.\n")
                f.write("# GHZ S RI R 50\n")
            for i, freq in enumerate(freqs_ghz):
                f.write(
                    f"{self._format_touchstone_freq(freq)} "
                    f"{s11[i].real:.12e} {s11[i].imag:.12e} "
                    f"{s21[i].real:.12e} {s21[i].imag:.12e} "
                    f"{s12[i].real:.12e} {s12[i].imag:.12e} "
                    f"{s22[i].real:.12e} {s22[i].imag:.12e}\n"
                )  # 변경: 2-port Touchstone 한 줄에 S11 S21 S12 S22 순서로 저장
    def _export_best_result(self, best_algo):
        """최적 역설계 결과를 s2p + npz 로 저장 (저장 경로 직접 선택)."""
        if not best_algo or best_algo not in self.all_results:
            return
        result = self.all_results.get(best_algo, {})
        layout = result.get('layout')
        if layout is None:
            return

        # ── 저장 폴더 선택 (사용자 직접 탐색) ──────────────────────────────
        out_dir_str = QFileDialog.getExistingDirectory(
            self, "s2p / npz 저장 폴더 선택", str(Path.home()))
        if not out_dir_str:
            self.log("저장 취소됨.")
            return
        out_dir = Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            layout_arr = (np.asarray(layout) >= 0.5).astype(np.uint8)
            lt = torch.FloatTensor(layout_arr).unsqueeze(0).unsqueeze(0).to(self.device)
            pred_norm = ensemble_predict(self.forward_models, lt, 512).cpu().numpy()[0]
            pred_raw  = self.denorm_clamp(pred_norm)
            p1_z0, p2_z0 = 50.0 + 0j, 50.0 + 0j  # 변경: export 저장은 항상 50Ω / 50Ω 기준으로 고정
            plot_ch = self._extract_plot_channels(pred_raw, p1_z0, p2_z0)
            s11_c = plot_ch['s11_complex']
            s12_c = plot_ch['s12_complex']
            s21_c = plot_ch['s21_complex']
            s22_c = plot_ch['s22_complex']
            src_freqs = np.asarray(
                self.frequencies if self.frequencies is not None else _FALLBACK_FREQS,
                dtype=np.float64,
            )

            # ── 91pt 정밀 주파수 축으로 리샘플 ──────────────────────────────
            export_freqs = self._build_export_freqs()
            s11_exp = self._resample_complex_trace(src_freqs, s11_c, export_freqs)
            s12_exp = self._resample_complex_trace(src_freqs, s12_c, export_freqs)
            s21_exp = self._resample_complex_trace(src_freqs, s21_c, export_freqs)
            s22_exp = self._resample_complex_trace(src_freqs, s22_c, export_freqs)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem     = f"best_{best_algo.lower()}_{timestamp}"
            npz_path = out_dir / f"{stem}.npz"
            s2p_path = out_dir / f"{stem}.s2p"

            db_from_c  = lambda a: 20.0 * np.log10(np.maximum(np.abs(a), 1e-15))
            deg_from_c = lambda a: np.degrees(np.angle(a))
            np.savez(
                npz_path,
                layout=layout_arr,
                frequencies_ghz=export_freqs.astype(np.float32),
                s11_db=db_from_c(s11_exp).astype(np.float32),
                s12_db=db_from_c(s12_exp).astype(np.float32),
                s21_db=db_from_c(s21_exp).astype(np.float32),
                s22_db=db_from_c(s22_exp).astype(np.float32),
                s11_deg=deg_from_c(s11_exp).astype(np.float32),
                s12_deg=deg_from_c(s12_exp).astype(np.float32),
                s21_deg=deg_from_c(s21_exp).astype(np.float32),
                s22_deg=deg_from_c(s22_exp).astype(np.float32),
                s11_real=s11_exp.real.astype(np.float32),
                s11_imag=s11_exp.imag.astype(np.float32),
                s12_real=s12_exp.real.astype(np.float32),
                s12_imag=s12_exp.imag.astype(np.float32),
                s21_real=s21_exp.real.astype(np.float32),
                s21_imag=s21_exp.imag.astype(np.float32),
                s22_real=s22_exp.real.astype(np.float32),
                s22_imag=s22_exp.imag.astype(np.float32),
                algorithm=np.array(best_algo),
                p1_z0=np.array(str(p1_z0)),
                p2_z0=np.array(str(p2_z0)),
            )
            self._write_touchstone_s2p(
                s2p_path, export_freqs,
                s11_exp, s12_exp, s21_exp, s22_exp,
                p1_z0, p2_z0)
            self.log(f"저장 완료: {s2p_path.name}, {npz_path.name}  →  {out_dir}")
            return best_algo, s2p_path, npz_path
        except Exception as e:
            self.log(f"저장 실패: {e}")
            QMessageBox.critical(self, "Export", f"저장 중 오류가 발생했습니다.\n{e}")
            return None
    def _export_current_best_result(self):
        """현재 best 결과를 수동으로 추출."""
        if not self.all_results:
            QMessageBox.information(self, "Export", "먼저 역설계를 실행해 결과를 만든 뒤 추출해 주세요.")
            return
        valid = [name for name, res in self.all_results.items() if res.get('layout') is not None]
        if not valid:
            QMessageBox.information(self, "Export", "추출할 수 있는 레이아웃 결과가 없습니다.")
            return
        best_algo = min(valid, key=lambda name: self.all_results[name].get('error', float('inf')))
        exported = self._export_best_result(best_algo)
        if exported is None:
            return
        _, s2p_path, npz_path = exported
        QMessageBox.information(
            self,
            "Export",
            f"{best_algo} 결과를 저장했습니다.\nS2P: {s2p_path}\nNPZ: {npz_path}",
        )

    def run_inverse(self):
        """target_keys 기준으로 dB-only/8채널 모델 모두에 대한 target 생성."""
        if not self.forward_models:
            QMessageBox.warning(self, "Error", "Forward 모델이 없습니다"); return
        params = self.validate()
        if not params:
            return

        t11, t21, t22 = self.gen_targets(
            self.frequencies, params['f_low'], params['f_high'],
            params['il'], params['rl'], params.get('sl', -6.0))
        norm = self.config['normalization']
        norm_tp = self.config.get('normalization_type', 'minmax')
        target_keys = get_target_keys(self.config)
        target_map = {key: np.zeros_like(self.frequencies, dtype=np.float32) for key in target_keys}
        if any(key.endswith('_real') or key.endswith('_imag') for key in target_keys):
            def _db_to_mag(db):
                return np.power(10.0, np.asarray(db, dtype=np.float32) / 20.0).astype(np.float32)
            for port, db_vals in (('s11', t11), ('s21', t21), ('s12', t21), ('s22', t22)):
                rk = f'{port}_real'
                ik = f'{port}_imag'
                if rk in target_map:
                    target_map[rk] = _db_to_mag(db_vals)
                if ik in target_map:
                    target_map[ik] = np.zeros_like(self.frequencies, dtype=np.float32)
        if 's11_db' in target_map:
            target_map['s11_db'] = t11.astype(np.float32)
        if 's12_db' in target_map:
            target_map['s12_db'] = t21.astype(np.float32)
        if 's21_db' in target_map:
            target_map['s21_db'] = t21.astype(np.float32)
        if 's22_db' in target_map:
            target_map['s22_db'] = t22.astype(np.float32)
        targets = np.concatenate([target_map[key] for key in target_keys])

        def _norm_ch(arr, key):
            p = norm.get(key, {})
            t = p.get('type', norm_tp if 's11_db' in norm else 'minmax')
            if t == 'standard_direct':
                return (arr - p['mean']) / (p['std'] + 1e-8)
            if t == 'standard':
                clip = p.get('clip', 3.0)
                z = (arr - p['mean']) / (p['std'] + 1e-8)
                z = np.clip(z, -clip, clip)
                return (z + clip) / (2.0 * clip)
            if 's11_db' in norm:
                lo, hi = p['min'], p['max']
                return (arr - lo) / (hi - lo + 1e-8)
            if 's11_min' in norm:
                if key.endswith('_db'):
                    lo = norm[key.replace('_db', '_min')]
                    hi = norm[key.replace('_db', '_max')]
                else:
                    lo = norm[key.replace('_deg', '_min')]
                    hi = norm[key.replace('_deg', '_max')]
                return (arr - lo) / (hi - lo + 1e-8)
            s_min = norm.get('s_min', -140.0)
            s_max = norm.get('s_max', 0.0)
            return (arr - s_min) / (s_max - s_min + 1e-8)

        targets_norm = np.concatenate([_norm_ch(target_map[key], key) for key in target_keys])

        self.config['f_low'] = params['f_low']
        self.config['f_high'] = params['f_high']
        self.config['targets_params'] = targets
        self.config['split_line_enabled'] = params.get('split_line_enabled', False)  # 변경: 코어가 현재 분리선 사용 여부를 읽을 수 있게 설정 저장
        self.config['split_line_mode'] = params.get('split_line_mode', 'none')  # 변경: 코어가 분리선 방향을 읽을 수 있게 설정 저장
        self.config['split_line_index'] = params.get('split_line_index', GRID_SIZE // 2)  # 변경: 코어가 분리선 위치를 읽을 수 있게 설정 저장
        self.config['split_line_width'] = params.get('split_line_width', 1)  # 변경: 코어가 분리선 폭을 읽을 수 있게 설정 저장
        self.config['split_line_pattern'] = params.get('split_line_pattern', 'straight')  # 변경: 코어가 직선/지그재그 분리선 패턴을 읽을 수 있게 설정 저장
        self.config['split_line_seed'] = params.get('split_line_seed', 0)  # 변경: 코어가 이번 실행의 랜덤 분리선 경로를 재현할 수 있게 seed 저장
        pb_mask = ((self.frequencies >= params['f_low']) & (self.frequencies <= params['f_high']))
        tl = params['f_low'] * 0.8
        tr = params['f_high'] * 1.2
        sb_mask = (self.frequencies <= tl) | (self.frequencies >= tr)

        self._pb_mask = pb_mask
        self._sb_mask = sb_mask
        if self.worker:
            self.worker.stop(); self.worker.wait(); self.worker = None
        algo_text = self.algo_combo.currentText()
        self.all_results = {}
        self._conv_data = {}
        self.targets_norm = targets_norm
        self.pending_algos = self._parse_algo(algo_text)
        self.log(f"알고리즘: {algo_text}  Pop={params['pop_size']}  Ensemble={len(self.forward_models)}")
        if params.get('split_line_enabled', False):
            self.log(f"분리선 적용: Band-pass preset ({params['split_line_mode']}, random index={params['split_line_index']}, width={params['split_line_width']}, pattern={params.get('split_line_pattern', 'straight')})")  # 변경: 지그재그/대각선 분리선 패턴까지 로그에 표시
        else:
            self.log("분리선 미사용: Low-pass preset")  # 변경: LPF 선택 시에도 현재 구조 프리셋 상태를 로그에 남김
        self._pop_size = params['pop_size']
        self._chunk_size = params['chunk_size']
        self._p1_z0 = params.get('p1_z0', 50.0+0j)
        self._p2_z0 = params.get('p2_z0', 50.0+0j)
        self._run_next()
        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def _on_all_done(self):
        """8채널 모델에서 채널명을 기준으로 후처리 검증."""
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.progress.setValue(100)
        if not self.all_results:
            return
        best = min(self.all_results, key=lambda x: self.all_results[x]['error'])
        self.log("=" * 50 + "\nRESULTS")
        for a in ['GD', 'GA', 'BPSO', 'BO', 'DE', 'GA_GD', 'BPSO_GD', 'GA_DBS', 'BPSO_DBS']:
            if a in self.all_results:
                self.log(f"  {a:7s}: {self.all_results[a]['error']:8.4f}{' <- BEST' if a == best else ''}")
        self.log("=" * 50)
        self._viz_compare()
        self._update_pass_fail(best)

        print("\n========== POST-OPTIMIZATION VERIFICATION ==========")
        n_f = len(self.frequencies)
        target_keys = get_target_keys(self.config)
        fl, fh = self.config['f_low'], self.config['f_high']
        pb_mask = (self.frequencies >= fl) & (self.frequencies <= fh)
        for name in ['GD', 'GA', 'BPSO', 'BO', 'DE', 'GA_GD', 'BPSO_GD', 'GA_DBS', 'BPSO_DBS']:
            if name in self.all_results and self.all_results[name]['layout'] is not None:
                try:
                    lt = torch.FloatTensor(self.all_results[name]['layout']).unsqueeze(0).unsqueeze(0).to(self.device)
                    pn = ensemble_predict(self.forward_models, lt, 512).cpu().numpy()[0]
                    pred = self.denorm_clamp(pn)
                    ch = split_prediction_channels(pred, target_keys, n_f)
                    s11 = ch['s11_db']
                    s21 = ch['s21_db']
                    print(f"{name}: S21_pass={np.mean(s21[pb_mask]) > -10}, S11_pass={np.mean(s11[pb_mask]) < -10}, Error={self.all_results[name]['error']:.4f}")
                except Exception as e:
                    print(f"{name}: 검증 실패 ({e})")
        print("=" * 50)

    def _viz_compare(self):
        """target_keys 기준 채널 분리와 phase 기반 renorm을 사용한 비교 그래프."""
        # 변경: Start 누르기 전에는 config['targets_params']가 아직 없음 → 안전하게 return
        if not getattr(self, 'config', None) or 'targets_params' not in self.config:
            return
        if getattr(self, 'frequencies', None) is None:
            return
        ALL_ALGOS = ['GD', 'GA', 'BPSO', 'BO', 'DE', 'GA_GD', 'BPSO_GD', 'GA_DBS', 'BPSO_DBS']
        ALGO_COLOR = {'GD': '#2ecc71', 'GA': '#3498db',
                      'BPSO': '#e74c3c', 'BO': '#9b59b6',
                      'DE': '#f39c12', 'GA_GD': '#1a5276',
                      'BPSO_GD': '#922b21', 'GA_DBS': '#16a085',
                      'BPSO_DBS': '#d35400'}

        self.layout_fig.clear()
        algos_all = [a for a in ALL_ALGOS if a in self.all_results]
        for i, a in enumerate(algos_all):
            if self.all_results[a]['layout'] is not None:
                ax = self.layout_fig.add_subplot(1, len(algos_all), i + 1)
                ax.imshow(self.all_results[a]['layout'], cmap='binary', interpolation='nearest')
                ax.set_title(f"{a}\n{self.all_results[a]['error']:.4f}", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        self.layout_fig.tight_layout(); self.layout_canvas.draw()

        self.sp_fig.clear()
        import matplotlib.gridspec as gridspec
        gs_outer = gridspec.GridSpec(3, 1, height_ratios=[2, 0.5, 2], hspace=0.0,
                                     left=0.12, right=0.97, top=0.97, bottom=0.05,
                                     figure=self.sp_fig)
        gs_top = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[0], hspace=0.6)
        gs_bot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[2], hspace=0.6)

        n_f = len(self.frequencies)
        target_keys = get_target_keys(self.config)
        target_ch = split_prediction_channels(self.config['targets_params'], target_keys, n_f)
        fl, fh = self.config['f_low'], self.config['f_high']
        fmin, fmax = self.frequencies[0], self.frequencies[-1]

        ax_s11 = self.sp_fig.add_subplot(gs_top[0])
        ax_s22 = self.sp_fig.add_subplot(gs_top[1])
        ax_s21 = self.sp_fig.add_subplot(gs_bot[0])
        ax_s12 = self.sp_fig.add_subplot(gs_bot[1])

        self.smith_fig.clear()
        ax_smith = self.smith_fig.add_subplot(111, aspect='equal')
        ax_smith.set_xlim(-1.30, 1.30); ax_smith.set_ylim(-1.30, 1.30)
        ax_smith.axis('off')
        self._ax_smith = ax_smith

        pb_mask = (self.frequencies >= fl) & (self.frequencies <= fh)
        try:
            target_il = float(self.il_input.text())
        except ValueError:
            target_il = -1.5
        try:
            target_rl = float(self.rl_input.text())
        except ValueError:
            target_rl = -15.0
        try:
            target_sl = float(self.sl_input.text())
        except (ValueError, AttributeError):
            target_sl = -6.0
        # 변경: S21/S12 stopband 타겟선 = IL - 3 + SL (3 dB 전이 마진 고정)
        stopband_level = target_il - 3.0 + target_sl
        # 변경: L-PB / R-PB는 전이대역 폭 결정 (램프 구간)
        try:
            l_pb_val = float(self.l_pb_freq.text())
        except (ValueError, AttributeError):
            l_pb_val = fl
        try:
            r_pb_val = float(self.r_pb_freq.text())
        except (ValueError, AttributeError):
            r_pb_val = fh
        l_pb_val = min(l_pb_val, fl)
        r_pb_val = max(r_pb_val, fh)

        # 변경: 타겟선 단순화
        #   S11/S22: 통과대역 외부에만 0 dB 점선 + 통과대역 안에 RL 평탄선 1개
        #   S21/S12: 통과대역 안에만 IL 평탄선 + IL-3 dB 평탄선 2개 (트라페조이드/stopband 제거)
        freqs_arr = np.asarray(self.frequencies, dtype=np.float32)
        # 통과대역 freq 와 외부 freq
        freqs_pb = freqs_arr[pb_mask]
        out_left_mask = freqs_arr < fl
        out_right_mask = freqs_arr > fh
        rl_pb = np.full(freqs_pb.shape, target_rl, dtype=np.float32)
        zero_left = np.zeros(int(out_left_mask.sum()), dtype=np.float32)
        zero_right = np.zeros(int(out_right_mask.sum()), dtype=np.float32)
        il_pb = np.full(freqs_pb.shape, target_il, dtype=np.float32)
        il_minus3_pb = np.full(freqs_pb.shape, target_il - 3.0, dtype=np.float32)

        self._target_lines = []
        # S11: 0 dB 점선 (FL 이전 + FH 이후) + FL~FH 평탄 RL
        ln1a, = ax_s11.plot(freqs_arr[out_left_mask], zero_left, 'k--', lw=2, zorder=5, label='Target')
        ln1b, = ax_s11.plot(freqs_arr[out_right_mask], zero_right, 'k--', lw=2, zorder=5)
        ln1c, = ax_s11.plot(freqs_pb, rl_pb, 'k--', lw=2, zorder=5)
        
        # S22: 동일 구조
        ln4a, = ax_s22.plot(freqs_arr[out_left_mask], zero_left, 'k--', lw=2, zorder=5, label='Target')
        ln4b, = ax_s22.plot(freqs_arr[out_right_mask], zero_right, 'k--', lw=2, zorder=5)
        ln4c, = ax_s22.plot(freqs_pb, rl_pb, 'k--', lw=2, zorder=5)
        
        # S21/S12: 통과대역 안의 IL 평탄선 + IL-3 dB 평탄선
        ln2a, = ax_s21.plot(freqs_pb, il_pb, 'k--', lw=2, zorder=5, label='Target')
        ln2b, = ax_s21.plot(freqs_pb, il_minus3_pb, 'k--', lw=2, zorder=5)
        ln3a, = ax_s12.plot(freqs_pb, il_pb, 'k--', lw=2, zorder=5, label='Target')
        ln3b, = ax_s12.plot(freqs_pb, il_minus3_pb, 'k--', lw=2, zorder=5)
        
        # 추가: S21/S12 저지대역(Stopband) 타겟선 (l_pb_val 이전 및 r_pb_val 이후)
        sb_left_mask = freqs_arr <= l_pb_val
        sb_right_mask = freqs_arr >= r_pb_val
        freqs_sb_l = freqs_arr[sb_left_mask]
        freqs_sb_r = freqs_arr[sb_right_mask]
        val_sb_l = np.full(freqs_sb_l.shape, stopband_level, dtype=np.float32)
        val_sb_r = np.full(freqs_sb_r.shape, stopband_level, dtype=np.float32)
        
        ln2c, = ax_s21.plot(freqs_sb_l, val_sb_l, 'k--', lw=2, zorder=5)
        ln2d, = ax_s21.plot(freqs_sb_r, val_sb_r, 'k--', lw=2, zorder=5)
        ln3c, = ax_s12.plot(freqs_sb_l, val_sb_l, 'k--', lw=2, zorder=5)
        ln3d, = ax_s12.plot(freqs_sb_r, val_sb_r, 'k--', lw=2, zorder=5)
        
        self._target_lines.extend([ln1a, ln1b, ln1c, ln2a, ln2b, ln2c, ln2d, ln3a, ln3b, ln3c, ln3d, ln4a, ln4b, ln4c])
        # 변경: 토글 상태가 Hide이면 바로 숨김 처리
        if hasattr(self, 'show_target_btn') and not self.show_target_btn.isChecked():
            for ln in self._target_lines:
                ln.set_visible(False)

        algos_sp = [a for a in algos_all if not (a in ALL_ALGOS and not self._sp_algo_visible(a))]
        best_algo = None
        best_err = float('inf')
        for a in algos_sp:
            if self.all_results[a]['layout'] is None:
                continue
            if self.all_results[a]['error'] < best_err:
                best_err = self.all_results[a]['error']; best_algo = a
            try:
                lt = torch.FloatTensor(self.all_results[a]['layout']).unsqueeze(0).unsqueeze(0).to(self.device)
                pn = ensemble_predict(self.forward_models, lt, 512).cpu().numpy()[0]
                pred_raw = self.denorm_clamp(pn)
                try:
                    _p1 = parse_z0_input(self.p1_z0_edit.text())
                    _p2 = parse_z0_input(self.p2_z0_edit.text())
                except Exception:
                    _p1, _p2 = 50.0 + 0j, 50.0 + 0j
                plot_ch = self._extract_plot_channels(pred_raw, _p1, _p2)
                c = ALGO_COLOR.get(a, '#888888')
                ax_s11.plot(self.frequencies, plot_ch['s11_db'], color=c, label=a, lw=1.3, alpha=0.85)
                ax_s21.plot(self.frequencies, plot_ch['s21_db'], color=c, label=a, lw=1.3, alpha=0.85)
                ax_s12.plot(self.frequencies, plot_ch['s12_db'], color=c, label=a, lw=1.3, alpha=0.85)
                ax_s22.plot(self.frequencies, plot_ch['s22_db'], color=c, label=a, lw=1.3, alpha=0.85)
            except Exception as e:
                self.log(f"Viz error {a}: {e}")

        try:
            p_fl = float(self.plot_fl_input.text())
            p_fh = float(self.plot_fh_input.text())
            p_pts = int(self.plot_pts_input.text())
        except ValueError:
            p_fl, p_fh, p_pts = fmin, fmax, 11

        # 변경: S11/S22 와 S21/S12 별 Y축 박스 입력값 (없거나 잘못되면 기본값 fallback)
        try:
            refl_y_lo = float(self.plot_y_lo_refl.text())
            refl_y_hi = float(self.plot_y_hi_refl.text())
            refl_y_ticks = int(self.plot_y_ticks_refl.text())
        except (ValueError, AttributeError):
            refl_y_lo, refl_y_hi, refl_y_ticks = -30.0, 0.5, 7
        try:
            trans_y_lo = float(self.plot_y_lo_trans.text())
            trans_y_hi = float(self.plot_y_hi_trans.text())
            trans_y_ticks = int(self.plot_y_ticks_trans.text())
        except (ValueError, AttributeError):
            trans_y_lo = min(float(stopband_level) - 2.0, -10.0)
            trans_y_lo = float(int(np.floor(trans_y_lo / 2.0)) * 2)
            trans_y_hi = 1.0
            trans_y_ticks = 8

        for ax, title, yl in [(ax_s11, 'S11 (Return Loss)', 'S11 (dB)'),
                              (ax_s22, 'S22 (Return Loss)', 'S22 (dB)'),
                              (ax_s21, 'S21 (Insertion Loss)', 'S21 (dB)'),
                              (ax_s12, 'S12 (Insertion Loss)', 'S12 (dB)')]:
            # 변경: S11/S22는 refl 모드, S21/S12는 trans 모드(사용자 L-PB/R-PB 반영)
            _mode = 'refl' if ('S11' in title or 'S22' in title) else 'trans'
            self._add_band_shading(ax, fl, fh, fmin, fmax, mode=_mode)
            ax.set_xlabel('Freq (GHz)', fontsize=8); ax.set_ylabel(yl, fontsize=8)
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.legend(fontsize=6, loc='lower left')
            ax.grid(True, alpha=0.3); ax.set_xlim(p_fl, p_fh)
            if p_pts > 1:
                ax.set_xticks(np.linspace(p_fl, p_fh, p_pts))
            if _mode == 'refl':
                # 변경: 사용자가 입력한 S11/S22 Y축 박스 적용
                ax.set_ylim(refl_y_lo, refl_y_hi)
                if refl_y_ticks > 1:
                    ax.set_yticks(np.linspace(refl_y_lo, refl_y_hi, refl_y_ticks))
            else:
                # 변경: 사용자가 입력한 S21/S12 Y축 박스 적용
                ax.set_ylim(trans_y_lo, trans_y_hi)
                if trans_y_ticks > 1:
                    ax.set_yticks(np.linspace(trans_y_lo, trans_y_hi, trans_y_ticks))

        if best_algo and self.all_results[best_algo]['layout'] is not None:
            try:
                lt = torch.FloatTensor(self.all_results[best_algo]['layout']).unsqueeze(0).unsqueeze(0).to(self.device)
                pn = ensemble_predict(self.forward_models, lt, 512).cpu().numpy()[0]
                pred_raw = self.denorm_clamp(pn)
                try:
                    p1_z0 = parse_z0_input(self.p1_z0_edit.text())
                    p2_z0 = parse_z0_input(self.p2_z0_edit.text())
                except Exception:
                    p1_z0, p2_z0 = 50.0 + 0j, 50.0 + 0j
                plot_ch = self._extract_plot_channels(pred_raw, p1_z0, p2_z0)
                gamma_s11 = plot_ch['s11_complex']
                gamma_s22 = plot_ch['s22_complex']
                z0_ref_s11 = p1_z0.real if p1_z0.real > 0 else 50.0
                z0_ref_s22 = p2_z0.real if p2_z0.real > 0 else 50.0
                Z_s11 = z0_ref_s11 * (1 + gamma_s11) / (1 - gamma_s11 + 1e-9)
                Z_s22 = z0_ref_s22 * (1 + gamma_s22) / (1 - gamma_s22 + 1e-9)
                self._smith_freqs = self.frequencies.copy()
                self._smith_Z = Z_s11
                self._smith_Z_s11 = Z_s11
                self._smith_Z_s22 = Z_s22
                self._smith_gamma_all = gamma_s11
                self._smith_gamma_s11 = gamma_s11
                self._smith_gamma_s22 = gamma_s22
                self._smith_z0_ref = z0_ref_s11
                self._smith_z0_ref_s11 = complex(p1_z0)
                self._smith_z0_ref_s22 = complex(p2_z0)
                self._smith_pb_mask = pb_mask.copy()
                self._smith_best_algo = best_algo
                self._smith_has_smith = True
                self._smith_color = ALGO_COLOR.get(best_algo, '#3498db')
                self._smith_s11_db = plot_ch['s11_db']
                self._smith_s21_db = plot_ch['s21_db']
                self._smith_s22_db = plot_ch['s22_db']
                self._smith_mag = np.abs(gamma_s11)
                self._smith_phase = np.angle(gamma_s11)
                self._draw_smith_grid(ax_smith)
                ax_smith.plot(gamma_s11.real, gamma_s11.imag,
                              color='#3498db', lw=1.4, alpha=0.7, zorder=3, label='S11')
                ax_smith.plot(gamma_s22.real, gamma_s22.imag,
                              color='#e74c3c', lw=1.4, alpha=0.7, zorder=3, label='S22')
                renorm_label = f"  [P1={complex(p1_z0).real:.0f}{complex(p1_z0).imag:+.0f}jΩ, P2={complex(p2_z0).real:.0f}{complex(p2_z0).imag:+.0f}jΩ]"
                ax_smith.set_title(f"S11 / S22 Smith Chart - {best_algo}{renorm_label}",
                                   fontsize=10, fontweight='bold')
                self._rebuild_smith_slider()
                self._redraw_smith(0)
            except Exception as e:
                self.log(f"Smith chart update failed: {e}")

        self.sp_fig.tight_layout(); self.sp_canvas.draw()
        self.smith_fig.tight_layout(); self.smith_canvas.draw()

    # 변경: 선택된 알고리즘 콤보에 따라 Result 행 표시/숨김 필터링
    def _filter_result_rows_by_algo(self, *_args):
        if not hasattr(self, '_result_row_widgets'):
            return
        text = self.algo_combo.currentText() if hasattr(self, 'algo_combo') else ""
        combo_map = {
            "GD Only":          ["GD"],
            "GA Only":          ["GA"],
            "BPSO Only":        ["BPSO"],
            "BO Only":          ["BO"],
            "DE Only":          ["DE"],
            "GA->GD Only":      ["GA_GD"],
            "BPSO->GD Only":    ["BPSO_GD"],
            "GA->DBS Only":     ["GA_DBS"],
            "BPSO->DBS Only":   ["BPSO_DBS"],
        }
        if text.startswith("All"):
            visible = list(self._result_row_widgets.keys())
        else:
            visible = combo_map.get(text, list(self._result_row_widgets.keys()))
        for name, widgets in self._result_row_widgets.items():
            show = name in visible
            for w in widgets:
                try:
                    w.setVisible(show)
                except Exception:
                    pass

    def _update_pass_fail(self, best_algo):
        """8채널 모델에서 s11_db/s21_db 채널을 기준으로 Pass/Fail 계산."""
        if not self.all_results:
            return
        n_f = len(self.frequencies)
        target_keys = get_target_keys(self.config)
        fl, fh = self.config['f_low'], self.config['f_high']
        pb_mask = (self.frequencies >= fl) & (self.frequencies <= fh)
        try:
            target_il = float(self.il_input.text())
            target_rl = float(self.rl_input.text())
        except Exception:
            target_il, target_rl = -1.5, -15.0

        for name in ['GD', 'GA', 'BPSO', 'BO', 'DE', 'GA_GD', 'BPSO_GD', 'GA_DBS', 'BPSO_DBS']:
            if name in self.all_results and name in self._result_rows:
                res = self.all_results[name]
                layout = res.get('layout')
                if layout is None:
                    continue
                try:
                    lt = torch.FloatTensor(layout).unsqueeze(0).unsqueeze(0).to(self.device)
                    pn = ensemble_predict(self.forward_models, lt, 512).cpu().numpy()[0]
                    pred = self.denorm_clamp(pn)
                    ch = split_prediction_channels(pred, target_keys, n_f)
                    s11_m = float(np.mean(ch['s11_db'][pb_mask]))
                    s21_m = float(np.mean(ch['s21_db'][pb_mask]))
                    s21_l, s11_l = self._result_rows[name]
                    s21_l.setText(f"{s21_m:.1f}"); s11_l.setText(f"{s11_m:.1f}")
                    p21 = s21_m >= target_il
                    p11 = s11_m <= target_rl
                    s21_l.setStyleSheet(f"font-size: 10px; border-radius: 2px; background: {'#d4edda' if p21 else '#f8d7da'}; color: {'#155724' if p21 else '#721c24'};")
                    s11_l.setStyleSheet(f"font-size: 10px; border-radius: 2px; background: {'#d4edda' if p11 else '#f8d7da'}; color: {'#155724' if p11 else '#721c24'};")
                except Exception:
                    pass
        if best_algo in self.all_results:
            self._pf_detail_lbl.setText(f"Best: {best_algo} (Error {self.all_results[best_algo]['error']:.4f})")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop(); self.worker.wait(1000)
        event.accept()


def main():
    app = QApplication(sys.argv); app.setStyle('Fusion')
    w = InverseDesignGUI(); w.show(); sys.exit(app.exec_())

if __name__ == '__main__':
    main()
