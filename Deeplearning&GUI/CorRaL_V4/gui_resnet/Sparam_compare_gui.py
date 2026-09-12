#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S-parameter comparison GUI for inverse / simulation / measurement s2p files.
"""

import sys
import numpy as np
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

try:
    import skrf as rf
    _HAS_SKRF = True
except ImportError:
    _HAS_SKRF = False


def _to_db(s_complex: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(s_complex), 1e-15))


def _parse_z0_input(text: str) -> complex:
    text = (text or "").strip().replace("Ω", "").replace("ohm", "").replace("Ohm", "")
    if not text:
        return 50.0 + 0.0j
    text = text.replace(" ", "").replace("i", "j").replace("J", "j")
    if text.endswith("+j"):
        text = text[:-2] + "+1j"
    elif text.endswith("-j"):
        text = text[:-2] + "-1j"
    elif text == "j":
        text = "1j"
    elif text == "-j":
        text = "-1j"
    try:
        value = complex(text)
    except Exception as exc:
        raise ValueError(f"Invalid impedance: {text}") from exc
    return value


def _error_rate(reference_db: np.ndarray, inverse_db: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            np.abs(reference_db) > 1e-6,
            (reference_db - inverse_db) / reference_db * 100.0,
            0.0,
        )


def _gamma_to_impedance(gamma: np.ndarray, z0: complex) -> np.ndarray:
    denom = 1.0 - gamma
    denom = np.where(np.abs(denom) < 1e-15, 1e-15 + 0j, denom)
    return complex(z0) * (1.0 + gamma) / denom


def _relative_percent_error(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    reference_mag = np.abs(reference)
    denom = np.where(reference_mag > 1e-12, reference_mag, 1e-12)
    return np.abs(reference - estimate) / denom * 100.0


def _load_s2p(filepath: str):
    filepath = str(filepath)
    if _HAS_SKRF:
        nw = rf.Network(filepath)
        return (
            nw.f / 1e9,
            nw.s[:, 0, 0],
            nw.s[:, 0, 1],
            nw.s[:, 1, 0],
            nw.s[:, 1, 1],
        )

    freqs = []
    s11 = []
    s12 = []
    s21 = []
    s22 = []
    freq_scale = 1.0
    fmt = "RI"

    with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                parts = line.upper().split()
                if "HZ" in parts:
                    freq_scale = 1e-9
                if "KHZ" in parts:
                    freq_scale = 1e-6
                if "MHZ" in parts:
                    freq_scale = 1e-3
                if "GHZ" in parts:
                    freq_scale = 1.0
                if "MA" in parts:
                    fmt = "MA"
                if "DB" in parts:
                    fmt = "DB"
                if "RI" in parts:
                    fmt = "RI"
                continue

            values = line.split()
            if len(values) < 9:
                continue

            def _pair(a, b):
                va = float(a)
                vb = float(b)
                if fmt == "RI":
                    return complex(va, vb)
                if fmt == "MA":
                    return va * np.exp(1j * np.radians(vb))
                return 10 ** (va / 20.0) * np.exp(1j * np.radians(vb))

            freq = float(values[0]) * freq_scale
            freqs.append(freq)
            s11.append(_pair(values[1], values[2]))
            s21.append(_pair(values[3], values[4]))
            s12.append(_pair(values[5], values[6]))
            s22.append(_pair(values[7], values[8]))

    return (
        np.asarray(freqs, dtype=np.float64),
        np.asarray(s11, dtype=np.complex128),
        np.asarray(s12, dtype=np.complex128),
        np.asarray(s21, dtype=np.complex128),
        np.asarray(s22, dtype=np.complex128),
    )


def _align_to_ref(ref_data, other_data):
    if other_data is None:
        return None

    ref_freqs = ref_data[0]

    def _interp(trace):
        return (
            np.interp(ref_freqs, other_data[0], trace.real)
            + 1j * np.interp(ref_freqs, other_data[0], trace.imag)
        )

    return (
        ref_freqs,
        _interp(other_data[1]),
        _interp(other_data[2]),
        _interp(other_data[3]),
        _interp(other_data[4]),
    )


def _renormalize_dataset(data, p1_z0: complex, p2_z0: complex):
    if data is None or not _HAS_SKRF:
        return data

    freqs, s11, s12, s21, s22 = data
    s_matrix = np.stack(
        [
            np.stack([s11, s12], axis=-1),
            np.stack([s21, s22], axis=-1),
        ],
        axis=1,
    ).astype(np.complex128)

    z_old = np.array([50.0 + 0.0j, 50.0 + 0.0j], dtype=np.complex128)
    z_new = np.array([complex(p1_z0), complex(p2_z0)], dtype=np.complex128)
    s_renorm = rf.network.renormalize_s(s_matrix, z_old=z_old, z_new=z_new, s_def="power")

    return (
        np.asarray(freqs, dtype=np.float64),
        np.asarray(s_renorm[:, 0, 0], dtype=np.complex128),
        np.asarray(s_renorm[:, 0, 1], dtype=np.complex128),
        np.asarray(s_renorm[:, 1, 0], dtype=np.complex128),
        np.asarray(s_renorm[:, 1, 1], dtype=np.complex128),
    )


def _draw_smith_grid(ax, lw: float = 0.5):
    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    ax.plot(np.cos(theta), np.sin(theta), color="black", lw=1.1)

    r_values = [0.2, 0.5, 1.0, 2.0, 5.0]
    x_values = [0.2, 0.5, 1.0, 2.0, 5.0]

    for r in r_values:
        center_x = r / (1.0 + r)
        radius = 1.0 / (1.0 + r)
        x = center_x + radius * np.cos(theta)
        y = radius * np.sin(theta)
        mask = x * x + y * y <= 1.0001
        ax.plot(x[mask], y[mask], color="#b0b7bf", lw=lw)

    t = np.linspace(-20.0, 20.0, 2400)
    for x_value in x_values:
        for sign in (-1.0, 1.0):
            xc = 1.0 + 1j * sign / x_value
            radius = 1.0 / abs(x_value)
            gamma = xc + radius * np.exp(1j * t)
            mask = np.abs(gamma) <= 1.0001
            ax.plot(gamma.real[mask], gamma.imag[mask], color="#b0b7bf", lw=lw)

    ax.axhline(0.0, color="#d0d5da", lw=0.4)


def _set_freq_ticks(ax, fl: float, fh: float, tick_step: float = 0.5):
    ax.set_xlim(fl, fh)
    if fh > fl:
        start = np.ceil(fl / tick_step) * tick_step
        ticks = np.arange(start, fh + tick_step * 0.5, tick_step)
        ticks = ticks[(ticks >= fl - 1e-9) & (ticks <= fh + 1e-9)]
        ticks = np.unique(np.round(np.concatenate(([fl], ticks, [fh])), 6))
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def _set_freq_ticks_points(ax, fl: float, fh: float, tick_points: int):
    ax.set_xlim(fl, fh)
    tick_points = max(2, int(tick_points))
    ax.set_xticks(np.linspace(fl, fh, tick_points))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))


_COLORS = {
    "inv": "#3498db",
    "sim": "#2ecc71",
    "meas": "#e74c3c",
}

_S_PARAMS = [
    ("S11", lambda d: d[1]),
    ("S22", lambda d: d[4]),
    ("S21", lambda d: d[3]),
    ("S12", lambda d: d[2]),
]


class SParamCompareGUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._inv_data_raw = None
        self._sim_data_raw = None
        self._meas_data_raw = None
        self._inv_data = None
        self._sim_data = None
        self._meas_data = None
        self._err_hover_series = {}
        self._err_hover_ax = None
        self._err_hover_annot = None
        self._build_ui()
        self._wire_controls()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        root.addWidget(self._build_left_panel())
        right_side = QWidget()
        right_side_layout = QVBoxLayout(right_side)
        right_side_layout.setContentsMargins(0, 0, 0, 0)
        right_side_layout.setSpacing(4)
        # 변경: 상단 Plot X-Axis 컨트롤은 왼쪽 패널 "View Mode" 위로 이동 — 여기에선 제거

        panels_row = QWidget()
        panels_row_layout = QHBoxLayout(panels_row)
        panels_row_layout.setContentsMargins(0, 0, 0, 0)
        panels_row_layout.setSpacing(6)
        panels_row_layout.addWidget(self._build_compare_panel(), stretch=30)
        panels_row_layout.addWidget(self._build_right_panel(), stretch=30)
        panels_row_layout.addWidget(self._build_smith_panel(), stretch=30)
        right_side_layout.addWidget(panels_row, stretch=1)

        root.addWidget(right_side, stretch=1)

    # 변경: _build_plot_ctrl_bar는 이제 사용하지 않음 (왼쪽 패널로 이동)
    # 호출 코드가 남아있을 경우를 대비한 no-op 스텁
    def _build_plot_ctrl_bar(self) -> QWidget:
        stub = QWidget()
        stub.setVisible(False)
        return stub

    def _build_left_panel(self) -> QWidget:
        widget = QWidget()
        widget.setFixedWidth(350)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        file_group = QGroupBox("S2P Files")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(6)

        file_specs = [
            ("Inverse Design", "_inv_edit", _COLORS["inv"]),
            ("Simulation", "_sim_edit", _COLORS["sim"]),
            ("Measurement", "_meas_edit", _COLORS["meas"]),
        ]

        for label_text, attr, color in file_specs:
            label = QLabel(label_text)
            label.setStyleSheet(f"font-size:11px; font-weight:bold; color:{color};")
            file_layout.addWidget(label)

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            edit = QLineEdit()
            edit.setPlaceholderText("Select file")
            edit.setStyleSheet("font-size:11px;")
            edit.setFixedHeight(26)
            setattr(self, attr, edit)
            row_layout.addWidget(edit)

            button = QPushButton("...")
            button.setFixedWidth(34)
            button.setToolTip("Browse")

            def _make_cb(target_edit=edit):
                def _cb():
                    selected, _ = QFileDialog.getOpenFileName(
                        self,
                        "Select S2P File",
                        "",
                        "Touchstone (*.s2p *.S2P);;All Files (*.*)",
                    )
                    if selected:
                        target_edit.setText(selected)

                return _cb

            button.clicked.connect(_make_cb())
            row_layout.addWidget(button)
            file_layout.addWidget(row_widget)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        z0_group = QGroupBox("Reference Impedance (Renorm)")
        z0_layout = QVBoxLayout()
        z0_layout.setSpacing(5)

        port1_row = QWidget()
        port1_layout = QHBoxLayout(port1_row)
        port1_layout.setContentsMargins(0, 0, 0, 0)
        port1_layout.setSpacing(4)
        lbl1 = QLabel("P1 Z0")
        lbl1.setStyleSheet("font-size:11px;")
        port1_layout.addWidget(lbl1)
        self._p1_z0_edit = QLineEdit("50+0j")
        self._p1_z0_edit.setFixedHeight(26); self._p1_z0_edit.setStyleSheet("font-size:11px;")
        port1_layout.addWidget(self._p1_z0_edit)
        z0_layout.addWidget(port1_row)

        port2_row = QWidget()
        port2_layout = QHBoxLayout(port2_row)
        port2_layout.setContentsMargins(0, 0, 0, 0)
        port2_layout.setSpacing(4)
        lbl2 = QLabel("P2 Z0")
        lbl2.setStyleSheet("font-size:11px;")
        port2_layout.addWidget(lbl2)
        self._p2_z0_edit = QLineEdit("50+0j")
        self._p2_z0_edit.setFixedHeight(26); self._p2_z0_edit.setStyleSheet("font-size:11px;")
        port2_layout.addWidget(self._p2_z0_edit)
        z0_layout.addWidget(port2_row)

        z0_note = QLabel("Renorm: treated as 50 ohm data and recomputed.")
        z0_note.setStyleSheet("font-size:10px; color:#67727e;")
        z0_layout.addWidget(z0_note)
        z0_group.setLayout(z0_layout)
        layout.addWidget(z0_group)

        # 변경: Plot X-Axis 컨트롤을 왼쪽 패널의 View Mode 위로 이동
        plot_axis_group = QGroupBox("Plot X-Axis")
        plot_axis_group.setMinimumWidth(230)
        plot_axis_layout = QGridLayout()
        plot_axis_layout.setContentsMargins(12, 12, 12, 10)
        plot_axis_layout.setHorizontalSpacing(10)
        plot_axis_layout.setVerticalSpacing(4)
        _pfl_lbl = QLabel("Plot FL (GHz)")
        _pfl_lbl.setStyleSheet("font-size:11px;")
        plot_axis_layout.addWidget(_pfl_lbl, 0, 0)
        self._fl_edit = QLineEdit("0.1")
        self._fl_edit.setFixedHeight(26); self._fl_edit.setStyleSheet("font-size:11px;")
        plot_axis_layout.addWidget(self._fl_edit, 0, 1)
        _pfh_lbl = QLabel("Plot FH (GHz)")
        _pfh_lbl.setStyleSheet("font-size:11px;")
        plot_axis_layout.addWidget(_pfh_lbl, 1, 0)
        self._fh_edit = QLineEdit("30.0")
        self._fh_edit.setFixedHeight(26); self._fh_edit.setStyleSheet("font-size:11px;")
        plot_axis_layout.addWidget(self._fh_edit, 1, 1)
        _ptk_lbl = QLabel("X Ticks")
        _ptk_lbl.setStyleSheet("font-size:11px;")
        plot_axis_layout.addWidget(_ptk_lbl, 2, 0)
        self._ticks_edit = QLineEdit("11")
        self._ticks_edit.setFixedHeight(26); self._ticks_edit.setStyleSheet("font-size:11px;")
        plot_axis_layout.addWidget(self._ticks_edit, 2, 1)
        self._plot_apply_btn = QPushButton("Apply Plot X-Axis")
        self._plot_apply_btn.setFixedHeight(26)
        self._plot_apply_btn.setStyleSheet("font-size:11px;")
        plot_axis_layout.addWidget(self._plot_apply_btn, 3, 0, 1, 2)
        plot_axis_group.setLayout(plot_axis_layout)
        layout.addWidget(plot_axis_group)

        # 변경: S11/S22 와 S21/S12 Y축 박스를 좌우로 나란히 배치
        plot_y_pair_widget = QWidget()
        plot_y_pair_layout = QHBoxLayout(plot_y_pair_widget)
        plot_y_pair_layout.setContentsMargins(0, 0, 0, 0); plot_y_pair_layout.setSpacing(4)

        plot_y_refl_group = QGroupBox("Y-Axis (S11/S22)")
        plot_y_refl_group.setStyleSheet("font-size:11px;")
        plot_y_refl_layout = QGridLayout()
        plot_y_refl_layout.setContentsMargins(6, 6, 6, 6)
        plot_y_refl_layout.setHorizontalSpacing(2); plot_y_refl_layout.setVerticalSpacing(4)
        plot_y_refl_layout.setColumnStretch(0, 0); plot_y_refl_layout.setColumnStretch(1, 1)
        plot_y_refl_layout.addWidget(QLabel("min:"), 0, 0)
        self._y_lo_refl_edit = QLineEdit("-30")
        self._y_lo_refl_edit.setFixedHeight(26); self._y_lo_refl_edit.setStyleSheet("font-size:11px;")
        plot_y_refl_layout.addWidget(self._y_lo_refl_edit, 0, 1)
        plot_y_refl_layout.addWidget(QLabel("max:"), 1, 0)
        self._y_hi_refl_edit = QLineEdit("0")
        self._y_hi_refl_edit.setFixedHeight(26); self._y_hi_refl_edit.setStyleSheet("font-size:11px;")
        plot_y_refl_layout.addWidget(self._y_hi_refl_edit, 1, 1)
        plot_y_refl_layout.addWidget(QLabel("Ticks:"), 2, 0)
        self._y_ticks_refl_edit = QLineEdit("7")
        self._y_ticks_refl_edit.setFixedHeight(26); self._y_ticks_refl_edit.setStyleSheet("font-size:11px;")
        plot_y_refl_layout.addWidget(self._y_ticks_refl_edit, 2, 1)
        self._y_refl_apply_btn = QPushButton("Apply")
        self._y_refl_apply_btn.setFixedHeight(26); self._y_refl_apply_btn.setStyleSheet("font-size:11px;")
        plot_y_refl_layout.addWidget(self._y_refl_apply_btn, 3, 0, 1, 2)
        plot_y_refl_group.setLayout(plot_y_refl_layout)
        plot_y_pair_layout.addWidget(plot_y_refl_group)

        plot_y_trans_group = QGroupBox("Y-Axis (S21/S12)")
        plot_y_trans_group.setStyleSheet("font-size:11px;")
        plot_y_trans_layout = QGridLayout()
        plot_y_trans_layout.setContentsMargins(4, 4, 4, 4)
        plot_y_trans_layout.setHorizontalSpacing(2); plot_y_trans_layout.setVerticalSpacing(2)
        plot_y_trans_layout.addWidget(QLabel("min:"), 0, 0)
        self._y_lo_trans_edit = QLineEdit("-14")
        self._y_lo_trans_edit.setFixedHeight(26); self._y_lo_trans_edit.setStyleSheet("font-size:11px;")
        plot_y_trans_layout.addWidget(self._y_lo_trans_edit, 0, 1)
        plot_y_trans_layout.addWidget(QLabel("max:"), 1, 0)
        self._y_hi_trans_edit = QLineEdit("1")
        self._y_hi_trans_edit.setFixedHeight(26); self._y_hi_trans_edit.setStyleSheet("font-size:11px;")
        plot_y_trans_layout.addWidget(self._y_hi_trans_edit, 1, 1)
        plot_y_trans_layout.addWidget(QLabel("Ticks:"), 2, 0)
        self._y_ticks_trans_edit = QLineEdit("6")
        self._y_ticks_trans_edit.setFixedHeight(26); self._y_ticks_trans_edit.setStyleSheet("font-size:11px;")
        plot_y_trans_layout.addWidget(self._y_ticks_trans_edit, 2, 1)
        self._y_trans_apply_btn = QPushButton("Apply")
        self._y_trans_apply_btn.setFixedHeight(26); self._y_trans_apply_btn.setStyleSheet("font-size:11px;")
        plot_y_trans_layout.addWidget(self._y_trans_apply_btn, 3, 0, 1, 2)
        plot_y_trans_group.setLayout(plot_y_trans_layout)
        plot_y_pair_layout.addWidget(plot_y_trans_group)

        layout.addWidget(plot_y_pair_widget)

        view_group = QGroupBox("View Mode")
        view_layout = QVBoxLayout()
        view_layout.setSpacing(4)

        self._center_mode_combo = QComboBox()
        self._center_mode_combo.addItems([
            "Cartesian (dB) x 4",
            "Smith Chart",
            "Smith + Cartesian",
        ])
        self._center_mode_combo.setCurrentText("Smith + Cartesian")
        view_layout.addWidget(QLabel("Center Panel"))
        view_layout.addWidget(self._center_mode_combo)

        self._align_chk = QCheckBox("Resample inverse and measurement to simulation frequencies")
        self._align_chk.setChecked(True)
        self._align_chk.setStyleSheet("font-size:11px;")
        view_layout.addWidget(self._align_chk)

        plot_y_err_group = QGroupBox("Y-Axis (Error %)")
        plot_y_err_group.setStyleSheet("font-size:11px;")
        plot_y_err_layout = QGridLayout()
        plot_y_err_layout.setContentsMargins(4, 4, 4, 4)
        plot_y_err_layout.setHorizontalSpacing(4); plot_y_err_layout.setVerticalSpacing(2)
        plot_y_err_layout.addWidget(QLabel("min:"), 0, 0)
        self._y_lo_err_edit = QLineEdit("0"); self._y_lo_err_edit.setFixedWidth(40)
        self._y_lo_err_edit.setFixedHeight(26); self._y_lo_err_edit.setStyleSheet("font-size:11px;")
        plot_y_err_layout.addWidget(self._y_lo_err_edit, 0, 1)
        plot_y_err_layout.addWidget(QLabel("max:"), 0, 2)
        self._y_hi_err_edit = QLineEdit("50"); self._y_hi_err_edit.setFixedWidth(40)
        self._y_hi_err_edit.setFixedHeight(26); self._y_hi_err_edit.setStyleSheet("font-size:11px;")
        plot_y_err_layout.addWidget(self._y_hi_err_edit, 0, 3)
        plot_y_err_layout.addWidget(QLabel("Ticks:"), 0, 4)
        self._y_ticks_err_edit = QLineEdit("6"); self._y_ticks_err_edit.setFixedWidth(30)
        self._y_ticks_err_edit.setFixedHeight(26); self._y_ticks_err_edit.setStyleSheet("font-size:11px;")
        plot_y_err_layout.addWidget(self._y_ticks_err_edit, 0, 5)
        self._y_err_apply_btn = QPushButton("Apply")
        self._y_err_apply_btn.setFixedHeight(26); self._y_err_apply_btn.setStyleSheet("font-size:11px;")
        plot_y_err_layout.addWidget(self._y_err_apply_btn, 1, 0, 1, 6)
        plot_y_err_group.setLayout(plot_y_err_layout)
        view_layout.addWidget(plot_y_err_group)

        view_layout.addWidget(QLabel("Smith Data", styleSheet="font-size:11px;"))
        smith_toggle_row = QWidget()
        smith_toggle_layout = QHBoxLayout(smith_toggle_row)
        smith_toggle_layout.setContentsMargins(0, 0, 0, 0)
        smith_toggle_layout.setSpacing(6)

        self._smith_inv_chk = QCheckBox("Inv")
        self._smith_inv_chk.setChecked(True)
        self._smith_inv_chk.setStyleSheet(f"font-size:11px; color:{_COLORS['inv']};")
        smith_toggle_layout.addWidget(self._smith_inv_chk)

        self._smith_sim_chk = QCheckBox("Sim")
        self._smith_sim_chk.setChecked(True)
        self._smith_sim_chk.setStyleSheet(f"font-size:11px; color:{_COLORS['sim']};")
        smith_toggle_layout.addWidget(self._smith_sim_chk)

        self._smith_meas_chk = QCheckBox("Meas")
        self._smith_meas_chk.setChecked(True)
        self._smith_meas_chk.setStyleSheet(f"font-size:11px; color:{_COLORS['meas']};")
        smith_toggle_layout.addWidget(self._smith_meas_chk)
        smith_toggle_layout.addStretch(1)
        view_layout.addWidget(smith_toggle_row)

        view_layout.addWidget(QLabel("Smith Trace Toggle"))
        smith_trace_row = QWidget()
        smith_trace_layout = QHBoxLayout(smith_trace_row)
        smith_trace_layout.setContentsMargins(0, 0, 0, 0)
        smith_trace_layout.setSpacing(6)

        self._smith_s11_chk = QCheckBox("S11")
        self._smith_s11_chk.setChecked(True)
        self._smith_s11_chk.setStyleSheet("font-size:11px; color:#34495e;")
        smith_trace_layout.addWidget(self._smith_s11_chk)

        self._smith_s22_chk = QCheckBox("S22")
        self._smith_s22_chk.setChecked(True)
        self._smith_s22_chk.setStyleSheet("font-size:11px; color:#34495e;")
        smith_trace_layout.addWidget(self._smith_s22_chk)
        smith_trace_layout.addStretch(1)
        view_layout.addWidget(smith_trace_row)

        smith_note = QLabel("S11 (solid), S22 (dashed)")
        smith_note.setStyleSheet("font-size:10px; color:#67727e;")
        view_layout.addWidget(smith_note)
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        self._plot_btn = QPushButton("Plot / Refresh")
        self._plot_btn.setFixedHeight(32)
        self._plot_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self._plot_btn.setStyleSheet(
            """
            QPushButton { background:#2980b9; color:#fff; border-radius:4px; padding:2px; }
            QPushButton:hover { background:#1a6fa8; }
            QPushButton:pressed { background:#16599a; }
            """
        )
        self._plot_btn.clicked.connect(self._on_plot)
        layout.addWidget(self._plot_btn)



        layout.addStretch()
        return widget

    def _build_compare_panel(self) -> QWidget:
        widget = QWidget()
        self._compare_panel = widget
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        compare_title = QLabel("S-Parameter Comparison")
        compare_title.setAlignment(Qt.AlignCenter)
        compare_title.setStyleSheet(
            "font-size:11px; font-weight:bold; color:#2c3e50; "
            "background:#eaf0f6; border-radius:4px; padding:4px;"
        )
        compare_title.setFixedHeight(28)
        layout.addWidget(compare_title)

        self._compare_legend_lbl = QLabel()
        self._compare_legend_lbl.setAlignment(Qt.AlignCenter)
        self._compare_legend_lbl.setStyleSheet(
            "font-size:11px; color:#576574; border:1px solid #dee2e6; "
            "border-radius:3px; padding:1px 3px; background:#f8f9fa;"
        )
        self._compare_legend_lbl.setFixedHeight(34)
        layout.addWidget(self._compare_legend_lbl)

        self._compare_fig = Figure()
        self._compare_canvas = FigureCanvas(self._compare_fig)
        self._compare_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._compare_canvas, stretch=1)
        self._draw_empty(self._compare_fig, self._compare_canvas, ["S11", "S22", "S21", "S12"], 1)
        return widget

    def _build_smith_panel(self) -> QWidget:
        self._smith_panel = QWidget()
        smith_layout = QVBoxLayout(self._smith_panel)
        smith_layout.setContentsMargins(0, 0, 0, 0)
        smith_layout.setSpacing(3)

        smith_title = QLabel("Overall Smith Chart")
        smith_title.setAlignment(Qt.AlignCenter)
        smith_title.setStyleSheet(
            "font-size:11px; font-weight:bold; color:#2c3e50; "
            "background:#eaf0f6; border-radius:4px; padding:4px;"
        )
        smith_title.setFixedHeight(28)
        smith_layout.addWidget(smith_title)

        self._smith_legend_lbl = QLabel()
        self._smith_legend_lbl.setAlignment(Qt.AlignCenter)
        self._smith_legend_lbl.setWordWrap(True)
        self._smith_legend_lbl.setStyleSheet(
            "font-size:11px; color:#576574; border:1px solid #dee2e6; "
            "border-radius:3px; padding:2px; background:#f8f9fa;"
        )
        self._smith_legend_lbl.setFixedHeight(34)
        smith_layout.addWidget(self._smith_legend_lbl)

        self._smith_fig = Figure()
        self._smith_canvas = FigureCanvas(self._smith_fig)
        self._smith_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        smith_layout.addWidget(self._smith_canvas, stretch=1)
        self._smith_z0_lbl = QLabel("Port1 Z0 = 50Ω   |   Port2 Z0 = 50Ω")
        self._smith_z0_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._smith_z0_lbl.setStyleSheet(
            "font-size:11px; color:#555; border:1px solid #ccc; border-radius:3px; padding:3px 6px; background:#f8f8f8;"
        )
        smith_layout.addWidget(self._smith_z0_lbl)
        self._draw_empty(self._smith_fig, self._smith_canvas, ["Overall Smith"], 1)
        return self._smith_panel

    def _build_right_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        title = QLabel("Error Rate (%)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size:11px; font-weight:bold; color:#2c3e50; "
            "background:#eaf0f6; border-radius:4px; padding:4px;"
        )
        title.setFixedHeight(28)
        layout.addWidget(title)

        self._err_legend_lbl = QLabel()
        self._err_legend_lbl.setAlignment(Qt.AlignCenter)
        self._err_legend_lbl.setStyleSheet(
            "font-size:11px; color:#555; border:1px solid #dee2e6; "
            "border-radius:3px; padding:3px; background:#f8f9fa;"
        )
        self._err_legend_lbl.setFixedHeight(34)
        layout.addWidget(self._err_legend_lbl)

        self._err_fig = Figure()
        self._err_canvas = FigureCanvas(self._err_fig)
        self._err_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._err_canvas, stretch=1)
        self._draw_empty(
            self._err_fig,
            self._err_canvas,
            ["S11 err%", "S22 err%", "S21 err%", "S12 err%"],
            1,
        )
        return widget

    def _wire_controls(self):
        self._center_mode_combo.currentTextChanged.connect(lambda _: self._redraw_center())
        self._y_refl_apply_btn.clicked.connect(self._redraw_center)
        self._y_trans_apply_btn.clicked.connect(self._redraw_center)
        self._y_err_apply_btn.clicked.connect(self._redraw_error)
        self._y_hi_trans_edit.editingFinished.connect(self._redraw_center)
        self._y_ticks_trans_edit.editingFinished.connect(self._redraw_center)
        # Error Y축 자동 반영
        self._y_lo_err_edit.editingFinished.connect(self._redraw_error)
        self._y_hi_err_edit.editingFinished.connect(self._redraw_error)
        self._y_ticks_err_edit.editingFinished.connect(self._redraw_error)
        
        self._plot_apply_btn.clicked.connect(self._redraw_center)
        self._plot_apply_btn.clicked.connect(self._redraw_error)
        self._fl_edit.editingFinished.connect(self._redraw_center)
        self._fl_edit.editingFinished.connect(self._redraw_error)
        self._fh_edit.editingFinished.connect(self._redraw_center)
        self._fh_edit.editingFinished.connect(self._redraw_error)
        self._ticks_edit.editingFinished.connect(self._redraw_center)
        self._ticks_edit.editingFinished.connect(self._redraw_error)
        self._p1_z0_edit.editingFinished.connect(self._redraw_center)
        self._p1_z0_edit.editingFinished.connect(self._redraw_error)
        self._p2_z0_edit.editingFinished.connect(self._redraw_center)
        self._p2_z0_edit.editingFinished.connect(self._redraw_error)
        self._smith_inv_chk.stateChanged.connect(lambda _: self._redraw_center())
        self._smith_sim_chk.stateChanged.connect(lambda _: self._redraw_center())
        self._smith_meas_chk.stateChanged.connect(lambda _: self._redraw_center())
        self._smith_s11_chk.stateChanged.connect(lambda _: self._redraw_center())
        self._smith_s22_chk.stateChanged.connect(lambda _: self._redraw_center())
        self._err_canvas.mpl_connect("motion_notify_event", self._on_error_hover)
        self._err_canvas.mpl_connect("figure_leave_event", self._hide_error_hover)
        self._smith_canvas.mpl_connect("motion_notify_event", self._on_smith_hover)
        self._smith_canvas.mpl_connect("figure_leave_event", lambda _e: self._hide_smith_hover())
        self._smith_hover_points = []
        self._smith_hover_ax = None
        self._smith_hover_annot = None
        self._update_master_legend_labels()

    def _draw_empty(self, fig: Figure, canvas: FigureCanvas, titles: list, ncols: int):
        fig.clear()
        nrows = (len(titles) + ncols - 1) // ncols
        for index, title in enumerate(titles):
            ax = fig.add_subplot(nrows, ncols, index + 1)
            ax.set_title(title, fontsize=9, fontweight="bold")
            # 변경: 로고(wCoRaL)와 겹쳐 보여서 "No data" 텍스트 제거
            ax.set_xticks([])
            ax.set_yticks([])
        canvas.draw()

    def _on_plot(self):
        inv_path = self._inv_edit.text().strip()
        sim_path = self._sim_edit.text().strip()
        meas_path = self._meas_edit.text().strip()

        if not inv_path:
            QMessageBox.warning(self, "Missing File", "Select the inverse-design .s2p file.")
            return
        if not sim_path:
            QMessageBox.warning(self, "Missing File", "Select the simulation .s2p file.")
            return

        status_lines = []

        try:
            self._inv_data_raw = _load_s2p(inv_path)
            status_lines.append(
                f"Inverse: {Path(inv_path).name} "
                f"({len(self._inv_data_raw[0])} pts, {self._inv_data_raw[0][0]:.2f}~{self._inv_data_raw[0][-1]:.2f} GHz)"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", f"Failed to load inverse file:\n{exc}")
            return

        try:
            self._sim_data_raw = _load_s2p(sim_path)
            status_lines.append(
                f"Simulation: {Path(sim_path).name} "
                f"({len(self._sim_data_raw[0])} pts, {self._sim_data_raw[0][0]:.2f}~{self._sim_data_raw[0][-1]:.2f} GHz)"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", f"Failed to load simulation file:\n{exc}")
            return

        self._meas_data_raw = None
        if meas_path:
            try:
                self._meas_data_raw = _load_s2p(meas_path)
                status_lines.append(
                    f"Measurement: {Path(meas_path).name} "
                    f"({len(self._meas_data_raw[0])} pts, {self._meas_data_raw[0][0]:.2f}~{self._meas_data_raw[0][-1]:.2f} GHz)"
                )
            except Exception as exc:
                status_lines.append(f"Measurement load skipped: {exc}")

        self._sim_data = self._sim_data_raw
        self._inv_data = self._inv_data_raw
        self._meas_data = self._meas_data_raw
        if self._align_chk.isChecked():
            self._inv_data = _align_to_ref(self._sim_data_raw, self._inv_data_raw)
            self._meas_data = _align_to_ref(self._sim_data_raw, self._meas_data_raw)


        self._update_master_legend_labels()
        self._redraw_center()
        self._redraw_error()

    def _get_fl_fh(self):
        try:
            fl = float(self._fl_edit.text())
        except ValueError:
            fl = 0.1
        try:
            fh = float(self._fh_edit.text())
        except ValueError:
            fh = 30.0
        if fl > fh:
            fl, fh = fh, fl
        return fl, fh

    def _get_tick_points(self):
        try:
            pts = int(self._ticks_edit.text())
        except ValueError:
            pts = 11
        return max(2, pts)

    def _get_visual_dataset_specs(self):
        try:
            p1_z0 = _parse_z0_input(self._p1_z0_edit.text())
            p2_z0 = _parse_z0_input(self._p2_z0_edit.text())
        except ValueError:
            p1_z0, p2_z0 = 50.0 + 0.0j, 50.0 + 0.0j
        return [
            ("Inverse", _renormalize_dataset(self._inv_data, p1_z0, p2_z0), _COLORS["inv"], "-"),
            ("Sim", _renormalize_dataset(self._sim_data, p1_z0, p2_z0), _COLORS["sim"], "-"),
            ("Meas", _renormalize_dataset(self._meas_data, p1_z0, p2_z0), _COLORS["meas"], "--"),
        ]

    def _redraw_center(self):
        self._update_center_panel_visibility()
        self._update_master_legend_labels()

        mode = self._center_mode_combo.currentText()
        fl, fh = self._get_fl_fh()

        if mode != "Smith Chart":
            fig = self._compare_fig
            fig.clear()
            grid = gridspec.GridSpec(
                4, 1, figure=fig, hspace=0.60, top=0.935, bottom=0.100, left=0.150, right=0.960
            )
            shared_ax = None
            for index, (name, getter) in enumerate(_S_PARAMS):
                ax = fig.add_subplot(grid[index], sharex=shared_ax)
                if shared_ax is None:
                    shared_ax = ax
                self._plot_db(ax, name, getter, fl, fh, show_x=index == len(_S_PARAMS) - 1)
            self._compare_canvas.draw()
        else:
            self._draw_empty(self._compare_fig, self._compare_canvas, ["S11", "S22", "S21", "S12"], 1)

        if mode != "Cartesian (dB) x 4":
            smith_fig = self._smith_fig
            smith_fig.clear()
            ax = smith_fig.add_subplot(1, 1, 1)
            smith_fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)  # 수정: 0.9x 크기 축소
            self._plot_smith_combined(ax, fl, fh)
            self._smith_canvas.draw()
        else:
            self._draw_empty(self._smith_fig, self._smith_canvas, ["Overall Smith"], 1)

    def _update_center_panel_visibility(self):
        mode = self._center_mode_combo.currentText()
        self._compare_panel.setVisible(mode != "Smith Chart")
        self._smith_panel.setVisible(mode != "Cartesian (dB) x 4")

    def _plot_db(self, ax, pname: str, getter, fl: float, fh: float, show_x: bool):
        ax.set_title(f"{pname} (dB)", fontsize=8.5, fontweight="bold", pad=6)
        ax.set_xlabel("Freq (GHz)", fontsize=7.5)   # 수정: 모든 서브플롯에 x축 라벨 표시
        ax.set_ylabel("dB", fontsize=7.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=6.5, pad=2)
        ax.tick_params(axis="x", labelbottom=True)   # 수정: 모든 서브플롯에 x 눈금값 표시
        _set_freq_ticks_points(ax, fl, fh, self._get_tick_points())
        ax.margins(x=0.01)
        # 변경: S11/S22 와 S21/S12 별 Y축 박스 입력값 적용
        is_refl = pname in ("S11", "S22")
        try:
            if is_refl:
                ylo = float(self._y_lo_refl_edit.text())
                yhi = float(self._y_hi_refl_edit.text())
                yticks = int(self._y_ticks_refl_edit.text())
            else:
                ylo = float(self._y_lo_trans_edit.text())
                yhi = float(self._y_hi_trans_edit.text())
                yticks = int(self._y_ticks_trans_edit.text())
            ax.set_ylim(ylo, yhi)
            if yticks >= 2:
                ax.set_yticks(np.linspace(ylo, yhi, yticks))
        except (ValueError, AttributeError):
            pass

        plotted = []
        for label, data, color, line_style in self._get_visual_dataset_specs():
            if data is None:
                continue
            freqs = data[0]
            mask = (freqs >= fl) & (freqs <= fh)
            if not mask.any():
                continue
            ax.plot(
                freqs[mask],
                _to_db(getter(data)[mask]),
                color=color,
                lw=1.4,
                ls=line_style,
                label=label,
                alpha=0.88,
            )
            plotted.append(label)

        if plotted:
            ax.legend(loc="upper right", fontsize=6.2, framealpha=0.92, borderpad=0.3, handlelength=2.4)

    def _get_error_y_range(self):
        try:
            ylo = float(self._y_lo_err_edit.text())
            yhi = float(self._y_hi_err_edit.text())
            yticks = int(self._y_ticks_err_edit.text())
            return ylo, yhi, max(2, yticks)
        except ValueError:
            return 0.0, 50.0, 6

    def _get_smith_dataset_specs(self):
        try:
            p1_z0 = _parse_z0_input(self._p1_z0_edit.text())
            p2_z0 = _parse_z0_input(self._p2_z0_edit.text())
        except ValueError:
            p1_z0, p2_z0 = 50.0 + 0.0j, 50.0 + 0.0j
        specs = []
        if self._smith_inv_chk.isChecked() and self._inv_data_raw is not None:
            specs.append(("Inverse", _renormalize_dataset(self._inv_data_raw, p1_z0, p2_z0), _COLORS["inv"]))
        if self._smith_sim_chk.isChecked() and self._sim_data_raw is not None:
            specs.append(("Sim", _renormalize_dataset(self._sim_data_raw, p1_z0, p2_z0), _COLORS["sim"]))
        if self._smith_meas_chk.isChecked() and self._meas_data_raw is not None:
            specs.append(("Meas", _renormalize_dataset(self._meas_data_raw, p1_z0, p2_z0), _COLORS["meas"]))
        return specs

    def _update_master_legend_labels(self):
        self._compare_legend_lbl.setText(
            f"<span style='color:{_COLORS['inv']}; font-weight:600;'>Inverse</span>   |   "
            f"<span style='color:{_COLORS['sim']}; font-weight:600;'>Sim</span>   |   "
            f"<span style='color:{_COLORS['meas']}; font-weight:600;'>Meas</span>"
        )

        visible_names = [name for name, _, _ in self._get_smith_dataset_specs()]
        smith_visible_text = ", ".join(visible_names) if visible_names else "None"
        visible_traces = []
        if self._smith_s11_chk.isChecked():
            visible_traces.append("S11")
        if self._smith_s22_chk.isChecked():
            visible_traces.append("S22")
        smith_trace_text = ", ".join(visible_traces) if visible_traces else "None"
        self._smith_legend_lbl.setText(
            f"Data: <span style='color:{_COLORS['inv']}; font-weight:600;'>Inverse</span> / "
            f"<span style='color:{_COLORS['sim']}; font-weight:600;'>Sim</span> / "
            f"<span style='color:{_COLORS['meas']}; font-weight:600;'>Meas</span><br>"
            f"Trace: Solid = S11, Dashed = S22 | Data: {smith_visible_text} | Trace: {smith_trace_text}"
        )

        ylo, yhi, yticks = self._get_error_y_range()
        self._err_legend_lbl.setText(
            "S11/S22: ||Zref| - |Zinv|| / |Zref| x 100% &nbsp;&nbsp;|&nbsp;&nbsp; "
            "S12/S21: |Vref - Vinv| / |Vref| x 100% (V = |S|)<br>"
            f"<span style='color:{_COLORS['inv']}; font-weight:600;'>Blue: Sim vs Inv</span>   |   "
            f"<span style='color:{_COLORS['meas']}; font-weight:600;'>Red: Meas vs Inv</span>   |   "
            f"Y-Axis: {ylo:.0f}~{yhi:.0f}%"
        )

    def _plot_smith_combined(self, ax, fl: float, fh: float):
        ax.set_aspect("equal")
        ax.set_title("Overall Smith Chart", fontsize=10, fontweight="bold")
        self._smith_hover_points = []
        try:
            p1_z0 = _parse_z0_input(self._p1_z0_edit.text())
            p2_z0 = _parse_z0_input(self._p2_z0_edit.text())
        except ValueError:
            p1_z0, p2_z0 = 50.0 + 0j, 50.0 + 0j
        self._smith_z0_lbl.setText(
            f"Port1 Z0 = {p1_z0.real:.0f}{p1_z0.imag:+.0f}jΩ   |   Port2 Z0 = {p2_z0.real:.0f}{p2_z0.imag:+.0f}jΩ"
        )

        if _HAS_SKRF:
            rf.plotting.smith(
                ax=ax,
                draw_labels=True,
                chart_type="z",
                ref_imm=1.0,
                draw_vswr=False,
            )
        else:
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.axis("off")
            _draw_smith_grid(ax)

        dataset_specs = self._get_smith_dataset_specs()
        trace_specs = []
        if self._smith_s11_chk.isChecked():
            trace_specs.append(("S11", lambda d: d[1], "-"))
        if self._smith_s22_chk.isChecked():
            trace_specs.append(("S22", lambda d: d[4], "--"))
        any_trace = False
        for _, data, color in dataset_specs:
            if data is None:
                continue
            freqs = data[0]
            mask = (freqs >= fl) & (freqs <= fh)
            if not mask.any():
                continue
            for trace_name, getter, line_style in trace_specs:
                any_trace = True
                trace = getter(data)[mask]
                ax.plot(trace.real, trace.imag, color=color, lw=1.35, ls=line_style, alpha=0.88)
                ax.scatter(trace.real, trace.imag, color=color, s=12, alpha=0.75, zorder=4)
                ax.plot(trace[0].real, trace[0].imag, "o", color=color, markersize=3.5, zorder=5, alpha=0.9)
                ax.plot(trace[-1].real, trace[-1].imag, "s", color=color, markersize=4.0, zorder=5, alpha=0.9)
                z_ref = p1_z0 if trace_name == "S11" else p2_z0
                z_series = _gamma_to_impedance(trace, z_ref)
                for freq, gamma, z_val in zip(freqs[mask], trace, z_series):
                    self._smith_hover_points.append((trace_name, float(freq), complex(gamma), complex(z_val)))

        legend_handles = [
            Line2D([0], [0], color=_COLORS["inv"], lw=1.8, label="Inverse"),
            Line2D([0], [0], color=_COLORS["sim"], lw=1.8, label="Sim"),
            Line2D([0], [0], color=_COLORS["meas"], lw=1.8, label="Meas"),
        ]
        if self._smith_s11_chk.isChecked():
            legend_handles.append(Line2D([0], [0], color="#444", lw=1.8, ls="-", label="Solid = S11"))
        if self._smith_s22_chk.isChecked():
            legend_handles.append(Line2D([0], [0], color="#444", lw=1.8, ls="--", label="Dashed = S22"))
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=6.6,
                framealpha=0.95,
                borderpad=0.35,
                handlelength=2.6,
            )

        if not any_trace:
            ax.text(
                0.5,
                0.5,
                "No enabled reflection trace",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#8a8f98",
                fontsize=9,
            )

    def _ensure_smith_hover_annot(self, ax):
        if self._smith_hover_ax is ax and self._smith_hover_annot is not None:
            return self._smith_hover_annot
        if self._smith_hover_annot is not None:
            try:
                self._smith_hover_annot.remove()
            except Exception:
                pass
        self._smith_hover_ax = ax
        self._smith_hover_annot = ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.35", fc="#1f2933", ec="#34495e", alpha=0.95),
            color="white",
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color="#34495e", lw=0.8),
        )
        self._smith_hover_annot.set_visible(False)
        return self._smith_hover_annot

    def _hide_smith_hover(self):
        if self._smith_hover_annot is not None and self._smith_hover_annot.get_visible():
            self._smith_hover_annot.set_visible(False)
            self._smith_canvas.draw_idle()

    def _on_smith_hover(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None or not self._smith_hover_points:
            self._hide_smith_hover()
            return
        best = None
        for trace_name, freq, gamma, z_val in self._smith_hover_points:
            dist = (gamma.real - event.xdata) ** 2 + (gamma.imag - event.ydata) ** 2
            if best is None or dist < best[0]:
                best = (dist, trace_name, freq, gamma, z_val)
        if best is None or best[0] > 0.04 ** 2:
            self._hide_smith_hover()
            return
        _, trace_name, freq, gamma, z_val = best
        annot = self._ensure_smith_hover_annot(ax)
        annot.xy = (gamma.real, gamma.imag)
        annot.set_text(
            f"freq = {freq:.3f} GHz\n"
            f"{trace_name} = {gamma.real:.4f}{gamma.imag:+.4f}j\n"
            f"impedance = {z_val.real:.2f}{z_val.imag:+.2f}j Ω"
        )
        annot.set_visible(True)
        self._smith_canvas.draw_idle()

    def _redraw_error(self):
        fig = self._err_fig
        fig.clear()
        self._err_hover_series = {}
        self._err_hover_ax = None
        self._err_hover_annot = None
        self._update_master_legend_labels()

        if self._inv_data is None or self._sim_data is None:
            self._draw_empty(fig, self._err_canvas, ["S11 err%", "S22 err%", "S21 err%", "S12 err%"], 1)
            return

        try:
            p1_z0 = _parse_z0_input(self._p1_z0_edit.text())
            p2_z0 = _parse_z0_input(self._p2_z0_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Impedance", str(exc))
            return

        sim_data = _renormalize_dataset(self._sim_data, p1_z0, p2_z0)
        inv_data = _renormalize_dataset(self._inv_data, p1_z0, p2_z0)
        meas_data = _renormalize_dataset(self._meas_data, p1_z0, p2_z0)

        fl, fh = self._get_fl_fh()
        freqs = sim_data[0]
        mask = (freqs >= fl) & (freqs <= fh)
        if not mask.any():
            self._draw_empty(fig, self._err_canvas, ["S11 err%", "S22 err%", "S21 err%", "S12 err%"], 1)
            return

        f_plot = freqs[mask]
        ylo, yhi, yticks = self._get_error_y_range()
        grid = gridspec.GridSpec(
            4, 1, figure=fig, hspace=0.60, top=0.935, bottom=0.100, left=0.155, right=0.960
        )

        shared_ax = None
        for index, (name, getter) in enumerate(_S_PARAMS):
            ax = fig.add_subplot(grid[index], sharex=shared_ax)
            if shared_ax is None:
                shared_ax = ax
            show_x = index == len(_S_PARAMS) - 1
            metric_label = "Z Error (%)" if name in {"S11", "S22"} else "V Ratio Error (%)"
            ax.set_title(f"{name} {metric_label}", fontsize=8.5, fontweight="bold", pad=6)
            ax.set_ylabel("%", fontsize=7)
            ax.set_xlabel("Freq (GHz)", fontsize=7)   # 수정: 모든 서브플롯에 x축 라벨 표시
            ax.tick_params(axis="both", labelsize=6, pad=2)
            ax.tick_params(axis="x", labelbottom=True)   # 수정: 모든 서브플롯에 x 눈금값 표시
            ax.grid(True, alpha=0.3, lw=0.5)
            ax.axhline(0.0, color="black", lw=0.9, alpha=0.6)
            _set_freq_ticks_points(ax, fl, fh, self._get_tick_points())
            ax.margins(x=0.01)

            sim_trace = getter(sim_data)[mask]
            inv_trace = getter(inv_data)[mask]
            if name == "S11":
                sim_metric = np.abs(_gamma_to_impedance(sim_trace, p1_z0))
                inv_metric = np.abs(_gamma_to_impedance(inv_trace, p1_z0))
            elif name == "S22":
                sim_metric = np.abs(_gamma_to_impedance(sim_trace, p2_z0))
                inv_metric = np.abs(_gamma_to_impedance(inv_trace, p2_z0))
            else:
                sim_metric = np.abs(sim_trace)
                inv_metric = np.abs(inv_trace)
            err_sim_inv = _relative_percent_error(sim_metric, inv_metric)
            ax.plot(f_plot, err_sim_inv, color=_COLORS["inv"], lw=1.3, label="Sim vs Inv", alpha=0.9)
            series = [("Sim vs Inv", f_plot, err_sim_inv)]

            if meas_data is not None:
                meas_trace = getter(meas_data)[mask]
                if name == "S11":
                    meas_metric = np.abs(_gamma_to_impedance(meas_trace, p1_z0))
                elif name == "S22":
                    meas_metric = np.abs(_gamma_to_impedance(meas_trace, p2_z0))
                else:
                    meas_metric = np.abs(meas_trace)
                err_meas_inv = _relative_percent_error(meas_metric, inv_metric)
                ax.plot(
                    f_plot,
                    err_meas_inv,
                    color=_COLORS["meas"],
                    lw=1.3,
                    ls="--",
                    label="Meas vs Inv",
                    alpha=0.9,
                )
                series.append(("Meas vs Inv", f_plot, err_meas_inv))

            ax.set_ylim(ylo, yhi)
            if yticks >= 2:
                ax.set_yticks(np.linspace(ylo, yhi, yticks))
            ax.legend(loc="upper right", fontsize=6, framealpha=0.92, borderpad=0.3, handlelength=2.4)
            self._err_hover_series[ax] = series

        self._err_canvas.draw()

    def _ensure_error_hover_annot(self, ax):
        if self._err_hover_ax is ax and self._err_hover_annot is not None:
            return self._err_hover_annot

        if self._err_hover_annot is not None:
            try:
                self._err_hover_annot.remove()
            except Exception:
                pass

        self._err_hover_ax = ax
        self._err_hover_annot = ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.35", fc="#1f2933", ec="#34495e", alpha=0.95),
            color="white",
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color="#34495e", lw=0.8),
        )
        self._err_hover_annot.set_visible(False)
        return self._err_hover_annot

    def _hide_error_hover(self, _event=None):
        if self._err_hover_annot is not None and self._err_hover_annot.get_visible():
            self._err_hover_annot.set_visible(False)
            self._err_canvas.draw_idle()

    def _on_error_hover(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None or ax not in self._err_hover_series:
            self._hide_error_hover()
            return

        series = self._err_hover_series.get(ax, [])
        if not series:
            self._hide_error_hover()
            return

        x_values = series[0][1]
        if len(x_values) == 0:
            self._hide_error_hover()
            return

        idx = int(np.argmin(np.abs(x_values - event.xdata)))
        hover_x = float(x_values[idx])
        text_lines = [f"Freq: {hover_x:.3f} GHz"]
        anchor_y = None
        for label, _, y_values in series:
            hover_y = float(y_values[idx])
            text_lines.append(f"{label}: {hover_y:.2f}%")
            if anchor_y is None:
                anchor_y = hover_y

        if anchor_y is None:
            self._hide_error_hover()
            return

        annot = self._ensure_error_hover_annot(ax)
        annot.xy = (hover_x, anchor_y)
        annot.set_text("\n".join(text_lines))
        annot.set_visible(True)
        self._err_canvas.draw_idle()


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SParamCompareGUI()
    window.setWindowTitle("S-Param Compare")
    window.resize(1400, 800)
    window.show()
    sys.exit(app.exec_())
