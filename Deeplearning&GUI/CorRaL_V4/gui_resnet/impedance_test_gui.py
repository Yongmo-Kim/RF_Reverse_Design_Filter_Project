#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Required packages:
#   pip install PyQt5 matplotlib scikit-rf numpy

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import skrf as rf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QGraphicsOpacityEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


S2P_PATH = Path(r"C:\Users\User\OneDrive - 금오공과대학교\바탕 화면\zxcv\4.s2p")
DEFAULT_Z0 = 50.0 + 0.0j


def format_freq_value(freq_ghz: float) -> str:
    return f"{freq_ghz:.6f}".rstrip("0").rstrip(".")


def gamma_to_db_angle(gamma: complex) -> str:
    mag_db = 20.0 * np.log10(max(abs(gamma), 1e-15))
    ang_deg = np.degrees(np.angle(gamma))
    return f"{mag_db:.2f} dB / {ang_deg:.2f} deg"


def format_impedance(z: complex) -> str:
    imag_prefix = "+j" if z.imag >= 0 else "-j"
    return f"{z.real:.3g} {imag_prefix}{abs(z.imag):.3g} ohm"


def parse_z0_input(text: str) -> complex:
    text = (text or "").strip().replace("Ω", "").replace("ohm", "").replace("Ohm", "")
    if not text:
        return DEFAULT_Z0
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
        return complex(text)
    except Exception as exc:
        raise ValueError(f"Invalid impedance: {text}") from exc


def renorm_s2p_ads_style(
    s_matrix_f: np.ndarray,
    z0_old_vec: np.ndarray,
    z0_new_vec: np.ndarray,
) -> np.ndarray:
    """Match ADS-style renormalization using power-wave S-parameters."""
    return rf.network.renormalize_s(
        s_matrix_f,
        z_old=z0_old_vec,
        z_new=z0_new_vec,
        s_def="power",
        s_def_old="power",
    )


class SmithCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 8), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._hover_annot = None
        self._hover_points = []
        self.mpl_connect("motion_notify_event", self._on_hover)
        self.mpl_connect("figure_leave_event", lambda _e: self._hide_hover())

    def redraw(
        self,
        network: rf.Network,
        z01: complex,
        z02: complex,
        mask: np.ndarray,
        f_start_ghz: float,
        f_end_ghz: float,
        visible_map: dict[str, bool] | None = None,
    ) -> None:
        self.figure.clear()
        # 변경: 재정규화/Z0 변경 등으로 redraw가 다시 호출될 때
        #       이전 차트의 stale hover 포인트가 누적되어 빈 배경에서도
        #       툴팁이 뜨는 버그가 있었음 → 매 redraw 시작 시 hover 상태 초기화
        self._hover_points = []
        if self._hover_annot is not None:
            try:
                self._hover_annot.set_visible(False)
            except Exception:
                pass
            self._hover_annot = None
        ax = self.figure.add_subplot(111)
        ax.set_aspect("equal")

        rf.plotting.smith(
            ax=ax,
            draw_labels=True,
            chart_type="z",
            ref_imm=1.0,
            draw_vswr=False,
        )

        s11 = network.s[mask, 0, 0]
        s12 = network.s[mask, 0, 1]
        s21 = network.s[mask, 1, 0]
        s22 = network.s[mask, 1, 1]
        visible_map = visible_map or {"S11": True, "S12": True, "S21": True, "S22": True}
        traces = [
            ("S11", s11, "#1f77b4"),
            ("S12", s12, "#2ca02c"),
            ("S21", s21, "#9467bd"),
            ("S22", s22, "#d62728"),
        ]
        for label, data, color in traces:
            if not visible_map.get(label, True):
                continue
            ax.plot(np.real(data), np.imag(data), color=color, lw=1.8, label=label)
            if len(data):
                ax.scatter(np.real(data), np.imag(data), color=color, s=16, alpha=0.85, zorder=4)
                ax.plot(np.real(data[0]), np.imag(data[0]), "o", color=color, ms=4)
                ax.plot(np.real(data[-1]), np.imag(data[-1]), "s", color=color, ms=5)
                z_series = None
                if label == "S11":
                    z_series = z01 * (1.0 + data) / np.where(np.abs(1.0 - data) < 1e-15, 1e-15 + 0j, 1.0 - data)
                elif label == "S22":
                    z_series = z02 * (1.0 + data) / np.where(np.abs(1.0 - data) < 1e-15, 1e-15 + 0j, 1.0 - data)
                for freq, gamma, z_val in zip(network.f[mask] / 1e9, data, z_series if z_series is not None else [None] * len(data)):
                    self._hover_points.append((label, float(freq), complex(gamma), None if z_val is None else complex(z_val), color))

        ax.set_title(
            f"Renormalized Smith Chart\n"
            f"Freq band = {f_start_ghz:.4f} ~ {f_end_ghz:.4f} GHz | "
            f"Port1 Z0 = {format_impedance(z01)}, "
            f"Port2 Z0 = {format_impedance(z02)}",
            fontsize=11,
        )
        if ax.lines:
            ax.legend(loc="upper right", fontsize=9)
        self._hover_annot = ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.35", fc="#1f2933", ec="#34495e", alpha=0.95),
            color="white",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#34495e", lw=0.8),
        )
        self._hover_annot.set_visible(False)
        self.draw_idle()

    def _hide_hover(self):
        if self._hover_annot is not None and self._hover_annot.get_visible():
            self._hover_annot.set_visible(False)
            self.draw_idle()

    def _on_hover(self, event):
        # 변경: hover annotation이 없는 상태(첫 redraw 직전 등)에서도 안전하게 종료
        if self._hover_annot is None or not self._hover_points:
            self._hide_hover()
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            self._hide_hover()
            return
        # 변경: Smith 단위원(|Γ|≤1) 밖이면 데이터가 있을 리 없으므로 즉시 숨김
        if event.xdata * event.xdata + event.ydata * event.ydata > 1.05 * 1.05:
            self._hide_hover()
            return
        best = None
        for label, freq, gamma, z_val, color in self._hover_points:
            dist = (gamma.real - event.xdata) ** 2 + (gamma.imag - event.ydata) ** 2
            if best is None or dist < best[0]:
                best = (dist, label, freq, gamma, z_val, color)
        # 변경: threshold를 0.04 → 0.025로 축소 (과거: 단위원의 ~4%, 현재: ~2.5%)
        #       데이터 점 바로 위에서만 툴팁이 뜨도록 강제
        if best is None or best[0] > 0.025 ** 2:
            self._hide_hover()
            return
        _, label, freq, gamma, z_val, _ = best
        lines = [
            f"freq = {freq:.3f} GHz",
            f"{label} = {gamma.real:.4f}{gamma.imag:+.4f}j",
        ]
        if z_val is not None:
            lines.append(f"impedance = {z_val.real:.2f}{z_val.imag:+.2f}j Ω")
        self._hover_annot.xy = (gamma.real, gamma.imag)
        self._hover_annot.set_text("\n".join(lines))
        self._hover_annot.set_visible(True)
        self.draw_idle()


class SParameterCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure()  # 변경: Inverse Design 탭과 같은 기본 Figure 크기 기준 사용
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def redraw(
        self,
        network: rf.Network,
        f_start_ghz: float,
        f_end_ghz: float,
        x_ticks: int,
        refl_y: tuple = (-30.0, 0.0, 7),
        trans_y: tuple = (-8.0, 0.0, 9),
    ) -> None:
        self.figure.clear()
        freqs = network.f / 1e9
        plot_mask = (freqs >= f_start_ghz) & (freqs <= f_end_ghz)
        pf = freqs[plot_mask]
        if pf.size == 0:
            self.draw_idle()
            return

        s11_db = 20.0 * np.log10(np.maximum(np.abs(network.s[plot_mask, 0, 0]), 1e-15))
        s12_db = 20.0 * np.log10(np.maximum(np.abs(network.s[plot_mask, 0, 1]), 1e-15))
        s21_db = 20.0 * np.log10(np.maximum(np.abs(network.s[plot_mask, 1, 0]), 1e-15))
        s22_db = 20.0 * np.log10(np.maximum(np.abs(network.s[plot_mask, 1, 1]), 1e-15))

        titles = [
            "S11 (Return Loss)",
            "S22 (Return Loss)",
            "S21 (Insertion Loss)",
            "S12 (Insertion Loss)",
        ]
        colors = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c"]
        series = [s11_db, s22_db, s21_db, s12_db]

        axes = [
            self.figure.add_subplot(4, 1, 1),
            self.figure.add_subplot(4, 1, 2),
            self.figure.add_subplot(4, 1, 3),
            self.figure.add_subplot(4, 1, 4),
        ]  # 변경: Inverse Design 탭과 같은 세로 4단 구성으로 정렬

        tick_count = max(2, int(x_ticks))
        xticks = np.linspace(f_start_ghz, f_end_ghz, tick_count)
        for ax, title, color, values in zip(axes, titles, colors, series):
            ax.plot(pf, values, color=color, lw=1.6)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlim(f_start_ghz, f_end_ghz)
            ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_xlabel("Freq (GHz)", fontsize=8)
            ax.set_ylabel("dB", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if "Return Loss" in title:
                # 변경: 사용자가 입력한 S11/S22 Y축 박스 적용
                _ylo, _yhi, _yticks = refl_y
                ax.set_ylim(_ylo, _yhi)
                if _yticks > 1:
                    ax.set_yticks(np.linspace(_ylo, _yhi, int(_yticks)))
            else:
                # 변경: 사용자가 입력한 S21/S12 Y축 박스 적용
                _ylo, _yhi, _yticks = trans_y
                ax.set_ylim(_ylo, _yhi)
                if _yticks > 1:
                    ax.set_yticks(np.linspace(_ylo, _yhi, int(_yticks)))

        self.figure.tight_layout(pad=0.6)  # 변경: Inverse Design 탭처럼 그래프 내부 여백만 정리하고 과도한 박스 확장을 줄임
        self.draw_idle()


class ImpedanceTestGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S-Parameter Impedance Renormalization Test")
        self.setMinimumSize(1240, 780)

        self.source_network: rf.Network | None = None
        self.current_network: rf.Network | None = None
        self.freq_ghz: np.ndarray = np.array([], dtype=float)
        self.source_path: Path | None = None
        self._smith_overlay_label = None  # 변경: 두 번째 탭 Smith 차트 워터마크 라벨
        self._smith_overlay_pixmap = None  # 변경: 현재 상태용 워터마크 픽스맵
        self._smith_overlay_center_pixmap = None  # 변경: 데이터 없을 때 중앙 대형 로고용 픽스맵
        self._smith_overlay_corner_pixmap = None  # 변경: 데이터 있을 때 우하단 소형 로고용 픽스맵
        self._smith_overlay_label = None  # 변경: 첫 번째 탭처럼 두 번째 탭도 스미스 차트 워터마크를 직접 관리
        self._smith_overlay_pixmap = None  # 변경: 리사이즈 시 재사용할 원본 이미지를 저장

        self._build_ui()
        self._setup_smith_overlay()  # 변경: 두 번째 탭 스미스 차트에도 동일한 사진 마크를 부착

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)  # 변경: Inverse Design 탭과 같은 외곽 여백 기준으로 정렬
        root.setSpacing(8)  # 변경: 좌우 패널 간격을 고정해 탭 전환 시 위치 차이를 줄임

        left_panel = QWidget()
        left_panel.setMinimumWidth(360)  # 변경: Inverse Design 탭과 동일한 왼쪽 패널 폭 확보
        left_panel.setMaximumWidth(360)  # 변경: Inverse Design 탭과 동일한 왼쪽 패널 폭 고정
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)  # 변경: 왼쪽 그룹들의 시작 x 위치를 탭 간 동일하게 맞춤
        left_layout.setSpacing(8)  # 변경: 그룹 간 세로 간격을 일정하게 맞춤

        file_group = QGroupBox("File")
        file_layout = QVBoxLayout()
        self.path_label = QLabel("No s2p file selected")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.file_button = QPushButton("Select s2p File")
        self.file_button.clicked.connect(self._choose_s2p_file)
        file_layout.addWidget(QLabel("Loaded s2p path"))
        file_layout.addWidget(self.path_label)
        file_layout.addWidget(self.file_button)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        z0_group = QGroupBox("Reference Impedance")
        z0_layout = QVBoxLayout()
        z0_layout.setSpacing(6)

        p1_row = QWidget()
        p1_row_lay = QHBoxLayout(p1_row)
        p1_row_lay.setContentsMargins(0, 0, 0, 0)
        p1_row_lay.setSpacing(4)
        p1_row_lay.addWidget(QLabel("P1 Z0 (Ω)"))
        self.p1_z0_edit = QLineEdit("50+0j")
        self.p1_z0_edit.setPlaceholderText("e.g. 100+25j")
        p1_row_lay.addWidget(self.p1_z0_edit)
        z0_layout.addWidget(p1_row)

        p2_row = QWidget()
        p2_row_lay = QHBoxLayout(p2_row)
        p2_row_lay.setContentsMargins(0, 0, 0, 0)
        p2_row_lay.setSpacing(4)
        p2_row_lay.addWidget(QLabel("P2 Z0 (Ω)"))
        self.p2_z0_edit = QLineEdit("50+0j")
        self.p2_z0_edit.setPlaceholderText("e.g. 50-25j")
        p2_row_lay.addWidget(self.p2_z0_edit)
        z0_layout.addWidget(p2_row)

        z0_group.setLayout(z0_layout)
        left_layout.addWidget(z0_group)

        self.update_button = QPushButton("Apply And Update Smith Chart")
        self.update_button.setMinimumHeight(44)
        self.update_button.clicked.connect(self._apply_update)
        left_layout.addWidget(self.update_button)

        # 변경: Plot X-Axis 컨트롤을 왼쪽 패널(Info 위)로 이동
        plot_axis_group = QGroupBox("Plot X-Axis")
        plot_axis_group.setMinimumWidth(230)
        plot_axis_layout = QGridLayout()
        plot_axis_layout.setContentsMargins(12, 12, 12, 10)
        plot_axis_layout.setHorizontalSpacing(10)
        plot_axis_layout.setVerticalSpacing(6)
        plot_axis_layout.setColumnStretch(0, 0)
        plot_axis_layout.setColumnStretch(1, 1)
        plot_axis_layout.setColumnMinimumWidth(0, 110)
        _pfl_lbl = QLabel("Plot FL (GHz)")
        _pfl_lbl.setMinimumWidth(110)
        plot_axis_layout.addWidget(_pfl_lbl, 0, 0)
        self.plot_fl_edit = QLineEdit("0.1")
        self.plot_fl_edit.setMinimumWidth(70)
        plot_axis_layout.addWidget(self.plot_fl_edit, 0, 1)
        _pfh_lbl = QLabel("Plot FH (GHz)")
        _pfh_lbl.setMinimumWidth(110)
        plot_axis_layout.addWidget(_pfh_lbl, 1, 0)
        self.plot_fh_edit = QLineEdit("10.0")
        self.plot_fh_edit.setMinimumWidth(70)
        plot_axis_layout.addWidget(self.plot_fh_edit, 1, 1)
        _ptk_lbl = QLabel("X-Axis Ticks")
        _ptk_lbl.setMinimumWidth(110)
        plot_axis_layout.addWidget(_ptk_lbl, 2, 0)
        self.plot_ticks_edit = QLineEdit("11")
        self.plot_ticks_edit.setMinimumWidth(70)
        plot_axis_layout.addWidget(self.plot_ticks_edit, 2, 1)
        self.plot_update_button = QPushButton("Apply Plot X-Axis")
        self.plot_update_button.setMinimumHeight(28)
        self.plot_update_button.clicked.connect(self._apply_update)
        plot_axis_layout.addWidget(self.plot_update_button, 3, 0, 1, 2)
        plot_axis_group.setLayout(plot_axis_layout)
        left_layout.addWidget(plot_axis_group)

        # 변경: S11/S22 와 S21/S12 Y축 박스를 좌우로 나란히 배치
        plot_y_pair_widget = QWidget()
        plot_y_pair_layout = QHBoxLayout(plot_y_pair_widget)
        plot_y_pair_layout.setContentsMargins(0, 0, 0, 0); plot_y_pair_layout.setSpacing(4)

        plot_y_refl_group = QGroupBox("Y-Axis (S11/S22)")
        plot_y_refl_layout = QGridLayout()
        plot_y_refl_layout.setContentsMargins(6, 6, 6, 6)
        plot_y_refl_layout.setHorizontalSpacing(2); plot_y_refl_layout.setVerticalSpacing(4)
        plot_y_refl_layout.setColumnStretch(0, 0); plot_y_refl_layout.setColumnStretch(1, 1)
        plot_y_refl_layout.addWidget(QLabel("min:"), 0, 0)
        self.y_lo_refl_edit = QLineEdit("-30"); self.y_lo_refl_edit.setMinimumWidth(40)
        plot_y_refl_layout.addWidget(self.y_lo_refl_edit, 0, 1)
        plot_y_refl_layout.addWidget(QLabel("max:"), 1, 0)
        self.y_hi_refl_edit = QLineEdit("0"); self.y_hi_refl_edit.setMinimumWidth(40)
        plot_y_refl_layout.addWidget(self.y_hi_refl_edit, 1, 1)
        plot_y_refl_layout.addWidget(QLabel("Ticks:"), 2, 0)
        self.y_ticks_refl_edit = QLineEdit("7"); self.y_ticks_refl_edit.setMinimumWidth(40)
        plot_y_refl_layout.addWidget(self.y_ticks_refl_edit, 2, 1)
        self.plot_y_refl_btn = QPushButton("Apply")
        self.plot_y_refl_btn.setMinimumHeight(26)
        self.plot_y_refl_btn.clicked.connect(self._apply_update)
        plot_y_refl_layout.addWidget(self.plot_y_refl_btn, 3, 0, 1, 2)
        plot_y_refl_group.setLayout(plot_y_refl_layout)
        plot_y_pair_layout.addWidget(plot_y_refl_group)

        plot_y_trans_group = QGroupBox("Y-Axis (S21/S12)")
        plot_y_trans_layout = QGridLayout()
        plot_y_trans_layout.setContentsMargins(6, 6, 6, 6)
        plot_y_trans_layout.setHorizontalSpacing(2); plot_y_trans_layout.setVerticalSpacing(4)
        plot_y_trans_layout.setColumnStretch(0, 0); plot_y_trans_layout.setColumnStretch(1, 1)
        plot_y_trans_layout.addWidget(QLabel("min:"), 0, 0)
        self.y_lo_trans_edit = QLineEdit("-8"); self.y_lo_trans_edit.setMinimumWidth(40)
        plot_y_trans_layout.addWidget(self.y_lo_trans_edit, 0, 1)
        plot_y_trans_layout.addWidget(QLabel("max:"), 1, 0)
        self.y_hi_trans_edit = QLineEdit("0"); self.y_hi_trans_edit.setMinimumWidth(40)
        plot_y_trans_layout.addWidget(self.y_hi_trans_edit, 1, 1)
        plot_y_trans_layout.addWidget(QLabel("Ticks:"), 2, 0)
        self.y_ticks_trans_edit = QLineEdit("9"); self.y_ticks_trans_edit.setMinimumWidth(40)
        plot_y_trans_layout.addWidget(self.y_ticks_trans_edit, 2, 1)
        self.plot_y_trans_btn = QPushButton("Apply")
        self.plot_y_trans_btn.setMinimumHeight(26)
        self.plot_y_trans_btn.clicked.connect(self._apply_update)
        plot_y_trans_layout.addWidget(self.plot_y_trans_btn, 3, 0, 1, 2)
        plot_y_trans_group.setLayout(plot_y_trans_layout)
        plot_y_pair_layout.addWidget(plot_y_trans_group)

        left_layout.addWidget(plot_y_pair_widget)

        info_group = QGroupBox("Info")
        info_layout = QFormLayout()
        self.status_value = QLabel("Loading...")
        self.freq_range_value = QLabel("-")
        self.points_value = QLabel("-")
        self.band_points_value = QLabel("-")
        self.s11_start_value = QLabel("-")
        self.s22_start_value = QLabel("-")
        self.s11_end_value = QLabel("-")
        self.s22_end_value = QLabel("-")
        info_layout.addRow("Status", self.status_value)
        info_layout.addRow("File range", self.freq_range_value)
        info_layout.addRow("Total points", self.points_value)
        info_layout.addRow("Band points", self.band_points_value)
        info_layout.addRow("S11 @ start", self.s11_start_value)
        info_layout.addRow("S22 @ start", self.s22_start_value)
        info_layout.addRow("S11 @ end", self.s11_end_value)
        info_layout.addRow("S22 @ end", self.s22_end_value)
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)
        left_layout.addStretch()

        root.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)  # 변경: 오른쪽 그래프 영역 시작 위치를 Inverse Design 탭과 맞춤
        right_layout.setSpacing(4)  # 변경: 상단 컨트롤과 그래프 사이 간격을 Inverse Design 탭과 맞춤
        # 변경: 상단 Plot X-Axis 컨트롤은 이제 왼쪽 Info 그룹 위로 이동 — 중복 생성 제거

        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        response_group = QGroupBox("S-Parameter Response")
        response_layout = QVBoxLayout()
        self.response_canvas = SParameterCanvas()
        response_layout.addWidget(self.response_canvas)
        response_group.setLayout(response_layout)
        content_layout.addWidget(response_group, stretch=7)  # 변경: Inverse Design 탭과 동일한 왼쪽 응답 그래프 비율 사용

        right_col_widget = QWidget()
        right_col_layout = QVBoxLayout(right_col_widget)
        right_col_layout.setContentsMargins(0, 0, 0, 0)
        right_col_layout.setSpacing(4)

        chart_group = QGroupBox("Smith Chart")
        chart_layout = QVBoxLayout()
        self.canvas = SmithCanvas()
        chart_layout.addWidget(self.canvas)
        self.smith_z0_label = QLabel("Port1 Z0 = 50Ω   |   Port2 Z0 = 50Ω")
        self.smith_z0_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.smith_z0_label.setStyleSheet(
            "font-size:10px; color:#555; border:1px solid #ccc; border-radius:3px; padding:3px 6px; background:#f8f8f8;"
        )
        chart_layout.addWidget(self.smith_z0_label)

        smith_toggle_widget = QWidget()
        smith_toggle = QHBoxLayout(smith_toggle_widget)
        smith_toggle.setContentsMargins(4, 2, 4, 2)
        smith_toggle.setSpacing(6)
        smith_toggle.addWidget(QLabel("Show:"))
        self.smith_trace_buttons = {}
        for name, color in [("S11", "#1f77b4"), ("S12", "#2ca02c"), ("S21", "#9467bd"), ("S22", "#d62728")]:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedWidth(48)
            btn.setStyleSheet(f"""
                QPushButton {{ border: 1px solid {color}; border-radius: 3px; background: #ffffff; font-size: 10px; padding: 2px 6px; }}
                QPushButton:checked {{ background: {color}; color: #ffffff; }}
                QPushButton:hover {{ background: #f3f6f9; }}
            """)
            btn.toggled.connect(self._refresh_smith_only)
            self.smith_trace_buttons[name] = btn
            smith_toggle.addWidget(btn)
        smith_toggle.addStretch()
        chart_layout.addWidget(smith_toggle_widget)
        chart_group.setLayout(chart_layout)
        right_col_layout.addWidget(chart_group, stretch=1)
        content_layout.addWidget(right_col_widget, stretch=13)  # 변경: Inverse Design 탭과 동일한 오른쪽 컬럼 비율 사용

        right_layout.addWidget(content_widget, stretch=1)
        root.addWidget(right_panel, stretch=1)

    def _picture_overlay_path(self) -> str:
        picture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picture")  # 변경: 친구 PC 절대경로 대신 현재 프로젝트 picture 폴더 사용
        return os.path.join(picture_dir, "KakaoTalk_20260325_165731666.png")  # 변경: 첫 번째 탭과 같은 이미지 자산을 사용

    def _setup_smith_overlay(self) -> None:
        try:
            img_path = self._picture_overlay_path()
            if not os.path.exists(img_path):
                return
            self._smith_overlay_pixmap = QPixmap(img_path)  # 변경: 첫 번째 탭처럼 스미스 차트 내부에 오버레이용 이미지를 로드
            if self._smith_overlay_pixmap.isNull():
                return
            self._smith_overlay_label = QLabel(self.canvas)  # 변경: 두 번째 탭의 실제 스미스 캔버스(self.canvas)에 직접 부착
            self._smith_overlay_label.setAlignment(Qt.AlignCenter)
            self._smith_overlay_label.setAttribute(Qt.WA_TransparentForMouseEvents)
            effect = QGraphicsOpacityEffect()
            effect.setOpacity(0.20)  # 변경: 그래프 가독성을 해치지 않도록 반투명 처리
            self._smith_overlay_label.setGraphicsEffect(effect)
            self.canvas.installEventFilter(self)  # 변경: 탭 크기 변경 시 워터마크 위치를 자동으로 다시 맞춤
            self._reposition_smith_overlay()
            self._smith_overlay_label.raise_()
        except Exception:
            self._smith_overlay_label = None  # 변경: 이미지 로드 실패 시 GUI 동작은 유지

    def _reposition_smith_overlay(self) -> None:
        if self._smith_overlay_label is None or self._smith_overlay_pixmap is None:
            return
        cw = self.canvas.width()
        ch = self.canvas.height()
        if cw < 20 or ch < 20:
            return
        size = max(16, int(min(cw, ch) * 0.22))  # 변경: 첫 번째 탭과 비슷한 체감 크기로 워터마크를 스케일
        scaled = self._smith_overlay_pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lw = scaled.width()
        lh = scaled.height()
        self._smith_overlay_label.setPixmap(scaled)
        self._smith_overlay_label.setGeometry(cw - lw - 6, ch - lh - 6, lw, lh)  # 변경: 스미스 차트 우하단에 고정 배치
        self._smith_overlay_label.raise_()

    def eventFilter(self, watched, event):
        try:
            if watched is self.canvas and event.type() in (QEvent.Resize, QEvent.Show):
                self._reposition_smith_overlay()  # 변경: 스미스 차트가 다시 그려질 때마다 사진 위치를 유지
        except Exception:
            pass
        return super().eventFilter(watched, event)

    def _choose_s2p_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select s2p File",
            str(Path.home()),
            "Touchstone Files (*.s2p);;All Files (*.*)",
        )
        if not selected:
            return
        self.source_path = Path(selected)
        self._load_network()
        self._set_default_band()

    def _load_network(self) -> None:
        path = self.source_path
        if path is None:
            raise FileNotFoundError("Please select an s2p file first.")
        if not path.exists():
            raise FileNotFoundError(f"s2p file not found: {path}")

        self.source_path = path
        self.source_network = rf.Network(str(path))
        self.freq_ghz = self.source_network.f / 1e9
        self.path_label.setText(str(path))
        self.status_value.setText("Loaded")
        self.freq_range_value.setText(
            f"{format_freq_value(float(self.freq_ghz[0]))} ~ {format_freq_value(float(self.freq_ghz[-1]))} GHz"
        )
        self.points_value.setText(str(len(self.freq_ghz)))

    def _set_default_band(self) -> None:
        if len(self.freq_ghz) == 0:
            return
        self.f_start_edit.setText(format_freq_value(float(self.freq_ghz[0])))  # 변경: 화면에는 0.1처럼 깔끔한 값으로 표시
        self.f_end_edit.setText(format_freq_value(float(self.freq_ghz[-1])))  # 변경: 화면에는 30처럼 깔끔한 값으로 표시
        self.plot_fl_edit.setText(format_freq_value(float(self.freq_ghz[0])))  # 변경: Plot FL도 동일한 표시 형식 사용
        self.plot_fh_edit.setText(format_freq_value(float(self.freq_ghz[-1])))  # 변경: Plot FH도 동일한 표시 형식 사용

    def _parse_complex_inputs(self) -> tuple[complex, complex]:
        try:
            z01 = parse_z0_input(self.p1_z0_edit.text())
            z02 = parse_z0_input(self.p2_z0_edit.text())
        except ValueError as exc:
            raise ValueError("Please enter valid values for P1 Z0 and P2 Z0.") from exc

        if z01.real <= 0 or z02.real <= 0:
            raise ValueError("The real part of both reference impedances must be positive.")

        return z01, z02

    def _parse_band(self) -> tuple[float, float, np.ndarray]:
        try:
            f_start = float(self.f_start_edit.text())
            f_end = float(self.f_end_edit.text())
        except ValueError as exc:
            raise ValueError("Please enter valid numeric values for the frequency band.") from exc

        if f_start > f_end:
            raise ValueError("Start frequency must be smaller than or equal to end frequency.")

        low = float(self.freq_ghz[0])  # 변경: 비교 기준을 지역 변수로 분리해 오차 보정에 사용
        high = float(self.freq_ghz[-1])  # 변경: 비교 기준을 지역 변수로 분리해 오차 보정에 사용
        tol = 1e-6  # 변경: s2p 주파수 반올림으로 생기는 미세 비교 오차 허용

        # 사용자 입력값 그대로 사용 (0 입력 시 축이 0부터 시작하도록)
        mask = (self.freq_ghz >= f_start) & (self.freq_ghz <= f_end)
        if not np.any(mask):
            raise ValueError("No frequency points exist inside the selected band.")

        return f_start, f_end, mask

    def _parse_plot_band(self) -> tuple[float, float, int]:
        try:
            f_start = float(self.plot_fl_edit.text())
            f_end = float(self.plot_fh_edit.text())
            x_ticks = int(self.plot_ticks_edit.text())
        except ValueError as exc:
            raise ValueError("Please enter valid numeric values for the plot range and tick count.") from exc

        if f_start > f_end:
            raise ValueError("Plot FL must be smaller than or equal to Plot FH.")

        # 사용자 입력값 그대로 사용
        return f_start, f_end, x_ticks

    def _update_band_info(self, mask: np.ndarray) -> None:
        if self.current_network is None:
            self.band_points_value.setText("-")
            self.s11_start_value.setText("-")
            self.s22_start_value.setText("-")
            self.s11_end_value.setText("-")
            self.s22_end_value.setText("-")
            return

        indices = np.flatnonzero(mask)
        start_idx = int(indices[0])
        end_idx = int(indices[-1])

        self.band_points_value.setText(str(len(indices)))
        self.s11_start_value.setText(gamma_to_db_angle(self.current_network.s[start_idx, 0, 0]))
        self.s22_start_value.setText(gamma_to_db_angle(self.current_network.s[start_idx, 1, 1]))
        self.s11_end_value.setText(gamma_to_db_angle(self.current_network.s[end_idx, 0, 0]))
        self.s22_end_value.setText(gamma_to_db_angle(self.current_network.s[end_idx, 1, 1]))

    def _smith_visible_map(self) -> dict[str, bool]:
        return {name: btn.isChecked() for name, btn in self.smith_trace_buttons.items()}

    def _refresh_smith_only(self) -> None:
        if self.current_network is None:
            return
        try:
            z01, z02 = self._parse_complex_inputs()
            f_start, f_end, mask = self._parse_band()
            self.canvas.redraw(self.current_network, z01, z02, mask, f_start, f_end, self._smith_visible_map())
            self._reposition_smith_overlay()  # 변경: 스미스 차트 redraw 뒤에도 워터마크가 계속 보이도록 재배치
            self.smith_z0_label.setText(
                f"Port1 Z0 = {format_impedance(z01)}   |   Port2 Z0 = {format_impedance(z02)}"
            )
        except Exception:
            pass

    def _apply_update(self) -> None:
        try:
            if self.source_network is None:
                self._load_network()

            z01, z02 = self._parse_complex_inputs()
            f_start, f_end, mask = self._parse_band()
            plot_f_start, plot_f_end, x_ticks = self._parse_plot_band()

            renorm = self.source_network.copy()
            z0_old = np.array([50.0 + 0.0j, 50.0 + 0.0j], dtype=complex)
            z0_new = np.array([z01, z02], dtype=complex)
            renorm.s = renorm_s2p_ads_style(renorm.s, z0_old, z0_new)
            renorm.z0 = np.tile(z0_new, (len(renorm.f), 1))
            self.current_network = renorm

            self.canvas.redraw(renorm, z01, z02, mask, f_start, f_end, self._smith_visible_map())
            self._reposition_smith_overlay()  # 변경: 전체 업데이트 후에도 사진 마크가 스미스 차트 위에 유지되도록 보강
            # 변경: S11/S22, S21/S12 별 Y축 박스 입력값 읽기 (실패 시 기본값)
            try:
                refl_y = (
                    float(self.y_lo_refl_edit.text()),
                    float(self.y_hi_refl_edit.text()),
                    int(self.y_ticks_refl_edit.text()),
                )
            except (ValueError, AttributeError):
                refl_y = (-30.0, 0.0, 7)
            try:
                trans_y = (
                    float(self.y_lo_trans_edit.text()),
                    float(self.y_hi_trans_edit.text()),
                    int(self.y_ticks_trans_edit.text()),
                )
            except (ValueError, AttributeError):
                trans_y = (-8.0, 0.0, 9)
            self.response_canvas.redraw(renorm, plot_f_start, plot_f_end, x_ticks,
                                        refl_y=refl_y, trans_y=trans_y)
            self.smith_z0_label.setText(
                f"Port1 Z0 = {format_impedance(z01)}   |   Port2 Z0 = {format_impedance(z02)}"
            )
            self._update_band_info(mask)
            self.status_value.setText(
                f"Updated | {f_start:.4f}-{f_end:.4f} GHz | "
                f"Z01={format_impedance(z01)}, Z02={format_impedance(z02)}"
            )
        except Exception as exc:
            self.status_value.setText("Error")
            QMessageBox.critical(self, "Renormalization Error", str(exc))

    def _picture_overlay_paths(self) -> tuple[str, str]:
        """빈 상태용 중앙 로고와 데이터 상태용 코너 로고 경로를 돌려준다."""  # 변경: 첫 번째 탭과 같은 상태 전환 로고 자산을 사용
        picture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picture")
        return (
            os.path.join(picture_dir, "KakaoTalk_20260325_165731666_01.png"),
            os.path.join(picture_dir, "KakaoTalk_20260325_165731666.png"),
        )

    def _setup_smith_overlay(self) -> None:
        """두 번째 탭 Smith 차트 로고를 초기화한다."""  # 변경: 빈 상태는 중앙 대형, 데이터 상태는 우하단 소형 로고로 동작하도록 초기화
        try:
            img_center, img_corner = self._picture_overlay_paths()
            if not (os.path.exists(img_center) and os.path.exists(img_corner)):
                return
            self._smith_overlay_center_pixmap = QPixmap(img_center)
            self._smith_overlay_corner_pixmap = QPixmap(img_corner)
            self._smith_overlay_pixmap = self._smith_overlay_center_pixmap
            if self._smith_overlay_center_pixmap.isNull() or self._smith_overlay_corner_pixmap.isNull():
                return
            self._smith_overlay_label = QLabel(self.canvas)
            self._smith_overlay_label.setAlignment(Qt.AlignCenter)
            self._smith_overlay_label.setAttribute(Qt.WA_TransparentForMouseEvents)
            effect = QGraphicsOpacityEffect()
            effect.setOpacity(0.20)
            self._smith_overlay_label.setGraphicsEffect(effect)
            self.canvas.installEventFilter(self)
            self._reposition_smith_overlay()
            self._smith_overlay_label.raise_()
        except Exception:
            self._smith_overlay_label = None

    def _reposition_smith_overlay(self) -> None:
        """두 번째 탭 Smith 차트 로고 위치와 크기를 상태에 맞게 갱신한다."""  # 변경: 데이터 없으면 중앙 크게, 있으면 우하단 작게 표시
        if self._smith_overlay_label is None or self._smith_overlay_center_pixmap is None or self._smith_overlay_corner_pixmap is None:
            return
        cw = self.canvas.width()
        ch = self.canvas.height()
        if cw < 20 or ch < 20:
            return
        has_data = self.current_network is not None
        self._smith_overlay_pixmap = self._smith_overlay_corner_pixmap if has_data else self._smith_overlay_center_pixmap
        size = max(16, int(min(cw, ch) * (0.22 if has_data else 0.72)))
        scaled = self._smith_overlay_pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lw = scaled.width()
        lh = scaled.height()
        self._smith_overlay_label.setPixmap(scaled)
        if has_data:
            x = cw - lw - 6
            y = ch - lh - 6
        else:
            x = (cw - lw) // 2
            y = (ch - lh) // 2
        self._smith_overlay_label.setGeometry(x, y, lw, lh)
        self._smith_overlay_label.raise_()


def _impedance_test_set_default_band(self) -> None:
    if len(self.freq_ghz) == 0:
        return
    self.plot_fl_edit.setText(format_freq_value(float(self.freq_ghz[0])))
    self.plot_fh_edit.setText(format_freq_value(float(self.freq_ghz[-1])))


def _impedance_test_parse_band(self) -> tuple[float, float, np.ndarray]:
    try:
        f_start = float(self.plot_fl_edit.text())
        f_end = float(self.plot_fh_edit.text())
    except ValueError as exc:
        raise ValueError("Please enter valid numeric values for Plot FL and Plot FH.") from exc

    if f_start > f_end:
        raise ValueError("Plot FL must be smaller than or equal to Plot FH.")

    # 사용자 입력값 그대로 사용 (0 입력 시 축이 0부터 시작하도록)
    mask = (self.freq_ghz >= f_start) & (self.freq_ghz <= f_end)
    if not np.any(mask):
        raise ValueError("No frequency points exist inside the selected band.")

    return f_start, f_end, mask


ImpedanceTestGUI._set_default_band = _impedance_test_set_default_band
ImpedanceTestGUI._parse_band = _impedance_test_parse_band


def main() -> None:
    app = QApplication(sys.argv)
    window = ImpedanceTestGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
