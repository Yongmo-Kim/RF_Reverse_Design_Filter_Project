# -*- coding: utf-8 -*-
"""
Dataset Smith sampler GUI.

- .s2p folder mode
- preprocessed s_params.npz mode
- random sampling or spec-filtered sampling
- Smith chart overlay for S11 / S22
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import skrf as rf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


DEFAULT_Z0 = 50.0 + 0.0j


@dataclass
class SampleRecord:
    name: str
    freqs_ghz: np.ndarray
    s11: np.ndarray
    s12: np.ndarray
    s21: np.ndarray
    s22: np.ndarray


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
    return complex(text)


def format_impedance(z: complex) -> str:
    return f"{z.real:.3g}{z.imag:+.3g}jΩ"


def find_s2p_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.s2p") if p.is_file()])


def complex_from_db_deg(db: np.ndarray, deg: np.ndarray) -> np.ndarray:
    mag = 10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)
    phase = np.radians(np.asarray(deg, dtype=np.float64))
    return mag * np.exp(1j * phase)


def sample_passes_spec(
    sample: SampleRecord,
    f_low: float,
    f_high: float,
    il_db: float,
    rl_db: float,
    use_s12: bool,
) -> bool:
    mask = (sample.freqs_ghz >= f_low) & (sample.freqs_ghz <= f_high)
    if not np.any(mask):
        return False

    s11_db = 20.0 * np.log10(np.maximum(np.abs(sample.s11), 1e-15))
    s22_db = 20.0 * np.log10(np.maximum(np.abs(sample.s22), 1e-15))
    s21_db = 20.0 * np.log10(np.maximum(np.abs(sample.s21), 1e-15))
    s12_db = 20.0 * np.log10(np.maximum(np.abs(sample.s12), 1e-15))

    rl_ok = bool(np.all(s11_db[mask] <= rl_db) and np.all(s22_db[mask] <= rl_db))
    il_ok = bool(np.all(s21_db[mask] >= il_db))
    if use_s12:
        il_ok = il_ok and bool(np.all(s12_db[mask] >= il_db))
    return rl_ok and il_ok


def renormalize_sample(sample: SampleRecord, z01: complex, z02: complex) -> SampleRecord:
    n = len(sample.freqs_ghz)
    s_old = np.zeros((n, 2, 2), dtype=np.complex128)
    s_old[:, 0, 0] = sample.s11
    s_old[:, 0, 1] = sample.s12
    s_old[:, 1, 0] = sample.s21
    s_old[:, 1, 1] = sample.s22
    z_old = np.column_stack([
        np.full(n, 50.0 + 0.0j, dtype=np.complex128),
        np.full(n, 50.0 + 0.0j, dtype=np.complex128),
    ])
    z_new = np.column_stack([
        np.full(n, complex(z01), dtype=np.complex128),
        np.full(n, complex(z02), dtype=np.complex128),
    ])
    s_new = rf.network.renormalize_s(s_old, z_old=z_old, z_new=z_new, s_def="power", s_def_old="power")
    return SampleRecord(
        name=sample.name,
        freqs_ghz=sample.freqs_ghz.copy(),
        s11=np.asarray(s_new[:, 0, 0], dtype=np.complex128),
        s12=np.asarray(s_new[:, 0, 1], dtype=np.complex128),
        s21=np.asarray(s_new[:, 1, 0], dtype=np.complex128),
        s22=np.asarray(s_new[:, 1, 1], dtype=np.complex128),
    )


def load_s2p_dataset(root: Path) -> tuple[list[SampleRecord], list[str]]:
    records: list[SampleRecord] = []
    failed: list[str] = []
    for path in find_s2p_files(root):
        try:
            nw = rf.Network(str(path))
            records.append(
                SampleRecord(
                    name=str(path),
                    freqs_ghz=np.asarray(nw.f / 1e9, dtype=np.float64),
                    s11=np.asarray(nw.s[:, 0, 0], dtype=np.complex128),
                    s12=np.asarray(nw.s[:, 0, 1], dtype=np.complex128),
                    s21=np.asarray(nw.s[:, 1, 0], dtype=np.complex128),
                    s22=np.asarray(nw.s[:, 1, 1], dtype=np.complex128),
                )
            )
        except Exception as exc:
            failed.append(f"{path.name}: {exc}")
    return records, failed


def load_preprocessed_dataset(sparams_path: Path, freqs_path: Path | None = None) -> list[SampleRecord]:
    s_data = np.load(sparams_path, allow_pickle=True)
    if "frequencies" in s_data.files:
        freqs = np.asarray(s_data["frequencies"], dtype=np.float64).reshape(-1)  # 변경: npz 내부 frequencies가 있으면 별도 npy 없이 바로 사용
    elif freqs_path is not None:
        freqs = np.asarray(np.load(freqs_path), dtype=np.float64).reshape(-1)  # 변경: 내부 frequencies가 없을 때만 외부 frequencies.npy 사용
    else:
        raise KeyError("s_params.npz 안에 'frequencies' 키가 없고 frequencies.npy도 선택되지 않았습니다.")  # 변경: 두 경로 모두 없을 때만 명확한 오류 반환
    required = ["s11_db", "s12_db", "s21_db", "s22_db", "s11_deg", "s12_deg", "s21_deg", "s22_deg"]
    missing = [key for key in required if key not in s_data.files]
    if missing:
        raise KeyError(f"s_params.npz missing keys: {missing}")

    channel_map = {key: np.asarray(s_data[key], dtype=np.float64) for key in required}
    n_samples = channel_map["s11_db"].shape[0]
    n_freqs = channel_map["s11_db"].shape[1]
    if len(freqs) != n_freqs:
        raise ValueError(f"frequencies length mismatch: frequencies={len(freqs)}, channels={n_freqs}")

    records: list[SampleRecord] = []
    for idx in range(n_samples):
        records.append(
            SampleRecord(
                name=f"{sparams_path.name}::sample_{idx}",
                freqs_ghz=freqs.copy(),
                s11=complex_from_db_deg(channel_map["s11_db"][idx], channel_map["s11_deg"][idx]),
                s12=complex_from_db_deg(channel_map["s12_db"][idx], channel_map["s12_deg"][idx]),
                s21=complex_from_db_deg(channel_map["s21_db"][idx], channel_map["s21_deg"][idx]),
                s22=complex_from_db_deg(channel_map["s22_db"][idx], channel_map["s22_deg"][idx]),
            )
        )
    return records


class SmithSamplerCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 8), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def redraw(
        self,
        samples: list[SampleRecord],
        z01: complex,
        z02: complex,
        f_start_ghz: float,
        f_end_ghz: float,
        show_s11: bool,
        show_s22: bool,
    ) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_aspect("equal")

        rf.plotting.smith(
            ax=ax,
            draw_labels=True,
            chart_type="z",
            ref_imm=1.0,
            draw_vswr=False,
        )

        total_s11_pts = 0
        total_s22_pts = 0
        for sample in samples:
            mask = (sample.freqs_ghz >= f_start_ghz) & (sample.freqs_ghz <= f_end_ghz)
            if not np.any(mask):
                continue
            renorm = renormalize_sample(sample, z01, z02)
            alpha = 0.12 if len(samples) >= 40 else 0.22
            lw = 0.8 if len(samples) >= 40 else 1.0
            if show_s11:
                trace = renorm.s11[mask]
                total_s11_pts += len(trace)
                ax.plot(trace.real, trace.imag, color="#1f77b4", alpha=alpha, lw=lw)
            if show_s22:
                trace = renorm.s22[mask]
                total_s22_pts += len(trace)
                ax.plot(trace.real, trace.imag, color="#d62728", alpha=alpha, lw=lw)

        legend_items = []
        if show_s11:
            legend_items.append("S11")
        if show_s22:
            legend_items.append("S22")
        legend_text = " / ".join(legend_items) if legend_items else "None"

        ax.set_title(
            f"Dataset Smith Overlay\n"
            f"Shown samples = {len(samples)} | Freq = {f_start_ghz:.4f} ~ {f_end_ghz:.4f} GHz | "
            f"Port1 Z0 = {format_impedance(z01)} | Port2 Z0 = {format_impedance(z02)}\n"
            f"Visible traces = {legend_text} | Total pts: S11={total_s11_pts}, S22={total_s22_pts}",
            fontsize=10,
        )
        self.draw_idle()


class DatasetSmithSamplerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Smith Sampler")
        self.resize(1640, 1000)

        self.dataset_records: list[SampleRecord] = []
        self.dataset_failed: list[str] = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_panel.setFixedWidth(380)

        source_group = QGroupBox("Dataset Source")
        source_layout = QVBoxLayout(source_group)
        self.source_label = QLabel("No dataset selected")
        self.source_label.setWordWrap(True)
        source_layout.addWidget(self.source_label)
        source_btn_row = QHBoxLayout()
        self.select_s2p_folder_btn = QPushButton("Select s2p Folder")
        self.select_s2p_folder_btn.clicked.connect(self._select_s2p_folder)
        source_btn_row.addWidget(self.select_s2p_folder_btn)
        self.select_npz_btn = QPushButton("Select s_params.npz")
        self.select_npz_btn.clicked.connect(self._select_preprocessed_npz)
        source_btn_row.addWidget(self.select_npz_btn)
        source_layout.addLayout(source_btn_row)
        left_layout.addWidget(source_group)

        filter_group = QGroupBox("Sampling / Filter")
        filter_form = QFormLayout(filter_group)
        self.sample_count_spin = QSpinBox()
        self.sample_count_spin.setRange(1, 5000)
        self.sample_count_spin.setValue(100)
        filter_form.addRow("Sample Count", self.sample_count_spin)

        self.seed_edit = QLineEdit("")
        self.seed_edit.setPlaceholderText("empty = random")
        filter_form.addRow("Seed", self.seed_edit)

        self.fl_edit = QLineEdit("6.6")
        self.fh_edit = QLineEdit("7.4")
        self.il_edit = QLineEdit("-1.5")
        self.rl_edit = QLineEdit("-15.0")
        filter_form.addRow("Plot / Filter FL", self.fl_edit)
        filter_form.addRow("Plot / Filter FH", self.fh_edit)
        filter_form.addRow("Insertion Loss (dB)", self.il_edit)
        filter_form.addRow("Return Loss (dB)", self.rl_edit)

        self.filter_spec_chk = QCheckBox("Only samples satisfying spec")
        self.filter_spec_chk.setChecked(True)
        filter_form.addRow("Spec Filter", self.filter_spec_chk)

        self.use_s12_chk = QCheckBox("Also require S12 >= IL")
        self.use_s12_chk.setChecked(True)
        filter_form.addRow("Transmission Rule", self.use_s12_chk)

        self.p1_edit = QLineEdit("50+0j")
        self.p2_edit = QLineEdit("50+0j")
        filter_form.addRow("P1 Z0", self.p1_edit)
        filter_form.addRow("P2 Z0", self.p2_edit)

        self.show_s11_chk = QCheckBox("Show S11")
        self.show_s11_chk.setChecked(True)
        self.show_s22_chk = QCheckBox("Show S22")
        self.show_s22_chk.setChecked(True)
        trace_row = QHBoxLayout()
        trace_row.addWidget(self.show_s11_chk)
        trace_row.addWidget(self.show_s22_chk)
        filter_form.addRow("Visible Traces", trace_row)
        left_layout.addWidget(filter_group)

        button_row = QHBoxLayout()
        self.plot_btn = QPushButton("Filter / Sample / Plot")
        self.plot_btn.clicked.connect(self._sample_and_plot)
        button_row.addWidget(self.plot_btn)
        self.resample_btn = QPushButton("Resample")
        self.resample_btn.clicked.connect(self._sample_and_plot)
        button_row.addWidget(self.resample_btn)
        left_layout.addLayout(button_row)

        info_group = QGroupBox("Info")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("Load dataset first. Then filter and overlay Smith traces.")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        left_layout.addWidget(info_group)

        sample_group = QGroupBox("Selected Samples")
        sample_layout = QVBoxLayout(sample_group)
        self.sample_text = QTextEdit()
        self.sample_text.setReadOnly(True)
        sample_layout.addWidget(self.sample_text)
        left_layout.addWidget(sample_group, stretch=1)

        root.addWidget(left_panel)

        chart_group = QGroupBox("Smith Chart")
        chart_layout = QVBoxLayout(chart_group)
        chart_layout.setContentsMargins(4, 4, 4, 4)
        self.canvas = SmithSamplerCanvas()
        chart_layout.addWidget(self.canvas)
        root.addWidget(chart_group, stretch=1)

    def _select_s2p_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing s2p files")
        if not folder:
            return
        root = Path(folder)
        records, failed = load_s2p_dataset(root)
        self.dataset_records = records
        self.dataset_failed = failed
        self.source_label.setText(str(root))
        self.info_label.setText(
            f"Source: s2p folder\n"
            f"Loaded records: {len(records)}\n"
            f"Load failures: {len(failed)}"
        )
        if not records:
            QMessageBox.warning(self, "No data", "No usable .s2p records were loaded from the selected folder.")

    def _select_preprocessed_npz(self) -> None:
        sparams_file, _ = QFileDialog.getOpenFileName(self, "Select s_params.npz", "", "NumPy files (*.npz)")
        if not sparams_file:
            return
        try:
            records = load_preprocessed_dataset(Path(sparams_file), None)  # 변경: 먼저 단일 npz만으로 로드를 시도
        except Exception as exc:
            reply = QMessageBox.question(
                self,
                "Select frequencies.npy",
                f"npz 단일 파일 로드에 실패했습니다.\n\n{exc}\n\nfrequencies.npy를 추가로 선택할까요?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )  # 변경: 내부 frequencies가 없는 구버전 파일도 계속 지원하기 위해 선택적으로 npy를 추가 로드
            if reply != QMessageBox.Yes:
                return
            freqs_file, _ = QFileDialog.getOpenFileName(self, "Select frequencies.npy", str(Path(sparams_file).parent), "NumPy files (*.npy)")
            if not freqs_file:
                return
            try:
                records = load_preprocessed_dataset(Path(sparams_file), Path(freqs_file))  # 변경: 구버전 전처리 파일 호환용 fallback
            except Exception as exc2:
                QMessageBox.warning(self, "Load failed", str(exc2))
                return
        self.dataset_records = records
        self.dataset_failed = []
        self.source_label.setText(str(sparams_file))  # 변경: frequencies가 npz 내부에 있으면 단일 파일만 표시
        self.info_label.setText(
            f"Source: preprocessed npz\n"
            f"Loaded records: {len(records)}\n"
            f"Mode: uses embedded frequencies when available"
        )

    def _sample_and_plot(self) -> None:
        if not self.dataset_records:
            QMessageBox.warning(self, "No dataset", "Please load s2p folder or preprocessed npz first.")
            return

        try:
            fl = float(self.fl_edit.text())
            fh = float(self.fh_edit.text())
            il = float(self.il_edit.text())
            rl = float(self.rl_edit.text())
            if fl > fh:
                raise ValueError("Plot FL must be smaller than or equal to Plot FH.")
            p1 = parse_z0_input(self.p1_edit.text())
            p2 = parse_z0_input(self.p2_edit.text())
        except Exception as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return

        seed_text = self.seed_edit.text().strip()
        if seed_text:
            try:
                rng = random.Random(int(seed_text))
            except ValueError:
                QMessageBox.warning(self, "Invalid seed", "Seed must be an integer.")
                return
        else:
            rng = random.Random()

        working_pool = self.dataset_records
        if self.filter_spec_chk.isChecked():
            working_pool = [
                rec for rec in self.dataset_records
                if sample_passes_spec(rec, fl, fh, il, rl, self.use_s12_chk.isChecked())
            ]

        if not working_pool:
            QMessageBox.warning(self, "No matching samples", "No samples matched the current spec filter.")
            self.sample_text.clear()
            return

        count = min(int(self.sample_count_spin.value()), len(working_pool))
        selected = rng.sample(working_pool, count)
        self.canvas.redraw(
            selected,
            p1,
            p2,
            fl,
            fh,
            self.show_s11_chk.isChecked(),
            self.show_s22_chk.isChecked(),
        )

        self.sample_text.setPlainText("\n".join(rec.name for rec in selected))
        mode_text = "filtered" if self.filter_spec_chk.isChecked() else "random all"
        self.info_label.setText(
            f"Dataset records: {len(self.dataset_records)}\n"
            f"Matched pool: {len(working_pool)} ({mode_text})\n"
            f"Selected to plot: {len(selected)}\n"
            f"Renorm: P1={format_impedance(p1)}, P2={format_impedance(p2)}\n"
            f"Visible: {'S11 ' if self.show_s11_chk.isChecked() else ''}{'S22' if self.show_s22_chk.isChecked() else ''}".strip() or "None"
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DatasetSmithSamplerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
