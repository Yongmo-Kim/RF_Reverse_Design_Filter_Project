#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
renorm_ops.py  ─  S-파라미터 임피던스 재정규화 모듈
====================================================
NumPy 버전 (시각화 / 후처리용)  +  PyTorch 미분가능 버전 (최적화 루프용)

[RF 이론 — Kurokawa Power-Wave 재정규화]
─────────────────────────────────────────
  실수 Z0_old 기준, 새 임피던스 Z0i_new (복소수 허용):

    Γi = (Z0i_new − Z0_old) / (Z0i_new + Z0_old)
    G  = diag(Γ1, Γ2)
    Di = √(Re(Z0i_new) / Z0_old) · |Z0_old + Z0i_new| / |Z0_old + Z0i_new*|
         ─ 실수 임피던스 한정: Di = √(Z0i_new / Z0_old), |…|/|…| = 1

    S_new = D (S − G*)(I − G S)⁻¹ D⁻¹          ... (★ 핵심 공식)

  이 공식은 impedance_test_gui.py 의 skrf.renormalize_s(s_def="power") 와 동일합니다.

[최소위상(Minimum-Phase) 가정 — 제한사항]
──────────────────────────────────────────
  Forward 모델은 dB 크기(magnitude)만 출력하므로 위상을 추정해야 합니다.
  Hilbert Transform 을 이용한 최소위상 추정은 다음 조건에서 유효합니다:

    ✅ S21 (전송계수): 수동 네트워크의 전송 경로는 대부분 최소위상에 근접
    ⚠️ S11/S22 (반사계수): 공진형 필터·매칭 네트워크에서 비-최소위상(Non-MP) 가능

  실용적 결론:
    - 최적화 루프에서의 위상 추정은 *근사값*입니다.
    - S21 기반 Loss 비중을 높게 유지하는 것이 수렴 안정성에 유리합니다.
    - EM 시뮬레이션 완료 후 실제 .s2p 파일로 impedance_test_gui.py 에서 반드시 검증!

[핵심 워크플로우]
──────────────────
  1. Forward 모델: 50Ω 기준 S11/S21/S22 [dB] 예측
  2. _torch_renormalize_s(): dB → 최소위상 복소 S → 재정규화 → 정규화된 Loss 텐서
  3. Loss 계산: 재정규화된 S-파라미터 vs 목표 스펙
  4. 역전파 → 레이아웃 최적화
  5. 최적화 완료 후: numpy_renormalize_s() 로 시각화 (Hilbert 추정 사용)
  6. 실제 검증: impedance_test_gui.py 에 .s2p 파일 로드 → 정확한 스미스 차트 확인
"""

from __future__ import annotations
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# § 1. NumPy 유틸 (시각화 / 후처리용)
# ─────────────────────────────────────────────────────────────────────────────

def _minimum_phase_np(ln_mag: np.ndarray) -> np.ndarray:
    """
    주파수 응답 로그크기(nepers) → 최소위상 추정 [radians].

    Mirror-padding 을 사용하여 Hilbert 변환의 끝점 아티팩트를 억제합니다.
    ln_mag : (N,)  자연로그 크기  [nepers, 즉 dB / 20 * ln(10)]
    """
    from scipy.signal import hilbert
    N = len(ln_mag)
    # 좌우 거울 패딩 → 주기적 연속성 보장
    padded = np.concatenate([ln_mag[::-1], ln_mag, ln_mag[::-1]])
    phase  = -np.imag(hilbert(padded))
    return phase[N : 2 * N]


def _db_to_complex_np(mag_db: np.ndarray) -> np.ndarray:
    """dB 크기 배열 → 최소위상 복소수 배열"""
    ln_mag = mag_db / 20.0 * np.log(10.0)      # dB → nepers
    phase  = _minimum_phase_np(ln_mag)
    return np.exp(ln_mag + 1j * phase)


def numpy_renormalize_s(
    s11_db: np.ndarray,
    s21_db: np.ndarray,
    s22_db: np.ndarray,
    z0_old: float  = 50.0,
    z01_new: complex = 50.0 + 0j,
    z02_new: complex = 50.0 + 0j,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NumPy S-파라미터 재정규화 (시각화 / 역설계 완료 후 확인용).

    입력 : S11/S21/S22 dB 배열  (N_freqs,)  — Forward 모델 예측 또는 실측값
    출력 : 재정규화된 S11/S21/S22 dB 배열  (N_freqs,)

    ⚠️ 실제 .s2p 파일이 있으면 impedance_test_gui.py 를 사용하세요 (위상 정확).
       이 함수는 최소위상 근사를 사용합니다.
    """
    N = len(s11_db)

    # ── 복소 S 행렬 재구성 ─────────────────────────────────────
    s11 = _db_to_complex_np(s11_db)
    s21 = _db_to_complex_np(s21_db)
    s12 = s21.copy()          # 상호성 네트워크 (reciprocal) 가정
    s22 = _db_to_complex_np(s22_db)

    # ── 재정규화 계수 ──────────────────────────────────────────
    g1 = (z01_new - z0_old) / (z01_new + z0_old)
    g2 = (z02_new - z0_old) / (z02_new + z0_old)

    # D 스케일링: Di = √(Re(Z_new_i)/Z_old) · |Z_old+Z_new_i|/|Z_old+Z_new_i*|
    def _d_factor(z_new: complex, z_old: float) -> complex:
        mag  = np.sqrt(max(z_new.real, 1e-12) / z_old)
        phase_corr = (z_old + z_new) / abs(z_old + z_new) if abs(z_old + z_new) > 1e-15 else 1.0
        return mag * phase_corr

    d1 = _d_factor(z01_new, z0_old)
    d2 = _d_factor(z02_new, z0_old)

    s11_new = np.zeros(N, dtype=complex)
    s21_new = np.zeros(N, dtype=complex)
    s22_new = np.zeros(N, dtype=complex)

    for i in range(N):
        S  = np.array([[s11[i], s12[i]],
                       [s21[i], s22[i]]], dtype=complex)
        G  = np.diag([g1, g2])
        Gc = np.conj(G)
        D  = np.diag([d1, d2])
        Di = np.diag([1.0 / d1, 1.0 / d2])

        # S_new = D (S − G*)(I − G S)⁻¹ D⁻¹
        A     = S - Gc
        B     = np.eye(2, dtype=complex) - G @ S
        S_new = D @ A @ np.linalg.solve(B, np.eye(2, dtype=complex)) @ Di

        s11_new[i] = S_new[0, 0]
        s21_new[i] = S_new[1, 0]
        s22_new[i] = S_new[1, 1]

    _to_db = lambda c: 20.0 * np.log10(np.abs(c) + 1e-15)
    return _to_db(s11_new), _to_db(s21_new), _to_db(s22_new)


# ─────────────────────────────────────────────────────────────────────────────
# § 2. PyTorch 미분가능 버전 (최적화 루프 내장용)
# ─────────────────────────────────────────────────────────────────────────────

def _minimum_phase_torch(ln_mag: torch.Tensor) -> torch.Tensor:
    """
    Hilbert Transform (torch.fft) 을 이용한 최소위상 추정.

    ln_mag  : (N_freqs,)  자연로그 크기 텐서  [nepers]
    반환    : (N_freqs,)  위상 텐서  [radians]
    """
    N = ln_mag.shape[0]

    # Mirror-padding → 주기적 연속성 보장
    padded = torch.cat([ln_mag.flip(0), ln_mag, ln_mag.flip(0)])   # (3N,)
    L = padded.shape[0]

    # FFT-based analytic signal: imaginary part = Hilbert transform of input
    Xf    = torch.fft.fft(padded)                                   # (3N,) complex
    h_win = torch.zeros(L, dtype=torch.float32, device=ln_mag.device)
    h_win[0]         = 1.0
    h_win[1:L//2]    = 2.0
    if L % 2 == 0:
        h_win[L//2]  = 1.0
    analytic_sig = torch.fft.ifft(Xf * h_win.to(Xf.dtype))        # (3N,) complex
    phase_padded = torch.imag(analytic_sig)                         # (3N,) real
    return -phase_padded[N : 2 * N]                                 # (N,)


def _db_to_complex_torch(mag_db: torch.Tensor) -> torch.Tensor:
    """
    (N_freqs,) dB 크기 텐서 → (N_freqs,) 최소위상 복소 텐서

    반환 dtype: torch.cfloat (complex64)
    """
    ln_mag = mag_db / 20.0 * np.log(10.0)        # dB → nepers
    phase  = _minimum_phase_torch(ln_mag.float()) # float32 전용
    return torch.polar(torch.exp(ln_mag.float()), phase)  # (N,) cfloat


def _renorm_s_matrix_torch(
    S_cplx : torch.Tensor,    # (N_freqs, 2, 2)  complex64
    z0_old : float,
    z01_new: complex,
    z02_new: complex,
    device : torch.device,
) -> torch.Tensor:
    """
    (N_freqs, 2, 2) 복소 S 행렬 텐서 → 재정규화된 (N_freqs, 2, 2) 복소 텐서.

    공식: S_new = D (S − G*)(I − G S)⁻¹ D⁻¹
    단, D = diag(d1, d2),  di = √(Re(Zi_new)/Z0_old) · (Z0_old+Zi_new)/|Z0_old+Zi_new|
    """
    dt = torch.complex64

    def _gamma(z_new: complex) -> complex:
        return (z_new - z0_old) / (z_new + z0_old)

    def _d(z_new: complex) -> complex:
        mag  = np.sqrt(max(z_new.real, 1e-12) / z0_old)
        denom = abs(z0_old + z_new)
        corr = (z0_old + z_new) / denom if denom > 1e-15 else 1.0
        return complex(mag) * corr

    g1, g2 = _gamma(z01_new), _gamma(z02_new)
    d1, d2 = _d(z01_new),     _d(z02_new)

    # 상수 행렬 구성 (브로드캐스트용, batch=N)
    G  = torch.zeros(2, 2, dtype=dt, device=device)
    G[0, 0] = g1
    G[1, 1] = g2
    Gc = torch.conj(G)                                          # (2, 2)

    D    = torch.zeros(2, 2, dtype=dt, device=device)
    D[0, 0] = d1
    D[1, 1] = d2
    D_inv    = torch.zeros(2, 2, dtype=dt, device=device)
    D_inv[0, 0] = 1.0 / d1
    D_inv[1, 1] = 1.0 / d2

    I2 = torch.eye(2, dtype=dt, device=device)                  # (2, 2)

    # Vectorized over N_freqs
    N = S_cplx.shape[0]
    G_b  = G.unsqueeze(0).expand(N, -1, -1)                    # (N, 2, 2)
    Gc_b = Gc.unsqueeze(0).expand(N, -1, -1)
    D_b  = D.unsqueeze(0).expand(N, -1, -1)
    Di_b = D_inv.unsqueeze(0).expand(N, -1, -1)
    I_b  = I2.unsqueeze(0).expand(N, -1, -1)

    A = S_cplx - Gc_b                                           # (N, 2, 2)
    B = I_b    - torch.bmm(G_b, S_cplx)                        # (N, 2, 2)

    # B⁻¹: torch.linalg.solve 는 미분 가능
    B_inv  = torch.linalg.solve(B, I_b)                        # (N, 2, 2)

    S_new  = torch.bmm(D_b, torch.bmm(A, torch.bmm(B_inv, Di_b)))  # (N, 2, 2)
    return S_new


# ─────────────────────────────────────────────────────────────────────────────
# § 3. InverseDesignWorker 에 삽입할 메서드
# ─────────────────────────────────────────────────────────────────────────────

class _RenormMixin:
    """
    InverseDesignWorker 에 mix-in 형태로 추가하세요.

    __init__ 에서 다음을 추가해야 합니다:
        self.p1_z0 = p1_z0   # complex, 예: 50+0j
        self.p2_z0 = p2_z0   # complex, 예: 100+0j
        self.z0_ref = 50.0   # Forward 모델 기준 임피던스

    사용법:
        pred_renorm = self._torch_renormalize_s(pred_norm)
        loss = asymmetric_physics_loss(pred_renorm, target_norm)
    """

    def _torch_renormalize_s(
        self,
        pred_norm: torch.Tensor,            # (B, 3·N_freqs)  정규화된 예측값
    ) -> torch.Tensor:
        """
        정규화된 예측 텐서 → (p1_z0, p2_z0) 기준으로 재정규화 → 다시 정규화된 텐서 반환.

        파이프라인:
          [0,1] 정규화값 → dB (역정규화) → 최소위상 복소 S → 재정규화 → dB → [0,1]

        반환: (B, 3·N_freqs)  재정규화 후 정규화된 텐서 (Loss 계산에 바로 사용 가능)
        """
        # 50Ω 기준과 같으면 그대로 (연산 스킵)
        if (abs(self.p1_z0 - self.z0_ref) < 1e-6 and
                abs(self.p2_z0 - self.z0_ref) < 1e-6):
            return pred_norm

        # ── 역정규화: [0,1] → dB ────────────────────────────────────
        n_freqs = self.n_freqs          # 91
        norm    = self.norm_params      # dict with s11_db, s21_db, s22_db

        def _denorm(t, key):
            p = norm[key]
            if p.get('type', 'minmax') == 'standard':
                clip = float(p.get('clip', 3.0))
                z    = t * (2.0 * clip) - clip
                return z * p['std'] + p['mean']
            return t * (p['max'] - p['min']) + p['min']

        def _renorm(t, key):
            p = norm[key]
            if p.get('type', 'minmax') == 'standard':
                clip = float(p.get('clip', 3.0))
                z    = (t - p['mean']) / (p['std'] + 1e-8)
                return (torch.clamp(z, -clip, clip) + clip) / (2.0 * clip)
            lo, hi = p['min'], p['max']
            return torch.clamp((t - lo) / (hi - lo + 1e-8), 0.0, 1.0)

        # pred_norm: (B, 3·N)
        s11_db = _denorm(pred_norm[:, :n_freqs],           's11_db')  # (B, N)
        s21_db = _denorm(pred_norm[:, n_freqs:2*n_freqs],  's21_db')
        s22_db = _denorm(pred_norm[:, 2*n_freqs:],         's22_db')

        B     = pred_norm.shape[0]
        dev   = pred_norm.device
        dt    = torch.complex64

        s11_new_list, s21_new_list, s22_new_list = [], [], []

        for b in range(B):
            # ── dB → 최소위상 복소 S 행렬 (N, 2, 2) ──────────────
            S11_c = _db_to_complex_torch(s11_db[b])     # (N,)
            S21_c = _db_to_complex_torch(s21_db[b])
            S22_c = _db_to_complex_torch(s22_db[b])
            S12_c = S21_c.clone()                        # 상호성 가정

            # (N, 2, 2) 행렬 조립
            row0  = torch.stack([S11_c, S12_c], dim=-1)  # (N, 2)
            row1  = torch.stack([S21_c, S22_c], dim=-1)
            S_cplx = torch.stack([row0, row1], dim=-2)   # (N, 2, 2)

            # ── 재정규화 ──────────────────────────────────────────
            S_new = _renorm_s_matrix_torch(
                S_cplx, self.z0_ref, self.p1_z0, self.p2_z0, dev)

            # ── 크기 → dB ─────────────────────────────────────────
            eps = 1e-7
            s11_new_db = 20.0 * torch.log10(S_new[:, 0, 0].abs().clamp(min=eps))
            s21_new_db = 20.0 * torch.log10(S_new[:, 1, 0].abs().clamp(min=eps))
            s22_new_db = 20.0 * torch.log10(S_new[:, 1, 1].abs().clamp(min=eps))

            s11_new_list.append(s11_new_db)
            s21_new_list.append(s21_new_db)
            s22_new_list.append(s22_new_db)

        s11_t = torch.stack(s11_new_list)   # (B, N)
        s21_t = torch.stack(s21_new_list)
        s22_t = torch.stack(s22_new_list)

        # ── 재정규화 → [0,1] ──────────────────────────────────────
        out = torch.cat([
            _renorm(s11_t, 's11_db'),
            _renorm(s21_t, 's21_db'),
            _renorm(s22_t, 's22_db'),
        ], dim=1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# § 4. GUI 입력 파싱 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def parse_z0_input(text: str) -> complex:
    """
    GUI QLineEdit 입력 문자열 → complex 파싱.

    허용 형식:
      "50"         → 50+0j
      "50+0j"      → 50+0j
      "50+10j"     → 50+10j
      "100-5j"     → 100−5j
      "0+50j"      → 50j  (순수 리액턴스)
    """
    text = text.strip().replace(' ', '')
    try:
        z = complex(text)
    except ValueError:
        # "50+10j" → Python complex() 는 이미 처리 가능,
        # 혹시 "+j" 없이 입력되는 경우 대비
        try:
            z = complex(float(text), 0.0)
        except ValueError:
            raise ValueError(f"임피던스 파싱 실패: '{text}'  (예: 50, 50+0j, 100-5j)")

    if z.real <= 0:
        raise ValueError(f"임피던스 실수부는 양수여야 합니다: {z}")
    return z