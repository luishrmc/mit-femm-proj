"""
Pós-processamento simplificado FEMM a partir de arquivos .ANS já calculados.

Objetivo exclusivo:
1) Abrir diretamente os arquivos .ANS dos três motores, sem gerar nova malha e sem rodar solver.
2) Amostrar a componente radial B_r no entreferro usando o ponto de referência FEMM (0, 57.25) mm.
3) Gerar a forma de onda de B_r(theta) no entreferro.
4) Gerar os gráficos da análise harmônica de B_r(theta), incluindo harmônicos de ranhura.
5) Salvar três imagens do pós-processador FEMM por motor:
   - linhas de fluxo sem mapa colorido;
   - mapa colorido |B| com linhas de fluxo;
   - mapa colorido |B| sem linhas de fluxo.
6) Gerar o espectro normalizado por fundamental B_n/B_3.
7) Gerar a forma de onda de B_r(theta) alinhada pela fase da fundamental.

Dependências:
    pip install pyfemm numpy matplotlib

Observação:
    Este script pressupõe que os arquivos .ANS já foram gerados com as correntes finais.
    Não há chamadas a mi_createmesh(), mi_analyze(), mi_modifycircprop() ou mi_loadsolution().
"""

from __future__ import annotations

import csv
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import femm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


# =============================================================================
# CONFIGURAÇÕES DO USUÁRIO
# =============================================================================

# Pasta raiz onde o script procurará automaticamente os arquivos .ANS,
# se ans_file=None em CASES.
ANS_SEARCH_ROOT = Path(r"C:\Users\lumac\ufmg\braz\conv-trab\motor\femm")

# Pasta de saída.
RESULTS_DIR = Path(r"C:\Users\lumac\ufmg\braz\conv-trab\src\sim\out")

# Ponto retirado diretamente do FEMM para definir o raio de amostragem no entreferro.
AIRGAP_REFERENCE_POINT_MM = (0.0, 57.25)

# Amostragem no entreferro e espectro.
N_THETA = 4096
MAX_HARMONIC_TO_PLOT = 120
PEAK_REL_THRESHOLD = 0.035
MAX_LABELLED_PEAKS = 12

# Visualização no FEMM.
SHOW_FEMM = True
SMOOTHING = "off"
DENSITY_LOWER_B_T = 0.0
DENSITY_UPPER_B_T = 2.0
FLUX_CONTOUR_COUNT = 35
FLUX_CONTOUR_PERCENTILE_LOW = 1.0
FLUX_CONTOUR_PERCENTILE_HIGH = 99.0

# Figuras Matplotlib.
FIG_DPI = 300

# Paleta de cores.
CASE_COLORS = {
    "camada_simples_passo_pleno": "#1F77B4",
    "camada_dupla_passo_pleno": "#FF7F0E",
    "camada_dupla_passo_encurtado": "#2CA02C",
    "motor_cs_pp": "#1F77B4",
    "motor_cd_pp": "#FF7F0E",
    "motor_cd_pe": "#2CA02C",
}

CASE_LIGHT_COLORS = {
    "camada_simples_passo_pleno": "#9ECAE1",
    "camada_dupla_passo_pleno": "#FDBF6F",
    "camada_dupla_passo_encurtado": "#A1D99B",
    "motor_cs_pp": "#9ECAE1",
    "motor_cd_pp": "#FDBF6F",
    "motor_cd_pe": "#A1D99B",
}

CASE_LINESTYLES = {
    "camada_simples_passo_pleno": "-",
    "camada_dupla_passo_pleno": "--",
    "camada_dupla_passo_encurtado": "-.",
    "motor_cs_pp": "-",
    "motor_cd_pp": "--",
    "motor_cd_pe": "-.",
}

CASE_MARKERS = {
    "camada_simples_passo_pleno": "o",
    "camada_dupla_passo_pleno": "s",
    "camada_dupla_passo_encurtado": "^",
    "motor_cs_pp": "o",
    "motor_cd_pp": "s",
    "motor_cd_pe": "^",
}

PRETTY_NAMES = {
    "camada_simples_passo_pleno": "Camada simples, passo pleno",
    "camada_dupla_passo_pleno": "Camada dupla, passo pleno",
    "camada_dupla_passo_encurtado": "Camada dupla, passo encurtado",
    "motor_cs_pp": "Camada simples, passo pleno",
    "motor_cd_pp": "Camada dupla, passo pleno",
    "motor_cd_pe": "Camada dupla, passo encurtado",
}


@dataclass(frozen=True)
class FemAnsCase:
    name: str
    ans_file: Path | None
    stator_slots: int
    pole_pairs: int = 3
    rotor_slots: int | None = None
    i_base_a: float | None = None


# Para travar caminhos explícitos, substitua ans_file=None por:
# Path(r"C:\...\motor_cs_pp_calibrated.ans")
CASES: list[FemAnsCase] = [
    FemAnsCase(
        name="motor_cs_pp",
        ans_file=None,
        stator_slots=36,
        pole_pairs=3,
        i_base_a=12.136320,
    ),
    FemAnsCase(
        name="motor_cd_pp",
        ans_file=None,
        stator_slots=36,
        pole_pairs=3,
        i_base_a=6.047880,
    ),
    FemAnsCase(
        name="motor_cd_pe",
        ans_file=None,
        stator_slots=36,
        pole_pairs=3,
        i_base_a=6.466460,
    ),
]


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def airgap_radius_mm() -> float:
    x, y = AIRGAP_REFERENCE_POINT_MM
    radius = math.hypot(x, y)
    if radius <= 0:
        raise ValueError("AIRGAP_REFERENCE_POINT_MM deve definir um raio positivo.")
    return radius


def airgap_reference_angle_rad() -> float:
    x, y = AIRGAP_REFERENCE_POINT_MM
    return math.atan2(y, x)


def as_float(value) -> float:
    if isinstance(value, complex):
        return float(value.real)
    return float(value)


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Não é possível calcular percentil de uma lista vazia.")
    return float(np.percentile(np.asarray(values, dtype=float), p))


def pretty_case_name(name: str) -> str:
    return PRETTY_NAMES.get(name, name.replace("_", " "))


def case_color(name: str) -> str:
    return CASE_COLORS.get(name, "#333333")


def case_light_color(name: str) -> str:
    return CASE_LIGHT_COLORS.get(name, "#BDBDBD")


def case_linestyle(name: str) -> str:
    return CASE_LINESTYLES.get(name, "-")


def case_marker(name: str) -> str:
    return CASE_MARKERS.get(name, "o")


def apply_academic_axes(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, which="major", alpha=0.30, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", labelsize=10)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_femm_call(function_name: str, *args) -> bool:
    """Executa uma chamada FEMM se ela existir na instalação atual."""
    fn = getattr(femm, function_name, None)
    if fn is None:
        return False
    try:
        fn(*args)
        return True
    except Exception:
        return False


# =============================================================================
# BUSCA DE .ANS E EXTRAÇÃO NUMÉRICA
# =============================================================================

def find_latest_ans(case: FemAnsCase) -> Path:
    if case.ans_file is not None:
        ans = case.ans_file
        if not ans.exists():
            raise FileNotFoundError(f"Arquivo .ANS não encontrado para {case.name}: {ans}")
        return ans

    if not ANS_SEARCH_ROOT.exists():
        raise FileNotFoundError(f"ANS_SEARCH_ROOT não existe: {ANS_SEARCH_ROOT}")

    patterns = [
        f"**/{case.name}_calibrated.ans",
        f"**/{case.name}_work.ans",
        f"**/{case.name}.ans",
        f"**/*{case.name}*.ans",
    ]

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(ANS_SEARCH_ROOT.glob(pattern))

    unique = sorted(set(matches), key=lambda p: p.stat().st_mtime, reverse=True)
    if not unique:
        raise FileNotFoundError(
            f"Nenhum .ANS encontrado para {case.name} em {ANS_SEARCH_ROOT}. "
            "Defina ans_file explicitamente no caso correspondente."
        )
    return unique[0]


def sample_airgap() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Amostra B no círculo definido pelo ponto AIRGAP_REFERENCE_POINT_MM.

    A coordenada (0, 57.25) mm é usada como ponto de referência angular theta_m = 0°.
    Assim, a primeira amostra é exatamente no ponto informado pelo FEMM, e as demais
    amostras avançam no sentido anti-horário.
    """
    r_mm = airgap_radius_mm()
    theta_ref = airgap_reference_angle_rad()

    theta_rel_deg = np.linspace(0.0, 360.0, N_THETA, endpoint=False)
    theta_rel_rad = np.deg2rad(theta_rel_deg)
    theta_abs_rad = theta_ref + theta_rel_rad

    br = np.empty(N_THETA, dtype=float)
    bt = np.empty(N_THETA, dtype=float)
    bmag = np.empty(N_THETA, dtype=float)

    for k, theta in enumerate(theta_abs_rad):
        x = r_mm * math.cos(theta)
        y = r_mm * math.sin(theta)

        bx, by = femm.mo_getb(x, y)
        bx = as_float(bx)
        by = as_float(by)

        br[k] = bx * math.cos(theta) + by * math.sin(theta)
        bt[k] = -bx * math.sin(theta) + by * math.cos(theta)
        bmag[k] = math.hypot(bx, by)

    return theta_rel_deg, br, bt, bmag, theta_abs_rad


def fft_spectrum(br: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Espectro espacial de B_r(theta) em ordens harmônicas mecânicas.

    Como a amostragem cobre 0..360 graus mecânicos, o índice n da FFT
    representa a ordem harmônica mecânica n. A amplitude é normalizada como
    amplitude de pico da componente senoidal correspondente.
    """
    br = np.asarray(br, dtype=float)
    n = br.size
    coeff = np.fft.rfft(br)
    amp = np.abs(coeff) / n
    if n > 1:
        amp[1:-1] *= 2.0
    orders = np.arange(amp.size)
    return orders, amp




def fundamental_peak_angle_deg(br: np.ndarray, pole_pairs: int) -> float:
    """
    Retorna a posição angular mecânica do pico da fundamental espacial.

    A FFT é calculada na coordenada mecânica 0..360°. Para uma máquina de
    6 polos, pole_pairs = 3, logo a fundamental espacial está na ordem n=3.
    Pela convenção da FFT usada aqui, a componente pode ser escrita como:

        B_p(theta) = Bp*cos(p*theta + phi_p)

    O pico ocorre quando p*theta + phi_p = 0. A resposta é reduzida ao
    intervalo de um período fundamental mecânico, isto é, 360/p graus.
    """
    br = np.asarray(br, dtype=float)
    if pole_pairs <= 0:
        raise ValueError("pole_pairs deve ser positivo.")
    coeff = np.fft.rfft(br)
    if pole_pairs >= coeff.size:
        raise ValueError("A ordem fundamental está fora do espectro calculado.")
    phase_rad = float(np.angle(coeff[pole_pairs]))
    peak_deg = -math.degrees(phase_rad) / pole_pairs
    fundamental_period_deg = 360.0 / pole_pairs
    return peak_deg % fundamental_period_deg


def aligned_waveform_by_fundamental(theta_deg: np.ndarray, br: np.ndarray, pole_pairs: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Desloca a curva B_r(theta) para alinhar o pico da fundamental em theta=0°.

    Esse gráfico permite distinguir mudanças reais de forma de onda de simples
    deslocamentos de fase espacial entre modelos.
    """
    theta = np.asarray(theta_deg, dtype=float)
    values = np.asarray(br, dtype=float)
    peak_deg = fundamental_peak_angle_deg(values, pole_pairs)
    theta_aligned = (theta - peak_deg) % 360.0
    order = np.argsort(theta_aligned)
    return theta_aligned[order], values[order], peak_deg


def normalized_spectrum_amplitude(result: dict) -> np.ndarray:
    """Retorna o espectro de amplitude normalizado pela fundamental B_p."""
    fundamental = float(result["fundamental_amp_t"])
    if not math.isfinite(fundamental) or abs(fundamental) < 1e-12:
        return np.full_like(result["amp"], np.nan, dtype=float)
    return np.asarray(result["amp"], dtype=float) / fundamental

def slot_candidates(case: FemAnsCase) -> list[tuple[str, int]]:
    p = case.pole_pairs
    candidates: list[tuple[str, int]] = []

    for order in (case.stator_slots - p, case.stator_slots, case.stator_slots + p):
        if order > 0:
            candidates.append((f"estator_{order}", order))

    if case.rotor_slots is not None:
        for order in (case.rotor_slots - p, case.rotor_slots, case.rotor_slots + p):
            if order > 0:
                candidates.append((f"rotor_{order}", order))

    seen: set[int] = set()
    unique: list[tuple[str, int]] = []
    for label, order in candidates:
        if order not in seen:
            unique.append((label, order))
            seen.add(order)
    return unique


# =============================================================================
# IMAGENS DO PÓS-PROCESSADOR FEMM
# =============================================================================

def reset_post_view() -> None:
    """Remove pontos, malha, densidade e contornos antes de montar uma nova imagem."""
    safe_femm_call("mo_hidemesh")
    safe_femm_call("mo_hidepoints")
    safe_femm_call("mo_hidecontourplot")
    safe_femm_call("mo_hidedensityplot")
    safe_femm_call("mo_zoomnatural")
    safe_femm_call("mo_refreshview")


def estimate_vector_potential_limits() -> tuple[float, float]:
    """
    Estima o intervalo de A para desenhar linhas de fluxo.

    Em magnetostática 2D, as isolinhas do potencial vetor magnético A
    representam as linhas de fluxo. Usa-se percentil para evitar que
    valores extremos locais dominem a escala.
    """
    ne = int(femm.mo_numelements()) # type: ignore
    a_values: list[float] = []

    def scan(indices: Iterable[int]) -> None:
        for i in indices:
            try:
                elem = femm.mo_getelement(i)
                x = as_float(elem[3]) # type: ignore
                y = as_float(elem[4]) # type: ignore
                a_values.append(as_float(femm.mo_geta(x, y)))
            except Exception:
                continue

    scan(range(ne))
    if not a_values:
        scan(range(1, ne + 1))
    if not a_values:
        raise RuntimeError("Não foi possível estimar os limites de A para as linhas de fluxo.")

    a_low = percentile(a_values, FLUX_CONTOUR_PERCENTILE_LOW)
    a_high = percentile(a_values, FLUX_CONTOUR_PERCENTILE_HIGH)

    if abs(a_high - a_low) < 1e-12:
        a_min = min(a_values)
        a_max = max(a_values)
        if abs(a_max - a_min) < 1e-12:
            delta = max(abs(a_max), 1.0) * 1e-6
            return a_min - delta, a_max + delta
        return a_min, a_max

    return a_low, a_high


def show_flux_lines() -> None:
    a_low, a_high = estimate_vector_potential_limits()
    try:
        femm.mo_showcontourplot(FLUX_CONTOUR_COUNT, a_low, a_high, "real")
    except Exception:
        try:
            femm.mo_showcontourplot(FLUX_CONTOUR_COUNT, a_low, a_high) # type: ignore
        except Exception:
            femm.mo_showcontourplot(FLUX_CONTOUR_COUNT) # type: ignore


def show_density_plot() -> None:
    femm.mo_showdensityplot(1, 0, DENSITY_UPPER_B_T, DENSITY_LOWER_B_T, "bmag")


def save_postprocessor_images(case: FemAnsCase, case_dir: Path) -> dict[str, Path | None]:
    """Salva as três visualizações solicitadas no FEMM."""
    outputs: dict[str, Path | None] = {
        "flux_lines_only": None,
        "density_with_flux_lines": None,
        "density_only": None,
    }

    # 1) Linhas de fluxo sem mapa colorido.
    try:
        path = case_dir / f"{case.name}_flux_lines_only.bmp"
        reset_post_view()
        show_flux_lines()
        femm.mo_zoomnatural()
        femm.mo_refreshview()
        femm.mo_savebitmap(str(path))
        outputs["flux_lines_only"] = path
    except Exception as exc:
        print(f"[WARN] Falha ao salvar linhas de fluxo sem densidade de {case.name}: {exc}")

    # 2) Mapa colorido de |B| com linhas de fluxo.
    try:
        path = case_dir / f"{case.name}_density_Bmag_with_flux_lines.bmp"
        reset_post_view()
        show_density_plot()
        show_flux_lines()
        femm.mo_zoomnatural()
        femm.mo_refreshview()
        femm.mo_savebitmap(str(path))
        outputs["density_with_flux_lines"] = path
    except Exception as exc:
        print(f"[WARN] Falha ao salvar densidade com linhas de fluxo de {case.name}: {exc}")

    # 3) Mapa colorido de |B| sem linhas de fluxo.
    try:
        path = case_dir / f"{case.name}_density_Bmag_only.bmp"
        reset_post_view()
        show_density_plot()
        femm.mo_zoomnatural()
        femm.mo_refreshview()
        femm.mo_savebitmap(str(path))
        outputs["density_only"] = path
    except Exception as exc:
        print(f"[WARN] Falha ao salvar densidade sem linhas de fluxo de {case.name}: {exc}")

    return outputs


# =============================================================================
# PÓS-PROCESSAMENTO DE UM CASO
# =============================================================================

def postprocess_ans_case(case: FemAnsCase, out_dir: Path) -> dict:
    ans_path = find_latest_ans(case)

    case_dir = out_dir / "ans_postprocess" / case.name
    case_dir.mkdir(parents=True, exist_ok=True)

    copied_ans = case_dir / ans_path.name
    if ans_path.resolve() != copied_ans.resolve():
        shutil.copy2(ans_path, copied_ans)

    femm_open = False
    solution_open = False

    try:
        femm.openfemm(0 if SHOW_FEMM else 1)
        femm_open = True

        femm.opendocument(str(copied_ans))
        solution_open = True
        femm.mo_smooth(SMOOTHING)

        t0 = time.perf_counter()
        images = save_postprocessor_images(case, case_dir)
        theta_deg, br, bt, b_abs, theta_abs_rad = sample_airgap()
        orders, amp = fft_spectrum(br)
        elapsed = time.perf_counter() - t0

        p = case.pole_pairs
        fundamental_amp = float(amp[p]) if p < amp.size else float("nan")

        slot_rows: list[dict] = []
        for label, order in slot_candidates(case):
            if order < amp.size:
                slot_rows.append({
                    "label": label,
                    "order": order,
                    "amplitude_t": float(amp[order]),
                })
        dominant_slot = max(slot_rows, key=lambda r: r["amplitude_t"]) if slot_rows else None

        return {
            "case": case,
            "source_ans": ans_path,
            "copied_ans": copied_ans,
            "images": images,
            "elapsed_s": elapsed,
            "theta_deg": theta_deg,
            "theta_abs_rad": theta_abs_rad,
            "br": br,
            "bt": bt,
            "b_abs": b_abs,
            "orders": orders,
            "amp": amp,
            "fundamental_order": p,
            "fundamental_amp_t": fundamental_amp,
            "slot_rows": slot_rows,
            "dominant_slot": dominant_slot,
        }

    finally:
        if solution_open:
            try:
                femm.mo_close()
            except Exception:
                pass
        if femm_open:
            try:
                femm.closefemm()
            except Exception:
                pass


# =============================================================================
# CSVs MÍNIMOS
# =============================================================================

def save_airgap_waveform_csv(results: list[dict], out_dir: Path) -> None:
    rows: list[dict] = []
    r_mm = airgap_radius_mm()
    ref_x, ref_y = AIRGAP_REFERENCE_POINT_MM

    for r in results:
        name = r["case"].name
        for theta_rel, theta_abs, br, bt, b_abs in zip(
            r["theta_deg"], r["theta_abs_rad"], r["br"], r["bt"], r["b_abs"]
        ):
            rows.append({
                "case": name,
                "theta_mec_deg_from_reference": float(theta_rel),
                "theta_abs_deg_femm": float(math.degrees(theta_abs) % 360.0),
                "radius_mm": r_mm,
                "reference_x_mm": ref_x,
                "reference_y_mm": ref_y,
                "Br_T": float(br),
                "Bt_T": float(bt),
                "Bmag_T": float(b_abs),
            })

    write_csv(
        out_dir / "airgap_B_waveform.csv",
        rows,
        [
            "case",
            "theta_mec_deg_from_reference",
            "theta_abs_deg_femm",
            "radius_mm",
            "reference_x_mm",
            "reference_y_mm",
            "Br_T",
            "Bt_T",
            "Bmag_T",
        ],
    )


def save_airgap_spectrum_csv(results: list[dict], out_dir: Path) -> None:
    rows: list[dict] = []
    for r in results:
        name = r["case"].name
        max_order = min(MAX_HARMONIC_TO_PLOT, len(r["orders"]) - 1)
        for order in range(1, max_order + 1):
            rows.append({
                "case": name,
                "harmonic_order_mech": order,
                "amplitude_Br_T": float(r["amp"][order]),
            })

    write_csv(
        out_dir / "airgap_B_spectrum.csv",
        rows,
        ["case", "harmonic_order_mech", "amplitude_Br_T"],
    )




def save_airgap_normalized_spectrum_csv(results: list[dict], out_dir: Path) -> None:
    rows: list[dict] = []
    for r in results:
        name = r["case"].name
        norm_amp = normalized_spectrum_amplitude(r)
        max_order = min(MAX_HARMONIC_TO_PLOT, len(r["orders"]) - 1)
        for order in range(1, max_order + 1):
            rows.append({
                "case": name,
                "harmonic_order_mech": order,
                "amplitude_Br_over_fundamental": float(norm_amp[order]),
            })

    write_csv(
        out_dir / "airgap_B_spectrum_normalized.csv",
        rows,
        ["case", "harmonic_order_mech", "amplitude_Br_over_fundamental"],
    )


def save_airgap_aligned_waveform_csv(results: list[dict], out_dir: Path) -> None:
    rows: list[dict] = []
    for r in results:
        case: FemAnsCase = r["case"]
        theta_aligned, br_aligned, shift_deg = aligned_waveform_by_fundamental(
            r["theta_deg"], r["br"], case.pole_pairs
        )
        for theta, br in zip(theta_aligned, br_aligned):
            rows.append({
                "case": case.name,
                "theta_mec_deg_aligned_to_fundamental": float(theta),
                "fundamental_peak_shift_deg": float(shift_deg),
                "Br_T": float(br),
            })

    write_csv(
        out_dir / "airgap_B_waveform_fundamental_aligned.csv",
        rows,
        ["case", "theta_mec_deg_aligned_to_fundamental", "fundamental_peak_shift_deg", "Br_T"],
    )

def save_harmonic_summary_csv(results: list[dict], out_dir: Path) -> None:
    rows: list[dict] = []
    for r in results:
        case: FemAnsCase = r["case"]
        dominant = r["dominant_slot"] or {"label": "", "order": "", "amplitude_t": float("nan")}
        rows.append({
            "case": case.name,
            "pretty_name": pretty_case_name(case.name),
            "I_BASE_A": "" if case.i_base_a is None else case.i_base_a,
            "airgap_reference_x_mm": AIRGAP_REFERENCE_POINT_MM[0],
            "airgap_reference_y_mm": AIRGAP_REFERENCE_POINT_MM[1],
            "airgap_radius_mm": airgap_radius_mm(),
            "fundamental_order": r["fundamental_order"],
            "fundamental_amplitude_T": r["fundamental_amp_t"],
            "dominant_slot_label": dominant["label"],
            "dominant_slot_order": dominant["order"],
            "dominant_slot_amplitude_T": dominant["amplitude_t"],
            "source_ans": str(r["source_ans"]),
            "copied_ans": str(r["copied_ans"]),
            "image_flux_lines_only": str(r["images"].get("flux_lines_only") or ""),
            "image_density_with_flux_lines": str(r["images"].get("density_with_flux_lines") or ""),
            "image_density_only": str(r["images"].get("density_only") or ""),
        })

    write_csv(
        out_dir / "harmonic_summary.csv",
        rows,
        [
            "case",
            "pretty_name",
            "I_BASE_A",
            "airgap_reference_x_mm",
            "airgap_reference_y_mm",
            "airgap_radius_mm",
            "fundamental_order",
            "fundamental_amplitude_T",
            "dominant_slot_label",
            "dominant_slot_order",
            "dominant_slot_amplitude_T",
            "source_ans",
            "copied_ans",
            "image_flux_lines_only",
            "image_density_with_flux_lines",
            "image_density_only",
        ],
    )


# =============================================================================
# PLOTAGEM
# =============================================================================

def relevant_odd_harmonics(results: list[dict]) -> list[int]:
    relevant: set[int] = set()
    peak_candidates: list[tuple[float, int]] = []

    for r in results:
        case: FemAnsCase = r["case"]
        max_order = min(MAX_HARMONIC_TO_PLOT, len(r["amp"]) - 1)
        amp = r["amp"]

        scale = float(np.max(amp[1:max_order + 1])) if max_order >= 1 else 0.0
        threshold = PEAK_REL_THRESHOLD * scale

        p = case.pole_pairs
        if p % 2 == 1 and p <= max_order:
            relevant.add(p)

        for _label, order in slot_candidates(case):
            if order % 2 == 1 and order <= max_order:
                relevant.add(order)

        for order in range(1, max_order + 1, 2):
            a = float(amp[order])
            if a >= threshold:
                peak_candidates.append((a, order))

    for _a, order in sorted(peak_candidates, reverse=True):
        if len(relevant) >= MAX_LABELLED_PEAKS:
            break
        relevant.add(order)

    return sorted(relevant)


def y_limits_for_waveform(results: list[dict]) -> tuple[float, float]:
    values = np.concatenate([r["br"] for r in results])
    y_abs = float(np.max(np.abs(values)))
    margin = 0.08 * y_abs if y_abs > 0 else 0.1
    return -y_abs - margin, y_abs + margin


def y_limit_for_spectrum(results: list[dict]) -> float:
    y_max = 0.0
    for r in results:
        max_order = min(MAX_HARMONIC_TO_PLOT, len(r["amp"]) - 1)
        if max_order >= 1:
            y_max = max(y_max, float(np.max(r["amp"][1:max_order + 1])))
    return 1.12 * y_max if y_max > 0 else 1.0


def plot_airgap_waveform_merged(results: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    y_min, y_max = y_limits_for_waveform(results)

    for r in results:
        case: FemAnsCase = r["case"]
        ax.plot(
            r["theta_deg"],
            r["br"],
            color=case_color(case.name),
            linestyle=case_linestyle(case.name),
            linewidth=1.75,
            alpha=0.98,
            label=pretty_case_name(case.name),
        )

    ax.axhline(0.0, color="0.20", linewidth=0.8, alpha=0.65)
    apply_academic_axes(
        ax,
        xlabel=rf"Posição angular mecânica a partir de ({AIRGAP_REFERENCE_POINT_MM[0]:.2f}, {AIRGAP_REFERENCE_POINT_MM[1]:.2f}) mm, $\theta_m$ [graus]",
        ylabel=r"Componente radial da densidade de fluxo no gap, $B_r$ [T]",
        title=r"Distribuição espacial de $B_r$ no gap",
    )
    ax.set_xlim(0, 360)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "fig_airgap_Br_waveform_merged.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_airgap_waveform_individual(results: list[dict], out_dir: Path) -> None:
    y_min, y_max = y_limits_for_waveform(results)

    for r in results:
        case: FemAnsCase = r["case"]
        color = case_color(case.name)
        light = case_light_color(case.name)
        fig, ax = plt.subplots(figsize=(10.0, 5.0))

        ax.fill_between(r["theta_deg"], 0.0, r["br"], color=light, alpha=0.28, linewidth=0.0)
        ax.plot(r["theta_deg"], r["br"], color=color, linewidth=1.75, label=pretty_case_name(case.name))
        ax.axhline(0.0, color="0.20", linewidth=0.8, alpha=0.65)

        apply_academic_axes(
            ax,
            xlabel=rf"Posição angular mecânica a partir de ({AIRGAP_REFERENCE_POINT_MM[0]:.2f}, {AIRGAP_REFERENCE_POINT_MM[1]:.2f}) mm, $\theta_m$ [graus]",
            ylabel=r"Componente radial da densidade de fluxo no entreferro, $B_r$ [T]",
            title=rf"Distribuição espacial de $B_r$: {pretty_case_name(case.name)}",
        )
        ax.set_xlim(0, 360)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(0, 361, 30))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=True)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(out_dir / f"fig_airgap_Br_waveform_{case.name}.png", dpi=FIG_DPI)
        plt.close(fig)


def annotate_spectrum_peaks(ax, amp: np.ndarray, orders_to_annotate: list[int]) -> None:
    y_top = ax.get_ylim()[1]
    for order in orders_to_annotate:
        if order < len(amp):
            value = float(amp[order])
            if value <= 0:
                continue
            ax.annotate(
                f"{value:.3f}",
                xy=(order, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=True,
            )
            ax.hlines(value, xmin=max(1, order - 1.2), xmax=order + 1.2,
                      colors="0.35", linestyles=":", linewidth=0.7)
    ax.set_ylim(0, y_top)


def plot_spectrum_merged(results: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    tick_orders = relevant_odd_harmonics(results)
    y_top = y_limit_for_spectrum(results)

    for r in results:
        case: FemAnsCase = r["case"]
        max_order = min(MAX_HARMONIC_TO_PLOT, len(r["orders"]) - 1)
        x = r["orders"][1:max_order + 1]
        y = r["amp"][1:max_order + 1]
        odd_mask = (x % 2) == 1

        ax.plot(
            x[odd_mask],
            y[odd_mask],
            marker=case_marker(case.name),
            markersize=4.2,
            markerfacecolor=case_light_color(case.name),
            markeredgecolor=case_color(case.name),
            markeredgewidth=0.8,
            linewidth=1.35,
            color=case_color(case.name),
            linestyle=case_linestyle(case.name),
            label=pretty_case_name(case.name),
        )

    for order in tick_orders:
        ax.axvline(order, color="0.58", linestyle=":", linewidth=0.75, zorder=0)

    apply_academic_axes(
        ax,
        xlabel=r"Ordem harmônica espacial mecânica, $n$",
        ylabel=r"Amplitude de pico da componente radial, $B_r$ [T]",
        title=r"Distribuição espectral de $B_r$ no entreferro",
    )
    ax.set_xlim(1, MAX_HARMONIC_TO_PLOT)
    ax.set_ylim(0, y_top)
    ax.set_xticks(tick_orders)
    ax.set_xticklabels([str(o) for o in tick_orders], rotation=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "fig_airgap_Br_spectrum_merged.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_spectrum_individual(results: list[dict], out_dir: Path) -> None:
    tick_orders = relevant_odd_harmonics(results)
    y_top_global = y_limit_for_spectrum(results)

    for r in results:
        case: FemAnsCase = r["case"]
        max_order = min(MAX_HARMONIC_TO_PLOT, len(r["orders"]) - 1)
        x = r["orders"][1:max_order + 1]
        y = r["amp"][1:max_order + 1]
        odd_mask = (x % 2) == 1
        x_odd = x[odd_mask]
        y_odd = y[odd_mask]

        fig, ax = plt.subplots(figsize=(10.2, 5.2))
        color = case_color(case.name)
        light = case_light_color(case.name)

        ax.vlines(x_odd, 0, y_odd, color=light, alpha=0.95, linewidth=1.4)
        ax.plot(
            x_odd,
            y_odd,
            linestyle="None",
            marker=case_marker(case.name),
            markerfacecolor=light,
            markeredgecolor=color,
            markeredgewidth=0.9,
            markersize=4.8,
            label=pretty_case_name(case.name),
        )

        ax.set_ylim(0, y_top_global)
        annotate_orders = [o for o in tick_orders if o < len(r["amp"])]
        annotate_spectrum_peaks(ax, r["amp"], annotate_orders)

        apply_academic_axes(
            ax,
            xlabel=r"Ordem harmônica espacial mecânica, $n$",
            ylabel=r"Amplitude de pico da componente radial, $B_r$ [T]",
            title=rf"Espectro espacial de $B_r$: {pretty_case_name(case.name)}",
        )
        ax.set_xlim(1, MAX_HARMONIC_TO_PLOT)
        ax.set_ylim(0, y_top_global)
        ax.set_xticks(tick_orders)
        ax.set_xticklabels([str(o) for o in tick_orders], rotation=0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=True)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(out_dir / f"fig_airgap_Br_spectrum_{case.name}.png", dpi=FIG_DPI)
        plt.close(fig)




def y_limit_for_normalized_spectrum(results: list[dict]) -> float:
    y_max = 0.0
    for r in results:
        norm_amp = normalized_spectrum_amplitude(r)
        max_order = min(MAX_HARMONIC_TO_PLOT, len(norm_amp) - 1)
        if max_order >= 1:
            finite = norm_amp[1:max_order + 1]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                y_max = max(y_max, float(np.max(finite)))
    return 1.12 * y_max if y_max > 0 else 1.0


def plot_normalized_spectrum_merged(results: list[dict], out_dir: Path) -> None:
    """Plota B_n/B_p para evidenciar atenuação relativa de harmônicos."""
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    tick_orders = relevant_odd_harmonics(results)
    y_top = y_limit_for_normalized_spectrum(results)

    for r in results:
        case: FemAnsCase = r["case"]
        max_order = min(MAX_HARMONIC_TO_PLOT, len(r["orders"]) - 1)
        x = r["orders"][1:max_order + 1]
        y = normalized_spectrum_amplitude(r)[1:max_order + 1]
        odd_mask = (x % 2) == 1

        ax.plot(
            x[odd_mask],
            y[odd_mask],
            marker=case_marker(case.name),
            markersize=4.2,
            markerfacecolor=case_light_color(case.name),
            markeredgecolor=case_color(case.name),
            markeredgewidth=0.8,
            linewidth=1.35,
            color=case_color(case.name),
            linestyle=case_linestyle(case.name),
            label=pretty_case_name(case.name),
        )

    for order in tick_orders:
        ax.axvline(order, color="0.58", linestyle=":", linewidth=0.75, zorder=0)

    apply_academic_axes(
        ax,
        xlabel=r"Ordem harmônica espacial mecânica, $n$",
        ylabel=r"Amplitude normalizada, $B_n/B_3$ [-]",
        title=r"Espectro espacial normalizado de $B_r$ no entreferro",
    )
    ax.set_xlim(1, MAX_HARMONIC_TO_PLOT)
    ax.set_ylim(0, y_top)
    ax.set_xticks(tick_orders)
    ax.set_xticklabels([str(o) for o in tick_orders], rotation=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "fig_airgap_Br_spectrum_normalized_merged.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_fundamental_aligned_waveform_merged(results: list[dict], out_dir: Path) -> None:
    """Plota B_r(theta) após alinhar a fase da fundamental de cada caso."""
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    y_min, y_max = y_limits_for_waveform(results)

    for r in results:
        case: FemAnsCase = r["case"]
        theta_aligned, br_aligned, shift_deg = aligned_waveform_by_fundamental(
            r["theta_deg"], r["br"], case.pole_pairs
        )
        ax.plot(
            theta_aligned,
            br_aligned,
            color=case_color(case.name),
            linestyle=case_linestyle(case.name),
            linewidth=1.75,
            alpha=0.98,
            label=f"{pretty_case_name(case.name)} (desl. {shift_deg:.1f}°)",
        )

    ax.axhline(0.0, color="0.20", linewidth=0.8, alpha=0.65)
    apply_academic_axes(
        ax,
        xlabel=r"Posição angular mecânica alinhada pela fundamental, $\theta_m^*$ [graus]",
        ylabel=r"Componente radial da densidade de fluxo no entreferro, $B_r$ [T]",
        title=r"Distribuição espacial de $B_r$ alinhada pela fase da fundamental",
    )
    ax.set_xlim(0, 360)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "fig_airgap_Br_waveform_fundamental_aligned_merged.png", dpi=FIG_DPI)
    plt.close(fig)

def plot_slot_comparison(results: list[dict], out_dir: Path) -> None:
    labels = [pretty_case_name(r["case"].name) for r in results]
    amplitudes = [r["dominant_slot"]["amplitude_t"] if r["dominant_slot"] else 0.0 for r in results]
    orders = [r["dominant_slot"]["order"] if r["dominant_slot"] else "-" for r in results]
    colors = [case_color(r["case"].name) for r in results]
    light_colors = [case_light_color(r["case"].name) for r in results]

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, amplitudes, color=light_colors, edgecolor=colors, linewidth=1.2)

    for bar, amp, order, color in zip(bars, amplitudes, orders, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={order}\n{amp:.4f} T",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    apply_academic_axes(
        ax,
        xlabel="Configuração de enrolamento",
        ylabel=r"Amplitude de pico de $B_r$ no harmônico de ranhura [T]",
        title="Comparação do harmônico fundamental de ranhura",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0, max(amplitudes) * 1.18 if amplitudes else 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_slot_harmonic_comparison.png", dpi=FIG_DPI)
    plt.close(fig)


def generate_outputs(results: list[dict], out_dir: Path) -> None:
    save_airgap_waveform_csv(results, out_dir)
    save_airgap_spectrum_csv(results, out_dir)
    save_airgap_normalized_spectrum_csv(results, out_dir)
    save_airgap_aligned_waveform_csv(results, out_dir)
    save_harmonic_summary_csv(results, out_dir)
    plot_airgap_waveform_merged(results, out_dir)
    plot_airgap_waveform_individual(results, out_dir)
    plot_spectrum_merged(results, out_dir)
    plot_spectrum_individual(results, out_dir)
    plot_normalized_spectrum_merged(results, out_dir)
    plot_fundamental_aligned_waveform_merged(results, out_dir)
    plot_slot_comparison(results, out_dir)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main() -> None:
    print("=== Pós-processamento FEMM simplificado a partir de arquivos .ANS ===")
    print("Modo: somente leitura de solução; sem malha, sem solver e sem alteração de corrente.")
    print(f"Casos configurados: {len(CASES)}")
    print(
        "Raio de amostragem no entreferro: "
        f"ponto FEMM ({AIRGAP_REFERENCE_POINT_MM[0]:.3f}, {AIRGAP_REFERENCE_POINT_MM[1]:.3f}) mm "
        f"=> R = {airgap_radius_mm():.3f} mm"
    )
    print()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"postprocess_ans_simplified_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for case in CASES:
        print(f"[ANS] {case.name} — {pretty_case_name(case.name)}")
        result = postprocess_ans_case(case, out_dir)
        results.append(result)

        print(f"      .ANS usado: {result['source_ans']}")
        print(f"      fundamental ordem {result['fundamental_order']} = {result['fundamental_amp_t']:.5f} T")
        if result["dominant_slot"]:
            ds = result["dominant_slot"]
            print(f"      ranhura dominante: {ds['label']} | ordem {ds['order']} | amplitude {ds['amplitude_t']:.6f} T")
        print("      imagens FEMM:")
        print(f"        linhas de fluxo:        {result['images'].get('flux_lines_only')}")
        print(f"        densidade + fluxo:      {result['images'].get('density_with_flux_lines')}")
        print(f"        densidade apenas:       {result['images'].get('density_only')}")
        print(f"      tempo de extração: {result['elapsed_s']:.2f} s\n")

    generate_outputs(results, out_dir)

    print("=== Pós-processamento concluído ===")
    print(f"Resultados salvos em: {out_dir}")
    print("Arquivos principais:")
    print(f"  - {out_dir / 'airgap_B_waveform.csv'}")
    print(f"  - {out_dir / 'airgap_B_spectrum.csv'}")
    print(f"  - {out_dir / 'harmonic_summary.csv'}")
    print(f"  - {out_dir / 'fig_airgap_Br_waveform_merged.png'}")
    print(f"  - {out_dir / 'fig_airgap_Br_waveform_<caso>.png'}")
    print(f"  - {out_dir / 'fig_airgap_Br_spectrum_merged.png'}")
    print(f"  - {out_dir / 'fig_airgap_Br_spectrum_<caso>.png'}")
    print(f"  - {out_dir / 'fig_airgap_Br_spectrum_normalized_merged.png'}")
    print(f"  - {out_dir / 'fig_airgap_Br_waveform_fundamental_aligned_merged.png'}")
    print(f"  - {out_dir / 'airgap_B_spectrum_normalized.csv'}")
    print(f"  - {out_dir / 'airgap_B_waveform_fundamental_aligned.csv'}")
    print(f"  - {out_dir / 'fig_slot_harmonic_comparison.png'}")
    print(f"  - {out_dir / 'ans_postprocess' / '<caso>' / '<caso>_flux_lines_only.bmp'}")
    print(f"  - {out_dir / 'ans_postprocess' / '<caso>' / '<caso>_density_Bmag_with_flux_lines.bmp'}")
    print(f"  - {out_dir / 'ans_postprocess' / '<caso>' / '<caso>_density_Bmag_only.bmp'}")


if __name__ == "__main__":
    main()
