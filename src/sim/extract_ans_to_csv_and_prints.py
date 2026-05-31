"""
Extração isolada dos arquivos .ANS do FEMM.

Objetivo exclusivo:
1) Abrir diretamente os arquivos .ANS dos três motores, sem gerar malha e sem rodar solver.
2) Amostrar a componente radial B_r no entreferro usando o ponto FEMM (0, 57.25) mm.
3) Gerar somente os CSVs necessários para a etapa de plotagem.
4) Salvar três imagens do pós-processador FEMM por motor:
   - linhas de fluxo sem mapa colorido;
   - mapa colorido |B| com linhas de fluxo;
   - mapa colorido |B| sem linhas de fluxo.

Este script NÃO gera gráficos Matplotlib. Use plot_from_csv.py para a plotagem.

Arquivos CSV gerados:
    - airgap_B_waveform.csv
    - airgap_B_spectrum.csv
    - airgap_B_spectrum_normalized.csv
    - airgap_B_waveform_fundamental_aligned.csv
    - harmonic_summary.csv

Dependências:
    pip install pyfemm numpy
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
import numpy as np


# =============================================================================
# CONFIGURAÇÕES DO USUÁRIO
# =============================================================================

# Pasta raiz onde o script procurará automaticamente os arquivos .ANS,
# se ans_file=None em CASES.
ANS_SEARCH_ROOT = Path(r"C:\Users\lumac\ufmg\braz\conv-trab\motor\femm")

# Pasta de saída para os CSVs e prints FEMM.
RESULTS_DIR = Path(r"C:\Users\lumac\ufmg\braz\conv-trab\src\sim\out\ans_extraction")

# Ponto retirado diretamente do FEMM para definir o raio de amostragem no entreferro.
# O ponto (0, 57.25) mm define R = 57.25 mm e também a referência angular theta_m = 0°.
AIRGAP_REFERENCE_POINT_MM = (0.0, 57.25)

# Amostragem no entreferro e espectro.
N_THETA = 4096
MAX_HARMONIC_TO_SAVE = 120

# Visualização no FEMM.
SHOW_FEMM = True
SMOOTHING = "off"
DENSITY_LOWER_B_T = 0.0
DENSITY_UPPER_B_T = 2.0
FLUX_CONTOUR_COUNT = 35
FLUX_CONTOUR_PERCENTILE_LOW = 1.0
FLUX_CONTOUR_PERCENTILE_HIGH = 99.0

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
        i_base_a=6.047880,
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


# =============================================================================
# EXTRAÇÃO DE B NO ENTREFERRO E FFT
# =============================================================================

def sample_airgap() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Amostra B no círculo definido pelo ponto AIRGAP_REFERENCE_POINT_MM.

    A coordenada (0, 57.25) mm é usada como referência angular theta_m = 0°.
    Assim, a primeira amostra é exatamente no ponto informado pelo FEMM.
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
    Como a amostragem cobre 0..360 graus mecânicos, o índice n da FFT representa
    a ordem harmônica mecânica n.
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
    """Retorna a posição angular mecânica do pico da fundamental espacial."""
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
    """Desloca a curva B_r(theta) para alinhar o pico da fundamental em theta=0°."""
    theta = np.asarray(theta_deg, dtype=float)
    values = np.asarray(br, dtype=float)
    peak_deg = fundamental_peak_angle_deg(values, pole_pairs)
    theta_aligned = (theta - peak_deg) % 360.0
    order = np.argsort(theta_aligned)
    return theta_aligned[order], values[order], peak_deg


def normalized_spectrum(amp: np.ndarray, fundamental_order: int) -> np.ndarray:
    fundamental = float(amp[fundamental_order])
    if not math.isfinite(fundamental) or abs(fundamental) < 1e-12:
        return np.full_like(amp, np.nan, dtype=float)
    return amp / fundamental


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
    """Estima o intervalo de A para desenhar linhas de fluxo."""
    ne = int(femm.mo_numelements())  # type: ignore
    a_values: list[float] = []

    def scan(indices: Iterable[int]) -> None:
        for i in indices:
            try:
                elem = femm.mo_getelement(i)
                x = as_float(elem[3])  # type: ignore
                y = as_float(elem[4])  # type: ignore
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
            femm.mo_showcontourplot(FLUX_CONTOUR_COUNT, a_low, a_high)  # type: ignore
        except Exception:
            femm.mo_showcontourplot(FLUX_CONTOUR_COUNT)  # type: ignore


def show_density_plot() -> None:
    femm.mo_showdensityplot(1, 0, DENSITY_UPPER_B_T, DENSITY_LOWER_B_T, "bmag")


def save_postprocessor_images(case: FemAnsCase, case_dir: Path) -> dict[str, Path | None]:
    """Salva as três visualizações solicitadas no FEMM."""
    outputs: dict[str, Path | None] = {
        "flux_lines_only": None,
        "density_with_flux_lines": None,
        "density_only": None,
    }

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
# PROCESSAMENTO DE CADA CASO
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
        norm_amp = normalized_spectrum(amp, case.pole_pairs)
        theta_aligned, br_aligned, shift_deg = aligned_waveform_by_fundamental(theta_deg, br, case.pole_pairs)
        elapsed = time.perf_counter() - t0

        fundamental_amp = float(amp[case.pole_pairs])

        slot_rows: list[dict] = []
        for label, order in slot_candidates(case):
            if order < amp.size:
                slot_rows.append({
                    "label": label,
                    "order": order,
                    "amplitude_t": float(amp[order]),
                    "normalized_amplitude": float(norm_amp[order]),
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
            "norm_amp": norm_amp,
            "theta_aligned": theta_aligned,
            "br_aligned": br_aligned,
            "fundamental_peak_shift_deg": shift_deg,
            "fundamental_order": case.pole_pairs,
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
# CSVs
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
        max_order = min(MAX_HARMONIC_TO_SAVE, len(r["orders"]) - 1)
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
        max_order = min(MAX_HARMONIC_TO_SAVE, len(r["orders"]) - 1)
        for order in range(1, max_order + 1):
            rows.append({
                "case": name,
                "harmonic_order_mech": order,
                "amplitude_Br_over_fundamental": float(r["norm_amp"][order]),
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
        for theta, br in zip(r["theta_aligned"], r["br_aligned"]):
            rows.append({
                "case": case.name,
                "theta_mec_deg_aligned_to_fundamental": float(theta),
                "fundamental_peak_shift_deg": float(r["fundamental_peak_shift_deg"]),
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
        dominant = r["dominant_slot"] or {"label": "", "order": "", "amplitude_t": float("nan"), "normalized_amplitude": float("nan")}
        rows.append({
            "case": case.name,
            "pretty_name": pretty_case_name(case.name),
            "I_BASE_A": "" if case.i_base_a is None else case.i_base_a,
            "airgap_reference_x_mm": AIRGAP_REFERENCE_POINT_MM[0],
            "airgap_reference_y_mm": AIRGAP_REFERENCE_POINT_MM[1],
            "airgap_radius_mm": airgap_radius_mm(),
            "fundamental_order": r["fundamental_order"],
            "fundamental_amplitude_T": r["fundamental_amp_t"],
            "fundamental_peak_shift_deg": r["fundamental_peak_shift_deg"],
            "dominant_slot_label": dominant["label"],
            "dominant_slot_order": dominant["order"],
            "dominant_slot_amplitude_T": dominant["amplitude_t"],
            "dominant_slot_amplitude_over_B3": dominant["normalized_amplitude"],
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
            "fundamental_peak_shift_deg",
            "dominant_slot_label",
            "dominant_slot_order",
            "dominant_slot_amplitude_T",
            "dominant_slot_amplitude_over_B3",
            "source_ans",
            "copied_ans",
            "image_flux_lines_only",
            "image_density_with_flux_lines",
            "image_density_only",
        ],
    )


def save_all_csvs(results: list[dict], out_dir: Path) -> None:
    save_airgap_waveform_csv(results, out_dir)
    save_airgap_spectrum_csv(results, out_dir)
    save_airgap_normalized_spectrum_csv(results, out_dir)
    save_airgap_aligned_waveform_csv(results, out_dir)
    save_harmonic_summary_csv(results, out_dir)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main() -> None:
    print("=== Extração FEMM a partir de arquivos .ANS ===")
    print("Modo: somente leitura de solução; sem malha, sem solver e sem alteração de corrente.")
    print(f"Casos configurados: {len(CASES)}")
    print(
        "Raio de amostragem no entreferro: "
        f"ponto FEMM ({AIRGAP_REFERENCE_POINT_MM[0]:.3f}, {AIRGAP_REFERENCE_POINT_MM[1]:.3f}) mm "
        f"=> R = {airgap_radius_mm():.3f} mm"
    )
    print()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"ans_extraction_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for case in CASES:
        print(f"[ANS] {case.name} — {pretty_case_name(case.name)}")
        result = postprocess_ans_case(case, out_dir)
        results.append(result)

        print(f"      .ANS usado: {result['source_ans']}")
        print(f"      fundamental ordem {result['fundamental_order']} = {result['fundamental_amp_t']:.6f} T")
        print(f"      deslocamento da fundamental = {result['fundamental_peak_shift_deg']:.4f}°")
        if result["dominant_slot"]:
            ds = result["dominant_slot"]
            print(
                f"      ranhura dominante: {ds['label']} | ordem {ds['order']} | "
                f"amplitude {ds['amplitude_t']:.6f} T | normalizada {ds['normalized_amplitude']:.6f}"
            )
        print("      imagens FEMM:")
        print(f"        linhas de fluxo:        {result['images'].get('flux_lines_only')}")
        print(f"        densidade + fluxo:      {result['images'].get('density_with_flux_lines')}")
        print(f"        densidade apenas:       {result['images'].get('density_only')}")
        print(f"      tempo de extração: {result['elapsed_s']:.2f} s\n")

    save_all_csvs(results, out_dir)

    print("=== Extração concluída ===")
    print(f"Resultados salvos em: {out_dir}")
    print("Arquivos principais:")
    print(f"  - {out_dir / 'airgap_B_waveform.csv'}")
    print(f"  - {out_dir / 'airgap_B_spectrum.csv'}")
    print(f"  - {out_dir / 'airgap_B_spectrum_normalized.csv'}")
    print(f"  - {out_dir / 'airgap_B_waveform_fundamental_aligned.csv'}")
    print(f"  - {out_dir / 'harmonic_summary.csv'}")
    print(f"  - {out_dir / 'ans_postprocess' / '<caso>' / '<caso>_flux_lines_only.bmp'}")
    print(f"  - {out_dir / 'ans_postprocess' / '<caso>' / '<caso>_density_Bmag_with_flux_lines.bmp'}")
    print(f"  - {out_dir / 'ans_postprocess' / '<caso>' / '<caso>_density_Bmag_only.bmp'}")


if __name__ == "__main__":
    main()
