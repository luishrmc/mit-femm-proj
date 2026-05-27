"""
Pós-processamento FEMM a partir de arquivos .ANS já calculados.

Objetivo:
1) Não executar nova malha nem novo solver.
2) Abrir diretamente os arquivos .ANS pré-elaborados dos três motores.
3) Extrair B no núcleo e B_r no entreferro.
4) Gerar prints de tela do gradiente colorido |B| no FEMM.
5) Gerar os gráficos finais dos itens C e D, em versões mescladas e individuais.

Dependências:
    pip install pyfemm numpy matplotlib

Observação:
    Este script pressupõe que os arquivos .ANS foram gerados previamente com as
    correntes finais já definidas nos respectivos modelos .FEM.
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

# Pasta raiz onde o script procurará automaticamente os arquivos .ANS, caso
# ans_file=None nos casos abaixo. Use a pasta que contém os resultados anteriores.
ANS_SEARCH_ROOT = Path(r"C:\Users\lumac\ufmg\braz\conv-trab\motor\femm")

# Pasta de saída do pós-processamento atual.
RESULTS_DIR = Path(r"C:\Users\lumac\ufmg\braz\conv-trab\src\sim\out\postprocess")

# Critério apenas informativo de validação do campo no núcleo.
# O script NÃO ajusta corrente e NÃO bloqueia a geração dos gráficos.
TARGET_B_CORE_T = 1.70
B_TOLERANCE_T = 0.05
CORE_PERCENTILE = 99.5

# Grupos definidos nos block labels do FEMM.
GROUP_STATOR = 1
GROUP_ROTOR = 2

# Amostragem no entreferro e espectro.
N_THETA = 4096
MAX_HARMONIC_TO_PLOT = 120
PEAK_REL_THRESHOLD = 0.035
MAX_LABELLED_PEAKS = 12

# Visualização FEMM.
SHOW_FEMM = True
SMOOTHING = "off"
DENSITY_LOWER_B_T = 0.0
DENSITY_UPPER_B_T = 2.0

# Linhas de fluxo sobrepostas ao mapa de |B|.
# Em problemas magnetostáticos 2D, as linhas de fluxo são as isolinhas do
# potencial vetor magnético A. O intervalo de A é estimado automaticamente
# nos centróides dos elementos, usando percentis para evitar que valores
# extremos locais definam toda a escala do contorno.
SHOW_FLUX_LINES_ON_DENSITY = False
FLUX_CONTOUR_COUNT = 35
FLUX_CONTOUR_PERCENTILE_LOW = 1.0
FLUX_CONTOUR_PERCENTILE_HIGH = 99.0

# Estilo gráfico acadêmico.
FIG_DPI = 300

# Paleta principal para as curvas mescladas e para os gráficos individuais.
# A paleta foi definida com cores de alto contraste, adequadas para relatório
# técnico e impressão, evitando a aparência monocromática nos gráficos finais.
CASE_COLORS = {
    "camada_simples_passo_pleno": "#1F77B4",       # azul
    "camada_dupla_passo_pleno": "#FF7F0E",         # laranja
    "camada_dupla_passo_encurtado": "#2CA02C",     # verde
    # aliases opcionais, caso os arquivos .ANS usem nomes compactos
    "motor_cs_pp": "#1F77B4",
    "motor_cd_pp": "#FF7F0E",
    "motor_cd_pe": "#2CA02C",
}

# Tons claros da mesma paleta. São usados em preenchimentos e hastes
# para recuperar a visualização colorida que facilita a leitura dos resultados.
CASE_LIGHT_COLORS = {
    "camada_simples_passo_pleno": "#9ECAE1",       # azul claro
    "camada_dupla_passo_pleno": "#FDBF6F",         # laranja claro
    "camada_dupla_passo_encurtado": "#A1D99B",     # verde claro
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
    rg_mm: float
    stator_slots: int
    pole_pairs: int = 3
    rotor_slots: int | None = None
    i_base_a: float | None = None


# Se ans_file=None, o script busca automaticamente o .ANS mais recente em
# ANS_SEARCH_ROOT que combine com o nome do caso.
#
# Para travar caminhos explícitos, substitua None por:
# Path(r"C:\...\camada_simples_passo_pleno_work.ans")
CASES: list[FemAnsCase] = [
    FemAnsCase(
        name="motor_cs_pp",
        ans_file=None,
        rg_mm=74.250,
        stator_slots=36,
        pole_pairs=3,
        i_base_a=12.5,
    ),
    FemAnsCase(
        name="motor_cd_pp",
        ans_file=None,
        rg_mm=74.250,
        stator_slots=36,
        pole_pairs=3,
        i_base_a=5.9,
    ),
    FemAnsCase(
        name="motor_cd_pe",
        ans_file=None,
        rg_mm=74.250,
        stator_slots=36,
        pole_pairs=3,
        i_base_a=6.2,
    ),
]


# =============================================================================
# UTILITÁRIOS NUMÉRICOS E DE FORMATAÇÃO
# =============================================================================

def as_float(value) -> float:
    if isinstance(value, complex):
        return float(value.real)
    return float(value)


def bmag(bx: float, by: float) -> float:
    return math.hypot(bx, by)


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


def validation_status(b995: float) -> str:
    if abs(b995 - TARGET_B_CORE_T) <= B_TOLERANCE_T:
        return "OK"
    return "FORA_DA_FAIXA"


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


def slot_candidates(case: FemAnsCase) -> list[tuple[str, int]]:
    """
    Candidatos de harmônicos de ranhura.

    Para Qs ranhuras de estator e p pares de polos, as bandas laterais
    principais esperadas na escala mecânica aparecem em Qs - p e Qs + p.
    O termo Qs também é calculado para inspeção.
    """
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


def fft_spectrum(br: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Espectro espacial de B_r(theta) em ordens harmônicas mecânicas.

    A amostragem cobre 0..360 graus mecânicos; logo, o índice n da FFT
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


def find_latest_ans(case: FemAnsCase) -> Path:
    if case.ans_file is not None:
        ans = case.ans_file
        if not ans.exists():
            raise FileNotFoundError(f"Arquivo .ANS não encontrado para {case.name}: {ans}")
        return ans

    if not ANS_SEARCH_ROOT.exists():
        raise FileNotFoundError(f"ANS_SEARCH_ROOT não existe: {ANS_SEARCH_ROOT}")

    patterns = [
        f"**/{case.name}_work.ans",
        f"**/{case.name}.ans",
        f"**/*{case.name}*.ans",
    ]

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(ANS_SEARCH_ROOT.glob(pattern))

    # Remove duplicados e ordena pelo mais recente.
    unique = sorted(set(matches), key=lambda p: p.stat().st_mtime, reverse=True)
    if not unique:
        raise FileNotFoundError(
            f"Nenhum .ANS encontrado para {case.name} em {ANS_SEARCH_ROOT}. "
            "Defina ans_file explicitamente no caso correspondente."
        )

    return unique[0]


# =============================================================================
# FUNÇÕES FEMM — SOMENTE PÓS-PROCESSAMENTO DE .ANS
# =============================================================================

def collect_core_indicator() -> dict[str, float]:
    ne = int(femm.mo_numelements())  # type: ignore
    values: list[float] = []
    bmax = -1.0
    xmax = ymax = 0.0
    gmax = -1

    def scan(indices: Iterable[int]) -> None:
        nonlocal bmax, xmax, ymax, gmax
        for i in indices:
            try:
                elem = femm.mo_getelement(i)
                x = as_float(elem[3])  # type: ignore
                y = as_float(elem[4])  # type: ignore
                group = int(as_float(elem[6]))  # type: ignore
                if group not in (GROUP_STATOR, GROUP_ROTOR):
                    continue

                bx, by = femm.mo_getb(x, y)
                bm = bmag(as_float(bx), as_float(by))
                values.append(bm)

                if bm > bmax:
                    bmax, xmax, ymax, gmax = bm, x, y, group
            except Exception:
                continue

    scan(range(ne))
    if not values:
        scan(range(1, ne + 1))
    if not values:
        raise RuntimeError("Nenhum elemento dos grupos de estator/rotor foi encontrado no .ANS.")

    return {
        "bmax_abs_t": bmax,
        "b995_t": percentile(values, CORE_PERCENTILE),
        "x_bmax_mm": xmax,
        "y_bmax_mm": ymax,
        "group_bmax": float(gmax),
    }


def sample_airgap(case: FemAnsCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_deg = np.linspace(0.0, 360.0, N_THETA, endpoint=False)
    theta_rad = np.deg2rad(theta_deg)
    br = np.empty(N_THETA, dtype=float)
    bt = np.empty(N_THETA, dtype=float)

    for k, theta in enumerate(theta_rad):
        x = case.rg_mm * math.cos(theta)
        y = case.rg_mm * math.sin(theta)

        bx, by = femm.mo_getb(x, y)
        bx = as_float(bx)
        by = as_float(by)

        br[k] = bx * math.cos(theta) + by * math.sin(theta)
        bt[k] = -bx * math.sin(theta) + by * math.cos(theta)

    return theta_deg, br, bt


def estimate_vector_potential_limits() -> tuple[float, float]:
    """
    Estima automaticamente o intervalo de potencial vetor magnético A para
    desenhar linhas de fluxo no pós-processador.

    Em uma análise magnetostática 2D do FEMM, as isolinhas de A correspondem
    às linhas de fluxo. A estimativa por percentis evita que singularidades
    locais em quinas/dentes definam um intervalo exagerado de contorno.
    """
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

    # Garante intervalo não degenerado.
    if abs(a_high - a_low) < 1e-12:
        a_min = min(a_values)
        a_max = max(a_values)
        if abs(a_max - a_min) < 1e-12:
            delta = max(abs(a_max), 1.0) * 1e-6
            return a_min - delta, a_max + delta
        return a_min, a_max

    return a_low, a_high


def show_flux_lines_on_density() -> None:
    """Sobrepõe linhas de fluxo ao mapa colorido de |B|."""
    a_low, a_high = estimate_vector_potential_limits()

    # pyFEMM/FEMM aceita a forma completa com tipo = "real" para problemas DC.
    # As chamadas alternativas tornam a rotina mais robusta entre instalações.
    try:
        femm.mo_showcontourplot(FLUX_CONTOUR_COUNT, a_low, a_high, "real")
    except Exception:
        try:
            femm.mo_showcontourplot(FLUX_CONTOUR_COUNT, a_low, a_high) # type: ignore
        except Exception:
            femm.mo_showcontourplot(FLUX_CONTOUR_COUNT) # type: ignore


def save_density_gradient_bitmap(case: FemAnsCase, case_dir: Path) -> Path | None:
    bitmap_path = case_dir / f"{case.name}_density_gradient_Bmag_fluxlines.bmp"
    try:
        femm.mo_hidemesh()
        femm.mo_hidepoints()
        femm.mo_hidecontourplot()
        femm.mo_showdensityplot(1, 0, DENSITY_UPPER_B_T, DENSITY_LOWER_B_T, "bmag")

        if SHOW_FLUX_LINES_ON_DENSITY:
            show_flux_lines_on_density()

        femm.mo_zoomnatural()
        femm.mo_refreshview()
        femm.mo_savebitmap(str(bitmap_path))
        return bitmap_path
    except Exception as exc:
        print(f"[WARN] Não foi possível salvar o mapa colorido com linhas de fluxo de {case.name}: {exc}")
        return None


def postprocess_ans_case(case: FemAnsCase, out_dir: Path) -> dict:
    ans_path = find_latest_ans(case)

    case_dir = out_dir / "ans_postprocess" / case.name
    case_dir.mkdir(parents=True, exist_ok=True)

    copied_ans = case_dir / ans_path.name
    if ans_path.resolve() != copied_ans.resolve():
        shutil.copy2(ans_path, copied_ans)

    femm_open = False
    solution_open = False
    density_bitmap: Path | None = None

    try:
        femm.openfemm(0 if SHOW_FEMM else 1)
        femm_open = True

        # Abre diretamente o arquivo de solução .ANS. Não há mi_createmesh,
        # mi_analyze ou mi_loadsolution nesta versão.
        femm.opendocument(str(copied_ans))
        solution_open = True

        femm.mo_smooth(SMOOTHING)

        t0 = time.perf_counter()
        density_bitmap = save_density_gradient_bitmap(case, case_dir)
        core = collect_core_indicator()
        theta_deg, br, bt = sample_airgap(case)
        elapsed = time.perf_counter() - t0

        orders, amp = fft_spectrum(br)

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
            "density_bitmap": density_bitmap,
            "elapsed_s": elapsed,
            "core": core,
            "theta_deg": theta_deg,
            "br": br,
            "bt": bt,
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
# CSVs
# =============================================================================

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(results: list[dict], out_dir: Path) -> None:
    rows: list[dict] = []

    for r in results:
        case: FemAnsCase = r["case"]
        dominant = r["dominant_slot"] or {"label": "", "order": "", "amplitude_t": float("nan")}
        b995 = r["core"]["b995_t"]

        rows.append({
            "case": case.name,
            "I_BASE_A": "" if case.i_base_a is None else case.i_base_a,
            "fase_A_A": "" if case.i_base_a is None else case.i_base_a,
            "fase_B_A": "" if case.i_base_a is None else -0.5 * case.i_base_a,
            "fase_C_A": "" if case.i_base_a is None else -0.5 * case.i_base_a,
            "B99_5_core_T": b995,
            "Bmax_abs_core_T": r["core"]["bmax_abs_t"],
            "validation_status": validation_status(b995),
            "fundamental_order": r["fundamental_order"],
            "fundamental_amplitude_T": r["fundamental_amp_t"],
            "dominant_slot_label": dominant["label"],
            "dominant_slot_order": dominant["order"],
            "dominant_slot_amplitude_T": dominant["amplitude_t"],
            "source_ans": str(r["source_ans"]),
            "copied_ans": str(r["copied_ans"]),
            "density_bitmap": str(r["density_bitmap"]) if r["density_bitmap"] else "",
        })

    write_csv(
        out_dir / "summary_validation_and_harmonics.csv",
        rows,
        [
            "case",
            "I_BASE_A",
            "fase_A_A",
            "fase_B_A",
            "fase_C_A",
            "B99_5_core_T",
            "Bmax_abs_core_T",
            "validation_status",
            "fundamental_order",
            "fundamental_amplitude_T",
            "dominant_slot_label",
            "dominant_slot_order",
            "dominant_slot_amplitude_T",
            "source_ans",
            "copied_ans",
            "density_bitmap",
        ],
    )


def save_minimal_data_csv(results: list[dict], out_dir: Path) -> None:
    distribution_rows: list[dict] = []
    for r in results:
        name = r["case"].name
        for theta, br in zip(r["theta_deg"], r["br"]):
            distribution_rows.append({
                "case": name,
                "theta_mec_deg": float(theta),
                "Br_T": float(br),
            })

    write_csv(
        out_dir / "airgap_distribution.csv",
        distribution_rows,
        ["case", "theta_mec_deg", "Br_T"],
    )

    spectrum_rows: list[dict] = []
    for r in results:
        name = r["case"].name
        max_order = min(MAX_HARMONIC_TO_PLOT, len(r["orders"]) - 1)
        for order in range(1, max_order + 1):
            spectrum_rows.append({
                "case": name,
                "harmonic_order_mech": order,
                "amplitude_T": float(r["amp"][order]),
            })

    write_csv(
        out_dir / "airgap_spectrum.csv",
        spectrum_rows,
        ["case", "harmonic_order_mech", "amplitude_T"],
    )


# =============================================================================
# SELEÇÃO DE HARMÔNICOS RELEVANTES E PLOTAGEM
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


def y_limits_for_distribution(results: list[dict]) -> tuple[float, float]:
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


def plot_airgap_distribution_merged(results: list[dict], out_dir: Path) -> None:
    """Gráfico mesclado do item C com paleta colorida e boa separação visual."""
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    y_min, y_max = y_limits_for_distribution(results)

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
        xlabel=r"Posição angular mecânica, $\theta_m$ [graus]",
        ylabel=r"Componente radial da densidade de fluxo no entreferro, $B_r$ [T]",
        title=r"Item C — Distribuição espacial de $B_r$ no entreferro",
    )
    ax.set_xlim(0, 360)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "fig_c_airgap_distribution_merged.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_airgap_distribution_individual(results: list[dict], out_dir: Path) -> None:
    """Gráficos individuais do item C com tom principal e preenchimento suave."""
    y_min, y_max = y_limits_for_distribution(results)

    for r in results:
        case: FemAnsCase = r["case"]
        color = case_color(case.name)
        light = case_light_color(case.name)
        fig, ax = plt.subplots(figsize=(10.0, 5.0))

        # Preenchimento em tom claro para melhorar a leitura da polaridade de Br.
        ax.fill_between(
            r["theta_deg"],
            0.0,
            r["br"],
            color=light,
            alpha=0.28,
            linewidth=0.0,
        )
        ax.plot(
            r["theta_deg"],
            r["br"],
            color=color,
            linewidth=1.75,
            label=pretty_case_name(case.name),
        )
        ax.axhline(0.0, color="0.20", linewidth=0.8, alpha=0.65)

        apply_academic_axes(
            ax,
            xlabel=r"Posição angular mecânica, $\theta_m$ [graus]",
            ylabel=r"Componente radial da densidade de fluxo no entreferro, $B_r$ [T]",
            title=rf"Item C — Distribuição espacial de $B_r$: {pretty_case_name(case.name)}",
        )
        ax.set_xlim(0, 360)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(0, 361, 30))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=True)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(out_dir / f"fig_c_airgap_distribution_{case.name}.png", dpi=FIG_DPI)
        plt.close(fig)


def annotate_spectrum_peaks(ax, orders: np.ndarray, amp: np.ndarray, orders_to_annotate: list[int]) -> None:
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
            ax.hlines(
                value,
                xmin=max(1, order - 1.2),
                xmax=order + 1.2,
                colors="0.35",
                linestyles=":",
                linewidth=0.7,
            )
    ax.set_ylim(0, y_top)


def plot_spectrum_merged(results: list[dict], out_dir: Path) -> None:
    """Gráfico mesclado do item D com cores por configuração e harmônicos ímpares."""
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
        title=r"Item D — Distribuição espectral de $B_r$ no entreferro",
    )
    ax.set_xlim(1, MAX_HARMONIC_TO_PLOT)
    ax.set_ylim(0, y_top)
    ax.set_xticks(tick_orders)
    ax.set_xticklabels([str(o) for o in tick_orders], rotation=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "fig_d_airgap_spectrum_merged.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_spectrum_individual(results: list[dict], out_dir: Path) -> None:
    """Gráficos individuais do item D com hastes e marcadores coloridos."""
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
        annotate_spectrum_peaks(ax, r["orders"], r["amp"], annotate_orders)

        apply_academic_axes(
            ax,
            xlabel=r"Ordem harmônica espacial mecânica, $n$",
            ylabel=r"Amplitude de pico da componente radial, $B_r$ [T]",
            title=rf"Item D — Espectro espacial de $B_r$: {pretty_case_name(case.name)}",
        )
        ax.set_xlim(1, MAX_HARMONIC_TO_PLOT)
        ax.set_ylim(0, y_top_global)
        ax.set_xticks(tick_orders)
        ax.set_xticklabels([str(o) for o in tick_orders], rotation=0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=True)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(out_dir / f"fig_d_airgap_spectrum_{case.name}.png", dpi=FIG_DPI)
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
        title="Item D — Comparação do harmônico fundamental de ranhura",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0, max(amplitudes) * 1.18 if amplitudes else 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_d_slot_harmonic_comparison.png", dpi=FIG_DPI)
    plt.close(fig)


def generate_final_outputs(results: list[dict], out_dir: Path) -> None:
    save_minimal_data_csv(results, out_dir)
    save_summary_csv(results, out_dir)
    plot_airgap_distribution_merged(results, out_dir)
    plot_airgap_distribution_individual(results, out_dir)
    plot_spectrum_merged(results, out_dir)
    plot_spectrum_individual(results, out_dir)
    plot_slot_comparison(results, out_dir)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main() -> None:
    print("=== Pós-processamento FEMM a partir de arquivos .ANS ===")
    print(f"Casos configurados: {len(CASES)}")
    print("Modo: somente leitura de solução; sem mi_createmesh, sem mi_analyze e sem alteração de corrente.")
    print(f"Critério informativo: B{CORE_PERCENTILE:.1f}% no núcleo = {TARGET_B_CORE_T:.2f} ± {B_TOLERANCE_T:.2f} T\n")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"postprocess_ans_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for case in CASES:
        print(f"[ANS] {case.name}")
        print(f"      raio médio do entreferro: {case.rg_mm:.3f} mm")

        result = postprocess_ans_case(case, out_dir)
        results.append(result)

        b995 = result["core"]["b995_t"]
        bmax = result["core"]["bmax_abs_t"]
        print(f"      .ANS usado: {result['source_ans']}")
        print(f"      B99,5% núcleo = {b995:.5f} T [{validation_status(b995)}]")
        print(f"      Bmax absoluto = {bmax:.5f} T")
        print(f"      fundamental ordem {result['fundamental_order']} = {result['fundamental_amp_t']:.5f} T")
        if result["dominant_slot"]:
            ds = result["dominant_slot"]
            print(f"      ranhura dominante: {ds['label']} | ordem {ds['order']} | amplitude {ds['amplitude_t']:.6f} T")
        print(f"      mapa colorido |B|: {result['density_bitmap']}")
        print(f"      tempo de extração: {result['elapsed_s']:.2f} s\n")

    generate_final_outputs(results, out_dir)

    print("=== Pós-processamento concluído ===")
    print(f"Resultados salvos em: {out_dir}")
    print("Arquivos principais:")
    print(f"  - {out_dir / 'summary_validation_and_harmonics.csv'}")
    print(f"  - {out_dir / 'airgap_distribution.csv'}")
    print(f"  - {out_dir / 'airgap_spectrum.csv'}")
    print(f"  - {out_dir / 'fig_c_airgap_distribution_merged.png'}")
    print(f"  - {out_dir / 'fig_c_airgap_distribution_<caso>.png'}")
    print(f"  - {out_dir / 'fig_d_airgap_spectrum_merged.png'}")
    print(f"  - {out_dir / 'fig_d_airgap_spectrum_<caso>.png'}")
    print(f"  - {out_dir / 'fig_d_slot_harmonic_comparison.png'}")
    print(f"  - {out_dir / 'ans_postprocess' / '<caso>' / '<caso>_density_gradient_Bmag_fluxlines.bmp'}")


if __name__ == "__main__":
    main()
