"""
Análise FEMM para o projeto do motor C.A. trifásico.

Alterações principais desta versão:
1) Executa cada modelo .FEM em uma cópia de trabalho persistente.
2) Preserva o arquivo .ANS gerado para cada motor/configuração.
3) Salva uma imagem do pós-processador com mapa colorido de |B|, sem pontos/nós,
   com legenda do gradiente.
4) Se todos os casos forem validados por B99,5% ≈ 1,7 T, gera os gráficos do item C
   e do item D em versões mescladas e individuais.
5) Se algum caso não for validado, não gera os gráficos finais; imprime apenas a
   corrente sugerida para ajuste, mas mantém os .FEM/.ANS de trabalho e o print
   do mapa colorido para inspeção.

Dependências:
    pip install pyfemm numpy matplotlib

Observação de segurança:
    Os arquivos .FEM originais nunca são modificados. Cada caso é copiado para
    uma pasta persistente em RESULTS_DIR antes da análise.
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

RESULTS_DIR = Path(r"C:\Users\lumac\ufmg\braz\conv-trab\src\sim\out\final_plots")

# Meta definida no projeto: densidade de fluxo no núcleo próxima de 1,7 T.
TARGET_B_CORE_T = 1.70
B_TOLERANCE_T = 0.05
CORE_PERCENTILE = 99.5

# Grupos definidos nos block labels do FEMM.
GROUP_STATOR = 1
GROUP_ROTOR = 2

# Amostragem no entreferro e espectro.
N_THETA = 4096
MAX_HARMONIC_TO_PLOT = 120
PEAK_REL_THRESHOLD = 0.035      # limiar para harmônicos ímpares relevantes
MAX_LABELLED_PEAKS = 12         # limita o número de marcações no eixo x

# Visualização FEMM.
SHOW_FEMM = True
SMOOTHING = "off"
DENSITY_LOWER_B_T = 0.0
DENSITY_UPPER_B_T = 2.0

# Estilo gráfico acadêmico.
FIG_DPI = 300
CASE_COLORS = {
    "camada_simples_passo_pleno": "#1F4E79",      # azul profundo
    "camada_dupla_passo_pleno": "#D55E00",        # laranja/vermelho escuro
    "camada_dupla_passo_encurtado": "#009E73",    # verde escuro
}
CASE_LINESTYLES = {
    "camada_simples_passo_pleno": "-",
    "camada_dupla_passo_pleno": "--",
    "camada_dupla_passo_encurtado": "-.",
}
PRETTY_NAMES = {
    "camada_simples_passo_pleno": "Camada simples, passo pleno",
    "camada_dupla_passo_pleno": "Camada dupla, passo pleno",
    "camada_dupla_passo_encurtado": "Camada dupla, passo encurtado",
}


@dataclass(frozen=True)
class FemCase:
    name: str
    fem_file: Path
    i_base_a: float
    rg_mm: float
    stator_slots: int
    pole_pairs: int = 3
    rotor_slots: int | None = None


# Edite os caminhos e correntes das configurações disponíveis.
CASES: list[FemCase] = [
    FemCase(
        name="camada_simples_passo_pleno",
        fem_file=Path(r"C:\Users\lumac\ufmg\braz\conv-trab\motor\femm\motor_cs.FEM"),
        i_base_a=12.544,
        rg_mm=74.250,
        stator_slots=36,
        pole_pairs=3,
    ),
    FemCase(
        name="camada_dupla_passo_pleno",
        fem_file=Path(r"C:\Users\lumac\ufmg\braz\conv-trab\motor\femm\motor_cd_pp.FEM"),
        i_base_a=5.988,
        rg_mm=74.250,
        stator_slots=36,
        pole_pairs=3,
    ),
    FemCase(
        name="camada_dupla_passo_encurtado",
        fem_file=Path(r"C:\Users\lumac\ufmg\braz\conv-trab\motor\femm\motor_cd_pe.FEM"),
        i_base_a=6.188,
        rg_mm=74.250,
        stator_slots=36,
        pole_pairs=3,
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


def case_linestyle(name: str) -> str:
    return CASE_LINESTYLES.get(name, "-")


def apply_academic_axes(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", labelsize=10)


def phase_currents(i_base: float) -> dict[str, float]:
    return {
        "fase_A": i_base,
        "fase_B": -0.5 * i_base,
        "fase_C": -0.5 * i_base,
    }


def slot_candidates(case: FemCase) -> list[tuple[str, int]]:
    """
    Candidatos de harmônicos de ranhura.

    Para Qs ranhuras de estator e p pares de polos, as bandas laterais
    principais esperadas na escala mecânica aparecem em Qs - p e Qs + p.
    O termo Qs também é calculado, mas pode ser par; na plotagem final do
    item D só são marcados no eixo x os harmônicos ímpares relevantes.
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


# =============================================================================
# FUNÇÕES FEMM
# =============================================================================

def prepare_case_work_file(case: FemCase, solutions_dir: Path) -> Path:
    if not case.fem_file.exists():
        raise FileNotFoundError(f"Arquivo .FEM não encontrado para {case.name}: {case.fem_file}")
    if case.fem_file.suffix.lower() != ".fem":
        raise ValueError(f"Arquivo inválido para {case.name}: {case.fem_file}")

    case_dir = solutions_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    work_fem = case_dir / f"{case.name}_work.FEM"

    if work_fem.resolve() == case.fem_file.resolve():
        raise RuntimeError("A cópia de trabalho coincide com o arquivo original. Operação abortada.")

    shutil.copy2(case.fem_file, work_fem)
    return work_fem


def set_currents(i_base: float) -> None:
    for circuit, current in phase_currents(i_base).items():
        femm.mi_modifycircprop(circuit, 1, current)


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
        raise RuntimeError("Nenhum elemento dos grupos de estator/rotor foi encontrado.")

    return {
        "bmax_abs_t": bmax,
        "b995_t": percentile(values, CORE_PERCENTILE),
        "x_bmax_mm": xmax,
        "y_bmax_mm": ymax,
        "group_bmax": float(gmax),
    }


def sample_airgap(case: FemCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def save_density_gradient_bitmap(case: FemCase, case_dir: Path) -> Path | None:
    """
    Salva um print do pós-processador mostrando |B| por gradiente colorido.

    A imagem é salva sem pontos/nós e com legenda do gradiente. O formato BMP é
    usado porque é o formato nativo e mais robusto do mo_savebitmap no FEMM.
    """
    bitmap_path = case_dir / f"{case.name}_density_gradient_Bmag.bmp"
    try:
        femm.mo_hidemesh()
        femm.mo_hidepoints()
        femm.mo_hidecontourplot()
        femm.mo_showdensityplot(1, 0, DENSITY_UPPER_B_T, DENSITY_LOWER_B_T, "bmag")
        femm.mo_zoomnatural()
        femm.mo_refreshview()
        femm.mo_savebitmap(str(bitmap_path))
        return bitmap_path
    except Exception as exc:
        print(f"[WARN] Não foi possível salvar o mapa colorido de {case.name}: {exc}")
        return None


def run_femm_case(case: FemCase, solutions_dir: Path) -> dict:
    work_fem = prepare_case_work_file(case, solutions_dir)
    work_ans = work_fem.with_suffix(".ans")
    case_dir = work_fem.parent

    solver_flag = 0 if SHOW_FEMM else 1
    femm_open = False
    solution_open = False
    density_bitmap: Path | None = None

    try:
        femm.openfemm(0 if SHOW_FEMM else 1)
        femm_open = True
        femm.opendocument(str(work_fem))
        set_currents(case.i_base_a)

        t0 = time.perf_counter()
        mesh_elems = femm.mi_createmesh()
        femm.mi_analyze(solver_flag)
        femm.mi_loadsolution()
        solution_open = True
        femm.mo_smooth(SMOOTHING)
        elapsed = time.perf_counter() - t0

        density_bitmap = save_density_gradient_bitmap(case, case_dir)
        core = collect_core_indicator()
        theta_deg, br, bt = sample_airgap(case)
        orders, amp = fft_spectrum(br)

        p = case.pole_pairs
        fundamental_amp = float(amp[p]) if p < amp.size else float("nan")

        slot_rows = []
        for label, order in slot_candidates(case):
            if order < amp.size:
                slot_rows.append({"label": label, "order": order, "amplitude_t": float(amp[order])})
        dominant_slot = max(slot_rows, key=lambda r: r["amplitude_t"]) if slot_rows else None

        return {
            "case": case,
            "work_fem": work_fem,
            "work_ans": work_ans,
            "density_bitmap": density_bitmap,
            "mesh_elements": int(mesh_elems),  # type: ignore
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
                femm.mi_close()
            except Exception:
                pass
            try:
                femm.closefemm()
            except Exception:
                pass


# =============================================================================
# VALIDAÇÃO E CSVs
# =============================================================================

def validate_case(result: dict) -> tuple[bool, float]:
    case: FemCase = result["case"]
    b995 = result["core"]["b995_t"]
    ok = abs(b995 - TARGET_B_CORE_T) <= B_TOLERANCE_T
    suggested_current = case.i_base_a * TARGET_B_CORE_T / b995 if b995 > 0 else math.nan
    return ok, suggested_current


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(results: list[dict], out_dir: Path) -> None:
    rows: list[dict] = []
    for r in results:
        case: FemCase = r["case"]
        dominant = r["dominant_slot"] or {"label": "", "order": "", "amplitude_t": float("nan")}
        rows.append({
            "case": case.name,
            "I_BASE_A": case.i_base_a,
            "fase_A_A": case.i_base_a,
            "fase_B_A": -0.5 * case.i_base_a,
            "fase_C_A": -0.5 * case.i_base_a,
            "B99_5_core_T": r["core"]["b995_t"],
            "Bmax_abs_core_T": r["core"]["bmax_abs_t"],
            "fundamental_order": r["fundamental_order"],
            "fundamental_amplitude_T": r["fundamental_amp_t"],
            "dominant_slot_label": dominant["label"],
            "dominant_slot_order": dominant["order"],
            "dominant_slot_amplitude_T": dominant["amplitude_t"],
            "work_fem": str(r["work_fem"]),
            "work_ans": str(r["work_ans"]),
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
            "fundamental_order",
            "fundamental_amplitude_T",
            "dominant_slot_label",
            "dominant_slot_order",
            "dominant_slot_amplitude_T",
            "work_fem",
            "work_ans",
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
    write_csv(out_dir / "airgap_distribution.csv", distribution_rows, ["case", "theta_mec_deg", "Br_T"])

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
    write_csv(out_dir / "airgap_spectrum.csv", spectrum_rows, ["case", "harmonic_order_mech", "amplitude_T"])


# =============================================================================
# SELEÇÃO DE HARMÔNICOS RELEVANTES E PLOTAGEM
# =============================================================================

def relevant_odd_harmonics(results: list[dict]) -> list[int]:
    """
    Seleciona os harmônicos ímpares relevantes para marcação do eixo horizontal.

    Inclui sempre a fundamental p e as bandas de ranhura ímpares Qs±p.
    Além disso, inclui os maiores picos ímpares acima de um limiar relativo.
    """
    relevant: set[int] = set()
    peak_candidates: list[tuple[float, int]] = []

    for r in results:
        case: FemCase = r["case"]
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

    # Maiores picos globais, sem repetir ordens.
    for _amp, order in sorted(peak_candidates, reverse=True):
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
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    y_min, y_max = y_limits_for_distribution(results)

    for r in results:
        case: FemCase = r["case"]
        ax.plot(
            r["theta_deg"],
            r["br"],
            color=case_color(case.name),
            linestyle=case_linestyle(case.name),
            linewidth=1.45,
            label=pretty_case_name(case.name),
        )

    apply_academic_axes(
        ax,
        xlabel=r"Posição angular mecânica, $\theta_m$ [graus]",
        ylabel=r"Componente radial da densidade de fluxo, $B_r$ [T]",
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
    y_min, y_max = y_limits_for_distribution(results)

    for r in results:
        case: FemCase = r["case"]
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        ax.plot(
            r["theta_deg"],
            r["br"],
            color=case_color(case.name),
            linewidth=1.55,
            label=pretty_case_name(case.name),
        )
        apply_academic_axes(
            ax,
            xlabel=r"Posição angular mecânica, $\theta_m$ [graus]",
            ylabel=r"Componente radial da densidade de fluxo, $B_r$ [T]",
            title=rf"Item C — $B_r$ no entreferro: {pretty_case_name(case.name)}",
        )
        ax.set_xlim(0, 360)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(0, 361, 30))
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
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
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                clip_on=True,
            )
            # Linha guia discreta para facilitar leitura no eixo vertical.
            ax.hlines(value, xmin=max(1, order - 1.2), xmax=order + 1.2, colors="0.35", linestyles=":", linewidth=0.6)
    ax.set_ylim(0, y_top)


def plot_spectrum_merged(results: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    tick_orders = relevant_odd_harmonics(results)
    y_top = y_limit_for_spectrum(results)

    for r in results:
        case: FemCase = r["case"]
        max_order = min(MAX_HARMONIC_TO_PLOT, len(r["orders"]) - 1)
        x = r["orders"][1:max_order + 1]
        y = r["amp"][1:max_order + 1]
        odd_mask = (x % 2) == 1
        ax.plot(
            x[odd_mask],
            y[odd_mask],
            marker="o",
            markersize=3.2,
            linewidth=1.2,
            color=case_color(case.name),
            linestyle=case_linestyle(case.name),
            label=pretty_case_name(case.name),
        )

    # Linhas verticais apenas para os harmônicos ímpares selecionados.
    for order in tick_orders:
        ax.axvline(order, color="0.55", linestyle=":", linewidth=0.75, zorder=0)

    apply_academic_axes(
        ax,
        xlabel="Ordem harmônica espacial mecânica, n",
        ylabel=r"Amplitude de pico de $B_r$ [T]",
        title=r"Item D — Espectro espacial de $B_r$ no entreferro",
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
    tick_orders = relevant_odd_harmonics(results)
    y_top_global = y_limit_for_spectrum(results)

    for r in results:
        case: FemCase = r["case"]
        max_order = min(MAX_HARMONIC_TO_PLOT, len(r["orders"]) - 1)
        x = r["orders"][1:max_order + 1]
        y = r["amp"][1:max_order + 1]
        odd_mask = (x % 2) == 1
        x_odd = x[odd_mask]
        y_odd = y[odd_mask]

        fig, ax = plt.subplots(figsize=(10.0, 5.0))
        color = case_color(case.name)
        ax.vlines(x_odd, 0, y_odd, color=color, alpha=0.65, linewidth=0.8)
        ax.plot(x_odd, y_odd, "o", color=color, markersize=3.5, label=pretty_case_name(case.name))

        # Anota os valores dos harmônicos relevantes que existem neste caso.
        annotate_orders = [o for o in tick_orders if o < len(r["amp"])]
        ax.set_ylim(0, y_top_global)
        annotate_spectrum_peaks(ax, r["orders"], r["amp"], annotate_orders)

        apply_academic_axes(
            ax,
            xlabel="Ordem harmônica espacial mecânica, n",
            ylabel=r"Amplitude de pico de $B_r$ [T]",
            title=rf"Item D — Espectro de $B_r$: {pretty_case_name(case.name)}",
        )
        ax.set_xlim(1, MAX_HARMONIC_TO_PLOT)
        ax.set_ylim(0, y_top_global)
        ax.set_xticks(tick_orders)
        ax.set_xticklabels([str(o) for o in tick_orders], rotation=0)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_d_airgap_spectrum_{case.name}.png", dpi=FIG_DPI)
        plt.close(fig)


def plot_slot_comparison(results: list[dict], out_dir: Path) -> None:
    labels = [pretty_case_name(r["case"].name) for r in results]
    amplitudes = [r["dominant_slot"]["amplitude_t"] if r["dominant_slot"] else 0.0 for r in results]
    orders = [r["dominant_slot"]["order"] if r["dominant_slot"] else "-" for r in results]
    colors = [case_color(r["case"].name) for r in results]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, amplitudes, color=colors, edgecolor="black", linewidth=0.6)

    for bar, amp, order in zip(bars, amplitudes, orders):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={order}\n{amp:.4f} T",
            ha="center",
            va="bottom",
            fontsize=9,
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
    plot_airgap_distribution_merged(results, out_dir)
    plot_airgap_distribution_individual(results, out_dir)
    plot_spectrum_merged(results, out_dir)
    plot_spectrum_individual(results, out_dir)
    plot_slot_comparison(results, out_dir)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main() -> None:
    print("=== Análise FEMM — projeto motor C.A. trifásico ===")
    print(f"Casos configurados: {len(CASES)}")
    print(f"Critério de validação: B{CORE_PERCENTILE:.1f}% no núcleo = {TARGET_B_CORE_T:.2f} ± {B_TOLERANCE_T:.2f} T\n")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"validated_results_{stamp}"
    solutions_dir = out_dir / "femm_solutions"
    out_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    invalid: list[tuple[FemCase, dict, float]] = []

    for case in CASES:
        print(f"[RUN] {case.name}")
        print(f"      arquivo original: {case.fem_file}")
        print(f"      I_BASE: {case.i_base_a:.6g} A")
        result = run_femm_case(case, solutions_dir)
        ok, suggested = validate_case(result)

        b995 = result["core"]["b995_t"]
        bmax = result["core"]["bmax_abs_t"]
        print(f"      B99,5% núcleo = {b995:.5f} T")
        print(f"      Bmax absoluto = {bmax:.5f} T")
        print(f"      fundamental ordem {result['fundamental_order']} = {result['fundamental_amp_t']:.5f} T")
        if result["dominant_slot"]:
            ds = result["dominant_slot"]
            print(f"      ranhura dominante: {ds['label']} | ordem {ds['order']} | amplitude {ds['amplitude_t']:.6f} T")
        print(f"      arquivo .ANS: {result['work_ans']}")
        print(f"      mapa colorido |B|: {result['density_bitmap']}")
        print(f"      validação: {'OK' if ok else 'NÃO OK'}\n")

        if not ok:
            invalid.append((case, result, suggested))
        results.append(result)

    # O resumo é salvo sempre, mesmo se alguma corrente precisar ser ajustada.
    save_summary_csv(results, out_dir)

    if invalid:
        print("=== Corrente fora da faixa de validação ===")
        print("Os arquivos .FEM/.ANS de trabalho e os mapas coloridos foram salvos.")
        print("Os gráficos finais dos itens C e D NÃO foram gerados. Ajuste a corrente e rode novamente.\n")
        for case, result, suggested in invalid:
            print(f"Caso: {case.name}")
            print(f"  I_BASE atual      = {case.i_base_a:.6g} A")
            print(f"  B99,5% medido     = {result['core']['b995_t']:.5f} T")
            print(f"  I_BASE sugerido   = {suggested:.6g} A")
            print(f"  Correntes de fase = A:{suggested:.6g} A, B:{-0.5 * suggested:.6g} A, C:{-0.5 * suggested:.6g} A")
        print(f"\nDiretório de saída parcial: {out_dir}")
        return

    generate_final_outputs(results, out_dir)

    print("=== Validação aceita ===")
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
    print(f"  - {solutions_dir / '<caso>' / '<caso>_work.ans'}")
    print(f"  - {solutions_dir / '<caso>' / '<caso>_density_gradient_Bmag.bmp'}")


if __name__ == "__main__":
    main()
