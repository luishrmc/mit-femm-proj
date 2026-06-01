"""
Plotagem isolada dos resultados FEMM a partir dos CSVs já gerados.

Objetivo exclusivo:
1) Ler os CSVs produzidos pelo script de extração dos arquivos .ANS.
2) Gerar os gráficos equivalentes aos do script original ans_plot.py.
3) Gerar gráficos adicionais com zoom nas faixas dos principais harmônicos.
4) Inserir valores numéricos nos pontos/legendas para facilitar a interpretação.

Este script NÃO usa pyFEMM e NÃO abre arquivos .ANS. Ele trabalha apenas com CSVs.

Entradas esperadas em CSV_DIR:
    - airgap_B_waveform.csv
    - airgap_B_spectrum.csv
    - airgap_B_spectrum_normalized.csv
    - airgap_B_waveform_fundamental_aligned.csv
    - harmonic_summary.csv

Dependências:
    pip install numpy matplotlib
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


# =============================================================================
# CONFIGURAÇÕES DO USUÁRIO
# =============================================================================

# Pasta onde estão os CSVs gerados pelo script de extração dos .ANS.
CSV_DIR = Path(r"C:\Users\lumac\ufmg\braz\conv-trab\src\sim\out\ans_extraction\ans_extraction_20260531_135954")

# Pasta raiz para salvar as figuras.
RESULTS_DIR = Path(r"C:\Users\lumac\ufmg\braz\conv-trab\src\sim\out\plots_from_csv")

# Arquivos de entrada.
WAVEFORM_CSV = "airgap_B_waveform.csv"
SPECTRUM_CSV = "airgap_B_spectrum.csv"
NORMALIZED_SPECTRUM_CSV = "airgap_B_spectrum_normalized.csv"
ALIGNED_WAVEFORM_CSV = "airgap_B_waveform_fundamental_aligned.csv"
SUMMARY_CSV = "harmonic_summary.csv"

# Limite geral do espectro.
MAX_HARMONIC_TO_PLOT = 120

# Harmônicos explicitamente anotados nos gráficos.
# Para 36 ranhuras e p=3: fundamental=3; harmônicos de enrolamento relevantes=15,21;
# harmônicos de ranhura principais=33,39; secundários=69,75; superiores=105,111.
IMPORTANT_HARMONICS = [3, 15, 21, 33, 39, 69, 75, 105, 111]

# Os gráficos de barras com zoom agora usam todas as ordens harmônicas
# presentes em cada janela curta definida em HARMONIC_ZOOM_WINDOWS.

# Janelas de zoom dos principais grupos harmônicos.
HARMONIC_ZOOM_WINDOWS = [
    (1, 25, "baixa_ordem_enrolamento"),
    (29, 43, "ranhura_principal"),
    (63, 81, "ranhura_secundaria"),
    (99, 113, "ranhura_superior"),
]

# Nos gráficos de zoom, os valores numéricos não são mais escritos diretamente
# sobre cada ponto da curva, pois isso polui a figura quando há três modelos.
# A nova abordagem usa:
#   1) linhas verticais discretas nos harmônicos destacados;
#   2) uma tabela numérica abaixo do gráfico;
#   3) gráficos de barras agrupadas para evidenciar o efeito do passo encurtado.
ZOOM_TABLE_MAX_ORDERS = 6
ZOOM_BAR_MAX_ORDERS = 6
EXCLUDE_FUNDAMENTAL_FROM_LOW_ORDER_BAR_ZOOM = True

# Apenas harmônicos ímpares nos gráficos principais, conforme adotado no relatório.
PLOT_ONLY_ODD_HARMONICS = True

# Figuras.
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


# =============================================================================
# ESTRUTURAS E UTILITÁRIOS
# =============================================================================

@dataclass
class CaseData:
    name: str
    pretty_name: str
    theta_deg: np.ndarray | None = None
    br_t: np.ndarray | None = None
    bt_t: np.ndarray | None = None
    bmag_t: np.ndarray | None = None
    theta_aligned_deg: np.ndarray | None = None
    br_aligned_t: np.ndarray | None = None
    fundamental_peak_shift_deg: float | None = None
    orders: np.ndarray | None = None
    spectrum_t: np.ndarray | None = None
    normalized_spectrum: np.ndarray | None = None
    fundamental_order: int | None = None
    fundamental_amp_t: float | None = None
    dominant_slot_order: int | None = None
    dominant_slot_amp_t: float | None = None
    i_base_a: float | None = None


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


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def as_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def apply_academic_axes(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, which="major", alpha=0.30, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis="both", labelsize=10)


def odd_filter(orders: np.ndarray) -> np.ndarray:
    if PLOT_ONLY_ODD_HARMONICS:
        return (orders % 2) == 1
    return np.ones_like(orders, dtype=bool)


def ensure_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"csv_plots_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def select_annotation_orders(x: np.ndarray, y: np.ndarray, *, window: tuple[int, int] | None = None, max_labels: int = 10) -> list[int]:
    """Seleciona harmônicos para anotação numérica sem poluir o gráfico."""
    if x.size == 0:
        return []

    if window is not None:
        lo, hi = window
        candidate_mask = (x >= lo) & (x <= hi)
    else:
        candidate_mask = np.ones_like(x, dtype=bool)

    if PLOT_ONLY_ODD_HARMONICS:
        candidate_mask &= (x % 2 == 1)

    candidates = [(float(amp), int(order)) for order, amp in zip(x[candidate_mask], y[candidate_mask])]
    if not candidates:
        return []

    important = [h for h in IMPORTANT_HARMONICS if h in set(x[candidate_mask].astype(int))]
    selected = list(dict.fromkeys(important))

    for _amp, order in sorted(candidates, reverse=True):
        if order not in selected:
            selected.append(order)
        if len(selected) >= max_labels:
            break

    return sorted(selected)


def annotate_points(ax, x: np.ndarray, y: np.ndarray, orders: list[int], *, fmt: str = "{:.4f}") -> None:
    """Anota valores numéricos nos pontos selecionados."""
    x_to_y = {int(order): float(value) for order, value in zip(x, y)}
    y_top = ax.get_ylim()[1]
    for order in orders:
        if order not in x_to_y:
            continue
        value = x_to_y[order]
        if not math.isfinite(value) or value <= 0:
            continue
        ax.annotate(
            fmt.format(value),
            xy=(order, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=True,
        )
        ax.hlines(value, xmin=max(1, order - 0.7), xmax=order + 0.7, colors="0.35", linestyles=":", linewidth=0.7)
    ax.set_ylim(0, y_top)


# =============================================================================
# CARREGAMENTO DOS CSVs
# =============================================================================

def load_case_data(csv_dir: Path) -> dict[str, CaseData]:
    cases: dict[str, CaseData] = {}

    def get_case(name: str) -> CaseData:
        if name not in cases:
            cases[name] = CaseData(name=name, pretty_name=pretty_case_name(name))
        return cases[name]

    # Resumo harmônico.
    summary_path = csv_dir / SUMMARY_CSV
    if summary_path.exists():
        for row in read_csv_rows(summary_path):
            name = row["case"]
            c = get_case(name)
            c.pretty_name = row.get("pretty_name") or pretty_case_name(name)
            c.i_base_a = as_float(row.get("I_BASE_A"))
            c.fundamental_order = as_int(row.get("fundamental_order"))
            c.fundamental_amp_t = as_float(row.get("fundamental_amplitude_T"))
            c.dominant_slot_order = as_int(row.get("dominant_slot_order"))
            c.dominant_slot_amp_t = as_float(row.get("dominant_slot_amplitude_T"))

    # Forma de onda original.
    wave_rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(csv_dir / WAVEFORM_CSV):
        wave_rows_by_case[row["case"]].append(row)
    for name, rows in wave_rows_by_case.items():
        rows = sorted(rows, key=lambda r: float(r["theta_mec_deg_from_reference"]))
        c = get_case(name)
        c.theta_deg = np.asarray([float(r["theta_mec_deg_from_reference"]) for r in rows])
        c.br_t = np.asarray([float(r["Br_T"]) for r in rows])
        c.bt_t = np.asarray([float(r["Bt_T"]) for r in rows])
        c.bmag_t = np.asarray([float(r["Bmag_T"]) for r in rows])

    # Forma de onda alinhada.
    aligned_path = csv_dir / ALIGNED_WAVEFORM_CSV
    if aligned_path.exists():
        aligned_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in read_csv_rows(aligned_path):
            aligned_by_case[row["case"]].append(row)
        for name, rows in aligned_by_case.items():
            rows = sorted(rows, key=lambda r: float(r["theta_mec_deg_aligned_to_fundamental"]))
            c = get_case(name)
            c.theta_aligned_deg = np.asarray([float(r["theta_mec_deg_aligned_to_fundamental"]) for r in rows])
            c.br_aligned_t = np.asarray([float(r["Br_T"]) for r in rows])
            c.fundamental_peak_shift_deg = float(rows[0]["fundamental_peak_shift_deg"])

    # Espectro absoluto.
    spec_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(csv_dir / SPECTRUM_CSV):
        spec_by_case[row["case"]].append(row)
    for name, rows in spec_by_case.items():
        rows = sorted(rows, key=lambda r: int(r["harmonic_order_mech"]))
        c = get_case(name)
        c.orders = np.asarray([int(r["harmonic_order_mech"]) for r in rows], dtype=int)
        c.spectrum_t = np.asarray([float(r["amplitude_Br_T"]) for r in rows], dtype=float)
        if c.fundamental_order is not None and c.fundamental_amp_t is None:
            mask = c.orders == c.fundamental_order
            if np.any(mask):
                c.fundamental_amp_t = float(c.spectrum_t[mask][0])

    # Espectro normalizado.
    norm_path = csv_dir / NORMALIZED_SPECTRUM_CSV
    if norm_path.exists():
        norm_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in read_csv_rows(norm_path):
            norm_by_case[row["case"]].append(row)
        for name, rows in norm_by_case.items():
            rows = sorted(rows, key=lambda r: int(r["harmonic_order_mech"]))
            c = get_case(name)
            c.normalized_spectrum = np.asarray([float(r["amplitude_Br_over_fundamental"]) for r in rows], dtype=float)
    else:
        for c in cases.values():
            if c.spectrum_t is not None and c.fundamental_order is not None and c.fundamental_amp_t:
                c.normalized_spectrum = c.spectrum_t / c.fundamental_amp_t

    return cases


def ordered_cases(cases: dict[str, CaseData]) -> list[CaseData]:
    preferred_order = ["motor_cs_pp", "motor_cd_pp", "motor_cd_pe",
                       "camada_simples_passo_pleno", "camada_dupla_passo_pleno", "camada_dupla_passo_encurtado"]
    seen = set()
    ordered: list[CaseData] = []
    for name in preferred_order:
        if name in cases and name not in seen:
            ordered.append(cases[name])
            seen.add(name)
    for name in sorted(cases):
        if name not in seen:
            ordered.append(cases[name])
    return ordered


# =============================================================================
# PLOTAGEM DAS FORMAS DE ONDA
# =============================================================================

def waveform_y_limits(cases: list[CaseData], *, aligned: bool = False) -> tuple[float, float]:
    values = []
    for c in cases:
        arr = c.br_aligned_t if aligned else c.br_t
        if arr is not None:
            values.append(arr)
    if not values:
        return -1, 1
    all_values = np.concatenate(values)
    y_abs = float(np.max(np.abs(all_values)))
    margin = 0.08 * y_abs if y_abs > 0 else 0.1
    return -y_abs - margin, y_abs + margin


def add_waveform_textbox(ax, c: CaseData) -> None:
    if c.br_t is None:
        return
    br_max = float(np.max(c.br_t))
    br_min = float(np.min(c.br_t))
    text = f"$B_{{r,max}}$ = {br_max:.4f} T\n$B_{{r,min}}$ = {br_min:.4f} T"
    if c.fundamental_amp_t is not None:
        text += f"\n$B_3$ = {c.fundamental_amp_t:.4f} T"
    ax.text(
        0.985, 0.965, text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.75", alpha=0.88),
    )


def plot_waveform_merged(cases: list[CaseData], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    y_min, y_max = waveform_y_limits(cases)

    for c in cases:
        if c.theta_deg is None or c.br_t is None:
            continue
        label = c.pretty_name
        if c.fundamental_amp_t is not None:
            label += f" | $B_3$={c.fundamental_amp_t:.4f} T"
        ax.plot(
            c.theta_deg,
            c.br_t,
            color=case_color(c.name),
            linestyle=case_linestyle(c.name),
            linewidth=1.75,
            alpha=0.98,
            label=label,
        )

    ax.axhline(0.0, color="0.20", linewidth=0.8, alpha=0.65)
    apply_academic_axes(
        ax,
        xlabel=r"Posição angular mecânica, $\theta_m$ [graus]",
        ylabel=r"Componente radial da densidade de fluxo no gap, $B_r$ [T]",
        title=r"Distribuição espacial de $B_r$ no gap",
    )
    ax.set_xlim(0, 360)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=True)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out_dir / "fig_airgap_Br_waveform_merged.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_waveform_individual(cases: list[CaseData], out_dir: Path) -> None:
    y_min, y_max = waveform_y_limits(cases)
    for c in cases:
        if c.theta_deg is None or c.br_t is None:
            continue
        fig, ax = plt.subplots(figsize=(10.0, 5.0))
        color = case_color(c.name)
        light = case_light_color(c.name)
        ax.fill_between(c.theta_deg, 0.0, c.br_t, color=light, alpha=0.28, linewidth=0.0)
        ax.plot(c.theta_deg, c.br_t, color=color, linewidth=1.75, label=c.pretty_name)
        ax.axhline(0.0, color="0.20", linewidth=0.8, alpha=0.65)
        add_waveform_textbox(ax, c)
        apply_academic_axes(
            ax,
            xlabel=r"Posição angular mecânica, $\theta_m$ [graus]",
            ylabel=r"Componente radial da densidade de fluxo no gap, $B_r$ [T]",
            title=rf"Distribuição espacial de $B_r$: {c.pretty_name}",
        )
        ax.set_xlim(0, 360)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(0, 361, 30))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=True)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(out_dir / f"fig_airgap_Br_waveform_{c.name}.png", dpi=FIG_DPI)
        plt.close(fig)


def plot_aligned_waveform_merged(cases: list[CaseData], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    y_min, y_max = waveform_y_limits(cases, aligned=True)

    for c in cases:
        if c.theta_aligned_deg is None or c.br_aligned_t is None:
            continue
        shift = "" if c.fundamental_peak_shift_deg is None else f" | desloc.={c.fundamental_peak_shift_deg:.2f}°"
        ax.plot(
            c.theta_aligned_deg,
            c.br_aligned_t,
            color=case_color(c.name),
            linestyle=case_linestyle(c.name),
            linewidth=1.75,
            label=f"{c.pretty_name}{shift}",
        )

    ax.axhline(0.0, color="0.20", linewidth=0.8, alpha=0.65)
    apply_academic_axes(
        ax,
        xlabel=r"Posição angular mecânica alinhada pela fundamental, $\theta_m^*$ [graus]",
        ylabel=r"Componente radial da densidade de fluxo no entreferro, $B_r$ [T]",
        title=r"Distribuição de $B_r$ alinhada pela fase da fundamental",
    )
    ax.set_xlim(0, 360)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=True)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out_dir / "fig_airgap_Br_waveform_fundamental_aligned_merged.png", dpi=FIG_DPI)
    plt.close(fig)


# =============================================================================
# PLOTAGEM DOS ESPECTROS
# =============================================================================

def spectrum_y_limit(cases: list[CaseData], *, normalized: bool = False, window: tuple[int, int] | None = None) -> float:
    y_max = 0.0
    for c in cases:
        if c.orders is None:
            continue
        y = c.normalized_spectrum if normalized else c.spectrum_t
        if y is None:
            continue
        x = c.orders
        mask = x <= MAX_HARMONIC_TO_PLOT
        if window is not None:
            lo, hi = window
            mask &= (x >= lo) & (x <= hi)
        if PLOT_ONLY_ODD_HARMONICS:
            mask &= (x % 2 == 1)
        if np.any(mask):
            y_max = max(y_max, float(np.nanmax(y[mask])))
    return 1.18 * y_max if y_max > 0 else 1.0


def relevant_orders_for_axis(cases: list[CaseData], *, window: tuple[int, int] | None = None) -> list[int]:
    orders = set()
    for c in cases:
        if c.orders is None or c.spectrum_t is None:
            continue
        x = c.orders
        y = c.spectrum_t
        mask = x <= MAX_HARMONIC_TO_PLOT
        if window is not None:
            lo, hi = window
            mask &= (x >= lo) & (x <= hi)
        if PLOT_ONLY_ODD_HARMONICS:
            mask &= (x % 2 == 1)
        if not np.any(mask):
            continue
        selected = select_annotation_orders(x[mask], y[mask], window=window, max_labels=10)
        orders.update(selected)
    for h in IMPORTANT_HARMONICS:
        if window is None:
            if h <= MAX_HARMONIC_TO_PLOT:
                orders.add(h)
        else:
            lo, hi = window
            if lo <= h <= hi:
                orders.add(h)
    return sorted(orders)


def plot_spectrum_merged(cases: list[CaseData], out_dir: Path, *, normalized: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    y_top = spectrum_y_limit(cases, normalized=normalized)
    tick_orders = relevant_orders_for_axis(cases)

    for c in cases:
        if c.orders is None:
            continue
        y_data = c.normalized_spectrum if normalized else c.spectrum_t
        if y_data is None:
            continue
        x = c.orders
        mask = (x >= 1) & (x <= MAX_HARMONIC_TO_PLOT)
        if PLOT_ONLY_ODD_HARMONICS:
            mask &= (x % 2 == 1)
        # Legenda limpa: apenas a identificação completa de cada motor.
        # Os valores específicos de harmônicos de ranhura são apresentados
        # nos gráficos de barras com zoom, não na legenda do gráfico mesclado.
        label = c.pretty_name
        ax.plot(
            x[mask],
            y_data[mask],
            marker=case_marker(c.name),
            markersize=4.2,
            markerfacecolor=case_light_color(c.name),
            markeredgecolor=case_color(c.name),
            markeredgewidth=0.8,
            linewidth=1.35,
            color=case_color(c.name),
            linestyle=case_linestyle(c.name),
            label=label,
        )

    for order in tick_orders:
        ax.axvline(order, color="0.58", linestyle=":", linewidth=0.75, zorder=0)

    ylabel = r"Amplitude normalizada, $B_n/B_3$ [-]" if normalized else r"Amplitude de pico da componente radial, $B_r$ [T]"
    title = r"Espectro espacial normalizado de $B_r$" if normalized else r"Distribuição espectral de $B_r$ no entreferro"
    apply_academic_axes(ax, xlabel=r"Ordem harmônica espacial mecânica, $n$", ylabel=ylabel, title=title)
    ax.set_xlim(1, MAX_HARMONIC_TO_PLOT)
    ax.set_ylim(0, y_top)
    ax.set_xticks(tick_orders)
    ax.set_xticklabels([str(o) for o in tick_orders])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=True)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    suffix = "normalized_merged" if normalized else "merged"
    fig.savefig(out_dir / f"fig_airgap_Br_spectrum_{suffix}.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_spectrum_individual(cases: list[CaseData], out_dir: Path, *, normalized: bool = False) -> None:
    """Gera a análise harmônica individual como gráfico de linha.

    Mantém a leitura por ordem harmônica, mas usa linha com marcadores em vez de
    hastes verticais, deixando o estilo consistente com a análise mesclada.
    """
    y_top = spectrum_y_limit(cases, normalized=normalized)
    tick_orders = relevant_orders_for_axis(cases)
    for c in cases:
        if c.orders is None:
            continue
        y_data = c.normalized_spectrum if normalized else c.spectrum_t
        if y_data is None:
            continue
        x = c.orders
        mask = (x >= 1) & (x <= MAX_HARMONIC_TO_PLOT)
        if PLOT_ONLY_ODD_HARMONICS:
            mask &= (x % 2 == 1)
        if not np.any(mask):
            continue

        fig, ax = plt.subplots(figsize=(10.2, 5.2))
        color = case_color(c.name)
        light = case_light_color(c.name)
        ax.plot(
            x[mask],
            y_data[mask],
            marker=case_marker(c.name),
            markersize=4.8,
            markerfacecolor=light,
            markeredgecolor=color,
            markeredgewidth=0.9,
            linewidth=1.45,
            color=color,
            linestyle="-",
            label=c.pretty_name,
        )
        ax.set_ylim(0, y_top)
        annotation_orders = [o for o in tick_orders if o in set(x[mask].astype(int))]
        annotate_points(ax, x, y_data, annotation_orders, fmt="{:.4f}" if not normalized else "{:.3f}")

        ylabel = r"Amplitude normalizada, $B_n/B_3$ [-]" if normalized else r"Amplitude de pico da componente radial, $B_r$ [T]"
        title_kind = "Espectro normalizado" if normalized else "Espectro espacial"
        apply_academic_axes(
            ax,
            xlabel=r"Ordem harmônica espacial mecânica, $n$",
            ylabel=ylabel,
            title=rf"{title_kind} de $B_r$: {c.pretty_name}",
        )
        ax.set_xlim(1, MAX_HARMONIC_TO_PLOT)
        ax.set_ylim(0, y_top)
        ax.set_xticks(tick_orders)
        ax.set_xticklabels([str(o) for o in tick_orders])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=True)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        suffix = "normalized" if normalized else "absolute"
        fig.savefig(out_dir / f"fig_airgap_Br_spectrum_{suffix}_{c.name}.png", dpi=FIG_DPI)
        plt.close(fig)


def short_case_label(c: CaseData) -> str:
    """Rótulo curto para tabelas e barras."""
    mapping = {
        "motor_cs_pp": "CS-PP",
        "motor_cd_pp": "CD-PP",
        "motor_cd_pe": "CD-PE",
        "camada_simples_passo_pleno": "CS-PP",
        "camada_dupla_passo_pleno": "CD-PP",
        "camada_dupla_passo_encurtado": "CD-PE",
    }
    return mapping.get(c.name, c.pretty_name)


def get_harmonic_value(c: CaseData, order: int, *, normalized: bool) -> float:
    """Retorna a amplitude de um harmônico específico para um caso."""
    if c.orders is None:
        return float("nan")
    y_data = c.normalized_spectrum if normalized else c.spectrum_t
    if y_data is None:
        return float("nan")
    idx = np.where(c.orders == order)[0]
    if not idx.size:
        return float("nan")
    return float(y_data[idx[0]])


def format_harmonic_value(value: float, *, normalized: bool) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{value:.3f}" if normalized else f"{value:.4f}"


def relative_change_percent(new_value: float, ref_value: float) -> float:
    if not math.isfinite(new_value) or not math.isfinite(ref_value) or abs(ref_value) < 1e-12:
        return float("nan")
    return 100.0 * (new_value - ref_value) / ref_value


def find_case_by_names(cases: list[CaseData], names: set[str]) -> CaseData | None:
    for c in cases:
        if c.name in names:
            return c
    return None


def zoom_orders_for_table(cases: list[CaseData], *, normalized: bool, lo: int, hi: int, max_orders: int) -> list[int]:
    """
    Seleciona poucos harmônicos para a tabela do zoom.

    Prioriza harmônicos de interesse físico e depois completa com os maiores picos
    da janela. No zoom de baixa ordem, n=3 é mantido na tabela, mas o gráfico de
    barras pode omiti-lo para evidenciar n=15 e n=21.
    """
    selected: list[int] = []
    for h in IMPORTANT_HARMONICS:
        if lo <= h <= hi and (not PLOT_ONLY_ODD_HARMONICS or h % 2 == 1):
            selected.append(h)

    peaks: list[tuple[float, int]] = []
    for c in cases:
        if c.orders is None:
            continue
        y_data = c.normalized_spectrum if normalized else c.spectrum_t
        if y_data is None:
            continue
        x = c.orders
        mask = (x >= lo) & (x <= hi)
        if PLOT_ONLY_ODD_HARMONICS:
            mask &= (x % 2 == 1)
        for order, value in zip(x[mask], y_data[mask]):
            if math.isfinite(float(value)):
                peaks.append((float(value), int(order)))

    for _value, order in sorted(peaks, reverse=True):
        if order not in selected:
            selected.append(order)
        if len(selected) >= max_orders:
            break

    return sorted(selected[:max_orders])


def add_zoom_value_table(ax, cases: list[CaseData], orders: list[int], *, normalized: bool) -> None:
    """Adiciona tabela abaixo do gráfico de zoom, evitando sobreposição nos pontos."""
    if not orders:
        return

    cd_pp = find_case_by_names(cases, {"motor_cd_pp", "camada_dupla_passo_pleno"})
    cd_pe = find_case_by_names(cases, {"motor_cd_pe", "camada_dupla_passo_encurtado"})

    col_labels = ["n"] + [short_case_label(c) for c in cases]
    if cd_pp is not None and cd_pe is not None:
        col_labels.append("Δ PE/PP")

    table_rows: list[list[str]] = []
    for order in orders:
        row = [str(order)]
        for c in cases:
            row.append(format_harmonic_value(get_harmonic_value(c, order, normalized=normalized), normalized=normalized))
        if cd_pp is not None and cd_pe is not None:
            ref = get_harmonic_value(cd_pp, order, normalized=normalized)
            new = get_harmonic_value(cd_pe, order, normalized=normalized)
            delta = relative_change_percent(new, ref)
            row.append("—" if not math.isfinite(delta) else f"{delta:+.1f}%")
        table_rows.append(row)

    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="bottom",
        bbox=[0.0, -0.48, 1.0, 0.34],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.06)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EFEFEF")
        if col > 0 and row > 0 and col <= len(cases):
            case = cases[col - 1]
            cell.set_edgecolor(case_color(case.name))

    unit = "[-]" if normalized else "[T]"
    ax.text(
        0.0,
        -0.08,
        f"Valores da tabela em {unit}; Δ PE/PP = variação do passo encurtado em relação à camada dupla de passo pleno.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
    )


def zoom_orders_for_bars(lo: int, hi: int) -> list[int]:
    """Retorna todas as ordens harmônicas da janela curta de zoom."""
    orders = list(range(lo, hi + 1))
    if PLOT_ONLY_ODD_HARMONICS:
        orders = [o for o in orders if o % 2 == 1]
    return orders


def plot_harmonic_bars_zoom(cases: list[CaseData], out_dir: Path, *, normalized: bool, lo: int, hi: int, tag: str) -> None:
    """Gera gráfico de barras agrupadas para todas as ordens de uma janela curta.

    Esta figura substitui o zoom por linhas nas janelas de harmônicos, porque o
    gráfico de barras preserva o efeito de zoom e permite comparar, em cada
    ordem n, as três configurações de enrolamento sem a sobreposição visual da
    tabela e das anotações.
    """
    orders = zoom_orders_for_bars(lo, hi)
    if not orders:
        return

    # Remove ordens que não existem em nenhum caso, mantendo robustez para CSVs
    # com limite harmônico menor que a janela solicitada.
    available_orders = set()
    for c in cases:
        if c.orders is not None:
            available_orders.update(int(o) for o in c.orders)
    orders = [o for o in orders if o in available_orders]
    if not orders:
        return

    # A largura cresce levemente com o número de ordens para evitar rótulos
    # comprimidos nas janelas maiores, mas mantém proporção de relatório.
    fig_width = max(10.8, 0.62 * len(orders) + 4.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5.9))
    x = np.arange(len(orders), dtype=float)
    width = 0.78 / max(len(cases), 1)

    for idx, c in enumerate(cases):
        offsets = x - 0.39 + width / 2.0 + idx * width
        values = [get_harmonic_value(c, order, normalized=normalized) for order in orders]
        bars = ax.bar(
            offsets,
            values,
            width=width,
            color=case_light_color(c.name),
            edgecolor=case_color(c.name),
            linewidth=1.0,
            label=c.pretty_name,
        )
        for bar, value in zip(bars, values):
            if not math.isfinite(value) or value <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                format_harmonic_value(value, normalized=normalized),
                ha="center",
                va="bottom",
                fontsize=7.6,
                rotation=90,
            )

    ylabel = r"Amplitude normalizada, $B_n/B_3$ [-]" if normalized else r"Amplitude de pico da componente radial, $B_r$ [T]"
    norm_txt = "normalizado" if normalized else "não normalizado"
    apply_academic_axes(
        ax,
        xlabel=r"Ordem harmônica espacial mecânica, $n$",
        ylabel=ylabel,
        title=fr"Zoom em barras do espectro de $B_r$: {lo} ≤ n ≤ {hi}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={o}" for o in orders], rotation=0)

    ymax = 0.0
    for c in cases:
        for order in orders:
            value = get_harmonic_value(c, order, normalized=normalized)
            if math.isfinite(value):
                ymax = max(ymax, value)
    ax.set_ylim(0, ymax * 1.34 if ymax > 0 else 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=True)
    ax.grid(True, axis="y", alpha=0.28)
    ax.grid(False, axis="x")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    suffix = "normalized" if normalized else "absolute"
    fig.savefig(out_dir / f"fig_airgap_Br_spectrum_zoom_bars_{tag}_{suffix}.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_spectrum_zoom(cases: list[CaseData], out_dir: Path, *, normalized: bool, lo: int, hi: int, tag: str) -> None:
    """Gera somente o zoom em barras agrupadas para a janela solicitada.

    Os arquivos de zoom por linhas não são produzidos, evitando especialmente a
    figura `fig_airgap_Br_spectrum_zoom_ranhura_principal_absolute.png`, cuja
    leitura fica prejudicada pela sobreposição de anotações. A comparação em
    cada faixa curta passa a ser feita diretamente por barras agrupadas.
    """
    plot_harmonic_bars_zoom(cases, out_dir, normalized=normalized, lo=lo, hi=hi, tag=tag)


def plot_slot_comparison(cases: list[CaseData], out_dir: Path) -> None:
    filtered = [c for c in cases if c.dominant_slot_amp_t is not None]
    if not filtered:
        return
    labels = [c.pretty_name for c in filtered]
    amplitudes = [float(c.dominant_slot_amp_t) for c in filtered] # type: ignore
    orders = [c.dominant_slot_order for c in filtered]
    colors = [case_color(c.name) for c in filtered]
    light_colors = [case_light_color(c.name) for c in filtered]

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, amplitudes, color=light_colors, edgecolor=colors, linewidth=1.2)
    for bar, amp, order, color in zip(bars, amplitudes, orders, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={order}\n{amp:.4f} T",
            ha="center", va="bottom", fontsize=9, color=color, fontweight="bold",
        )
    apply_academic_axes(
        ax,
        xlabel="Configuração de enrolamento",
        ylabel=r"Amplitude de pico de $B_r$ no harmônico de ranhura [T]",
        title="Comparação do harmônico fundamental de ranhura",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0, max(amplitudes) * 1.18)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_slot_harmonic_comparison.png", dpi=FIG_DPI)
    plt.close(fig)


# =============================================================================
# EXECUÇÃO
# =============================================================================

def generate_all_plots(cases: list[CaseData], out_dir: Path) -> None:
    # Forma de onda no entreferro: somente gráficos individuais, sem figura mesclada.
    plot_waveform_individual(cases, out_dir)

    # Análise harmônica individual: gráficos de linha por motor.
    plot_spectrum_individual(cases, out_dir, normalized=False)
    plot_spectrum_individual(cases, out_dir, normalized=True)

    # Análise harmônica mesclada: gráficos de linhas para os três motores.
    plot_spectrum_merged(cases, out_dir, normalized=False)
    plot_spectrum_merged(cases, out_dir, normalized=True)

    # Zooms mesclados em barras agrupadas, para todas as ordens de cada faixa curta.
    for lo, hi, tag in HARMONIC_ZOOM_WINDOWS:
        plot_spectrum_zoom(cases, out_dir, normalized=False, lo=lo, hi=hi, tag=tag)
        plot_spectrum_zoom(cases, out_dir, normalized=True, lo=lo, hi=hi, tag=tag)

    plot_slot_comparison(cases, out_dir)


def main() -> None:
    print("=== Plotagem isolada a partir dos CSVs FEMM ===")
    print(f"CSV_DIR: {CSV_DIR}")
    require_file(CSV_DIR / WAVEFORM_CSV)
    require_file(CSV_DIR / SPECTRUM_CSV)

    out_dir = ensure_output_dir()
    cases_dict = load_case_data(CSV_DIR)
    cases = ordered_cases(cases_dict)

    if not cases:
        raise RuntimeError("Nenhum caso foi encontrado nos CSVs.")

    print("Casos encontrados:")
    for c in cases:
        print(f"  - {c.name}: {c.pretty_name}")
        if c.fundamental_amp_t is not None:
            print(f"      B3 = {c.fundamental_amp_t:.6f} T")
        if c.dominant_slot_order is not None and c.dominant_slot_amp_t is not None:
            print(f"      Harmônico de ranhura dominante n={c.dominant_slot_order}: {c.dominant_slot_amp_t:.6f} T")

    generate_all_plots(cases, out_dir)

    print("\n=== Plotagem concluída ===")
    print(f"Figuras salvas em: {out_dir}")
    print("Arquivos principais:")
    print(f"  - {out_dir / 'fig_airgap_Br_waveform_<caso>.png'}")
    print(f"  - {out_dir / 'fig_airgap_Br_spectrum_merged.png'}")
    print(f"  - {out_dir / 'fig_airgap_Br_spectrum_absolute_<caso>.png'}")
    print(f"  - {out_dir / 'fig_airgap_Br_spectrum_normalized_merged.png'}")
    print(f"  - {out_dir / 'fig_airgap_Br_spectrum_normalized_<caso>.png'}")
    print(f"  - {out_dir / 'fig_airgap_Br_spectrum_zoom_bars_<faixa>_absolute.png'}")
    print(f"  - {out_dir / 'fig_airgap_Br_spectrum_zoom_bars_<faixa>_normalized.png'}")
    print(f"  - {out_dir / 'fig_slot_harmonic_comparison.png'}")


if __name__ == "__main__":
    main()
