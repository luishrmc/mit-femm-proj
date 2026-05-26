"""
Análise FEMM simplificada para o projeto do motor C.A. trifásico.

Objetivo:
1) Executar a análise FEMM em cópia temporária do(s) arquivo(s) .FEM.
2) Validar se a densidade de fluxo no núcleo está compatível com a meta do projeto.
3) Se a validação for aceita, gerar apenas os gráficos necessários para o relatório:
   - distribuição espacial de B_r no entreferro;
   - espectro espacial de B_r, incluindo harmônicos espaciais e de ranhura;
   - comparação da amplitude do harmônico fundamental de ranhura entre configurações.
4) Se a validação falhar, não gerar gráficos; apenas imprimir a corrente sugerida.

Dependências:
    pip install pyfemm numpy matplotlib

Observação de segurança:
    O arquivo original .FEM nunca é modificado. Cada caso é copiado para uma pasta
    temporária de trabalho antes de qualquer modificação de corrente.
"""

from __future__ import annotations

import csv
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import femm
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURAÇÕES DO USUÁRIO
# =============================================================================

RESULTS_DIR = Path(r"C:\Users\lumac\ufmg\braz\conv-trab\src\sim\out\final_plots")

# Meta definida no projeto: densidade de fluxo no núcleo próxima de 1,7 T.
TARGET_B_CORE_T = 1.70
B_TOLERANCE_T = 0.05              # aceita B99,5% entre 1,65 T e 1,75 T
CORE_PERCENTILE = 99.5            # evita usar pico local de canto como critério principal

# A análise já validada usou grupos 1 e 2 para estator e rotor.
GROUP_STATOR = 1
GROUP_ROTOR = 2

# Amostragem no entreferro.
N_THETA = 4096                    # potência de 2 melhora resolução do FFT
MAX_HARMONIC_TO_PLOT = 120        # deve incluir p, Qs-p, Qs e Qs+p

# FEMM visível é mais seguro durante validação; depois pode mudar para False.
SHOW_FEMM = True
SMOOTHING = "off"                 # campo por elemento, sem suavização artificial


@dataclass(frozen=True)
class FemCase:
    name: str
    fem_file: Path
    i_base_a: float
    rg_mm: float
    stator_slots: int
    pole_pairs: int = 3            # 6 polos -> p = 3 pares de polos
    rotor_slots: int | None = None


# Edite os caminhos e correntes das configurações disponíveis.
# Deixe somente os casos que já existem no seu computador.
CASES: list[FemCase] = [
    FemCase(
        name="camada_simples_passo_pleno",
        fem_file=Path(r"C:\Users\lumac\ufmg\braz\conv-trab\motor\femm\motor_cs.FEM"),
        i_base_a=12.544,
        rg_mm=74.250,
        stator_slots=36,
        pole_pairs=3,
        rotor_slots=None,
    ),
    # Exemplo para ativar depois:
    FemCase(
        name="camada_dupla_passo_pleno",
        fem_file=Path(r"C:\Users\lumac\ufmg\braz\conv-trab\motor\femm\motor_cd_pp.FEM"),
        i_base_a=5.888,
        rg_mm=74.250,
        stator_slots=36,
        pole_pairs=3,
        rotor_slots=None,
    ),
    FemCase(
        name="camada_dupla_passo_encurtado",
        fem_file=Path(r"C:\Users\lumac\ufmg\braz\conv-trab\motor\femm\motor_cd_pe.FEM"),
        i_base_a=5.888,
        rg_mm=74.250,
        stator_slots=36,
        pole_pairs=3,
        rotor_slots=None,
    ),
]


# =============================================================================
# FUNÇÕES NUMÉRICAS
# =============================================================================

def as_float(value) -> float:
    if isinstance(value, complex):
        return float(value.real)
    return float(value)


def bmag(bx: float, by: float) -> float:
    return math.hypot(bx, by)


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), p))


def phase_currents(i_base: float) -> dict[str, float]:
    return {
        "fase_A": i_base,
        "fase_B": -0.5 * i_base,
        "fase_C": -0.5 * i_base,
    }


def slot_candidates(case: FemCase) -> list[tuple[str, int]]:
    """
    Candidatos de harmônicos associados ao ranhuramento.

    Em uma máquina com p pares de polos e Qs ranhuras de estator, as bandas de
    ranhura mais importantes aparecem em torno de Qs ± p. O termo Qs também é
    calculado para identificar a ondulação pura de permeância de ranhura.
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

    # Remove duplicados preservando ordem.
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

    Como a amostragem cobre 0..360 graus mecânicos, o índice n da FFT representa
    a ordem harmônica espacial mecânica n. A amplitude é normalizada como pico.
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

def safe_work_copy(case: FemCase, work_dir: Path) -> Path:
    if not case.fem_file.exists():
        raise FileNotFoundError(f"Arquivo .FEM não encontrado para {case.name}: {case.fem_file}")
    if case.fem_file.suffix.lower() != ".fem":
        raise ValueError(f"Arquivo inválido para {case.name}: {case.fem_file}")

    work_fem = work_dir / f"{case.name}_work.FEM"
    if work_fem.resolve() == case.fem_file.resolve():
        raise RuntimeError("A cópia de trabalho coincide com o arquivo original. Operação abortada.")

    shutil.copy2(case.fem_file, work_fem)
    return work_fem


def set_currents(i_base: float) -> None:
    for circuit, current in phase_currents(i_base).items():
        femm.mi_modifycircprop(circuit, 1, current)


def collect_core_indicator() -> dict[str, float]:
    ne = int(femm.mo_numelements()) # type: ignore
    values: list[float] = []
    bmax = -1.0
    xmax = ymax = 0.0
    gmax = -1

    def scan(indices: Iterable[int]) -> None:
        nonlocal bmax, xmax, ymax, gmax, values
        for i in indices:
            try:
                elem = femm.mo_getelement(i)
                x = as_float(elem[3]) # type: ignore
                y = as_float(elem[4]) # type: ignore
                group = int(as_float(elem[6])) # type: ignore
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

    b995 = percentile(values, CORE_PERCENTILE)
    return {
        "bmax_abs_t": bmax,
        "b995_t": b995,
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


def run_femm_case(case: FemCase, work_dir: Path) -> dict:
    work_fem = safe_work_copy(case, work_dir)
    solver_flag = 0 if SHOW_FEMM else 1
    femm_open = False
    solution_open = False

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
            "mesh_elements": int(mesh_elems), # type: ignore
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
# VALIDAÇÃO E SAÍDAS MÍNIMAS
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


def save_minimal_csv(results: list[dict], out_dir: Path) -> None:
    # Dados mínimos da distribuição no entreferro: necessários para reproduzir o item c.
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

    # Dados mínimos do espectro: necessários para reproduzir o item d.
    spectrum_rows: list[dict] = []
    for r in results:
        name = r["case"].name
        orders = r["orders"]
        amp = r["amp"]
        max_order = min(MAX_HARMONIC_TO_PLOT, len(orders) - 1)
        for order in range(1, max_order + 1):
            spectrum_rows.append({
                "case": name,
                "harmonic_order_mech": order,
                "amplitude_T": float(amp[order]),
            })
    write_csv(out_dir / "airgap_spectrum.csv", spectrum_rows, ["case", "harmonic_order_mech", "amplitude_T"])

    # Resumo de validação e harmônicos de ranhura.
    summary_rows: list[dict] = []
    for r in results:
        case = r["case"]
        dominant = r["dominant_slot"] or {"label": "", "order": "", "amplitude_t": float("nan")}
        summary_rows.append({
            "case": case.name,
            "I_BASE_A": case.i_base_a,
            "B99_5_core_T": r["core"]["b995_t"],
            "Bmax_abs_core_T": r["core"]["bmax_abs_t"],
            "fundamental_order": r["fundamental_order"],
            "fundamental_amplitude_T": r["fundamental_amp_t"],
            "dominant_slot_label": dominant["label"],
            "dominant_slot_order": dominant["order"],
            "dominant_slot_amplitude_T": dominant["amplitude_t"],
        })
    write_csv(
        out_dir / "summary_validation_and_harmonics.csv",
        summary_rows,
        [
            "case",
            "I_BASE_A",
            "B99_5_core_T",
            "Bmax_abs_core_T",
            "fundamental_order",
            "fundamental_amplitude_T",
            "dominant_slot_label",
            "dominant_slot_order",
            "dominant_slot_amplitude_T",
        ],
    )


def plot_airgap_distribution(results: list[dict], out_dir: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    for r in results:
        plt.plot(r["theta_deg"], r["br"], label=r["case"].name)
    plt.xlabel("posição angular mecânica θ [graus]")
    plt.ylabel("B_r no entreferro [T]")
    plt.title("Distribuição espacial da densidade de fluxo no entreferro")
    plt.xlim(0, 360)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_c_airgap_distribution.png", dpi=300)
    plt.close()


def plot_spectrum(results: list[dict], out_dir: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    for r in results:
        max_order = min(MAX_HARMONIC_TO_PLOT, len(r["orders"]) - 1)
        x = r["orders"][1:max_order + 1]
        y = r["amp"][1:max_order + 1]
        plt.plot(x, y, marker="o", markersize=2.5, linewidth=1.0, label=r["case"].name)

    # Marcadores para a fundamental de campo e para candidatos de ranhura do primeiro caso.
    first_case: FemCase = results[0]["case"]
    plt.axvline(first_case.pole_pairs, linestyle="--", linewidth=1.0, label=f"fundamental p={first_case.pole_pairs}")
    for label, order in slot_candidates(first_case):
        if order <= MAX_HARMONIC_TO_PLOT:
            plt.axvline(order, linestyle=":", linewidth=0.9)
            plt.text(order, plt.ylim()[1] * 0.92, str(order), rotation=90, va="top", ha="center", fontsize=8)

    plt.xlabel("ordem harmônica espacial mecânica")
    plt.ylabel("amplitude de B_r [T]")
    plt.title("Distribuição espectral da densidade de fluxo no entreferro")
    plt.xlim(1, MAX_HARMONIC_TO_PLOT)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_d_airgap_spectrum.png", dpi=300)
    plt.close()


def plot_slot_comparison(results: list[dict], out_dir: Path) -> None:
    labels = [r["case"].name for r in results]
    amplitudes = [r["dominant_slot"]["amplitude_t"] if r["dominant_slot"] else 0.0 for r in results]
    orders = [r["dominant_slot"]["order"] if r["dominant_slot"] else "-" for r in results]

    plt.figure(figsize=(8.5, 4.8))
    x = np.arange(len(labels))
    plt.bar(x, amplitudes)
    plt.xticks(x, [f"{lab}\nordem {ord_}" for lab, ord_ in zip(labels, orders)], rotation=0)
    plt.ylabel("amplitude do harmônico de ranhura dominante [T]")
    plt.title("Comparação do harmônico fundamental de ranhura")
    plt.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_d_slot_harmonic_comparison.png", dpi=300)
    plt.close()


def generate_final_outputs(results: list[dict]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"validated_results_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_minimal_csv(results, out_dir)
    plot_airgap_distribution(results, out_dir)
    plot_spectrum(results, out_dir)
    plot_slot_comparison(results, out_dir)
    return out_dir


def main() -> None:
    print("=== Análise FEMM simplificada ===")
    print(f"Casos configurados: {len(CASES)}")
    print(f"Critério: B{CORE_PERCENTILE:.1f}% no núcleo = {TARGET_B_CORE_T:.2f} ± {B_TOLERANCE_T:.2f} T\n")

    results: list[dict] = []
    invalid: list[tuple[FemCase, dict, float]] = []

    with tempfile.TemporaryDirectory(prefix="femm_motor_work_") as tmp:
        work_dir = Path(tmp)

        for case in CASES:
            print(f"[RUN] {case.name}")
            print(f"      arquivo: {case.fem_file}")
            print(f"      I_BASE: {case.i_base_a:.6g} A")
            result = run_femm_case(case, work_dir)
            ok, suggested = validate_case(result)

            b995 = result["core"]["b995_t"]
            bmax = result["core"]["bmax_abs_t"]
            print(f"      B99,5% núcleo = {b995:.5f} T")
            print(f"      Bmax absoluto = {bmax:.5f} T")
            print(f"      fundamental ordem {result['fundamental_order']} = {result['fundamental_amp_t']:.5f} T")
            if result["dominant_slot"]:
                ds = result["dominant_slot"]
                print(f"      ranhura dominante: {ds['label']} | ordem {ds['order']} | amplitude {ds['amplitude_t']:.6f} T")
            print(f"      validação: {'OK' if ok else 'NÃO OK'}")

            if not ok:
                invalid.append((case, result, suggested))
            results.append(result)

    if invalid:
        print("\n=== Corrente fora da faixa de validação ===")
        print("Nenhum gráfico final foi gerado. Atualize a corrente no código e rode novamente.\n")
        for case, result, suggested in invalid:
            print(f"Caso: {case.name}")
            print(f"  I_BASE atual      = {case.i_base_a:.6g} A")
            print(f"  B99,5% medido     = {result['core']['b995_t']:.5f} T")
            print(f"  I_BASE sugerido   = {suggested:.6g} A")
            print(f"  Correntes de fase = A:{suggested:.6g} A, B:{-0.5*suggested:.6g} A, C:{-0.5*suggested:.6g} A")
        return

    out_dir = generate_final_outputs(results)
    print("\n=== Validação aceita ===")
    print(f"Gráficos e CSVs mínimos salvos em: {out_dir}")
    print("Arquivos principais:")
    print(f"  - {out_dir / 'fig_c_airgap_distribution.png'}")
    print(f"  - {out_dir / 'fig_d_airgap_spectrum.png'}")
    print(f"  - {out_dir / 'fig_d_slot_harmonic_comparison.png'}")
    print(f"  - {out_dir / 'summary_validation_and_harmonics.csv'}")


if __name__ == "__main__":
    main()
