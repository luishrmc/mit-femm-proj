"""
Calibração automática das correntes dos modelos FEMM para B99,5% ~= 1,7 T.

Objetivo exclusivo:
1) Abrir os arquivos .FEM dos três motores.
2) Ajustar a corrente base de cada caso, mantendo:
       fase_A = I_BASE
       fase_B = -0.5 * I_BASE
       fase_C = -0.5 * I_BASE
3) Executar o solver até encontrar o maior I_BASE tal que:
       B99,5% no núcleo <= 1,700 T
   e, preferencialmente, dentro de TARGET_TOLERANCE_T abaixo da meta.
4) Salvar o .FEM e o .ANS finais correspondentes à corrente calibrada.

Observações:
- O script preserva os arquivos .FEM originais. Cada caso é salvo como cópia de trabalho.
- O critério usa os grupos 1 e 2 como estator e rotor ferromagnéticos.
- O máximo absoluto Bmax é registrado apenas como diagnóstico; ele não é usado como critério
  principal porque pode ocorrer em quinas ou pontos localizados da malha.

Dependências:
    pip install pyfemm numpy

Antes de executar:
- Ajuste FEM_SEARCH_ROOT, RESULTS_DIR e, se necessário, defina fem_file explicitamente em CASES.
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

# Pasta onde estão os arquivos .FEM originais.
FEM_SEARCH_ROOT = Path(r"C:\Users\lumac\ufmg\braz\conv-trab\motor\femm")

# Pasta de saída da calibração.
RESULTS_DIR = Path(r"C:\Users\lumac\ufmg\braz\conv-trab\src\sim\out\current_calibration")

# Meta do projeto: B representativo no núcleo próximo de 1,7 T, mas sem ultrapassar.
TARGET_B995_T = 1.700
TARGET_TOLERANCE_T = 0.005       # aceita B99,5% em [1,695 ; 1,700] T
CORE_PERCENTILE = 99.5

# Grupos do FEMM definidos nos block labels.
GROUP_STATOR = 1
GROUP_ROTOR = 2

# Controle de execução do FEMM.
SHOW_FEMM = True                 # mantém a interface FEMM visível
SHOW_SOLVER_WINDOW = True        # mantém a janela do solver visível
SMOOTHING = "off"                # recomendado para medição direta por elementos

# Controle do algoritmo de busca.
MAX_EXPANSIONS = 10              # tentativas para encontrar faixa [abaixo, acima]
MAX_BISECTION_ITERS = 14         # iterações de bisseção dentro da faixa encontrada
INITIAL_INCREASE_FACTOR = 1.08   # se B inicial estiver abaixo da meta
INITIAL_DECREASE_FACTOR = 0.92   # se B inicial estiver acima da meta
MIN_CURRENT_A = 0.01

# Circuitos usados nos modelos.
CIRCUIT_A = "fase_A"
CIRCUIT_B = "fase_B"
CIRCUIT_C = "fase_C"


@dataclass(frozen=True)
class FemCalibrationCase:
    name: str
    initial_i_base_a: float
    fem_file: Path | None = None
    search_patterns: tuple[str, ...] = ()


# Se fem_file=None, o script procura automaticamente o arquivo .FEM usando
# search_patterns dentro de FEM_SEARCH_ROOT.
# Se a busca automática não encontrar o arquivo correto, defina fem_file explicitamente.
CASES: list[FemCalibrationCase] = [
    FemCalibrationCase(
        name="motor_cs_pp",
        initial_i_base_a=12.544,
        fem_file=None,
        search_patterns=(
            "motor_cs_pp.FEM",
            "motor_cs.FEM",
            "*motor_cs_pp*.FEM",
            "*motor_cs*.FEM",
            "*cs*pp*.FEM",
        ),
    ),
    FemCalibrationCase(
        name="motor_cd_pp",
        initial_i_base_a=5.988,
        fem_file=None,
        search_patterns=(
            "motor_cd_pp.FEM",
            "*motor_cd_pp*.FEM",
            "*cd*pp*.FEM",
            "*dupla*pleno*.FEM",
        ),
    ),
    FemCalibrationCase(
        name="motor_cd_pe",
        initial_i_base_a=6.188,
        fem_file=None,
        search_patterns=(
            "motor_cd_pe.FEM",
            "*motor_cd_pe*.FEM",
            "*cd*pe*.FEM",
            "*encurtado*.FEM",
        ),
    ),
]


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def as_float(value) -> float:
    if isinstance(value, complex):
        return float(value.real)
    return float(value)


def bmag(bx: float, by: float) -> float:
    return math.hypot(bx, by)


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Lista vazia para cálculo de percentil.")
    return float(np.percentile(np.asarray(values, dtype=float), p))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_fem_file(case: FemCalibrationCase) -> Path:
    if case.fem_file is not None:
        if not case.fem_file.exists():
            raise FileNotFoundError(f"Arquivo .FEM não encontrado para {case.name}: {case.fem_file}")
        return case.fem_file

    if not FEM_SEARCH_ROOT.exists():
        raise FileNotFoundError(f"FEM_SEARCH_ROOT não existe: {FEM_SEARCH_ROOT}")

    matches: list[Path] = []
    for pattern in case.search_patterns:
        matches.extend(FEM_SEARCH_ROOT.glob(pattern))

    unique = sorted(set(matches), key=lambda p: p.stat().st_mtime, reverse=True)
    if not unique:
        raise FileNotFoundError(
            f"Nenhum .FEM encontrado para {case.name} em {FEM_SEARCH_ROOT}. "
            "Defina fem_file explicitamente no caso correspondente."
        )

    return unique[0]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_phase_currents(i_base_a: float) -> None:
    """Define as correntes trifásicas instantâneas balanceadas no FEMM."""
    femm.mi_modifycircprop(CIRCUIT_A, 1, i_base_a)
    femm.mi_modifycircprop(CIRCUIT_B, 1, -0.5 * i_base_a)
    femm.mi_modifycircprop(CIRCUIT_C, 1, -0.5 * i_base_a)


def collect_core_b_indicator() -> dict[str, float]:
    """Calcula B99,5% e Bmax nos elementos pertencentes aos grupos do núcleo."""
    ne = int(femm.mo_numelements()) # type: ignore
    values: list[float] = []
    bmax = -1.0
    xmax = 0.0
    ymax = 0.0
    gmax = -1

    def scan(indices: Iterable[int]) -> None:
        nonlocal bmax, xmax, ymax, gmax
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
                    bmax = bm
                    xmax = x
                    ymax = y
                    gmax = group
            except Exception:
                continue

    # FEMM geralmente usa indexação 0..ne-1. A segunda tentativa cobre instalações
    # que porventura respondam melhor com 1..ne.
    scan(range(ne))
    if not values:
        scan(range(1, ne + 1))

    if not values:
        raise RuntimeError(
            "Nenhum elemento dos grupos 1 e 2 foi encontrado. "
            "Verifique os grupos do estator e rotor nos block labels."
        )

    return {
        "b995_t": percentile(values, CORE_PERCENTILE),
        "bmax_abs_t": bmax,
        "x_bmax_mm": xmax,
        "y_bmax_mm": ymax,
        "group_bmax": float(gmax),
        "n_core_samples": float(len(values)),
    }


def solve_and_measure(i_base_a: float, solver_flag: int) -> dict[str, float]:
    """Atualiza corrente, executa análise e mede B99,5% no núcleo."""
    set_phase_currents(i_base_a)

    t0 = time.perf_counter()
    femm.mi_analyze(solver_flag)
    solve_time = time.perf_counter() - t0

    femm.mi_loadsolution()
    femm.mo_smooth(SMOOTHING)
    indicator = collect_core_b_indicator()
    femm.mo_close()

    indicator["i_base_a"] = i_base_a
    indicator["phase_A_A"] = i_base_a
    indicator["phase_B_A"] = -0.5 * i_base_a
    indicator["phase_C_A"] = -0.5 * i_base_a
    indicator["solve_time_s"] = solve_time
    return indicator


def within_target_without_exceeding(b995_t: float) -> bool:
    return (TARGET_B995_T - TARGET_TOLERANCE_T) <= b995_t <= TARGET_B995_T


def status_from_b995(b995_t: float) -> str:
    if within_target_without_exceeding(b995_t):
        return "OK"
    if b995_t > TARGET_B995_T:
        return "ACIMA_DA_META"
    return "ABAIXO_DA_META"


# =============================================================================
# CALIBRAÇÃO DE UM CASO
# =============================================================================

def calibrate_case(case: FemCalibrationCase, out_root: Path, solver_flag: int) -> dict:
    original_fem = find_fem_file(case)
    case_dir = out_root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)

    backup_fem = case_dir / f"{case.name}_original_backup.FEM"
    work_fem = case_dir / f"{case.name}_calibration_work.FEM"
    final_fem = case_dir / f"{case.name}_calibrated.FEM"
    final_ans = case_dir / f"{case.name}_calibrated.ans"
    iteration_csv = case_dir / f"{case.name}_calibration_iterations.csv"

    shutil.copy2(original_fem, backup_fem)

    print("=" * 78)
    print(f"[CASO] {case.name}")
    print(f"  FEM original : {original_fem}")
    print(f"  FEM trabalho : {work_fem}")
    print(f"  I inicial    : {case.initial_i_base_a:.6f} A")
    print("=" * 78)

    femm.opendocument(str(original_fem))
    femm.mi_saveas(str(work_fem))

    rows: list[dict] = []

    def evaluate(i_base: float, stage: str, step: int) -> dict[str, float]:
        print(f"[{case.name}] {stage:<12s} #{step:02d} | I_BASE = {i_base:.6f} A")
        r = solve_and_measure(i_base, solver_flag)
        st = status_from_b995(r["b995_t"])
        print(
            f"    B99,5% = {r['b995_t']:.6f} T [{st}] | "
            f"Bmax = {r['bmax_abs_t']:.6f} T | tempo = {r['solve_time_s']:.2f} s"
        )
        row = {
            "case": case.name,
            "stage": stage,
            "step": step,
            "I_BASE_A": r["i_base_a"],
            "fase_A_A": r["phase_A_A"],
            "fase_B_A": r["phase_B_A"],
            "fase_C_A": r["phase_C_A"],
            "B99_5_core_T": r["b995_t"],
            "Bmax_abs_core_T": r["bmax_abs_t"],
            "status": st,
            "solve_time_s": r["solve_time_s"],
            "x_Bmax_mm": r["x_bmax_mm"],
            "y_Bmax_mm": r["y_bmax_mm"],
            "group_Bmax": int(r["group_bmax"]),
            "n_core_samples": int(r["n_core_samples"]),
        }
        rows.append(row)
        write_csv(
            iteration_csv,
            rows,
            [
                "case", "stage", "step", "I_BASE_A", "fase_A_A", "fase_B_A", "fase_C_A",
                "B99_5_core_T", "Bmax_abs_core_T", "status", "solve_time_s",
                "x_Bmax_mm", "y_Bmax_mm", "group_Bmax", "n_core_samples",
            ],
        )
        return r

    # 1) Avaliação inicial.
    step = 0
    r0 = evaluate(case.initial_i_base_a, "inicial", step)

    if within_target_without_exceeding(r0["b995_t"]):
        low_i = case.initial_i_base_a
        low_r = r0
        bracket_found = False
    elif r0["b995_t"] < TARGET_B995_T:
        # Precisa aumentar até passar da meta para formar faixa.
        low_i = case.initial_i_base_a
        low_r = r0
        high_i = None
        high_r = None
        bracket_found = False

        test_i = low_i
        for k in range(1, MAX_EXPANSIONS + 1):
            test_i *= INITIAL_INCREASE_FACTOR
            rk = evaluate(test_i, "expande_up", k)
            if rk["b995_t"] <= TARGET_B995_T:
                low_i, low_r = test_i, rk
                if within_target_without_exceeding(rk["b995_t"]):
                    bracket_found = False
                    break
            else:
                high_i, high_r = test_i, rk
                bracket_found = True
                break

        if bracket_found:
            low_i, low_r = bisect_current(case, low_i, low_r, high_i, high_r, evaluate) # type: ignore

    else:
        # Corrente inicial está acima da meta. Reduz até ficar abaixo da meta.
        high_i = case.initial_i_base_a
        high_r = r0
        low_i = None
        low_r = None
        bracket_found = False

        test_i = high_i
        for k in range(1, MAX_EXPANSIONS + 1):
            test_i *= INITIAL_DECREASE_FACTOR
            test_i = max(test_i, MIN_CURRENT_A)
            rk = evaluate(test_i, "expande_down", k)
            if rk["b995_t"] <= TARGET_B995_T:
                low_i, low_r = test_i, rk
                bracket_found = True
                break
            high_i, high_r = test_i, rk

        if not bracket_found or low_i is None or low_r is None:
            raise RuntimeError(f"Não foi possível encontrar corrente abaixo da meta para {case.name}.")

        low_i, low_r = bisect_current(case, low_i, low_r, high_i, high_r, evaluate)

    # 2) Garante que o arquivo .ANS final corresponde exatamente ao ponto aceito.
    print(f"[{case.name}] Reexecutando solução final aceita: I_BASE = {low_i:.6f} A")
    final_r = solve_and_measure(low_i, solver_flag)
    rows.append({
        "case": case.name,
        "stage": "final",
        "step": 0,
        "I_BASE_A": final_r["i_base_a"],
        "fase_A_A": final_r["phase_A_A"],
        "fase_B_A": final_r["phase_B_A"],
        "fase_C_A": final_r["phase_C_A"],
        "B99_5_core_T": final_r["b995_t"],
        "Bmax_abs_core_T": final_r["bmax_abs_t"],
        "status": status_from_b995(final_r["b995_t"]),
        "solve_time_s": final_r["solve_time_s"],
        "x_Bmax_mm": final_r["x_bmax_mm"],
        "y_Bmax_mm": final_r["y_bmax_mm"],
        "group_Bmax": int(final_r["group_bmax"]),
        "n_core_samples": int(final_r["n_core_samples"]),
    })
    write_csv(
        iteration_csv,
        rows,
        [
            "case", "stage", "step", "I_BASE_A", "fase_A_A", "fase_B_A", "fase_C_A",
            "B99_5_core_T", "Bmax_abs_core_T", "status", "solve_time_s",
            "x_Bmax_mm", "y_Bmax_mm", "group_Bmax", "n_core_samples",
        ],
    )

    # Salva .FEM final com as correntes finais.
    femm.mi_saveas(str(final_fem))
    femm.mi_close()

    # A solução .ANS é gerada com o mesmo nome do arquivo .FEM ativo antes do mi_saveas final.
    # Tentamos copiar do .ANS de trabalho e, se necessário, do .ANS final.
    candidate_ans = [work_fem.with_suffix(".ans"), final_fem.with_suffix(".ans")]
    copied = False
    for ans in candidate_ans:
        if ans.exists():
            if ans.resolve() != final_ans.resolve():
                shutil.copy2(ans, final_ans)
            copied = True
            break

    if not copied:
        raise FileNotFoundError(
            f"Não foi possível localizar o .ANS final de {case.name}. "
            f"Tentativas: {candidate_ans}"
        )

    print(f"[{case.name}] FINAL")
    print(f"  I_BASE final  = {final_r['i_base_a']:.6f} A")
    print(f"  fase_A        = {final_r['phase_A_A']:.6f} A")
    print(f"  fase_B        = {final_r['phase_B_A']:.6f} A")
    print(f"  fase_C        = {final_r['phase_C_A']:.6f} A")
    print(f"  B99,5% núcleo = {final_r['b995_t']:.6f} T [{status_from_b995(final_r['b995_t'])}]")
    print(f"  Bmax absoluto = {final_r['bmax_abs_t']:.6f} T")
    print(f"  FEM salvo     = {final_fem}")
    print(f"  ANS salvo     = {final_ans}\n")

    return {
        "case": case.name,
        "original_fem": str(original_fem),
        "backup_fem": str(backup_fem),
        "final_fem": str(final_fem),
        "final_ans": str(final_ans),
        "iterations_csv": str(iteration_csv),
        "I_BASE_A": final_r["i_base_a"],
        "fase_A_A": final_r["phase_A_A"],
        "fase_B_A": final_r["phase_B_A"],
        "fase_C_A": final_r["phase_C_A"],
        "B99_5_core_T": final_r["b995_t"],
        "Bmax_abs_core_T": final_r["bmax_abs_t"],
        "status": status_from_b995(final_r["b995_t"]),
        "x_Bmax_mm": final_r["x_bmax_mm"],
        "y_Bmax_mm": final_r["y_bmax_mm"],
        "group_Bmax": int(final_r["group_bmax"]),
    }


def bisect_current(
    case: FemCalibrationCase,
    low_i: float,
    low_r: dict[str, float],
    high_i: float,
    high_r: dict[str, float],
    evaluate_func,
) -> tuple[float, dict[str, float]]:
    """
    Bisseção entre:
      low_i  -> B99,5% <= TARGET_B995_T
      high_i -> B99,5% >  TARGET_B995_T

    Retorna sempre o melhor ponto abaixo ou igual à meta.
    """
    if low_r["b995_t"] > TARGET_B995_T:
        raise ValueError("low_r está acima da meta; bisseção inválida.")
    if high_r["b995_t"] <= TARGET_B995_T:
        raise ValueError("high_r não está acima da meta; bisseção inválida.")

    best_i = low_i
    best_r = low_r

    for k in range(1, MAX_BISECTION_ITERS + 1):
        mid_i = 0.5 * (low_i + high_i)
        mid_r = evaluate_func(mid_i, "bissecao", k)
        mid_b = mid_r["b995_t"]

        if mid_b <= TARGET_B995_T:
            low_i, low_r = mid_i, mid_r
            best_i, best_r = mid_i, mid_r
            if within_target_without_exceeding(mid_b):
                # Continua mesmo assim para aproximar o maior valor seguro, sem ultrapassar.
                pass
        else:
            high_i, high_r = mid_i, mid_r

        if 0.0 <= TARGET_B995_T - best_r["b995_t"] <= TARGET_TOLERANCE_T:
            # Já atingiu a faixa desejada sem ultrapassar.
            break

    return best_i, best_r


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main() -> None:
    print("=== Calibração de corrente FEMM para B99,5% ~= 1,7 T sem ultrapassar ===")
    print(f"Casos configurados: {len(CASES)}")
    print(f"Meta: B{CORE_PERCENTILE:.1f}% <= {TARGET_B995_T:.3f} T")
    print(f"Faixa desejada: {TARGET_B995_T - TARGET_TOLERANCE_T:.3f} T a {TARGET_B995_T:.3f} T")
    print("Correntes impostas como: fase_A=I, fase_B=-0,5I, fase_C=-0,5I\n")

    out_root = RESULTS_DIR / f"calibration_{timestamp()}"
    out_root.mkdir(parents=True, exist_ok=True)

    solver_flag = 0 if SHOW_SOLVER_WINDOW else 1
    summaries: list[dict] = []

    femm_open = False
    t_total = time.perf_counter()

    try:
        femm.openfemm(0 if SHOW_FEMM else 1)
        femm_open = True

        for case in CASES:
            summary = calibrate_case(case, out_root, solver_flag)
            summaries.append(summary)

        summary_csv = out_root / "calibrated_currents_summary.csv"
        write_csv(
            summary_csv,
            summaries,
            [
                "case", "I_BASE_A", "fase_A_A", "fase_B_A", "fase_C_A",
                "B99_5_core_T", "Bmax_abs_core_T", "status",
                "x_Bmax_mm", "y_Bmax_mm", "group_Bmax",
                "original_fem", "backup_fem", "final_fem", "final_ans", "iterations_csv",
            ],
        )

        print("=" * 78)
        print("RESUMO FINAL")
        print("=" * 78)
        for s in summaries:
            print(
                f"{s['case']:<14s} | I_BASE={s['I_BASE_A']:.6f} A | "
                f"B99,5%={s['B99_5_core_T']:.6f} T | {s['status']}"
            )
        print(f"\nResumo CSV: {summary_csv}")
        print(f"Pasta de saída: {out_root}")

    finally:
        if femm_open:
            try:
                femm.closefemm()
            except Exception:
                pass
        print(f"Tempo total: {time.perf_counter() - t_total:.2f} s")


if __name__ == "__main__":
    main()
