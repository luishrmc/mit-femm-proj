import matplotlib.pyplot as plt
import numpy as np
import csv

# ============================================================
# Distribuição de enrolamentos — máquina CA trifásica
# 36 ranhuras, 6 polos, q = 2
#
# Configurações:
# 1) Camada simples, passo pleno
# 2) Camada dupla, passo pleno
# 3) Camada dupla, passo encurtado 5/6
# ============================================================

# -----------------------------
# Dados principais da máquina
# -----------------------------
S = 36                 # número de ranhuras
P = 6                  # número de polos
m = 3                  # número de fases

slots_per_pole = S // P
q = S // (m * P)

pitch_full = slots_per_pole      # passo pleno = 6 ranhuras
pitch_short = 5                  # passo 5/6 = 5 ranhuras

slot_angle_mech_deg = 360 / S
slot_angle_elec_deg = (P / 2) * slot_angle_mech_deg

print("Dados geométricos:")
print(f"S = {S} ranhuras")
print(f"P = {P} polos")
print(f"m = {m} fases")
print(f"q = S/(mP) = {q}")
print(f"Passo mecânico de ranhura = {slot_angle_mech_deg:.1f} graus mecânicos")
print(f"Passo elétrico de ranhura = {slot_angle_elec_deg:.1f} graus elétricos")
print(f"Passo pleno = {pitch_full} ranhuras = {pitch_full * slot_angle_elec_deg:.1f} graus elétricos")
print(f"Passo encurtado 5/6 = {pitch_short} ranhuras = {pitch_short * slot_angle_elec_deg:.1f} graus elétricos")


# ============================================================
# 1. Sequência espacial base dos cinturões de fase
# ============================================================

def build_phase_sign_sequence():
    """
    Sequência espacial de fase e polaridade sem numeração de bobina.

    Convenção:
    - ranhura 1 no topo;
    - numeração no sentido horário;
    - q = 2 ranhuras por cinturão de fase.

    Um período de 12 ranhuras corresponde a dois polos.
    """
    one_two_pole_period = [
        ("A", "+"), ("A", "+"),
        ("C", "-"), ("C", "-"),
        ("B", "+"), ("B", "+"),
        ("A", "-"), ("A", "-"),
        ("C", "+"), ("C", "+"),
        ("B", "-"), ("B", "-"),
    ]

    return one_two_pole_period * (S // len(one_two_pole_period))


base_seq = build_phase_sign_sequence()


# ============================================================
# 2. Funções auxiliares
# ============================================================

def phase_of(label):
    return label[0]


def sign_of(label):
    return label[-1]


def invert_sign(sign):
    return "+" if sign == "-" else "-"


def invert_label(label):
    return label[:-1] + invert_sign(sign_of(label))


def get_color(label):
    phase = phase_of(label)

    if phase == "A":
        return "#d62728"   # vermelho
    if phase == "B":
        return "#1f77b4"   # azul
    if phase == "C":
        return "#2ca02c"   # verde

    return "gray"


def slot_to_angle(slot_number):
    """
    Converte ranhura 1..S para ângulo no desenho.
    Ranhura 1 fica em 12h e a numeração segue no sentido horário.
    """
    return np.pi / 2 - (slot_number - 1) * 2 * np.pi / S


def clockwise_arc(theta_start, pitch_slots, npts=80):
    """
    Gera arco no sentido horário correspondente ao passo da bobina.
    """
    dtheta = pitch_slots * 2 * np.pi / S
    return theta_start - np.linspace(0, dtheta, npts)


# ============================================================
# 3. Construção da camada simples
# ============================================================

def build_single_layer_full_pitch(base_seq, pitch):
    """
    Camada simples:
    - há um lado de bobina por ranhura;
    - cada bobina tem dois lados;
    - com 36 ranhuras, há 18 bobinas totais;
    - em passo pleno, os lados ficam separados por 6 ranhuras.

    Aqui os dois lados da mesma bobina recebem o mesmo índice.
    Exemplo:
    ranhura 1: A1+
    ranhura 7: A1-
    """
    labels = [""] * S
    visited = [False] * S

    counters = {"A": 1, "B": 1, "C": 1}
    pairs = []

    for i in range(S):
        if visited[i]:
            continue

        j = (i + pitch) % S

        phase_i, sign_i = base_seq[i]
        phase_j, sign_j = base_seq[j]

        if phase_i != phase_j:
            raise ValueError(
                f"Erro de pareamento: ranhuras {i+1} e {j+1} têm fases diferentes "
                f"({phase_i} e {phase_j})."
            )

        if sign_j != invert_sign(sign_i):
            raise ValueError(
                f"Erro de pareamento: ranhuras {i+1} e {j+1} não têm sinais opostos "
                f"({sign_i} e {sign_j})."
            )

        coil_id = counters[phase_i]
        counters[phase_i] += 1

        label_i = f"{phase_i}{coil_id}{sign_i}"
        label_j = f"{phase_j}{coil_id}{sign_j}"

        labels[i] = label_i
        labels[j] = label_j

        visited[i] = True
        visited[j] = True

        pairs.append({
            "coil": f"{phase_i}{coil_id}",
            "phase": phase_i,
            "slot_1": i + 1,
            "label_1": label_i,
            "slot_2": j + 1,
            "label_2": label_j,
            "pitch": pitch,
            "layer": "camada unica"
        })

    return labels, pairs


# ============================================================
# 4. Construção da camada dupla
# ============================================================

def build_double_layer(base_seq, pitch):
    """
    Camada dupla:
    - há um lado superior em cada ranhura;
    - cada lado superior define uma bobina;
    - o retorno é colocado na camada inferior da ranhura i + pitch;
    - com 36 ranhuras, há 36 bobinas totais.

    Cada bobina recebe um índice único por fase.
    Exemplo, passo pleno:
    topo ranhura 1: A1+
    fundo ranhura 7: A1-
    """
    top = [""] * S
    bottom = [""] * S
    counters = {"A": 1, "B": 1, "C": 1}
    pairs = []

    for i, (phase, sign) in enumerate(base_seq):
        coil_id = counters[phase]
        counters[phase] += 1

        top_label = f"{phase}{coil_id}{sign}"
        bottom_label = f"{phase}{coil_id}{invert_sign(sign)}"

        target = (i + pitch) % S

        top[i] = top_label
        bottom[target] = bottom_label

        pairs.append({
            "coil": f"{phase}{coil_id}",
            "phase": phase,
            "slot_top": i + 1,
            "label_top": top_label,
            "slot_bottom": target + 1,
            "label_bottom": bottom_label,
            "pitch": pitch,
            "layer": "dupla"
        })

    if any(x == "" for x in bottom):
        raise ValueError("Erro: alguma ranhura ficou sem lado inferior.")

    return top, bottom, pairs


# ============================================================
# 5. Geração das três configurações
# ============================================================

# 1) Camada simples, passo pleno
single_labels, single_pairs = build_single_layer_full_pitch(base_seq, pitch_full)

# 2) Camada dupla, passo pleno
top_full, bottom_full, double_full_pairs = build_double_layer(base_seq, pitch_full)

# 3) Camada dupla, passo encurtado 5/6
top_short, bottom_short, double_short_pairs = build_double_layer(base_seq, pitch_short)


# ============================================================
# 6. Impressão de tabelas
# ============================================================

def print_single_layer_table(labels):
    print("\n" + "=" * 72)
    print("1) Camada simples — passo pleno")
    print("=" * 72)
    print(f"{'Ranhura':>8} | {'Lado de bobina':>16}")
    print("-" * 72)

    for k, label in enumerate(labels, start=1):
        print(f"{k:8d} | {label:>16}")


def print_double_layer_table(title, top, bottom):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'Ranhura':>8} | {'Camada superior':>18} | {'Camada inferior':>18}")
    print("-" * 72)

    for k in range(S):
        print(f"{k+1:8d} | {top[k]:>18} | {bottom[k]:>18}")


def print_pairs(title, pairs, double_layer=False):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    if double_layer:
        print(f"{'Bobina':>8} | {'Topo':>14} | {'Fundo':>14} | {'Passo':>6}")
        print("-" * 72)
        for p in pairs:
            print(
                f"{p['coil']:>8} | "
                f"{p['slot_top']:02d} {p['label_top']:>6} | "
                f"{p['slot_bottom']:02d} {p['label_bottom']:>6} | "
                f"{p['pitch']:6d}"
            )
    else:
        print(f"{'Bobina':>8} | {'Lado 1':>14} | {'Lado 2':>14} | {'Passo':>6}")
        print("-" * 72)
        for p in pairs:
            print(
                f"{p['coil']:>8} | "
                f"{p['slot_1']:02d} {p['label_1']:>6} | "
                f"{p['slot_2']:02d} {p['label_2']:>6} | "
                f"{p['pitch']:6d}"
            )


# print_single_layer_table(single_labels)
# print_pairs("Pares de bobina — camada simples, passo pleno", single_pairs, double_layer=False)

# print_double_layer_table("2) Camada dupla — passo pleno", top_full, bottom_full)
# print_pairs("Pares de bobina — camada dupla, passo pleno", double_full_pairs, double_layer=True)

# print_double_layer_table("3) Camada dupla — passo encurtado 5/6", top_short, bottom_short)
# print_pairs("Pares de bobina — camada dupla, passo encurtado 5/6", double_short_pairs, double_layer=True)


# ============================================================
# 7. Exportação das tabelas em CSV
# ============================================================

def export_pairs_csv(filename, pairs, double_layer=False):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if double_layer:
            writer.writerow(["bobina", "fase", "ranhura_topo", "rotulo_topo",
                             "ranhura_fundo", "rotulo_fundo", "passo"])
            for p in pairs:
                writer.writerow([
                    p["coil"], p["phase"], p["slot_top"], p["label_top"],
                    p["slot_bottom"], p["label_bottom"], p["pitch"]
                ])
        else:
            writer.writerow(["bobina", "fase", "ranhura_lado_1", "rotulo_lado_1",
                             "ranhura_lado_2", "rotulo_lado_2", "passo"])
            for p in pairs:
                writer.writerow([
                    p["coil"], p["phase"], p["slot_1"], p["label_1"],
                    p["slot_2"], p["label_2"], p["pitch"]
                ])


export_pairs_csv("bobinas_camada_simples_passo_pleno.csv", single_pairs, double_layer=False)
export_pairs_csv("bobinas_camada_dupla_passo_pleno.csv", double_full_pairs, double_layer=True)
export_pairs_csv("bobinas_camada_dupla_passo_encurtado_5_6.csv", double_short_pairs, double_layer=True)


# ============================================================
# 8. Plotagem
# ============================================================

def draw_slot_boundaries(ax):
    r_in = 7.5
    r_out = 10.5

    for i in range(S):
        theta = np.pi / 2 - (i - 0.5) * 2 * np.pi / S
        ax.plot(
            [r_in * np.cos(theta), r_out * np.cos(theta)],
            [r_in * np.sin(theta), r_out * np.sin(theta)],
            color="lightgray",
            lw=0.8,
            zorder=1
        )

def setup_motor_axes(ax, title):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, y=1.05, fontsize=12, fontweight="bold")

    # contornos básicos
    ax.add_patch(plt.Circle((0, 0), 10.5, fill=False, color="gray", lw=1.4))
    ax.add_patch(plt.Circle((0, 0), 7.5, fill=False, color="gray", lw=1.4))
    ax.add_patch(plt.Circle((0, 0), 6.4, fill=False, color="gray", lw=0.9))
    ax.add_patch(plt.Circle((0, 0), 3.1, fill=False, color="gray", lw=0.9))

    draw_slot_boundaries(ax)

    # numeração das ranhuras
    for i in range(1, S + 1):
        theta = slot_to_angle(i)
        ax.text(
            11 * np.cos(theta),
            11 * np.sin(theta),
            str(i),
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            zorder=5
        )


def plot_single_layer(ax, labels, pairs, title):
    setup_motor_axes(ax, title)

    r = 8.85

    for slot, label in enumerate(labels, start=1):
        theta = slot_to_angle(slot)
        color = get_color(label)

        ax.plot(
            r * np.cos(theta),
            r * np.sin(theta),
            "o",
            color=color,
            markersize=17,
            zorder=10
        )
        ax.text(
            r * np.cos(theta),
            r * np.sin(theta),
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=6.3,
            fontweight="bold",
            zorder=11
        )


def plot_double_layer(ax, top, bottom, pairs, title):
    setup_motor_axes(ax, title)

    r_top = 8.20       # mais próximo do entreferro
    r_bottom = 9.35    # mais externo

    for slot in range(1, S + 1):
        theta = slot_to_angle(slot)

        top_label = top[slot - 1]
        bottom_label = bottom[slot - 1]

        # Camada inferior
        ax.plot(
            r_bottom * np.cos(theta),
            r_bottom * np.sin(theta),
            "o",
            color=get_color(bottom_label),
            markersize=13,
            zorder=9
        )
        ax.text(
            r_bottom * np.cos(theta),
            r_bottom * np.sin(theta),
            bottom_label,
            ha="center",
            va="center",
            color="white",
            fontsize=5.2,
            fontweight="bold",
            zorder=10
        )

        # Camada superior
        ax.plot(
            r_top * np.cos(theta),
            r_top * np.sin(theta),
            "o",
            color=get_color(top_label),
            markersize=13,
            zorder=11
        )
        ax.text(
            r_top * np.cos(theta),
            r_top * np.sin(theta),
            top_label,
            ha="center",
            va="center",
            color="white",
            fontsize=5.2,
            fontweight="bold",
            zorder=12
        )

fig, axes = plt.subplots(1, 3, figsize=(23, 7.8))

plot_single_layer(
    axes[0],
    single_labels,
    single_pairs,
    "1) Camada simples — passo pleno\npasso = 6 ranhuras"
)

plot_double_layer(
    axes[1],
    top_full,
    bottom_full,
    double_full_pairs,
    "2) Camada dupla — passo pleno\npasso = 6 ranhuras"
)

plot_double_layer(
    axes[2],
    top_short,
    bottom_short,
    double_short_pairs,
    "3) Camada dupla — passo encurtado 5/6\npasso = 5 ranhuras"
)

plt.tight_layout()
plt.savefig("distribuicao_enrolamentos_final_corrigida.png", dpi=300, bbox_inches="tight")
plt.show()