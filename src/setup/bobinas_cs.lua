-- ============================================================
-- FEMM Lua script
-- Posicionamento dos block labels das bobinas do estator
-- Camada simples - passo pleno
-- 36 ranhuras, passo = 6 ranhuras
-- Material: Air
-- Espiras configuráveis
-- ============================================================

clearconsole()

-- ------------------------------------------------------------
-- 1. Parâmetros configuráveis
-- ------------------------------------------------------------

N_ESPIRAS = 20        -- altere aqui o número de espiras por lado de bobina

R_BOBINA = 72.0       -- raio médio das regiões de bobina [mm]
N_RANHURAS = 36

-- Ângulo empírico da primeira bobina/ranhura.
-- Obtido aproximadamente de atan2(71, 6) = 85.17 graus.
THETA_RANHURA_1 = 85.17

-- Como a numeração segue no sentido horário,
-- o ângulo diminui a cada ranhura.
PASSO_ANGULAR = 360.0 / N_RANHURAS

MATERIAL_BOBINA = "Air"
MESH_BOBINA = 1.0

-- Circuitos FEMM
CIRCUITO_A = "fase_A"
CIRCUITO_B = "fase_B"
CIRCUITO_C = "fase_C"

-- Grupos FEMM
GRUPO_A = 5
GRUPO_B = 6
GRUPO_C = 7

-- ------------------------------------------------------------
-- 2. Funções auxiliares compatíveis com Lua do FEMM
-- ------------------------------------------------------------

PI = 4 * atan(1)

function deg2rad(ang)
    return ang * PI / 180.0
end

function angulo_ranhura(n)
    return THETA_RANHURA_1 - (n - 1) * PASSO_ANGULAR
end

function adiciona_label_bobina(ranhura, circuito, sinal, grupo)
    local theta_deg = angulo_ranhura(ranhura)
    local theta = deg2rad(theta_deg)

    local x = R_BOBINA * cos(theta)
    local y = R_BOBINA * sin(theta)

    local espiras = sinal * N_ESPIRAS

    mi_addblocklabel(x, y)
    mi_selectlabel(x, y)

    -- mi_setblockprop(material, automesh, meshsize, circuito, magdir, grupo, turns)
    mi_setblockprop(MATERIAL_BOBINA, 0, MESH_BOBINA, circuito, 0, grupo, espiras)

    mi_clearselected()

    print("Ranhura", ranhura, "x=", x, "y=", y, "circuito=", circuito, "espiras=", espiras)
end

-- ------------------------------------------------------------
-- 3. Mapeamento das bobinas
-- Camada simples, passo pleno
-- ------------------------------------------------------------

-- Formato:
-- {ranhura, circuito, sinal, grupo}
--
-- sinal = +1  -> N_ESPIRAS positivo
-- sinal = -1  -> N_ESPIRAS negativo

bobinas = {
    {1,  CIRCUITO_A,  1, GRUPO_A},
    {2,  CIRCUITO_A,  1, GRUPO_A},
    {3,  CIRCUITO_C, -1, GRUPO_C},
    {4,  CIRCUITO_C, -1, GRUPO_C},
    {5,  CIRCUITO_B,  1, GRUPO_B},
    {6,  CIRCUITO_B,  1, GRUPO_B},

    {7,  CIRCUITO_A, -1, GRUPO_A},
    {8,  CIRCUITO_A, -1, GRUPO_A},
    {9,  CIRCUITO_C,  1, GRUPO_C},
    {10, CIRCUITO_C,  1, GRUPO_C},
    {11, CIRCUITO_B, -1, GRUPO_B},
    {12, CIRCUITO_B, -1, GRUPO_B},

    {13, CIRCUITO_A,  1, GRUPO_A},
    {14, CIRCUITO_A,  1, GRUPO_A},
    {15, CIRCUITO_C, -1, GRUPO_C},
    {16, CIRCUITO_C, -1, GRUPO_C},
    {17, CIRCUITO_B,  1, GRUPO_B},
    {18, CIRCUITO_B,  1, GRUPO_B},

    {19, CIRCUITO_A, -1, GRUPO_A},
    {20, CIRCUITO_A, -1, GRUPO_A},
    {21, CIRCUITO_C,  1, GRUPO_C},
    {22, CIRCUITO_C,  1, GRUPO_C},
    {23, CIRCUITO_B, -1, GRUPO_B},
    {24, CIRCUITO_B, -1, GRUPO_B},

    {25, CIRCUITO_A,  1, GRUPO_A},
    {26, CIRCUITO_A,  1, GRUPO_A},
    {27, CIRCUITO_C, -1, GRUPO_C},
    {28, CIRCUITO_C, -1, GRUPO_C},
    {29, CIRCUITO_B,  1, GRUPO_B},
    {30, CIRCUITO_B,  1, GRUPO_B},

    {31, CIRCUITO_A, -1, GRUPO_A},
    {32, CIRCUITO_A, -1, GRUPO_A},
    {33, CIRCUITO_C,  1, GRUPO_C},
    {34, CIRCUITO_C,  1, GRUPO_C},
    {35, CIRCUITO_B, -1, GRUPO_B},
    {36, CIRCUITO_B, -1, GRUPO_B}
}

-- ------------------------------------------------------------
-- 4. Inserção dos block labels
-- ------------------------------------------------------------

for k = 1, 36 do
    adiciona_label_bobina(
        bobinas[k][1],
        bobinas[k][2],
        bobinas[k][3],
        bobinas[k][4]
    )
end

mi_zoomnatural()
mi_refreshview()

print("Block labels das bobinas inseridos.")
print("Numero de espiras por lado de bobina:", N_ESPIRAS)