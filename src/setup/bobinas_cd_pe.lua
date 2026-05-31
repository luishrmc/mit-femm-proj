-- ============================================================
-- FEMM Lua script
-- Posicionamento dos block labels das bobinas do estator
-- Camada dupla - passo encurtado
-- 36 ranhuras
-- Camada superior: R = 72 mm
-- Camada inferior: R = 62 mm
-- Material: Air
-- Espiras configuráveis
-- ============================================================

clearconsole()

-- ------------------------------------------------------------
-- 1. Parâmetros configuráveis
-- ------------------------------------------------------------

N_ESPIRAS = 20        -- altere aqui o número de espiras por lado de bobina

N_RANHURAS = 36
PASSO_RANHURAS = 5    -- passo encurtado para máquina de 36 ranhuras e 6 polos

R_SUPERIOR = 72.0     -- raio da camada superior [mm]
R_INFERIOR = 62.0     -- raio da camada inferior [mm]

-- Ângulo empírico da primeira ranhura.
-- Obtido aproximadamente de atan2(71, 6) = 85.17 graus.
THETA_RANHURA_1 = 85.17

-- Numeração das ranhuras no sentido horário.
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

function normaliza_ranhura(n)
    while n > N_RANHURAS do
        n = n - N_RANHURAS
    end

    while n < 1 do
        n = n + N_RANHURAS
    end

    return n
end

function adiciona_label_bobina(ranhura, raio, circuito, sinal, grupo, camada)
    local theta_deg = angulo_ranhura(ranhura)
    local theta = deg2rad(theta_deg)

    local x = raio * cos(theta)
    local y = raio * sin(theta)

    local espiras = sinal * N_ESPIRAS

    mi_addblocklabel(x, y)
    mi_selectlabel(x, y)

    -- mi_setblockprop(material, automesh, meshsize, circuito, magdir, grupo, turns)
    mi_setblockprop(MATERIAL_BOBINA, 0, MESH_BOBINA, circuito, 0, grupo, espiras)

    mi_clearselected()

    print(
        "Camada=", camada,
        " Ranhura=", ranhura,
        " x=", x,
        " y=", y,
        " circuito=", circuito,
        " espiras=", espiras,
        " grupo=", grupo
    )
end

-- ------------------------------------------------------------
-- 3. Mapeamento da camada superior
-- Camada dupla, passo pleno
-- ------------------------------------------------------------
-- Formato:
-- {ranhura_superior, circuito, sinal_superior, grupo}
--
-- A camada inferior é gerada automaticamente:
-- ranhura_inferior = ranhura_superior + PASSO_RANHURAS
-- sinal_inferior = -sinal_superior
--
-- Isso representa os dois lados da mesma bobina em passo pleno.

bobinas_superior = {
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
    local ranhura_sup = bobinas_superior[k][1]
    local circuito    = bobinas_superior[k][2]
    local sinal_sup   = bobinas_superior[k][3]
    local grupo       = bobinas_superior[k][4]

    local ranhura_inf = normaliza_ranhura(ranhura_sup + PASSO_RANHURAS)
    local sinal_inf   = -sinal_sup

    -- Camada superior
    adiciona_label_bobina(
        ranhura_sup,
        R_SUPERIOR,
        circuito,
        sinal_sup,
        grupo,
        "superior"
    )

    -- Camada inferior
    adiciona_label_bobina(
        ranhura_inf,
        R_INFERIOR,
        circuito,
        sinal_inf,
        grupo,
        "inferior"
    )
end

mi_zoomnatural()
mi_refreshview()

print("Block labels das bobinas de camada dupla e passo pleno inseridos.")
print("Numero de espiras por lado de bobina:", N_ESPIRAS)
print("Raio camada superior:", R_SUPERIOR)
print("Raio camada inferior:", R_INFERIOR)