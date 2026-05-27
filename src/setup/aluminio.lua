-- ============================================================
-- FEMM Lua script
-- Posicionamento dos block labels das barras de alumínio do rotor
-- 28 barras igualmente distribuídas
-- Centro: (0,0)
-- Raio: 50 mm
-- Primeira barra: (0,50)
-- ============================================================

clearconsole()

-- -----------------------------
-- Parâmetros principais
-- -----------------------------

N_BARRAS = 28
R_ALUMINIO = 50.0        -- mm

MATERIAL_ALUMINIO = "Aluminum, 1100"   -- altere para "Aluminio_rotor" se for o nome usado no modelo
MESH_ALUMINIO = 0.8                    -- mm
GRUPO_ALUMINIO = 8

-- -----------------------------
-- Constantes e funções auxiliares
-- -----------------------------

PI = 4 * atan(1)

function deg2rad(ang)
    return ang * PI / 180.0
end

function adiciona_label_aluminio(theta_deg)
    local theta = deg2rad(theta_deg)

    local x = R_ALUMINIO * cos(theta)
    local y = R_ALUMINIO * sin(theta)

    mi_addblocklabel(x, y)
    mi_selectlabel(x, y)

    -- mi_setblockprop(material, automesh, meshsize, circuito, magdir, grupo, turns)
    mi_setblockprop(MATERIAL_ALUMINIO, 0, MESH_ALUMINIO, "<None>", 0, GRUPO_ALUMINIO, 0)

    mi_clearselected()
end

-- -----------------------------
-- Inserção das 28 barras
-- -----------------------------
-- Barra 1 em theta = 90 graus -> ponto (0,50)
-- As demais são distribuídas no sentido horário.

for k = 1, N_BARRAS do
    theta_deg = 90.0 - (k - 1) * 360.0 / N_BARRAS
    adiciona_label_aluminio(theta_deg)
end

mi_zoomnatural()
mi_refreshview()

print("28 block labels de aluminio inseridos no rotor.")