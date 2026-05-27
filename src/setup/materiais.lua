-- ============================================================
-- FEMM Lua script
-- Inserção dos block labels dos domínios principais
-- Eixo, rotor, estator, entreferro e ar externo
-- ============================================================

clearconsole()

-- ------------------------------------------------------------
-- Materiais
-- ------------------------------------------------------------

MAT_EIXO  = "Eixo"
MAT_ACO   = "M250_50"
MAT_AR    = "Air"

-- ------------------------------------------------------------
-- Grupos
-- ------------------------------------------------------------

GRUPO_ESTATOR   = 1
GRUPO_ROTOR     = 2
GRUPO_ENTREFERRO = 3
GRUPO_AR_EXTERNO = 4
GRUPO_EIXO      = 9

-- ------------------------------------------------------------
-- Tamanhos de malha
-- ------------------------------------------------------------

MESH_EIXO       = 1.0
MESH_ROTOR      = 1.5
MESH_ESTATOR    = 1.5
MESH_ENTREFERRO = 0.10
MESH_AR_EXTERNO = 8.0

-- ------------------------------------------------------------
-- Função auxiliar
-- ------------------------------------------------------------

function adiciona_block_label(x, y, material, mesh, grupo)
    mi_addblocklabel(x, y)
    mi_selectlabel(x, y)

    -- mi_setblockprop(material, automesh, meshsize, circuito, magdir, grupo, turns)
    mi_setblockprop(material, 0, mesh, "<None>", 0, grupo, 0)

    mi_clearselected()

    print("Label inserido em x=", x, " y=", y, " material=", material, " grupo=", grupo)
end

-- ------------------------------------------------------------
-- Inserção dos labels
-- ------------------------------------------------------------

-- Eixo
adiciona_block_label(0, 0, MAT_EIXO, MESH_EIXO, GRUPO_EIXO)

-- Rotor ferromagnético
adiciona_block_label(0, 30, MAT_ACO, MESH_ROTOR, GRUPO_ROTOR)

-- Entreferro
adiciona_block_label(0, 57.3, MAT_AR, MESH_ENTREFERRO, GRUPO_ENTREFERRO)

-- Estator ferromagnético
adiciona_block_label(0, 80, MAT_ACO, MESH_ESTATOR, GRUPO_ESTATOR)

-- Ar externo ao estator
adiciona_block_label(0, 100, MAT_AR, MESH_AR_EXTERNO, GRUPO_AR_EXTERNO)

-- ------------------------------------------------------------
-- Atualização da visualização
-- ------------------------------------------------------------

mi_zoomnatural()
mi_refreshview()

print("Block labels dos dominios principais inseridos com sucesso.")