clear all, close all, clc

addpath('C:\femm42\mfiles')
openfemm
opendocument('mat.fem')
mi_saveas('temp.fem')

lambda = zeros(6,7);
corrente = [0 1 2 3 4 5];

for i = 1:6
    i_a = i - 1;
    mi_modifycircprop('a',1,i_a)
    for k = 1:7
        mi_analyze();
        mi_loadsolution;
        cir_a = mo_getcircuitproperties('a');
        lambda(k,i) = cir_a(3)
        mo_seteditmode('area');
        mo_groupselectblock(10);
        torque(k,i) = mo_blockintegral(22)
        mi_seteditmode('group')
        mi_selectgroup(1)
        mi_moverotate(0,0,5)
    end
    mi_seteditmode('group')
    mi_selectgroup(1)
    mi_moverotate(0,0,55)
end

plot(corrente,lambda(1,1:6),corrente,lambda(2,1:6),corrente,lambda(3,1:6),corrente,lambda(4,1:6),corrente,lambda(5,1:6),corrente,lambda(6,1:6),corrente,lambda(7,1:6))
