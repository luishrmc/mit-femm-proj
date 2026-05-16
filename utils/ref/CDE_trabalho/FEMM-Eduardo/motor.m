%%%%%%%%%%%%%%%%%%%%%%%%%%
% Limpeza de dados
clear;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%
% INICIALIZACOES

% Especificacoes
compef = 0.5; % Comprimento do entreferro em mm
raioef = 50;  % Raio do entreferro em mm
compaxial = 20; % Comprimento axial em mm
nespiras = 200; % Numero de espiras por bobina
femp = 0.96;    % Fator de empacotamento
arcest = pi/6;  % Arco coberto pelo estator em rad
arcrot = pi/4;  % Arco coberto pelo rotor em rad

% Vetores do estator e rotor
crot = [-23.5 0];
crote = [-49.75 0];
cest = [-50.25 0];
ceste = [-72.8 0];
cond1 = [-60 0];
cond2 = [-65 0];


angulos = zeros(1, 91);
torque = zeros(1, 91);

for rotacao = 0:90
    angulos(rotacao+1) = rotacao;
end

% Abre a FEMM
openfemm;
% Abertura de novo arquivo
newdocument(0);
% Configuracoes do problema
mi_probdef( 0, 'millimeters', 'planar', 0.00000001 , compaxial, 30);

% Angulo de rotacao
angulo = 3*5;
th = angulo*pi/180;

% Matriz de rotacao
R = [cos(th) -sin(th) ;sin(th) cos(th)];
crot = crot*R;
crote = crote*R;

% Angulo de rotacao
angulo = 30;
th = angulo*pi/180;
R = [cos(th) -sin(th) ;sin(th) cos(th)];

%%%%%%%%%%%%%%%%%%%%%%%%%%
% ITERACOES DA FEMM
for rotacao = 0:0
    
    fprintf('ITERACAO %d\n', rotacao);
    
    % Angulo de rotacao
    angulo = 45;
    th = angulo*pi/180;
    R = [cos(th) -sin(th) ;sin(th) cos(th)];
    
    % Geracao e rotacao dos vetores (rotor)
    for i = 1:360/angulo
        mi_addnode(crot(1), crot(2));
        mi_addnode(crote(1), crote(2));
        mi_addsegment(crot(1), crot(2), crote(1), crote(2));
        
        cr = crot*R;
        if mod(i,2) == 1
            mi_drawarc(cr(1), cr(2), crot(1), crot(2), angulo, 1);
        end
        crot = cr;
        
        cr = crote*R;
        if mod(i,2) == 0
            mi_drawarc(cr(1), cr(2), crote(1), crote(2), angulo, 1)
        end
        crote = cr;
    end
    
    % Angulo de rotacao
    angulo = 30;
    th = angulo*pi/180;
    
    % Matriz de rotacao
    R = [cos(th) -sin(th) ;sin(th) cos(th)];
    
    % Geracao e rotacao dos vetores (estator)
    for i = 1:360/angulo
        mi_addnode(cest(1), cest(2));
        mi_addnode(ceste(1), ceste(2));
        mi_addsegment(cest(1), cest(2), ceste(1), ceste(2));
        
        cr = cest*R;
        if mod(i,2) == 1
            mi_drawarc(cr(1), cr(2), cest(1), cest(2), angulo, 1);
        end
        cest = cr;
        
        cr = ceste*R;
        if mod(i,2) == 0
            mi_drawarc(cr(1), cr(2), ceste(1), ceste(2), angulo, 1);
        end
        ceste = cr;
    end
    
    % Circulo externo do condutores
    mi_addnode(-80, 0);
    mi_addnode(80, 0);
    mi_drawarc(-80, 0, 80, 0, 180, 2);
    mi_drawarc(80, 0, -80, 0,  180, 2);
    
    % Contorno
    mi_addnode(0, -186);
    mi_addnode(0, 186);
    mi_addnode(0, -209.500);
    mi_addnode(0, 209.500);
    mi_drawarc(0, -186, 0, 186, 180, 2);
    mi_drawarc(0, 186, 0, -186, 180, 2);
    mi_drawarc(0, -209.500, 0, 209.500, 180, 2);
    mi_drawarc(0, 209.500, 0, -209.500, 180, 2);
    mi_addblocklabel(148, 143);
    mi_selectlabel(148, 143);
    mi_addmaterial('u1', 10.5238095238095, 10.5238095238095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    mi_setblockprop('u1', 1, 0.05, '', 0, 0, 0);
    mi_clearselected;
    
    % Materiais
    mi_getmaterial('Air');
    mi_getmaterial('M-36 Steel');
    mi_getmaterial('18 AWG');
    mi_addcircprop('a', 2.4, 1);
    mi_addcircprop('b', 0, 1);
    mi_addcircprop('c', 0, 1);
    
    % Metal
    mi_addblocklabel(0, 0);
    mi_addblocklabel(-44, -46);
    mi_selectlabel(0,0);
    
    mi_selectlabel(-44, -46);
    mi_setblockprop('M-36 Steel', 1, 0.05, '', 0, 0, 0);
    mi_modifymaterial('M-36 Steel',8,femp)
    mi_clearselected;
    
    % Bloco Ar
    mi_addblocklabel(59.6, 17.5);
    mi_selectlabel(59.6, 17.5);
    mi_addblocklabel(20, 95);
    mi_selectlabel(20, 95);
    mi_setblockprop('Air', 1, 1.05, '', 0, 0, 0);
    mi_clearselected;
    
    % Bloco Condutores
    % Geracao e rotacao dos vetores (limites condutores)
    for i = 1:12
        
        % Angulo de rotacao
        angulo = 10;
        th = (-1)^i*angulo*pi/180;
        
        % Matriz de rotacao
        R = [cos(th) -sin(th) ;sin(th) cos(th)];
        
        % Adiciona pontos dos condutores
        mi_addnode(cond1(1), cond1(2));
        mi_addnode(cond2(1), cond2(2));
        
        cr1 = cond1*R;
        mi_addnode(cr1(1), cr1(2));
        cr2 = cond2*R;
        mi_addnode(cr2(1), cr2(2));
        
        % Adiciona os segmentos
        mi_addsegment(cond1(1), cond1(2), cond2(1), cond2(2));
        mi_addsegment(cond1(1), cond1(2), cr1(1), cr1(2));
        mi_addsegment(cond2(1), cond2(2), cr2(1), cr2(2));
        mi_addsegment(cr1(1), cr1(2), cr2(1), cr2(2));
        
        % Angulo de rotacao
        angulo = 30;
        th = angulo*pi/180;
        
        % Matriz de rotacao
        R = [cos(th) -sin(th) ;sin(th) cos(th)];
        cond1 = cond1*R;
        cond2 = cond2*R;
        
    end
    
    cond1 = [-56 -26];
    cond2 = [-62 -7];
    
    % Angulo de rotacao
    angulo = 60;
    th = angulo*pi/180;
    
    % Matriz de rotacao
    R = [cos(th) -sin(th) ;sin(th) cos(th)];
    
    % Geracao e rotacao dos vetores (condutores)
    for i = 1:360/angulo
        
        mi_addblocklabel(cond1(1), cond1(2));
        mi_addblocklabel(cond2(1), cond2(2));
        
        mi_selectlabel(cond1(1), cond1(2));
        if i == 1 || i == 4
            mi_setblockprop('18 AWG', 1, 0.05, 'a', 0, 0, (-1)^(i+1)*200);
        elseif i == 2 || i == 5
            mi_setblockprop('18 AWG', 1, 0.05, 'b', 0, 0, (-1)^(i+1)*200);
        else
            mi_setblockprop('18 AWG', 1, 0.05, 'c', 0, 0, (-1)^(i+1)*200);
        end
        mi_clearselected;
        
        mi_selectlabel(cond2(1), cond2(2));
        if i == 1 || i == 4
            mi_setblockprop('18 AWG', 1, 0.05, 'b', 0, 0, (-1)^(i+1)*200);
        elseif i == 2 || i == 5
            mi_setblockprop('18 AWG', 1, 0.05, 'c', 0, 0, (-1)^(i+1)*200);
        else
            mi_setblockprop('18 AWG', 1, 0.05, 'a', 0, 0, (-1)^(i+1)*200);
        end
        mi_clearselected;
        
        cond1 = cond1*R;
        cond2 = cond2*R;
        
    end
    
    mi_clearselected;
    
    % % Finalizacao
    mi_zoomnatural;
    mi_saveas('MotorRelut.fem');
    
    
    
    mi_modifycircprop('a',1,0)
    mi_modifycircprop('b',1,0)
    mi_modifycircprop('a',1,2.4)
    
%     mi_analyze;
%     mi_loadsolution;
%     mo_showdensityplot(1,0,1.62722863463902,0.00005,'mag')
    

    
%     
%         mo_resize(730,453)
%         mo_zoom(-88,-90,96,90)
    
%         filename = strcat('distr', '3a', '.jpg')
%         mo_savebitmap(filename)
        
        mi_drawarc(0,50,0,-50,180,1)
        mi_drawarc(0,-50,0,50,180,1)
        mi_addblocklabel(0, 49.975);
        mi_selectlabel(0, 49.975);
        mi_setblockprop('Air', 1, 1.05, '', 0, 0, 0);
        mi_analyze;
        mi_loadsolution;
        
        mo_selectpoint(1,50);
        mo_selectpoint(1,-50);
        mo_selectpoint(-2,50);
        mo_makeplot(1,1500,'myfile.txt',0)
        
%         mo_selectblock(0, 0);
%         torque(1) = mo_blockintegral(22);

%         mo_clearblock;
        
%         mi_drawarc(0,50,0,-50,180,1)
%         mi_drawarc(0,-50,0,50,180,1)
%         mi_addblocklabel(0, 49.975);
%         mi_selectlabel(0, 49.975);
%         mi_setblockprop('Air', 1, 1.05, '', 0, 0, 0);
        
        
        load ('myfile.txt')
  
        
%         for i = 1:1500
%             myfile(i,1) = 360*myfile(i,1)/myfile(end,1)
%         end;
%         
        plot(myfile(:,1),myfile(:,2))
        title('Gráfico B x Angulo')
        xlabel('Angulo \theta (º)')
        ylabel('B (T)')
        saveas(gcf,'grafico3.png')
%         
%         Iteração de rotação

%     for i = 1:91
%         mi_analyze;
%         mi_loadsolution;
%         mo_selectblock(0, 0);
%         torque(i) = mo_blockintegral(22);
%         mo_blockintegral(22)
%         mo_clearblock;
%         mi_selectcircle(0,0,50,4);
%         mi_moverotate(0,0,-1);
%     end;
%     
    
end;

% closefemm;
%
% plot( angulos, torque);