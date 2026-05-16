%% Item b) — Espectro harmonico de Bg(theta)
% P=6, S=36, n=2, gamma=30deg, B1=0.9T

clear; clc;

n=2; gamma=30; Bg1=0.9;
rho_pp=180; rho_pe=150;
harmonics=[1 3 5 7 9 11 13];

kp=@(rho,v) abs(sin(v.*deg2rad(rho/2)));
kd=@(n,g,v) abs(sin(n.*v.*deg2rad(g/2))./(n.*sin(v.*deg2rad(g/2))));
kw=@(rho,n,g,v) kp(rho,v).*kd(n,g,v);

% Mascara de harmonicas triplas
is_triplen = (mod(harmonics,3)==0);

configs = {rho_pp, rho_pp, rho_pe};
nomes   = {'CS/PP','CD/PP','CD/PE 5/6'};

Bfase  = zeros(3,length(harmonics));
B3phi  = zeros(3,length(harmonics));

for ic=1:3
    rho=configs{ic};
    kw1=kw(rho,n,gamma,1);
    for iv=1:length(harmonics)
        v=harmonics(iv);
        Bfase(ic,iv) = Bg1*(kw(rho,n,gamma,v)/kw1)/v;
    end
    B3phi(ic,:) = Bfase(ic,:);
    B3phi(ic,is_triplen) = 0;   % suprimir triplas no resultante trifasico
end

% Graficos lado a lado
figure('Position',[100 100 1100 420]);
subplot(1,2,1);
bar(harmonics, Bfase', 'grouped');
set(gca,'YScale','log'); ylim([1e-3 1]);
legend(nomes,'Location','northeast'); grid on;
xlabel('Ordem harmonica v'); ylabel('|B_v| [T]');
title('Espectro por fase');

subplot(1,2,2);
bar(harmonics, B3phi', 'grouped');
set(gca,'YScale','log'); ylim([1e-3 1]);
legend(nomes,'Location','northeast'); grid on;
xlabel('Ordem harmonica v'); ylabel('|B_v| [T]');
title('Resultante trifasica (triplas = 0)');

sgtitle('Espectro de B_g(\theta)');
