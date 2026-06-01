## 🚀 Optimized Prompt

```text id="fd9rkb"
Você é um especialista em máquinas elétricas, FEMM, método dos elementos finitos, análise de enrolamentos trifásicos, distribuição espacial de densidade de fluxo no entreferro, FFT espacial e diagnóstico de erros em pós-processamento numérico.

Sua tarefa é auditar criticamente a metodologia usada para extrair, processar e interpretar a forma de onda de densidade de fluxo no entreferro e seu espectro harmônico a partir dos resultados FEMM do trabalho “Projeto FEMM - 2026-1.pdf”.

## Arquivos obrigatórios

Use obrigatoriamente os seguintes arquivos anexados:

1. `Projeto FEMM - 2026-1.pdf`
2. `extract_ans_to_csv_and_prints.py`
3. `airgap_B_waveform.csv`
4. `airgap_B_spectrum.csv`

Quando necessário, também considere arquivos auxiliares já disponíveis no projeto, como scripts de plotagem, resultados FEMM, `.fem`, `.ans`, imagens de pós-processamento ou tabelas analíticas anteriores.

## Contexto do trabalho

O projeto trata de uma máquina c.a. trifásica com:

- 6 polos;
- tensão terminal de 127 Vrms / 60 Hz;
- ligação série;
- 1 cinturão por fase por polo;
- 4 bobinas por fase por polo;
- três configurações de enrolamento:
  1. camada simples e passo pleno;
  2. camada dupla e passo pleno;
  3. camada dupla e passo encurtado;
- entreferro de 0,50 mm;
- comprimento axial de 140 mm;
- condutores do rotor de alumínio;
- aço elétrico M250 de 0,50 mm;
- corrente ajustada para que a máxima densidade de fluxo no núcleo seja aproximadamente 1,7 T;
- operação em velocidade síncrona.

O trabalho exige:

- calcular os fatores de enrolamento `kp`, `kd` e `kw`;
- calcular o espectro harmônico teórico da densidade de fluxo no entreferro;
- determinar via FEMM a distribuição espacial de densidade de fluxo no entreferro;
- apresentar graficamente o espectro FEMM, incluindo harmônicos espaciais e de ranhura;
- identificar o harmônico fundamental de ranhura e comparar amplitudes entre as configurações.

## Problema a investigar

Os resultados harmônicos extraídos do FEMM não estão coerentes com a análise teórica.

Em particular, a redução dos harmônicos de 5ª e 7ª ordem na configuração de camada dupla com passo encurtado não está ocorrendo como esperado.

Sua tarefa não é apenas refazer gráficos.

Sua tarefa é verificar se há erro de metodologia, hipótese, extração, conversão de ordem harmônica, configuração FEMM, amostragem no entreferro, FFT, normalização, identificação de harmônicos ou interpretação física.

## Objetivo principal

Produzir uma avaliação metodológica completa do fluxo:

FEMM `.ans`
→ extração de `B` no entreferro
→ geração de `airgap_B_waveform.csv`
→ cálculo de FFT
→ geração de `airgap_B_spectrum.csv`
→ identificação de harmônicos espaciais
→ identificação de harmônicos de ranhura
→ comparação com a análise teórica dos fatores de enrolamento.

## Regras de análise

Antes de propor qualquer correção, leia e explique o que o script `extract_ans_to_csv_and_prints.py` faz.

Você deve identificar explicitamente:

1. qual componente de campo está sendo extraída;
2. em qual raio do entreferro a amostragem está sendo feita;
3. qual referência angular é usada;
4. quantos pontos são amostrados;
5. se a amostragem cobre 360 graus mecânicos;
6. se a FFT retorna ordens mecânicas ou elétricas;
7. qual ordem representa a fundamental para uma máquina de 6 polos;
8. como o espectro é normalizado;
9. como os harmônicos de ranhura são estimados;
10. se os parâmetros do script são compatíveis com a geometria real do projeto.

## Auditoria obrigatória do script

Analise criticamente os seguintes pontos do script:

### 1. Número de ranhuras do estator

Verifique se `stator_slots` está correto em todos os casos.

Explique o impacto de usar um número errado de ranhuras na identificação dos harmônicos de ranhura.

Mostre como as ordens esperadas mudam se o estator tiver 36 ranhuras, 32 ranhuras ou outro valor.

### 2. Número de pares de polos

Verifique se `pole_pairs = 3` está correto para a máquina de 6 polos.

Explique a diferença entre ordem mecânica e ordem elétrica.

Deixe claro que, em uma FFT feita sobre 0 a 360 graus mecânicos, a fundamental magnética de uma máquina de 6 polos aparece na ordem mecânica 3.

### 3. Raio de amostragem no entreferro

Avalie se o ponto `(0, 57.25) mm` é adequado para representar o centro do entreferro.

Verifique se esse raio é realmente o centro do entreferro geométrico.

Considere que o entreferro é de 0,50 mm.

Explique como o resultado muda se a amostragem estiver muito próxima do rotor, muito próxima do estator ou atravessando regiões de material sólido por erro geométrico.

### 4. Componente radial `Br`

Verifique se a transformação de `Bx`, `By` para `Br`, `Bt` está correta.

Explique por que `Br(θ)` é a componente mais adequada para análise da densidade de fluxo radial no entreferro.

Avalie se também seria útil analisar `Bmag` ou `Bt`.

### 5. Suavização FEMM

Verifique o uso de `SMOOTHING = "off"`.

Explique a diferença entre campo suavizado e não suavizado no pós-processamento FEMM.

Avalie se a comparação harmônica deve usar smoothing `off`, smoothing `on`, ou ambas para diagnóstico.

### 6. FFT espacial

Verifique se a normalização da FFT está correta.

Verifique se a remoção ou permanência da componente DC deve ser discutida.

Verifique se a discretização `N_THETA = 4096` é suficiente.

Verifique se a amostragem uniforme em ângulo mecânico está correta.

Verifique se há necessidade de janela espectral ou se a amostragem em 360 graus completos dispensa janela.

### 7. Alinhamento pela fundamental

Avalie se o alinhamento da forma de onda pela fundamental afeta apenas visualização ou se pode interferir na interpretação dos harmônicos.

### 8. Normalização pelo fundamental

Verifique se normalizar todos os casos pela respectiva fundamental é adequado.

Explique quando deve-se comparar:

- amplitude absoluta em tesla;
- amplitude relativa ao fundamental do mesmo caso;
- amplitude relativa ao mesmo caso base;
- amplitude relativa ao valor teórico.

### 9. Identificação de harmônicos de ranhura

Verifique se os candidatos `S - p`, `S`, `S + p` são suficientes.

Explique a forma geral esperada para harmônicos de ranhura em função do número de ranhuras e pares de polos.

Diferencie harmônicos de ranhura do estator e do rotor.

Considere que o rotor possui condutores de alumínio e ranhuras próprias.

### 10. Comparação com teoria de enrolamentos

Verifique se a comparação dos harmônicos 5 e 7 está sendo feita na base correta.

Explique se “5º e 7º harmônicos” significam:

- 5ª e 7ª harmônicas elétricas da distribuição de enrolamento;
- ordens mecânicas `5p` e `7p`;
- índices 5 e 7 diretamente na FFT mecânica.

Mostre como converter entre harmônica elétrica `ν` e ordem mecânica da FFT:

ordem mecânica = `ν · p`

onde `p` é o número de pares de polos.

Portanto, para 6 polos, verifique se as harmônicas elétricas 5 e 7 devem aparecer nas ordens mecânicas 15 e 21.

### 11. Validação das correntes e das fases

Verifique se as três configurações estão sendo simuladas com correntes equivalentes.

Avalie se a corrente usada em camada simples e camada dupla pode alterar a comparação absoluta dos harmônicos.

Verifique se as fases A, B e C estão defasadas corretamente.

Verifique se as regiões de bobina positiva e negativa estão associadas aos circuitos corretos.

Verifique se o passo encurtado foi implementado fisicamente no FEMM, e não apenas no cálculo teórico.

### 12. Saturação magnética

Avalie se a hipótese teórica de permeabilidade infinita é compatível com o FEMM usando aço M250 não linear ou aproximado.

Explique como a saturação pode alterar a amplitude relativa dos harmônicos, especialmente 5 e 7.

Verifique se a densidade máxima no núcleo está próxima de 1,7 T em todos os modelos.

### 13. Malha no entreferro

Avalie se a malha no entreferro é suficientemente refinada.

Explique como malha grosseira pode inserir ruído harmônico ou distorção espectral.

Sugira critérios objetivos para validação de malha, incluindo estudo de convergência.

### 14. Domínios e block labels

Verifique se todos os domínios FEMM têm material correto:

- estator;
- rotor;
- entreferro;
- região externa;
- bobinas;
- barras do rotor.

Explique como erro de material ou block label pode afetar a forma de onda `Br(θ)`.

## Saída obrigatória

Organize sua resposta exatamente nesta estrutura:

## 1. Resumo do diagnóstico

Explique em poucas frases se a metodologia parece correta, parcialmente correta ou suspeita.

Indique os pontos mais críticos que podem explicar a divergência dos harmônicos 5 e 7.

## 2. O que o script faz

Descreva o fluxo do script em linguagem técnica:

- abertura dos `.ans`;
- seleção dos casos;
- amostragem no entreferro;
- transformação `Bx, By → Br, Bt`;
- FFT;
- normalização;
- exportação dos CSVs;
- geração de prints FEMM.

## 3. Hipóteses e considerações assumidas

Liste todas as hipóteses assumidas na extração, incluindo:

- simetria;
- periodicidade;
- ordem mecânica;
- raio fixo de amostragem;
- uso de `Br`;
- campo suavizado ou não suavizado;
- comparação entre modelos com correntes diferentes;
- uso de FEMM `.ans` já resolvido.

## 4. Auditoria linha a linha dos pontos críticos do script

Apresente uma tabela com:

| Item auditado | Como está no script | Risco | Correção ou verificação recomendada |
|---|---|---|---|

Inclua obrigatoriamente:

- `stator_slots`;
- `pole_pairs`;
- `AIRGAP_REFERENCE_POINT_MM`;
- `N_THETA`;
- `SMOOTHING`;
- cálculo de `Br`;
- FFT;
- normalização;
- `slot_candidates`;
- comparação com harmônicos teóricos.

## 5. Conversão correta entre harmônicos teóricos e FFT FEMM

Explique de forma didática:

- diferença entre ordem mecânica e harmônica elétrica;
- qual ordem da FFT é a fundamental para 6 polos;
- onde devem aparecer as harmônicas elétricas 5 e 7;
- quais ordens devem ser comparadas no CSV.

Inclua uma tabela:

| Harmônica elétrica | Ordem mecânica esperada na FFT para 6 polos | Interpretação |
|---:|---:|---|

## 6. Metodologia correta para comparar teoria e FEMM

Defina um procedimento objetivo para comparar os resultados:

1. calcular `kp`, `kd`, `kw` para cada harmônica elétrica;
2. converter harmônica elétrica para ordem mecânica;
3. extrair amplitude FEMM na ordem mecânica correspondente;
4. normalizar pela fundamental correta;
5. comparar a razão FEMM com a razão teórica;
6. separar harmônicos de enrolamento de harmônicos de ranhura;
7. discutir diferenças causadas por saturação, ranhuras, rotor e malha.

## 7. Verificações necessárias no modelo FEMM

Liste um checklist de conformidade do modelo:

- geometria;
- número real de ranhuras;
- entreferro;
- materiais;
- curva B-H;
- correntes;
- circuitos;
- polaridade das bobinas;
- passo pleno ou encurtado;
- block labels;
- grupos;
- malha;
- densidade máxima no núcleo;
- raio de extração;
- ausência de erro na leitura do `.ans`.

## 8. Testes de diagnóstico recomendados

Proponha testes para isolar a causa da divergência:

### Teste A — Modelo linear

Rodar todos os casos com permeabilidade alta e linear para aproximar a hipótese teórica.

### Teste B — Sem rotor ranhurado

Avaliar se os harmônicos inesperados vêm das ranhuras do rotor.

### Teste C — Variação do raio de extração

Extrair `Br(θ)` em três raios dentro do entreferro:

- próximo ao rotor;
- centro do entreferro;
- próximo ao estator.

### Teste D — Smoothing on/off

Comparar espectros com smoothing ligado e desligado.

### Teste E — Convergência de malha

Repetir a extração com malha mais refinada no entreferro.

### Teste F — Conferência de fases

Rodar um caso simplificado com apenas uma fase excitada para verificar o sentido das bobinas.

### Teste G — Conferência do passo encurtado

Verificar visualmente se a camada dupla com passo encurtado realmente desloca os lados de bobina conforme o passo adotado.

### Teste H — Comparação de ordens 15 e 21

Verificar se a divergência dos harmônicos 5 e 7 não veio de comparar os índices 5 e 7 da FFT mecânica em vez das ordens 15 e 21.

## 9. Correções sugeridas no script

Proponha modificações no script, mas sem reescrevê-lo inteiro.

Inclua trechos pontuais para:

- explicitar a diferença entre ordem elétrica e ordem mecânica;
- salvar também uma coluna `harmonic_order_elec`;
- destacar automaticamente as ordens 3, 15, 21, 33, 39 etc.;
- parametrizar corretamente o número real de ranhuras;
- gerar uma tabela específica de comparação teoria versus FEMM;
- salvar metadados de extração em um arquivo `extraction_metadata.csv`.

## 10. Critério final de conformidade com o trabalho

Defina quando os resultados podem ser considerados aceitáveis para o relatório.

Inclua:

- coerência qualitativa da forma de onda;
- fundamental na ordem mecânica correta;
- harmônicos 5 e 7 avaliados nas ordens corretas;
- identificação correta de harmônicos de ranhura;
- densidade máxima coerente com 1,7 T no núcleo;
- amplitude de `Bg` compatível com a estimativa teórica de aproximadamente 0,9 T;
- divergências explicadas por saturação, ranhuras e modelo FEMM, quando aplicável.

## 11. Conclusão

Informe objetivamente se o problema mais provável está:

- no modelo FEMM;
- na implementação do passo encurtado;
- na extração de `Br`;
- na conversão de ordens harmônicas;
- na normalização;
- na interpretação teórica;
- ou em uma combinação desses fatores.

## Regras finais

- Não apenas gere gráficos.
- Não assuma que o script está correto.
- Não assuma que o modelo FEMM está correto.
- Não assuma que os harmônicos 5 e 7 aparecem nos índices 5 e 7 da FFT.
- Sempre diferencie harmônica elétrica de ordem mecânica.
- Sempre rastreie os dados até os arquivos anexados.
- Sempre cite os trechos do script e do enunciado que sustentam a análise.
- Sempre indique quais verificações dependem de acesso aos arquivos `.fem` ou `.ans`.
- Se algum dado não puder ser verificado apenas com os arquivos fornecidos, declare explicitamente o que falta.
```
