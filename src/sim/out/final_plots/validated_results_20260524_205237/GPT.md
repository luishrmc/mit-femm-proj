### Avaliação crítica geral

Os resultados estão **coerentes para a configuração de camada simples e passo pleno**, com uma ressalva importante: a densidade de fluxo foi corretamente ajustada pelo critério robusto `B99,5%`, mas o valor de `Bmax` absoluto ainda indica picos localizados muito altos no núcleo.

Pelos arquivos gerados:

| Grandeza                          |                                                         Valor |
| --------------------------------- | ------------------------------------------------------------: |
| Corrente base usada               |                                           `I_BASE = 12,544 A` |
| Correntes equivalentes            | `fase_A = 12,544 A`, `fase_B = -6,272 A`, `fase_C = -6,272 A` |
| `B99,5%` no núcleo                |                                                  `1,697688 T` |
| `Bmax` absoluto no núcleo         |                                                  `5,175344 T` |
| Fundamental mecânica              |                                                     ordem `3` |
| Amplitude da fundamental          |                                                  `0,635096 T` |
| Harmônico dominante de ranhura    |                                                    ordem `33` |
| Amplitude do harmônico de ranhura |                                                  `0,536439 T` |

A corrente está **aceita** pelo critério de projeto adotado, pois `B99,5% = 1,697688 T`, praticamente igual à meta de `1,7 T`. Esse critério é mais adequado que o `Bmax` absoluto porque a amostragem por elementos pode capturar singularidades ou concentrações artificiais em cantos de dentes, bordas de ranhuras ou regiões de malha muito estreitas. O pyFEMM permite justamente essa análise por elementos, pois `mo_getelement` retorna centróide e grupo do elemento, enquanto `mo_getb` retorna as componentes locais de densidade de fluxo. 

A ressalva é que `Bmax = 5,175 T` **não é fisicamente aceitável como valor representativo do aço**. Portanto, no relatório, não escreva que “a densidade máxima absoluta no núcleo foi 1,7 T”. Escreva que a corrente foi ajustada para que a densidade de fluxo representativa do núcleo, avaliada por percentil de elementos para evitar singularidades locais, fosse aproximadamente `1,7 T`. Se o professor exigir o máximo absoluto literal, será necessário refinar a geometria/malha nas quinas ou tratar os picos locais separadamente.

---

## Gráfico 1 — Distribuição espacial da densidade de fluxo no entreferro

O primeiro gráfico mostra `B_r(θ)` ao longo de `0°` a `360°` mecânicos.

A forma de onda apresenta três ciclos completos ao redor da circunferência mecânica, o que está correto para uma máquina de **6 polos**, pois há `p = 3` pares de polos. A simetria observada é coerente: a curva se repete aproximadamente a cada `120°` mecânicos e muda de sinal a cada `60°`, comportamento esperado para polos norte e sul alternados.

A forma não é senoidal. Isso também é esperado, porque a simulação FEMM usa a geometria ranhurada real, e não a hipótese analítica ideal de entreferro liso. As regiões com forte variação e trechos próximos de zero indicam modulação de permeância causada pelas ranhuras do estator/rotor e pela concentração espacial das bobinas. Para camada simples e passo pleno, essa distorção tende a ser mais pronunciada, pois há menor cancelamento espacial de harmônicos do enrolamento.

Conclusão do gráfico 1: **a distribuição espacial está fisicamente plausível e compatível com uma máquina de 6 polos**, mas fortemente distorcida por efeitos de ranhura. Esse gráfico atende ao item de distribuição espacial para a configuração de camada simples.

---

## Gráfico 2 — Distribuição espectral da densidade de fluxo no entreferro

O segundo gráfico mostra o espectro espacial de `B_r(θ)`.

O pico principal aparece na ordem mecânica:

```text
n = 3
```

Isso é correto. Em uma máquina de `6 polos`, a fundamental espacial magnética tem `3` ciclos por volta mecânica. Portanto:

```text
ordem mecânica 3 = fundamental magnética de 6 polos
```

A amplitude da fundamental foi:

```text
B_3 = 0,635096 T
```

O espectro também mostra componentes relevantes nas ordens:

```text
33, 39, 69, 105, 111
```

As ordens `33` e `39` são especialmente importantes. Para `36` ranhuras de estator e `p = 3` pares de polos, os principais harmônicos de ranhura aparecem em torno de:

```text
Q_s - p = 36 - 3 = 33
Q_s + p = 36 + 3 = 39
```

O resultado mostra exatamente isso:

| Harmônico |    Amplitude | Relação com a fundamental |
| --------: | -----------: | ------------------------: |
|       `3` | `0,635096 T` |                    `100%` |
|      `33` | `0,536439 T` |                  `84,47%` |
|      `39` | `0,319783 T` |                  `50,35%` |
|      `69` | `0,133228 T` |                  `20,98%` |
|     `105` | `0,130562 T` |                  `20,56%` |

Portanto, o espectro está de acordo com o esperado: a fundamental aparece na ordem `3`, e os harmônicos de ranhura aparecem como bandas laterais associadas ao número de ranhuras. O manual do pyFEMM também descreve recursos específicos para extração de campo no entreferro e harmônicos espaciais em problemas com elemento de entreferro, mas como seu modelo usa a máquina completa, a FFT de `B_r(θ)` é uma alternativa apropriada. 

Conclusão do gráfico 2: **o espectro confirma a fundamental de 6 polos e identifica claramente o harmônico de ranhura dominante na ordem 33**. O resultado está coerente com a geometria de `36` ranhuras e `6` polos.

---

## Gráfico 3 — Comparação do harmônico fundamental de ranhura

O terceiro gráfico mostra a amplitude do harmônico de ranhura dominante para a configuração atualmente simulada:

```text
camada_simples_passo_pleno
ordem 33
amplitude = 0,536439 T
```

Esse gráfico está correto como saída parcial, mas **ainda não atende completamente ao item comparativo do trabalho**, porque há apenas uma configuração. O enunciado pede comparar as amplitudes dos harmônicos de ranhura entre as três configurações:

1. camada simples e passo pleno;
2. camada dupla e passo pleno;
3. camada dupla e passo encurtado.

Com apenas a camada simples, o gráfico serve como referência inicial. A comparação final só será válida quando as outras duas configurações forem simuladas com o mesmo critério de corrente, mesmo raio médio de entreferro, mesmo número de pontos de amostragem e mesma normalização espectral.

Conclusão do gráfico 3: **a identificação do harmônico de ranhura dominante está correta, mas a comparação ainda está incompleta**.

---

## Conclusão do experimento da camada simples

A simulação da configuração **camada simples e passo pleno** pode ser considerada **validada para prosseguir**, com as seguintes conclusões:

A corrente `I_BASE = 12,544 A` está adequada pelo critério representativo de densidade de fluxo no núcleo, pois `B99,5% = 1,697688 T`, muito próximo da meta de `1,7 T`. Assim, não é necessário alterar a corrente neste momento.

A distribuição espacial `B_r(θ)` no entreferro apresenta comportamento coerente com uma máquina de 6 polos: três períodos mecânicos completos, alternância de sinal entre polos e forte modulação devido às ranhuras. A forma não senoidal é esperada para geometria ranhurada e enrolamento de camada simples.

O espectro espacial confirma a fundamental na ordem mecânica `3` e identifica o harmônico fundamental de ranhura dominante na ordem `33`, com amplitude `0,536439 T`. Essa amplitude é elevada, cerca de `84,5%` da fundamental, indicando forte conteúdo harmônico de ranhura na configuração de camada simples e passo pleno.

A única ressalva relevante é o `Bmax` absoluto de `5,175344 T`, que não deve ser interpretado como valor físico global do núcleo. Ele provavelmente representa pico local em elemento próximo a quina ou descontinuidade geométrica. Para o relatório, documente que o ajuste da corrente foi feito por percentil de campo no núcleo e que o máximo absoluto foi tratado como concentração local de campo.

Portanto, **o experimento está de acordo com o esperado para camada simples**, desde que a validação da densidade de fluxo seja baseada em `B99,5%` e não no máximo absoluto isolado. A próxima etapa é repetir exatamente o mesmo procedimento para camada dupla e passo pleno.

Aguardo os prints desta etapa para validar antes de avançar.
