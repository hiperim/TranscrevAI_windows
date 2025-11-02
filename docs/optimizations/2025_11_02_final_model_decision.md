# Relatório Final de Otimização de Modelo e Parâmetros - 2 de Novembro de 2025

Este documento resume a investigação e os testes realizados para otimizar a performance e a acurácia do motor de transcrição da aplicação TranscrevAI.

## Baseline Inicial

*   **Modelo:** `openai/whisper-medium`
*   **`beam_size`:** 3
*   **Acurácia (`q.speakers.wav`):** 80.43%
*   **Ratio (`q.speakers.wav`):** 2.07x

## Testes Realizados

Foram testados 3 modelos `medium` afinados para português (`pierreguillou`, `const-me`, `jlondonobo`) e diferentes configurações do parâmetro `beam_size`.

### Tabela Comparativa Final (Modelo `pierreguillou`)

| Configuração             | Ratio `d.speakers` | Ratio `q.speakers` | Acurácia `q.speakers` | Conclusão                                    |
| ------------------------ | ------------------ | ------------------ | --------------------- | -------------------------------------------- |
| `beam_size=1`, `best_of=1` | 2.18x              | 1.78x              | 76.09%                | Mais rápido, mas com **perda grave de acurácia**. |
| **`beam_size=3`, `best_of=3`** | **2.26x**          | **1.93x**          | **80.43%**            | **Melhor balanço entre performance e acurácia.** |
| `beam_size=5`, `best_of=5` | 2.40x              | 2.08x              | 80.43%                | Mais lento, sem ganho de acurácia.           |

## Decisão Final

A configuração que oferece o melhor balanço entre performance e acurácia para o nosso caso de uso é:

*   **Modelo:** `pierreguillou/whisper-medium-portuguese`
*   **`beam_size`:** `3`
*   **`best_of`:** `3`

Esta configuração é **mais rápida** que a nossa baseline original, mantendo **exatamente a mesma acurácia**.
