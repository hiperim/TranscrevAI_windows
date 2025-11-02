# Otimização: Modelo `pierreguillou` com `beam_size=3` - 2 de Novembro de 2025

Esta otimização substituiu o modelo padrão "medium" pelo modelo `pierreguillou/whisper-medium-portuguese`, que é otimizado para o idioma português. O `beam_size` foi mantido em `3` para comparação direta com a baseline.

## Mudanças Aplicadas

Em `src/transcription.py`:
*   `model_name`: de `"medium"` para `"pierreguillou/whisper-medium-portuguese"`
*   `beam_size`: mantido em `3`

## Análise de Impacto

| Métrica                  | Baseline (Modelo "medium") | `pierreguillou` (`beam_size=3`) | Mudança                               |
| ------------------------ | -------------------------- | ------------------------------- | ------------------------------------- |
| **CPU Time**             | ~81 segundos               | ~80.5 segundos                  | **-0.5 segundos (Melhoria marginal)** |
| **Pico de Memória RAM**  | **2305.69 MB**             | **2303.01 MB**                  | **-2.68 MB (Melhoria marginal)**      |
| **Ratio `d.speakers`**   | 2.34x                      | 2.26x                           | **-3.4% (Melhoria)**                  |
| **Ratio `q.speakers`**   | 2.07x                      | 1.93x                           | **-6.8% (Melhoria)**                  |
| **Acurácia `d.speakers`**| 92.00%                     | 92.00%                          | **Sem alteração**                     |
| **Acurácia `q.speakers`**| 80.43%                     | 80.43%                          | **Sem alteração**                     |

**Conclusão:** A troca para o modelo `pierreguillou/whisper-medium-portuguese` com `beam_size=3` trouxe uma **melhoria notável na performance (redução do `processing_ratio`)** e uma pequena redução no consumo de RAM, **sem qualquer impacto negativo na acurácia**.
