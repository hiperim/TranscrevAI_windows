# SPRINT 3 v2 - X-vectors Implementation Changes Log

**Data**: 2025-10-01
**Objetivo**: Substituir Resemblyzer por SpeechBrain X-vectors para melhorar diarização em áudios curtos (9-21s)

---

## Contexto da Mudança

### Problema Identificado
- **Resemblyzer não adequado para áudios curtos**: Pesquisa web mostrou que Resemblyzer precisa de 30s+ por speaker
- **Nossos áudios**: 9-21s (MUITO CURTOS para Resemblyzer)
- **Benchmarks falhando**: 0/4 testes passando com Resemblyzer
- **Segmentos "Unknown"**: Alta porcentagem de segmentos não atribuídos a nenhum speaker

### Solução Escolhida: Option 2 - Lightweight X-vectors (SpeechBrain)

**Justificativa**:
- RTF 0.09-0.25 em CPU (pesquisa web)
- Multi-scale aggregation para utterances curtas
- Melhor acurácia que MFCCs (85-95% vs 30-50%)
- Adequado para áudios 9-21s com técnica de multi-scale

---

## Mudanças Implementadas

### 1. requirements.txt

**Linhas modificadas**: 37-42

**ANTES**:
```txt
# Speaker Diarization - Embeddings
resemblyzer>=0.1.0,<1.0.0
```

**DEPOIS**:
```txt
# ML/Clustering
scikit-learn>=1.0.0,<2.0.0

# Speaker Diarization - X-vectors (lightweight, CPU-optimized)
speechbrain>=0.5.16,<1.0.0
```

**Razão**:
- Removida dependência Resemblyzer (não adequada para áudios curtos)
- Adicionada dependência SpeechBrain (x-vectors otimizados para CPU)
- Adicionada scikit-learn para SpectralClustering

---

### 2. src/diarization.py - Estrutura Inicial

**Linhas modificadas**: 98-128 (método `__init__` e novo método `_get_xvector_model`)

**ANTES** (linhas 98-128):
```python
class CPUSpeakerDiarization:
    def __init__(self, max_speakers: int = 5):
        self.max_speakers = max_speakers
        self.sample_rate = 16000

        # Lazy loading - Resemblyzer
        self.speaker_encoder = LazyVoiceEncoder()

        logger.info(f"[DIARIZATION] Initialized with max_speakers={max_speakers}")
```

**DEPOIS** (linhas 98-128):
```python
class CPUSpeakerDiarization:
    def __init__(self, max_speakers: int = 5):
        self.max_speakers = max_speakers
        self.sample_rate = 16000

        # Lazy loading - SpeechBrain X-vectors
        self._xvector_model = None

        logger.info(f"[DIARIZATION] Initialized with max_speakers={max_speakers}")

    def _get_xvector_model(self):
        """Lazy-load SpeechBrain x-vector model (CPU-optimized)

        Research-proven RTF 0.09-0.25 on CPU with multi-scale aggregation.
        Better for short utterances (9-21s) than Resemblyzer (requires 30s+).
        """
        if self._xvector_model is None:
            try:
                from speechbrain.pretrained import EncoderClassifier
                logger.info("[SPRINT 3 v2] Loading SpeechBrain x-vector model (lightweight, CPU-only)...")

                self._xvector_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-xvect-voxceleb",
                    savedir="models/xvector_voxceleb",
                    run_opts={"device": "cpu"}
                )

                logger.info("[SPRINT 3 v2] X-vector model loaded successfully")
            except Exception as e:
                logger.error(f"[SPRINT 3 v2] Failed to load x-vector model: {e}")
                raise

        return self._xvector_model
```

**Razão**:
- Substituído `LazyVoiceEncoder()` (Resemblyzer) por `self._xvector_model = None` (SpeechBrain)
- Criado método `_get_xvector_model()` para lazy loading do modelo SpeechBrain
- Modelo: "speechbrain/spkrec-xvect-voxceleb" (CPU-only, lightweight)
- RTF esperado: 0.09-0.25 (pesquisa web)

---

## Mudanças Pendentes (Próximos Passos)

### 3. src/diarization.py - Reescrever `_clustering_diarization`

**Implementação necessária**:

```python
def _clustering_diarization(self, audio_data: np.ndarray, num_speakers: int) -> List[Dict]:
    """
    Multi-scale X-vector extraction + SpectralClustering

    Research findings:
    - Multi-scale: Extract at 1s and 2s segments for short audio
    - SpectralClustering: Better than Agglomerative for irregular clusters
    - Cosine affinity: Standard for speaker embeddings
    """

    # STEP 1: Multi-scale segmentation
    # - 1s segments with 0.5s overlap
    # - 2s segments with 1s overlap

    # STEP 2: Extract x-vectors for each segment
    # model = self._get_xvector_model()
    # embeddings = model.encode_batch(segments)

    # STEP 3: SpectralClustering with cosine affinity
    # from sklearn.cluster import SpectralClustering
    # clustering = SpectralClustering(
    #     n_clusters=num_speakers,
    #     affinity='cosine',
    #     assign_labels='kmeans'
    # )

    # STEP 4: Map clusters back to timestamps
    # STEP 5: Refine segments
    # STEP 6: Calculate confidence scores
```

**Razão**:
- Multi-scale agregation: Solução para áudios curtos (pesquisa web)
- SpectralClustering: "Highly competitive" vs Agglomerative (pesquisa web)
- Cosine affinity: Padrão para embeddings de speaker

---

### 4. src/diarization.py - Remover Código Resemblyzer

**Código a ser REMOVIDO**:

1. **Classe `LazyVoiceEncoder`** (linhas ~40-60):
```python
class LazyVoiceEncoder:
    # REMOVER TODA A CLASSE
```

2. **Método `_extract_speaker_embedding`** (linhas ~200-250):
```python
def _extract_speaker_embedding(self, audio_chunk: np.ndarray) -> np.ndarray:
    # REMOVER TODO O MÉTODO
```

3. **Imports não utilizados**:
```python
# REMOVER imports relacionados a Resemblyzer
```

**Razão**: Código Resemblyzer não será mais utilizado

---

### 5. src/diarization.py - Implementar SpectralClustering

**Código a ser ADICIONADO**:

```python
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

# No método _clustering_diarization:
clustering = SpectralClustering(
    n_clusters=num_speakers,
    affinity='cosine',
    assign_labels='kmeans',
    random_state=42
)
labels = clustering.fit_predict(embeddings)
```

**Razão**:
- Pesquisa web: "SpectralClustering highly competitive, doesn't need statistical metric"
- Melhor para clusters irregulares (comum em speaker data)

---

## Benchmarks Esperados (Pós-Implementação)

### Targets de Performance

| Arquivo | Speakers | Duração | RT Ratio | Acurácia |
|---------|----------|---------|----------|----------|
| d.speakers.wav | 2 | 21s | <1.0x | ≥90% |
| q.speakers.wav | 4 | 14s | <1.0x | ≥90% |
| t.speakers.wav | 3 | 9s | <1.0x | ≥90% |
| t2.speakers.wav | 3 | 10s | <1.0x | ≥90% |

### Performance Esperada (Pesquisa Web)

- **RTF (Real-Time Factor)**: 0.09-0.25 em CPU
- **Acurácia**: 85-95% (vs 30-50% com MFCCs básicos)
- **Multi-scale**: +20-30% melhoria para áudios curtos
- **SpectralClustering**: +15-20% melhoria vs Agglomerative

---

## Fundamentação das Escolhas (Web Research)

### Por que X-vectors?

1. **Lightweight SE-ResNet-34**: RTF 0.09-0.25 em CPU (pesquisa 2024)
2. **Multi-scale aggregation**: Solução comprovada para utterances curtas
3. **Melhor que Resemblyzer para áudios curtos**: Não requer 30s+ por speaker
4. **Melhor que MFCCs**: 85-95% acurácia vs 30-50% com MFCCs

### Por que SpectralClustering?

1. **Pesquisa 2024**: "Highly competitive" vs Agglomerative
2. **Não depende de métrica estatística**: Funciona bem com embeddings
3. **Clusters irregulares**: Comum em speaker data
4. **Cosine affinity**: Padrão para embeddings de alta dimensão

### Por que remover Resemblyzer?

1. **Pesquisa web**: "Recommends 30s+ per speaker"
2. **Nossos áudios**: 9-21s (MUITO CURTOS)
3. **Segmentos**: 2s chunks (vs recomendação 5-30s)
4. **Resultados**: 0/4 testes passando, alta taxa de "Unknown"

---

## Cronograma de Implementação

- [x] **Step 1**: Atualizar requirements.txt
- [x] **Step 2**: Reescrever `_clustering_diarization` com x-vectors
- [x] **Step 3**: Remover código Resemblyzer
- [x] **Step 4**: Implementar SpectralClustering
- [x] **Step 5**: Documentar mudanças (ESTE ARQUIVO)
- [ ] **Step 6**: Testar benchmark (target: 4/4 passing)

---

## Referências de Pesquisa Web

1. **X-vectors lightweight**: SE-ResNet-34 RTF 0.09-0.25 em CPU
2. **Multi-scale aggregation**: Extração em 1s e 2s para áudios curtos
3. **SpectralClustering**: "Highly competitive", não precisa métrica estatística
4. **Resemblyzer limitations**: Requer 30s+ por speaker, áudios 30-60s detectam 8-10 speakers quando só tem 1-2
5. **Cosine affinity**: Padrão para clustering de speaker embeddings
6. **DISPLACE hybrid 2024**: 27.1% DER com ensemble ECAPA-TDNN

---

---

## Resumo das Mudanças Implementadas

### Arquivos Modificados

1. **requirements.txt**
   - ❌ Removido: `resemblyzer>=0.1.0,<1.0.0`
   - ✅ Adicionado: `speechbrain>=0.5.16,<1.0.0`
   - ✅ Adicionado: `scikit-learn>=1.0.0,<2.0.0`

2. **src/diarization.py**
   - ❌ Removida classe `LazyVoiceEncoder` (linhas 48-63)
   - ❌ Removido método `_extract_speaker_embedding` (linhas 162-191)
   - ✅ Adicionado método `_get_xvector_model()` (linhas 113-128)
   - ✅ Reescrito método `_clustering_diarization()` (linhas 545-708):
     - Multi-scale extraction (1s + 2s segments)
     - X-vector embeddings via SpeechBrain
     - SpectralClustering com affinity='cosine'
     - Confidence scoring com cosine distances

### Código Removido (Cleanup)

```python
# REMOVIDO: LazyVoiceEncoder class
class LazyVoiceEncoder:
    def __init__(self):
        self._encoder = None
    ...

# REMOVIDO: _extract_speaker_embedding method
def _extract_speaker_embedding(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
    from resemblyzer import preprocess_wav
    ...
```

### Código Adicionado

**Novo método `_get_xvector_model()`**:
```python
def _get_xvector_model(self):
    """Lazy-load SpeechBrain x-vector model (CPU-optimized)"""
    if self._xvector_model is None:
        from speechbrain.pretrained import EncoderClassifier
        self._xvector_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="models/xvector_voxceleb",
            run_opts={"device": "cpu"}
        )
    return self._xvector_model
```

**Novo método `_clustering_diarization()` (multi-scale x-vectors + SpectralClustering)**:
- Scale 1: 1s segments, 0.5s overlap
- Scale 2: 2s segments, 1s overlap
- SpectralClustering com `affinity='cosine'`
- Confidence scoring preservado com cosine distances

---

---

## Análise de Resultados SPRINT 3 v2 (X-vectors)

### Problemas Encontrados

1. **SpeechBrain incompatível com Windows**:
   - Erro: `OSError: [WinError 1314] O cliente não tem o privilégio necessário`
   - Causa: SpeechBrain usa symlinks que requerem privilégios admin no Windows
   - Tentativas de fix (SPEECHBRAIN_FETCH_STRATEGY, cache direto) falharam

2. **Performance pior que esperado**:
   - RT factor: 0.99x, 1.24x, 1.41x, 1.38x (3/4 falharam target <1.0x)
   - Speaker detection: 3, 3, 1, 2 vs esperado 2, 4, 3, 3 (0/4 correto)
   - Fallback para `simple_diarization` quando x-vectors falham

3. **Comparação com Resemblyzer**:
   - Resemblyzer: 0/4 testes, mas sem erros de privilégio
   - X-vectors: 0/4 testes + erros de sistema operacional

### Decisão: Reverter para Option 1 (MFCC + Prosodic)

**Justificativa**:
- **CPU-only**: Sem dependências PyTorch complexas
- **Windows-friendly**: Sem problemas de symlinks ou privilégios
- **Research-proven**: 30% melhoria com MFCC + pitch/formants
- **Já temos librosa**: Pode extrair pitch e formants

**Próximo passo**: Implementar Option 1 sem SpeechBrain/Resemblyzer

---

**Última atualização**: 2025-10-01
**Status**: X-vectors descartados - mudando para MFCC + Prosodic (Option 1)
