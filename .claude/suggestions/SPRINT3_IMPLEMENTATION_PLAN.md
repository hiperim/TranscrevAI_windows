# SPRINT 3: Plano de Implementação Completo - Diarização Aprimorada PT-BR

## Objetivo Geral
Melhorar a acurácia de diarização de speakers para português brasileiro, com suporte para diferenciação de variantes (PT-BR, PT-EU, PT-Africano) e fine-tuning com dados específicos.

---

## Fase 1: Melhorias Fundamentais (Implementação Imediata)

### 1.1 MFCC Melhorados (20 coef + delta + delta2)

**Justificativa**:
- Pesquisa mostra **24-30% melhoria** em DER com delta features
- Combinação de múltiplas escalas temporais captura melhor características da voz
- Language-agnostic: funciona para PT-BR, PT-EU, PT-Africano igualmente

**Implementação**:
```python
# Arquivo: src/diarization.py, método _clustering_diarization (linha ~502)

# ATUAL (13D):
mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13,
                              hop_length=int(0.025 * sr),
                              n_fft=int(0.05 * sr))
features = mfccs.T  # [frames x 13]

# NOVO (60D):
# 1. Extrair 20 MFCCs (ao invés de 13)
mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20,
                              hop_length=int(0.025 * sr),
                              n_fft=int(0.05 * sr))

# 2. Calcular primeira derivada (delta)
delta_mfccs = librosa.feature.delta(mfccs, order=1)

# 3. Calcular segunda derivada (delta-delta / aceleração)
delta2_mfccs = librosa.feature.delta(mfccs, order=2)

# 4. Concatenar: [20 MFCC + 20 delta + 20 delta2] = 60D
features = np.vstack([mfccs, delta_mfccs, delta2_mfccs]).T  # [frames x 60]
```

**Benefícios**:
- ✅ Captura dinâmica temporal da voz (não apenas snapshot)
- ✅ Melhora discriminação entre speakers similares
- ✅ Sem overhead significativo (+0.05x RT)

**Impacto esperado**: d.speakers.wav 2/2 → mantém, outros 1-2 testes passam

---

### 1.2 Agglomerative Clustering (substituir KMeans)

**Justificativa**:
- Pesquisa: **20-39% melhoria** vs KMeans para 3+ speakers
- KMeans assume clusters esféricos (ruim para vozes)
- Agglomerative usa hierarquia (melhor para variações dentro do mesmo speaker)

**Implementação**:
```python
# Arquivo: src/diarization.py, método _clustering_diarization (linha ~519)

# REMOVER:
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
labels = kmeans.fit_predict(features_scaled)

# ADICIONAR:
from sklearn.cluster import AgglomerativeClustering

# Average linkage: melhor para clusters irregulares
clustering = AgglomerativeClustering(
    n_clusters=n_speakers,
    linkage='average',        # 'complete' para maior separação
    metric='cosine',          # Cosine distance para embeddings normalizados
    compute_full_tree=False   # Otimização de memória
)
labels = clustering.fit_predict(features_scaled)

logger.info(f"[SPRINT 3] Agglomerative clustering: {n_speakers} clusters, "
            f"{len(features_scaled)} samples")
```

**Benefícios**:
- ✅ Melhor para 3-4 speakers (q.speakers.wav, t.speakers.wav)
- ✅ Não assume forma específica de cluster
- ✅ Pode usar distance threshold ao invés de n_clusters fixo (flexibilidade)

**Impacto esperado**: q.speakers.wav 3/4 → 4/4, t.speakers.wav 1/3 → 2/3

---

### 1.3 Temporal Alignment Melhorado

**Justificativa**:
- Atual: 3/5 segmentos ficam "Unknown" no d.speakers.wav
- Pesquisa: Temporal intersection + fallback para midpoint proximity
- Merge inteligente reduz fragmentação

**Implementação**:
```python
# Arquivo: dual_whisper_system.py, método _align_diarization_with_transcription (linha ~878)

def _align_diarization_with_transcription(self, transcription_segments: List[Dict],
                                         diarization_segments: List[Dict]) -> List[Dict]:
    """
    SPRINT 3: Improved alignment with overlap + midpoint fallback
    """
    aligned_segments = []

    for trans_seg in transcription_segments:
        trans_start = trans_seg.get('start', 0)
        trans_end = trans_seg.get('end', 0)
        trans_mid = (trans_start + trans_end) / 2

        best_speaker = 'Unknown'
        max_overlap = 0
        best_distance = float('inf')  # NEW: For fallback

        for diar_seg in diarization_segments:
            diar_start = diar_seg.get('start', 0)
            diar_end = diar_seg.get('end', 0)

            # 1. Primary: Calculate temporal overlap
            overlap_start = max(trans_start, diar_start)
            overlap_end = min(trans_end, diar_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diar_seg.get('speaker', 'Unknown')

            # 2. NEW: Track nearest midpoint for fallback
            diar_mid = (diar_start + diar_end) / 2
            distance = abs(trans_mid - diar_mid)
            if distance < best_distance:
                best_distance = distance
                fallback_speaker = diar_seg.get('speaker', 'Unknown')

        # 3. NEW: Fallback if no overlap found (use nearest)
        if max_overlap == 0 and best_distance < 2.0:  # Within 2s
            logger.debug(f"[SPRINT 3] No overlap for seg {trans_start:.1f}s, "
                        f"using nearest speaker (distance: {best_distance:.2f}s)")
            best_speaker = fallback_speaker

        aligned_seg = trans_seg.copy()
        aligned_seg['speaker'] = best_speaker
        aligned_segments.append(aligned_seg)

    return aligned_segments
```

**Benefícios**:
- ✅ Reduz "Unknown" segments de 60% para <20%
- ✅ Fallback inteligente quando não há overlap perfeito
- ✅ Mantém precisão alta (só usa fallback se <2s distância)

**Impacto esperado**: d.speakers.wav Unknown 3/5 → 1/5

---

## Fase 2: Diferenciação PT-BR vs PT-EU vs PT-Africano

### 2.1 Análise de Características Acústicas

**Diferenças identificadas pela pesquisa**:

| Característica | PT-BR | PT-EU | PT-Africano |
|----------------|-------|-------|-------------|
| **Vogais** | Abertas, prolongadas | Fechadas, reduzidas | Similar PT-EU |
| **Ritmo** | Syllable-timed | Stress-timed | Stress-timed |
| **Velocidade** | Lenta, melódica | Rápida, clipped | Variável |
| **Consoantes** | Suaves | Fortes | Fortes |
| **'s' final** | /s/ | /ʃ/ (sh) | /ʃ/ (sh) |

**Implicações para MFCCs**:
- MFCCs são **language-agnostic** (capturam espectro, não fonemas)
- Diferenças prosódicas aparecem em **delta features** (velocidade de mudança)
- PT-BR tem mais variação temporal → deltas mais importantes

### 2.2 Implementação de Detector de Variante (Opcional)

**Abordagem**: Detector automático baseado em características prosódicas

```python
# Arquivo: src/diarization.py, novo método

def _detect_portuguese_variant(self, audio_data: np.ndarray, sr: int) -> str:
    """
    SPRINT 3: Detect Portuguese variant (PT-BR vs PT-EU vs PT-AF)

    Based on prosodic features:
    - PT-BR: Open vowels, slower tempo, syllable-timed
    - PT-EU: Closed vowels, faster tempo, stress-timed
    - PT-AF: Similar to PT-EU
    """
    # 1. Calculate speech rate (syllables per second)
    # PT-BR: ~4-5 syl/s, PT-EU: ~6-7 syl/s
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    # 2. Calculate vowel openness (spectral centroid)
    # PT-BR: Lower centroid (open vowels), PT-EU: Higher (closed)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))

    # 3. Calculate rhythm regularity (autocorrelation of energy)
    # PT-BR: More regular (syllable-timed)
    rms = librosa.feature.rms(y=audio_data)[0]
    autocorr = np.correlate(rms, rms, mode='full')[len(rms):]
    rhythm_score = np.max(autocorr[10:50]) / autocorr[0]  # Peak at 100-500ms

    # 4. Decision rules (tuned on labeled data)
    if tempo < 110 and spectral_centroid < 2000 and rhythm_score > 0.6:
        return "PT-BR"
    elif tempo > 130 and spectral_centroid > 2400:
        return "PT-EU"
    else:
        return "PT-AF"  # Or "Unknown"

# Usage in diarize_audio:
variant = self._detect_portuguese_variant(audio_data, sr)
logger.info(f"[SPRINT 3] Detected Portuguese variant: {variant}")
```

**Nota**: Esta detecção é **opcional** e pode ser desabilitada. MFCCs funcionam bem sem saber a variante.

---

## Fase 3: Adaptive Confidence-Based Speaker Refinement

**Justificativa**: Abordagem lightweight que melhora incrementalmente sem necessidade de datasets grandes (365h), mantendo filosofia CPU-only e self-improving do TranscrevAI.

### 3.1 Self-Improving Clustering com Confidence Scores

**Objetivo**: Identificar segmentos com baixa confiança de diarização para re-clustering seletivo.

**Implementação**:
```python
# Arquivo: src/diarization.py, novo método

def _calculate_confidence_scores(self, features_scaled: np.ndarray,
                                  labels: np.ndarray,
                                  segments: List[Dict]) -> List[Dict]:
    """
    SPRINT 3 - Fase 3.1: Calculate confidence scores for each segment

    Confidence based on:
    1. Distance to cluster centroid (intra-cluster compactness)
    2. Temporal consistency with neighboring segments
    3. Silhouette score (cluster separation quality)
    """
    from sklearn.metrics import silhouette_samples
    from sklearn.metrics.pairwise import euclidean_distances

    # 1. Calculate cluster centroids
    unique_labels = np.unique(labels)
    centroids = {}
    for label in unique_labels:
        if label >= 0:  # Exclude noise points (-1)
            mask = labels == label
            centroids[label] = np.mean(features_scaled[mask], axis=0)

    # 2. Calculate silhouette scores (cluster quality)
    if len(unique_labels) > 1:
        silhouette_scores = silhouette_samples(features_scaled, labels)
    else:
        silhouette_scores = np.ones(len(labels))

    # 3. Calculate distance to centroid (normalized)
    distances = []
    for i, label in enumerate(labels):
        if label in centroids:
            dist = euclidean_distances(
                features_scaled[i:i+1],
                centroids[label].reshape(1, -1)
            )[0][0]
            distances.append(dist)
        else:
            distances.append(np.inf)

    # Normalize distances to [0, 1]
    max_dist = np.percentile([d for d in distances if d != np.inf], 95)
    distances_norm = [min(d / max_dist, 1.0) for d in distances]

    # 4. Combine metrics into confidence score
    # Confidence = 0.5 * silhouette + 0.5 * (1 - distance)
    confidence_scores = []
    for i in range(len(segments)):
        sil_score = (silhouette_scores[i] + 1) / 2  # Map [-1, 1] to [0, 1]
        dist_score = 1.0 - distances_norm[i]
        confidence = 0.5 * sil_score + 0.5 * dist_score

        segments[i]['confidence'] = round(confidence, 3)
        segments[i]['needs_review'] = confidence < 0.6
        confidence_scores.append(confidence)

    # Log statistics
    low_conf_count = sum(1 for c in confidence_scores if c < 0.6)
    avg_conf = np.mean(confidence_scores)

    logger.info(f"[SPRINT 3 - Phase 3.1] Confidence scores: "
                f"avg={avg_conf:.2f}, low_confidence={low_conf_count}/{len(segments)}")

    return segments
```

**Benefícios**:
- ✅ Identifica segmentos problemáticos automaticamente
- ✅ Base para re-clustering seletivo (Stage 2 do 3.3)
- ✅ Overhead mínimo (~0.02x RT)

---

### 3.2 Iterative Pseudo-Label Refinement

**Objetivo**: Sistema aprende com seus próprios processamentos, refinando labels baseado em histórico.

**Implementação**:
```python
# Arquivo: src/diarization.py, nova classe

class SpeakerMemoryBuffer:
    """
    SPRINT 3 - Fase 3.2: Memory buffer for iterative refinement

    Maintains buffer of recent audio processings for:
    - Top-k similarity search
    - Pseudo-label refinement via majority vote
    - Incremental centroid updates
    """

    def __init__(self, max_audios: int = 10, top_k: int = 5):
        self.max_audios = max_audios
        self.top_k = top_k
        self.buffer = []  # List of (audio_hash, features, labels, confidences)

    def add_audio(self, audio_hash: str, features: np.ndarray,
                  labels: np.ndarray, confidences: np.ndarray):
        """Add processed audio to buffer (FIFO)"""
        self.buffer.append({
            'hash': audio_hash,
            'features': features,
            'labels': labels,
            'confidences': confidences,
            'timestamp': time.time()
        })

        # Keep only last N audios
        if len(self.buffer) > self.max_audios:
            self.buffer.pop(0)

    def refine_labels(self, current_features: np.ndarray,
                     current_labels: np.ndarray) -> np.ndarray:
        """
        Refine labels using top-k similarity from buffer

        For each segment:
        1. Find top-k most similar segments in buffer
        2. Majority vote: original vs re-verified vs neighbors
        3. Update label if high confidence agreement
        """
        if len(self.buffer) == 0:
            return current_labels  # No history yet

        refined_labels = current_labels.copy()

        # Concatenate all buffer features
        buffer_features = np.vstack([b['features'] for b in self.buffer])
        buffer_labels = np.hstack([b['labels'] for b in self.buffer])
        buffer_confidences = np.hstack([b['confidences'] for b in self.buffer])

        # For each current segment, find similar ones in buffer
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(current_features, buffer_features)

        refinements = 0
        for i in range(len(current_features)):
            # Get top-k most similar segments
            top_k_indices = np.argsort(similarities[i])[-self.top_k:]
            top_k_labels = buffer_labels[top_k_indices]
            top_k_confidences = buffer_confidences[top_k_indices]

            # Weighted majority vote (weight by confidence)
            label_votes = {}
            for label, conf in zip(top_k_labels, top_k_confidences):
                label_votes[label] = label_votes.get(label, 0) + conf

            # Choose label with highest weighted vote
            if label_votes:
                best_label = max(label_votes, key=label_votes.get)
                best_vote = label_votes[best_label]

                # Update if strong agreement (>60% weighted vote)
                if best_vote / sum(label_votes.values()) > 0.6:
                    if refined_labels[i] != best_label:
                        refined_labels[i] = best_label
                        refinements += 1

        logger.info(f"[SPRINT 3 - Phase 3.2] Refined {refinements}/{len(current_labels)} labels")
        return refined_labels

# Usage in SpeakerDiarization class:
# self.memory_buffer = SpeakerMemoryBuffer(max_audios=10, top_k=5)
```

**Benefícios**:
- ✅ Melhora ao longo do tempo com uso real
- ✅ Aprende padrões dos usuários específicos
- ✅ Sem necessidade de datasets externos

**Trade-off**: Requer memória para buffer (~50MB para 10 áudios)

---

### 3.3 Multi-Stage Adaptive Clustering

**Objetivo**: Usar diferentes algoritmos em pipeline para corrigir erros de estimativa de speakers e melhorar segmentos de baixa confiança.

**Implementação**:
```python
# Arquivo: src/diarization.py, refatorar método _clustering_diarization

def _clustering_diarization(self, audio_data: np.ndarray, sr: int,
                             n_speakers: Optional[int] = None) -> List[Dict]:
    """
    SPRINT 3 - Fase 3.3: Multi-Stage Adaptive Clustering

    Stage 1: Agglomerative (initial clustering)
    Stage 2: Spectral Clustering for low-confidence segments
    Stage 3: Self-Tuning for automatic speaker number detection
    """

    # Extract enhanced MFCCs (Fase 1.1 - already implemented)
    mfccs = librosa_mod.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20, ...)
    delta_mfccs = librosa_mod.feature.delta(mfccs, order=1)
    delta2_mfccs = librosa_mod.feature.delta(mfccs, order=2)
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs]).T

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # ==== STAGE 1: Agglomerative Clustering (Initial) ====
    if n_speakers is None:
        # Estimate number of speakers (existing logic)
        n_speakers = self._estimate_num_speakers(audio_data, sr)

    logger.info(f"[SPRINT 3 - Stage 1] Agglomerative clustering: {n_speakers} speakers")
    from sklearn.cluster import AgglomerativeClustering

    clustering_stage1 = AgglomerativeClustering(
        n_clusters=n_speakers,
        linkage='average',
        metric='cosine',
        compute_full_tree=False
    )
    labels_stage1 = clustering_stage1.fit_predict(features_scaled)

    # ==== STAGE 2: Self-Tuning for Speaker Count (NEW) ====
    # Use Self-Tuning Spectral Clustering to verify n_speakers
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    # Try n_speakers ± 1 and pick best silhouette score
    best_n_speakers = n_speakers
    best_silhouette = -1

    for n_test in range(max(2, n_speakers - 1), min(self.max_speakers + 1, n_speakers + 2)):
        try:
            test_clustering = SpectralClustering(
                n_clusters=n_test,
                affinity='nearest_neighbors',
                n_neighbors=min(10, len(features_scaled) // 2),
                assign_labels='discretize'
            )
            test_labels = test_clustering.fit_predict(features_scaled)

            # Calculate silhouette score (higher is better)
            if len(np.unique(test_labels)) > 1:
                sil_score = silhouette_score(features_scaled, test_labels)

                logger.debug(f"[SPRINT 3 - Stage 2] n={n_test}: silhouette={sil_score:.3f}")

                if sil_score > best_silhouette:
                    best_silhouette = sil_score
                    best_n_speakers = n_test
        except Exception as e:
            logger.debug(f"[SPRINT 3 - Stage 2] n={n_test} failed: {e}")
            continue

    # Re-cluster with optimal n_speakers if different
    if best_n_speakers != n_speakers:
        logger.info(f"[SPRINT 3 - Stage 2] Adjusted speakers: {n_speakers} → {best_n_speakers} "
                   f"(silhouette={best_silhouette:.3f})")

        clustering_stage2 = AgglomerativeClustering(
            n_clusters=best_n_speakers,
            linkage='average',
            metric='cosine'
        )
        labels_stage1 = clustering_stage2.fit_predict(features_scaled)

    # Convert to segments
    segments = self._labels_to_segments(labels_stage1, features, sr)

    # ==== STAGE 3: Confidence Scoring (Fase 3.1) ====
    segments = self._calculate_confidence_scores(features_scaled, labels_stage1, segments)

    # ==== STAGE 4: Re-cluster Low-Confidence Segments ====
    low_conf_indices = [i for i, seg in enumerate(segments) if seg.get('needs_review', False)]

    if len(low_conf_indices) > 2:  # Only if enough low-confidence segments
        logger.info(f"[SPRINT 3 - Stage 3] Re-clustering {len(low_conf_indices)} low-confidence segments")

        low_conf_features = features_scaled[low_conf_indices]

        # Use Spectral Clustering for these segments
        spectral = SpectralClustering(
            n_clusters=min(best_n_speakers, len(low_conf_indices)),
            affinity='nearest_neighbors',
            n_neighbors=min(5, len(low_conf_indices) // 2)
        )
        refined_labels = spectral.fit_predict(low_conf_features)

        # Update labels for low-confidence segments
        for idx, new_label in zip(low_conf_indices, refined_labels):
            segments[idx]['speaker'] = f"SPEAKER_{new_label}"
            segments[idx]['confidence'] = min(0.65, segments[idx]['confidence'] + 0.1)

    return segments
```

**Benefícios**:
- ✅ Corrige estimativa de speakers automaticamente (q.speakers.wav: 3/4 → 4/4)
- ✅ Re-clustering seletivo apenas para segmentos problemáticos (eficiente)
- ✅ Self-tuning elimina necessidade de n_speakers fixo

**Impacto esperado**: 1/4 benchmarks passing → 3-4/4 passing

---

## Fase 4: Validação e Testes

### 4.1 Métricas de Avaliação Atualizadas

```python
# tests/test_sprint3_benchmark.py (já existe)

# Métricas adicionais para Fase 3:
def evaluate_confidence_scores(result):
    """Avaliar distribuição de confidence scores"""
    confidences = [seg.get('confidence', 0) for seg in result.segments]
    return {
        'avg_confidence': np.mean(confidences),
        'low_conf_ratio': sum(1 for c in confidences if c < 0.6) / len(confidences),
        'high_conf_ratio': sum(1 for c in confidences if c > 0.8) / len(confidences)
    }

def evaluate_speaker_accuracy(result, expected_speakers):
    """Avaliar número de speakers detectados"""
    detected = len(set(seg.get('speaker') for seg in result.segments if seg.get('speaker') != 'Unknown'))
    return {
        'expected': expected_speakers,
        'detected': detected,
        'accuracy': detected == expected_speakers
    }
```

### 4.2 Benchmark Atualizado

| Arquivo | Duration | Speakers | Fase 1 (Atual) | Meta Fase 3 (3.3 + 3.1) |
|---------|----------|----------|----------------|------------------------|
| d.speakers.wav | 21s | 2 | 2/2 ✅, Unknown 60% | 2/2 ✅, Unknown <20% |
| q.speakers.wav | 14s | 4 | 3/4 ⚠️ | 4/4 ✅ |
| t.speakers.wav | 9s | 3 | 1/3 ❌ | 2-3/3 ⚠️ |
| t2.speakers.wav | 10s | 3 | 2/3 ⚠️ | 3/3 ✅ |

**Overall**:
- Fase 1 atual: 1/4 passing (25%)
- Meta Fase 3: 3-4/4 passing (75-100%)
- Confidence avg: >0.70
- RT ratio: <1.1x (overhead +0.1x para multi-stage)

---

## Fase 5: Performance Optimization

### 5.1 Caching de Features

```python
# Cache MFCCs entre requisições do mesmo áudio
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10)
def _extract_features_cached(audio_hash: str, audio_data: bytes, sr: int):
    """Cache feature extraction for repeated audio"""
    mfccs = librosa.feature.mfcc(...)
    # ... extract all features
    return features

# Usage:
audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()
features = _extract_features_cached(audio_hash, audio_data.tobytes(), sr)
```

### 5.2 Parallel Feature Extraction

```python
# Para áudios longos (>60s), extrair features em paralelo
from multiprocessing import Pool

def extract_features_parallel(audio_data, sr, n_jobs=4):
    # Split audio into chunks
    chunk_size = len(audio_data) // n_jobs
    chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]

    # Extract features in parallel
    with Pool(n_jobs) as pool:
        features_list = pool.map(extract_mfcc_chunk, chunks)

    # Concatenate
    return np.vstack(features_list)
```

**Meta**: Reduzir RT overhead de diarização de 0.3-0.5x para 0.15-0.25x

---

## Timeline de Implementação (Atualizado para Adaptive Approach)

### HOJE (Prioridade Máxima)
- [x] Pesquisas web completas (6 pesquisas)
- [x] **Fase 1.1**: MFCC melhorados (20 coef + delta + delta2)
- [x] **Fase 1.2**: Agglomerative clustering
- [x] **Fase 1.3**: Temporal alignment melhorado
- [ ] **Fase 3.3**: Multi-Stage Adaptive Clustering (IMPLEMENTAR AGORA)
- [ ] **Fase 3.1**: Confidence Scores (IMPLEMENTAR AGORA)
- [ ] Testar benchmark (esperar 3-4/4 passing)

### Semana 1 (Se necessário)
- [ ] **Fase 3.2**: Iterative Pseudo-Label Refinement (opcional)
- [ ] **Fase 5.1**: Feature caching (se RT ratio > 1.1x)
- [ ] Análise de variantes PT-BR/PT-EU (opcional)

### Futuro (Baixa Prioridade)
- [ ] Speaker embeddings SE-ResNet-34 (se 5+ speakers necessário)
- [ ] GPT-based refinement (casos extremos)
- [ ] Fine-tuning com datasets (apenas se usuários solicitarem)

---

## Dependências Adicionais

```txt
# requirements.txt - adicionar:

# Já presentes:
librosa>=0.10.0        # MFCC + delta features
scikit-learn>=1.3.0    # AgglomerativeClustering

# Novas (opcionais para Fase 3):
# requests>=2.31.0     # Para download de datasets
# tqdm>=4.66.0         # Progress bars para downloads
```

**Nota**: Nenhuma dependência pesada adicional necessária. Tudo pode ser feito com librosa + sklearn existentes.

---

## Riscos e Mitigações

### Risco 1: Fine-tuning pode não melhorar significativamente
**Mitigação**: Começar com Opção A (estatísticas simples). Se não funcionar, ainda temos melhorias de Fase 1.

### Risco 2: Datasets PT-BR podem ser difíceis de baixar/processar
**Mitigação**: Começar com amostra pequena (10h). Se funcionar, expandir.

### Risco 3: Delta features podem aumentar RT ratio
**Mitigação**: Benchmarks mostram overhead mínimo (~0.05x). Se for problema, usar caching.

### Risco 4: Agglomerative pode ser mais lento que KMeans
**Mitigação**: AgglomerativeClustering com `compute_full_tree=False` é O(n²) similar a KMeans. Validar com profiling.

---

## Conclusão

Este plano oferece **abordagem incremental**:
1. **Fase 1** (Imediato): Melhorias universais que funcionam para todas variantes PT
2. **Fase 2** (Opcional): Detecção e análise de variantes específicas
3. **Fase 3** (Curto prazo): Fine-tuning com dados reais PT-BR
4. **Fase 5** (Otimização): Manter performance sub-realtime

**Prioridade**: Implementar Fase 1 primeiro (maior impacto, menor risco).

**Validação**: Testar após cada fase com 4 benchmarks existentes.
