# TranscrevAI - Test Suite Documentation

Documentação completa da suite de testes para validação de compliance e produção.

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Estrutura dos Testes](#estrutura-dos-testes)
3. [Testes de Compliance](#testes-de-compliance)
4. [Benchmark do pyannote.audio](#benchmark-do-pyannotea udio)
5. [Como Executar](#como-executar)
6. [Interpretando Resultados](#interpretando-resultados)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Visão Geral

A suite de testes do TranscrevAI valida três pilares críticos para produção:

| Pilar | Target | Limite Aceitável |
|-------|--------|------------------|
| **RAM** | ≤4.0 GB | ≤5.0 GB (hard limit) |
| **Velocidade** | ~0.75x | ≤2.0x (aceitável) |
| **Precisão** | 90%+ | 85%+ (mínimo) |

### Compliance Rules

Baseado em: `.claude/compliance.md`

- **Regra 1:** Processing Speed (~0.75s/1s audio)
- **Regra 7:** Memory Management (≤3.5GB RAM, flexível para 4.0GB)
- **Regra 22-23:** Quality Assurance (90%+ accuracy)

---

## 📁 Estrutura dos Testes

```
tests/
├── README.md                    # Esta documentação
├── conftest.py                  # Fixtures do pytest
├── test_unit.py                 # Testes de integração básicos
├── test_compliance.py           # ✨ Testes de compliance (RAM/Speed/Accuracy)
└── benchmark_pyannote.py        # ✨ Script standalone de benchmark
```

### Arquivos Principais

#### **test_compliance.py**
Testes automatizados do pytest para validação de compliance.

**Testes incluídos:**
- `test_ram_compliance_hard_limit` - Valida RAM ≤5.0GB
- `test_speed_compliance_acceptable` - Valida velocidade ≤2.0x
- `test_speaker_accuracy_compliance` - Valida precisão de diarização
- `test_compliance_summary_report` - Relatório agregado
- `test_benchmark_initialization_time` - Tempo de inicialização
- `test_benchmark_component_timing` - Breakdown por componente

#### **benchmark_pyannote.py**
Script standalone para validação empírica rápida do pyannote.audio.

**Funcionalidades:**
- Medição detalhada de RAM durante processamento
- Breakdown de tempo (transcrição vs diarização)
- Validação de precisão de speakers
- Relatório consolidado de compliance
- Exportação de resultados para JSON

---

## 🧪 Testes de Compliance

### Pré-requisitos

1. **Token do Hugging Face:**
   ```bash
   # Windows (PowerShell)
   $env:HUGGING_FACE_HUB_TOKEN="seu_token_aqui"

   # Windows (CMD)
   set HUGGING_FACE_HUB_TOKEN=seu_token_aqui

   # Linux/Mac
   export HUGGING_FACE_HUB_TOKEN="seu_token_aqui"
   ```

   Obtenha seu token em: https://hf.co/settings/tokens

2. **Arquivos de Áudio de Teste:**
   Coloque os arquivos de benchmark em: `data/recordings/`

   Arquivos esperados:
   - `d.speakers.wav` (2 speakers)
   - `q.speakers.wav` (4 speakers)
   - `t.speakers.wav` (3 speakers)
   - `t2.speakers.wav` (3 speakers)

### Executando Testes de Compliance

```bash
# Executar todos os testes de compliance
pytest tests/test_compliance.py -v

# Executar teste específico
pytest tests/test_compliance.py::test_ram_compliance_hard_limit -v

# Executar com relatório detalhado
pytest tests/test_compliance.py -v -s

# Executar apenas testes de benchmark (marcados como @pytest.mark.benchmark)
pytest tests/test_compliance.py -m benchmark -v
```

### Interpretando Resultados

**Exemplo de Output:**

```
============================================================
RAM Compliance Test: q.speakers.wav
Audio Duration: 15.23s
Peak RAM: 4.12 GB
Delta RAM: 1.85 GB
Hard Limit: 5.0 GB
Target: 4.0 GB
============================================================
PASSED
```

**Status:**
- ✅ **PASS** - Dentro do limite aceitável
- ❌ **FAIL** - Excede o limite (requer otimização)

---

## 📊 Benchmark do pyannote.audio

### Script Standalone

O `benchmark_pyannote.py` é um script standalone para validação rápida sem pytest.

### Uso Básico

```bash
# Benchmark de todos os arquivos em data/recordings/
python tests/benchmark_pyannote.py

# Benchmark de um arquivo específico
python tests/benchmark_pyannote.py --file data/recordings/q.speakers.wav

# Ajuda
python tests/benchmark_pyannote.py --help
```

### Output do Benchmark

O script gera relatórios detalhados em 4 fases:

#### **Fase 1: Initialization**
```
============================================================
🔧 INITIALIZATION BENCHMARK
============================================================
Loading transcription model...
Loading diarization model...

✓ Initialization Complete
  Time: 12.34s
  RAM Used: 1.23 GB
```

#### **Fase 2: Processamento Individual**
```
============================================================
📊 BENCHMARKING: q.speakers.wav
============================================================
Audio Duration: 15.23s
Expected Speakers: 4

🎤 Transcription Phase...
  ✓ Completed in 8.45s

👥 Diarization Phase...
  ✓ Completed in 12.67s
  ✓ Detected 4 speakers

────────────────────────────────────────────────────────────
RESULTS:
────────────────────────────────────────────────────────────
⏱️  Processing Time:
   Transcription: 8.45s (40.0%)
   Diarization:   12.67s (60.0%)
   Total:         21.12s
   Ratio:         1.39x

💾 Memory Usage:
   Peak RAM:      4.12 GB

🎯 Accuracy:
   Expected:      4 speakers
   Detected:      4 speakers
   Error:         0 (Tolerance: ±1)

────────────────────────────────────────────────────────────
COMPLIANCE:
────────────────────────────────────────────────────────────
Speed (≤2.0x):  1.39x  ✅ PASS
RAM (≤5.0GB):   4.12GB ✅ PASS
Accuracy (±1):     0      ✅ PASS
```

#### **Fase 3: Summary Report**
```
============================================================
📋 COMPREHENSIVE BENCHMARK SUMMARY
============================================================

       ⚡ SPEED PERFORMANCE
────────────────────────────────────────────────────────────
File                 Duration   Time       Ratio      Status
────────────────────────────────────────────────────────────
d.speakers.wav         8.5s     12.3s      1.45x     ✅
q.speakers.wav        15.2s     21.1s      1.39x     ✅
t.speakers.wav        10.8s     16.4s      1.52x     ✅
t2.speakers.wav       11.2s     17.8s      1.59x     ✅
────────────────────────────────────────────────────────────
Average Ratio: 1.49x
Pass Rate:     100.0% (Target: ≤2.0x)

       💾 MEMORY USAGE
────────────────────────────────────────────────────────────
File                 Peak RAM        Status
────────────────────────────────────────────────────────────
d.speakers.wav          3.85 GB  ✅
q.speakers.wav          4.12 GB  ✅
t.speakers.wav          3.92 GB  ✅
t2.speakers.wav         4.05 GB  ✅
────────────────────────────────────────────────────────────
Average RAM:   3.99 GB
Pass Rate:     100.0% (Target: ≤5.0GB)

       🎯 SPEAKER ACCURACY
────────────────────────────────────────────────────────────
File                 Expected     Detected     Error      Status
────────────────────────────────────────────────────────────
d.speakers.wav              2            2         0  ✅
q.speakers.wav              4            4         0  ✅
t.speakers.wav              3            3         0  ✅
t2.speakers.wav             3            3         0  ✅
────────────────────────────────────────────────────────────
Pass Rate:     100.0% (Tolerance: ±1)

════════════════════════════════════════════════════════════
           🏆 OVERALL ASSESSMENT
════════════════════════════════════════════════════════════

✅ PRODUCTION READY
   All compliance targets met. System is ready for production deployment.

════════════════════════════════════════════════════════════
```

#### **Fase 4: Exportação JSON**
```
📁 Results saved to: benchmarks/benchmark_20251012_143522.json
```

### Resultados JSON

Os resultados são salvos em `benchmarks/benchmark_YYYYMMDD_HHMMSS.json`:

```json
{
  "initialization": {
    "time_seconds": 12.34,
    "ram_gb": 1.23
  },
  "speed": [
    {
      "file": "q.speakers.wav",
      "audio_duration": 15.23,
      "transcription_time": 8.45,
      "diarization_time": 12.67,
      "total_time": 21.12,
      "processing_ratio": 1.39,
      "peak_ram_gb": 4.12,
      "expected_speakers": 4,
      "detected_speakers": 4,
      "speaker_error": 0,
      "speed_pass": true,
      "ram_pass": true,
      "accuracy_pass": true
    }
  ]
}
```

---

## 🚀 Como Executar

### 1. Configuração Inicial

```bash
# Instalar dependências de teste
pip install pytest pytest-asyncio psutil

# Configurar token do Hugging Face
export HUGGING_FACE_HUB_TOKEN="seu_token"
```

### 2. Validação Rápida (Benchmark Standalone)

**Recomendado para primeira validação:**

```bash
# Validação completa (todos os arquivos)
python tests/benchmark_pyannote.py

# Validação de arquivo único (mais rápido)
python tests/benchmark_pyannote.py --file data/recordings/q.speakers.wav
```

### 3. Testes Automatizados (Pytest)

**Recomendado para CI/CD:**

```bash
# Suite completa de compliance
pytest tests/test_compliance.py -v

# Com relatório de cobertura
pytest tests/test_compliance.py -v --cov=src

# Executar todos os testes (unit + compliance)
pytest tests/ -v
```

### 4. Fluxo de Validação Recomendado

```bash
# 1. Validação rápida de um arquivo
python tests/benchmark_pyannote.py --file data/recordings/q.speakers.wav

# 2. Se passar, validar todos
python tests/benchmark_pyannote.py

# 3. Executar suite completa de testes
pytest tests/test_compliance.py -v

# 4. Validar integração
pytest tests/test_unit.py -v
```

---

## 📈 Interpretando Resultados

### Critérios de Aprovação

| Métrica | Ideal | Aceitável | Falha |
|---------|-------|-----------|-------|
| **RAM** | ≤4.0GB | ≤5.0GB | >5.0GB |
| **Velocidade** | ≤0.75x | ≤2.0x | >2.0x |
| **Precisão (speakers)** | Exato | ±1 speaker | >±1 speaker |

### Decisões Baseadas em Resultados

#### ✅ Todos os Testes Passam
**Status:** PRODUCTION READY

**Ação:** Prosseguir com deployment.

#### ⚠️ RAM ou Speed no Limite Aceitável
**Status:** FUNCIONAL, mas requer otimização

**Ações possíveis:**
1. **RAM alto (4.0-5.0GB):**
   - Verificar se há memory leaks
   - Considerar quantização de modelos
   - Testar em hardware de produção

2. **Speed lento (1.5-2.0x):**
   - Acceptable para PT-BR com alta precisão
   - Documentar trade-off (precisão vs velocidade)
   - Considerar otimizações futuras

#### ❌ Testes Falhando
**Status:** NÃO PRONTO PARA PRODUÇÃO

**Investigação necessária:**
1. **RAM >5.0GB:**
   - Analisar memory profiling
   - Verificar modelos carregados
   - Testar alternativas (model quantization)

2. **Speed >2.0x:**
   - Verificar se GPU está sendo usado (deve ser CPU)
   - Analisar bottlenecks (benchmark_component_timing)
   - Considerar otimizações de pyannote

3. **Precisão baixa (>±1 speaker):**
   - Ajustar `clustering.threshold` do pyannote
   - Validar qualidade dos áudios de teste
   - Revisar pipeline de diarização

---

## 🔧 Troubleshooting

### Erro: "HUGGING_FACE_HUB_TOKEN not set"

**Solução:**
```bash
# Obter token em https://hf.co/settings/tokens
export HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxxx"
```

### Erro: "No benchmark audio files found"

**Solução:**
```bash
# Verificar diretório
ls data/recordings/

# Criar diretório se não existir
mkdir -p data/recordings

# Adicionar arquivos de teste (.wav)
# Arquivos esperados: d.speakers.wav, q.speakers.wav, t.speakers.wav, t2.speakers.wav
```

### Erro: "Pipeline not initialized"

**Solução:**
```bash
# Garantir que main.py inicializou serviços
# Verificar logs de startup
python main.py

# Se persistir, verificar imports no test file
```

### RAM Usage Muito Alto

**Investigação:**
```bash
# Executar com profiling detalhado
python -m memory_profiler tests/benchmark_pyannote.py

# Verificar modelos carregados
# Considerar quantização ou modelos menores
```

### Testes Lentos

**Otimização:**
```bash
# Executar apenas um arquivo para desenvolvimento
python tests/benchmark_pyannote.py --file data/recordings/d.speakers.wav

# Usar pytest com menos verbosity
pytest tests/test_compliance.py -q
```

---

## 📝 Notas Importantes

1. **Arquivos de Benchmark:** Os arquivos `.wav` de teste NÃO estão incluídos no repositório. Você deve fornecê-los em `data/recordings/`.

2. **Hardware:** Os benchmarks são executados em CPU para consistência. Resultados em GPU serão diferentes.

3. **Targets Atualizados:** Os targets de compliance foram atualizados baseados na análise do pivô para pyannote.audio (11/10/2025).

4. **Resultados Históricos:** Benchmarks são salvos em `benchmarks/` para comparação histórica.

---

## 📚 Referências

- Compliance Rules: `.claude/compliance.md`
- Pyannote Justification: `.claude/suggestions/GEMINI_SUGGESTIONS/project_status_and_pyannote_justification_v1.md`
- Change Log: `.claude/CHANGES_MADE/GEMINI/2025_10_11_pivot_to_pyannote_log.md`

---

**Última Atualização:** 2025-10-12
**Versão:** 1.0
**Autor:** TranscrevAI Development Team
