"""
Parâmetros Whisper Otimizados para PT-BR - FASE 1 Fine-tuning
Focado exclusivamente em português brasileiro
"""

# PARÂMETROS ORIGINAIS (BACKUP) - CPU-ONLY
ORIGINAL_PARAMS = {
    "language": "pt",  # Fixo PT-BR
    "task": "transcribe",
    "verbose": False,
    "beam_size": 1,
    "best_of": 1,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "compression_ratio_threshold": 1.8,
    "logprob_threshold": -0.6,
    "no_speech_threshold": 0.9,
    "word_timestamps": False,
    "prepend_punctuations": "",
    "append_punctuations": "",
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "hallucination_silence_threshold": 1.0
}

# PARÂMETROS OTIMIZADOS FASE 1 - PT-BR EXCLUSIVO
PHASE1_OPTIMIZED_PARAMS = {
    # CORE SPEED OPTIMIZATIONS
    "beam_size": 1,  # Mantido - já otimizado
    "best_of": 1,    # Mantido - já otimizado
    "temperature": 0.1,  # Ligeiramente mais alto para melhor accuracy

    # PT-BR SPECIFIC OPTIMIZATIONS
    "compression_ratio_threshold": 1.6,  # Otimizado para PT-BR
    "no_speech_threshold": 0.85,         # Ajustado para português
    "logprob_threshold": -0.8,           # Mais conservador para accuracy PT

    # AGGRESSIVE PROCESSING OPTIMIZATIONS
    "condition_on_previous_text": False,    # Mantido OFF para speed
    "word_timestamps": False,               # Mantido OFF
    "without_timestamps": True,             # Mantido ON para speed
    "suppress_blank": True,                 # Mantido ON
    "suppress_tokens": [-1, 50256],         # Extended suppression
    "hallucination_silence_threshold": 0.8, # Mais agressivo

    # NEW EXPERIMENTAL OPTIMIZATIONS
    "patience": None,                       # Remove patience for speed
    "length_penalty": None,                 # Remove length penalty
    "repetition_penalty": 1.0,              # Minimal repetition control
    "no_repeat_ngram_size": 0,              # Disable n-gram blocking for speed
}

def get_optimized_params(use_phase1: bool = True) -> dict:
    """
    Retorna parâmetros otimizados para PT-BR (CPU-ONLY)

    Args:
        use_phase1: Se True, usa otimizações FASE 1, senão usa original

    Returns:
        Dict com parâmetros otimizados para PT-BR
    """
    if not use_phase1:
        # Rollback para parâmetros originais
        return ORIGINAL_PARAMS.copy()

    # Usar parâmetros FASE 1 otimizados para PT-BR (CPU-ONLY)
    base_params = {
        "language": "pt",  # Fixo PT-BR
        "task": "transcribe",
        "verbose": False,
        "prepend_punctuations": "",
        "append_punctuations": "",
    }

    # Aplicar todas as otimizações FASE 1
    base_params.update(PHASE1_OPTIMIZED_PARAMS)

    return base_params

def validate_params_safety(params: dict) -> bool:
    """
    Valida se os parâmetros são seguros e não vão quebrar o modelo
    
    Args:
        params: Dict com parâmetros Whisper
        
    Returns:
        True se parâmetros são seguros, False caso contrário
    """
    # Validações básicas de segurança
    safety_checks = [
        params.get("beam_size", 1) >= 1,
        params.get("best_of", 1) >= 1,
        0 <= params.get("temperature", 0) <= 2.0,
        params.get("compression_ratio_threshold", 2.0) > 0,
        -1.0 <= params.get("logprob_threshold", 0) <= 1.0,
        0 <= params.get("no_speech_threshold", 0.6) <= 1.0,
    ]
    
    return all(safety_checks)

# Metrics tracking para comparação de performance
PERFORMANCE_METRICS = {
    "original": {
        "average_ratio": 1.17,  # Baseline from tests
        "accuracy": 0.857,      # 85.7% from tests
        "stability": "high"
    },
    "phase1_target": {
        "average_ratio": 0.95,  # Target 18% improvement
        "accuracy": 0.85,       # Maintain ≥85%
        "stability": "high"
    }
}