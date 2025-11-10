"""Error message translations for user-facing responses in PT-BR"""

ERROR_MESSAGES_PT = {
    # Validation errors
    "file_too_large": "Arquivo muito grande. Tamanho máximo: {max_size}MB",
    "invalid_format": "Formato de arquivo inválido ou não suportado.",
    "missing_field": "Campo obrigatório ausente: {field}",
    "duration_exceeded": "Gravação excedeu o tempo máximo de {max_duration} segundos",
    "invalid_action": "Ação inválida ou ausente. Use 'start', 'audio_chunk' ou 'stop'.",

    # Processing errors
    "transcription_failed": "Falha na transcrição do áudio",
    "diarization_failed": "Falha na identificação de falantes",
    "audio_quality_low": "Qualidade do áudio muito baixa",
    "model_load_failed": "Erro ao carregar modelo de IA",

    # Session errors
    "session_not_found": "Sessão não encontrada",
    "session_expired": "Sessão expirada",
    "concurrent_limit": "Limite de gravações simultâneas atingido",

    # Generic errors
    "internal_error": "Erro interno do servidor. Tente novamente",
    "unknown_error": "Erro desconhecido: {details}"
}

def get_user_message(error_key: str, **kwargs) -> str:
    """Get user-facing error message in PT-BR:

    Args:
        error_key: Error message key from ERROR_MESSAGES_PT
        **kwargs: Values to format into message template

    Returns:
        Formatted Portuguese error message"""
    template = ERROR_MESSAGES_PT.get(error_key, ERROR_MESSAGES_PT["unknown_error"])
    return template.format(**kwargs)
