# FINALIZED AND CORRECTED - Enhanced Transcription Module with Model Unloading
"""
Enhanced Transcription Module with complete PT-BR corrections, advanced confidence
scoring, automatic model unloading for memory optimization, and production-ready
thread-safety mechanisms.
"""

import logging
import asyncio
import gc
import time
import re
import unicodedata
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """Standardized transcription result data structure."""
    text: str
    segments: List[Dict[str, Any]]
    language: str
    confidence: float
    processing_time: float
    word_count: int

class TranscriptionService:
    """Handles the core transcription logic using faster-whisper with automatic model unloading."""

    def __init__(self, model_name: str = "medium", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.compute_type = "int8" # Default compute type
        self.model = None
        self.last_used = time.time()
        self.model_unload_delay = 600  # 10 minutes (optimized from 30min for web usage)
        self._model_lock = asyncio.Lock()  # Thread-safety for model operations
        self.model_loads_count = 0  # Monitoring metric
        self.model_unloads_count = 0  # Monitoring metric
        self._init_ptbr_corrections()

    async def initialize(self, model_unload_delay: int = 600, compute_type: str = "int8"):
        """Loads the transcription model."""
        self.model_unload_delay = model_unload_delay
        self.compute_type = compute_type
        await self._load_model()

    def _init_ptbr_corrections(self):
        self.ptbr_corrections = {
            # Original (5)
            "nao": "não", "voce": "você", "esta": "está", "eh": "é", "ate": "até",

            # BATCH 1 (50) - Common PT-BR corrections
            "tambem": "também", "so": "só", "ja": "já", "la": "lá", "ca": "cá",
            "pra": "para", "pro": "para o", "ta": "está", "to": "estou", "ce": "você",
            "cade": "cadê", "neh": "né", "ne": "né", "vc": "você", "tbm": "também",
            "pq": "por que", "porque": "por que", "oque": "o que", "oq": "o que",
            "voces": "vocês", "apos": "após", "sao": "são", "tem": "têm", "ha": "há",
            "po": "pô", "assim": "assim", "mais": "mais", "mas": "mas", "ai": "aí",
            "dai": "daí", "entao": "então", "tao": "tão", "mao": "mão", "maos": "mãos",
            "irmao": "irmão", "irmaos": "irmãos", "acao": "ação", "acoes": "ações",
            "atencao": "atenção", "opcao": "opção", "opcoes": "opções",
            "informacao": "informação", "informacoes": "informações", "sera": "será",
            "porem": "porém", "alem": "além", "proximo": "próximo", "proxima": "próxima",
            "ultimo": "último",

            # BATCH 2 (50)
            "ultima": "última", "unico": "único", "unica": "única", "basico": "básico",
            "basica": "básica", "publico": "público", "publica": "pública",
            "logico": "lógico", "logica": "lógica", "pratico": "prático",
            "pratica": "prática", "otimo": "ótimo", "otima": "ótima",
            "pessimo": "péssimo", "pessima": "péssima", "facil": "fácil",
            "dificil": "difícil", "util": "útil", "inutil": "inútil", "movel": "móvel",
            "imovel": "imóvel", "avel": "ável", "nivel": "nível", "possivel": "possível",
            "impossivel": "impossível", "incrivel": "incrível", "terrivel": "terrível",
            "cafe": "café", "cha": "chá", "pe": "pé", "pes": "pés", "mes": "mês",
            "meses": "meses", "pais": "país", "paises": "países",
            "portugues": "português", "portuguesa": "portuguesa", "ingles": "inglês",
            "inglesa": "inglesa", "frances": "francês", "francesa": "francesa",
            "japones": "japonês", "japonesa": "japonesa", "aviao": "avião",
            "avioes": "aviões", "cao": "cão", "caes": "cães", "paes": "pães",
            "alemao": "alemão", "alema": "alemã",

            # BATCH 3 (50)
            "orgao": "órgão", "orgaos": "órgãos", "orfao": "órfão", "orfa": "órfã",
            "cidadao": "cidadão", "cidadaos": "cidadãos", "capitao": "capitão",
            "capitaes": "capitães", "segunda": "segunda-feira", "terca": "terça-feira",
            "quarta": "quarta-feira", "quinta": "quinta-feira", "sexta": "sexta-feira",
            "sabado": "sábado", "domingo": "domingo", "janeiro": "janeiro",
            "fevereiro": "fevereiro", "marco": "março", "abril": "abril",
            "maio": "maio", "junho": "junho", "julho": "julho", "agosto": "agosto",
            "setembro": "setembro", "outubro": "outubro", "novembro": "novembro",
            "dezembro": "dezembro", "manha": "manhã", "manhas": "manhãs",
            "amanha": "amanhã", "hoje": "hoje", "ontem": "ontem", "agora": "agora",
            "depois": "depois", "antes": "antes", "durante": "durante",
            "sempre": "sempre", "nunca": "nunca", "talvez": "talvez", "quem": "quem",
            "quando": "quando", "onde": "onde", "como": "como", "quanto": "quanto",
            "qual": "qual", "quais": "quais", "algum": "algum", "alguma": "alguma",
            "nenhum": "nenhum", "nenhuma": "nenhuma",

            # BATCH 4 (50)
            "tudo": "tudo", "nada": "nada", "algo": "algo", "alguem": "alguém",
            "ninguem": "ninguém", "comigo": "comigo", "contigo": "contigo",
            "conosco": "conosco", "convosco": "convosco", "consigo": "consigo",
            "dele": "dele", "dela": "dela", "deles": "deles", "delas": "delas",
            "meu": "meu", "minha": "minha", "teu": "teu", "tua": "tua", "seu": "seu",
            "sua": "sua", "nosso": "nosso", "nossa": "nossa", "vosso": "vosso",
            "vossa": "vossa", "esse": "esse", "essa": "essa", "isso": "isso",
            "este": "este", "este": "este", "isto": "isto", "aquele": "aquele",
            "aquela": "aquela", "aquilo": "aquilo", "mesmo": "mesmo",
            "mesma": "mesma", "proprio": "próprio", "propria": "própria",
            "outro": "outro", "outra": "outra", "varios": "vários", "varias": "várias",
            "poucos": "poucos", "poucas": "poucas", "muitos": "muitos",
            "muitas": "muitas", "todos": "todos", "todas": "todas", "ambos": "ambos",
            "ambas": "ambas", "cada": "cada", "qualquer": "qualquer",

            # BATCH 5 (50)
            "quaisquer": "quaisquer", "certo": "certo", "certa": "certa",
            "certos": "certos", "certas": "certas", "tal": "tal", "tais": "tais",
            "bastante": "bastante", "bastantes": "bastantes", "demais": "demais",
            "menos": "menos", "muito": "muito", "muita": "muita", "pouco": "pouco",
            "pouca": "pouca", "tanto": "tanto", "tanta": "tanta", "quanto": "quanto",
            "quanta": "quanta", "bem": "bem", "mal": "mal", "melhor": "melhor",
            "pior": "pior", "maior": "maior", "menor": "menor", "maximo": "máximo",
            "maxima": "máxima", "minimo": "mínimo", "minima": "mínima",
            "medio": "médio", "media": "média", "superior": "superior",
            "inferior": "inferior", "anterior": "anterior", "posterior": "posterior",
            "exterior": "exterior", "interior": "interior", "distante": "distante",
            "perto": "perto", "longe": "longe", "dentro": "dentro", "fora": "fora",
            "acima": "acima", "abaixo": "abaixo", "cima": "cima", "baixo": "baixo",
            "direita": "direita", "esquerda": "esquerda", "frente": "frente",
            "tras": "trás",

            # BATCH 6 (45)
            "atras": "atrás", "lado": "lado", "meio": "meio", "centro": "centro",
            "comeco": "começo", "fim": "fim", "final": "final", "inicial": "inicial",
            "primeiro": "primeiro", "primeira": "primeira", "segundo": "segundo",
            "terceiro": "terceiro", "terceira": "terceira", "quarto": "quarto",
            "quinto": "quinto", "sexto": "sexto", "setimo": "sétimo",
            "setima": "sétima", "oitavo": "oitavo", "oitava": "oitava", "nono": "nono",
            "nona": "nona", "decimo": "décimo", "decima": "décima",
            "centesimo": "centésimo", "centesima": "centésima", "milesimo": "milésimo",
            "milesima": "milésima", "metade": "metade", "terco": "terço",
            "dobro": "dobro", "triplo": "triplo", "simples": "simples",
            "duplo": "duplo", "multiplo": "múltiplo", "multipla": "múltipla",
            "diversos": "diversos", "diversas": "diversas"
        }

    def _apply_ptbr_corrections(self, text: str) -> str:
        if not text: return ""
        corrected_text = text.lower()
        for wrong, correct in self.ptbr_corrections.items():
            corrected_text = corrected_text.replace(f" {wrong} ", f" {correct} ")
        return corrected_text.capitalize()

    def _calculate_confidence(self, segments: List[Dict[str, Any]]) -> float:
        if not segments: return 0.0
        logprobs = [s.get('avg_logprob', -2.0) for s in segments if s.get('avg_logprob') is not None]
        if not logprobs: return 0.0
        confidences = [np.exp(lp) for lp in logprobs]
        # CORRECTED: Cast numpy float to standard Python float
        return float(np.mean(confidences))

    async def transcribe_with_enhancements(self, audio_path: str, quantization: Optional[str] = None, word_timestamps: bool = False) -> TranscriptionResult:
        async with self._model_lock:  # CRITICAL: Thread-safety for concurrent requests
            start_time = time.time()
            requested_compute_type = quantization or self.compute_type

            if not self.model or self.compute_type != requested_compute_type:
                await self._load_model(compute_type=requested_compute_type)

            self.last_used = time.time()

            if not self.model:
                raise RuntimeError("Transcription model could not be loaded.")

            vad_parameters = dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=2000
            )

            segments_generator, info = self.model.transcribe(
                audio_path, 
                language="pt", 
                beam_size=5,
                best_of=5,
                word_timestamps=word_timestamps,
                vad_filter=True,
                vad_parameters=vad_parameters
            )

            raw_segments = []
            full_text = ""
            for seg in segments_generator:
                segment_dict = {
                    "start": seg.start, 
                    "end": seg.end, 
                    "text": seg.text, 
                    "avg_logprob": seg.avg_logprob
                }
                if word_timestamps and hasattr(seg, 'words') and seg.words is not None:
                    segment_dict['words'] = [
                        {'word': w.word, 'start': w.start, 'end': w.end, 'probability': w.probability}
                        for w in seg.words
                    ]
                
                raw_segments.append(segment_dict)
                corrected_text = self._apply_ptbr_corrections(seg.text)
                full_text += corrected_text + " "

            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(raw_segments)

            return TranscriptionResult(
                text=full_text.strip(),
                segments=raw_segments, # Now contains word timestamps if requested
                language=info.language,
                confidence=confidence,
                processing_time=processing_time,
                word_count=len(full_text.split())
            )

    async def _load_model(self, compute_type: str = "int8"):
        """Load model with retry logic for production reliability."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from faster_whisper import WhisperModel
                self.compute_type = compute_type
                logger.info(f"Loading Whisper model (attempt {attempt+1}/{max_retries}): {self.model_name} with {self.compute_type} precision...")
                load_start = time.time()

                self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

                load_time = time.time() - load_start
                self.model_loads_count += 1
                logger.info(f"Model loaded successfully in {load_time:.2f}s (total loads: {self.model_loads_count})")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.critical(f"Model reload failed after {max_retries} attempts: {e}")
                    self.model = None
                    raise
                logger.warning(f"Model load attempt {attempt+1} failed, retrying in {2**attempt}s...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def unload_model(self):
        """Unload model from memory with thread-safety to free ~1.5GB RAM."""
        async with self._model_lock:  # CRITICAL: Prevent race conditions during unload
            if self.model is not None:
                logger.info("Unloading Whisper model due to inactivity...")
                self.model = None
                self.model_unloads_count += 1
                gc.collect()
                logger.info(f"Model unloaded successfully (total unloads: {self.model_unloads_count})")

    def should_unload(self) -> bool:
        """Check if model should be unloaded based on inactivity period."""
        return (self.model is not None and
                time.time() - self.last_used > self.model_unload_delay)
