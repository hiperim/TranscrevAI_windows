# Contextual Corrector for TranscrevAI
# Lightweight corrections for common transcription errors

"""
SimpleContextualCorrector

Lightweight post-processing system for common transcription errors:
- Simple lookup-based corrections (minimal overhead)
- Language-specific corrections for PT, EN, ES
- Confidence-based corrections (only low confidence words)
- Context-aware homophone fixes
- Maintains real-time performance requirements
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import time

logger = logging.getLogger(__name__)

class SimpleContextualCorrector:
    """
    Lightweight contextual corrections for common transcription errors
    
    Features:
    - Fast lookup-based corrections
    - Language-specific error patterns
    - Confidence-based filtering
    - Context-aware replacements
    - Real-time performance optimized
    """
    
    def __init__(self):
        """Initialize with language-specific correction patterns"""
        
        # Portuguese corrections (Brazilian focus)
        self.portuguese_corrections = {
            # Common Whisper errors in Portuguese
            "voce": "você",
            "esta": "está", 
            "nao": "não",
            "cao": "cão",
            "coração": "coração",
            "pao": "pão",
            "irmao": "irmão",
            "mamae": "mamãe",
            "papai": "papai",
            "tambem": "também",
            "porem": "porém",
            "atraves": "através",
            "pos": "pós",
            "pre": "pré",
            "pro": "pró",
            
            # Common transcription mistakes
            "estúgio": "estojo",
            "camita": "caneta",
            "preciza": "precisa",
            "ezemplo": "exemplo",
            "compania": "companhia",
            "telfone": "telefone",
            "enderesso": "endereço",
            "emfim": "enfim",
            "intão": "então",
            
            # Technology/business terms
            "e-mail": "email",
            "site": "site",
            "online": "on-line",
            "download": "download",
            "software": "software",
            "hardware": "hardware"
        }
        
        # English corrections  
        self.english_corrections = {
            # Common homophones and contractions
            "cant": "can't",
            "wont": "won't", 
            "dont": "don't",
            "youre": "you're",
            "theyre": "they're",
            "were": "we're",  # Context dependent
            "its": "it's",     # Context dependent
            "your": "you're",  # Context dependent
            
            # Common transcription errors
            "recieve": "receive",
            "definately": "definitely",
            "seperate": "separate",
            "occured": "occurred",
            "begining": "beginning",
            "sucessful": "successful",
            "recomend": "recommend",
            
            # Business/tech terms
            "website": "website",
            "email": "email",
            "online": "online",
            "offline": "offline"
        }
        
        # Spanish corrections
        self.spanish_corrections = {
            # Accent corrections
            "medico": "médico",
            "telefono": "teléfono", 
            "musica": "música",
            "rapido": "rápido",
            "facil": "fácil",
            "dificil": "difícil",
            "ultimo": "último",
            "numero": "número",
            "camara": "cámara",
            "compania": "compañía",
            
            # Common errors
            "tambien": "también",
            "acion": "acción",
            "sion": "sión",
            "porque": "porque",  # vs "por qué" - context dependent
            "atraves": "a través",
            
            # Tech terms
            "correo": "correo",
            "internet": "internet",
            "computadora": "computadora"
        }
        
        # Context-based correction patterns
        self.context_patterns = {
            "en": [
                # "your" vs "you're" based on context
                (r'\byour\s+(going|coming|working|thinking)', r"you're \1"),
                (r'\bthere\s+(going|coming|working)', r"they're \1"),
                (r'\bto\s+(many|much|often)', r"too \1"),
                (r'\bits\s+(time|important|necessary)', r"it's \1"),
                
                # Capitalization fixes
                (r'\bi\s+', r"I "),
                (r'\bi\'', r"I'"),
            ],
            "pt": [
                # Common contractions and combinations
                (r'\bde\s+o\b', r"do"),
                (r'\bde\s+a\b', r"da"), 
                (r'\bem\s+o\b', r"no"),
                (r'\bem\s+a\b', r"na"),
                (r'\bpor\s+o\b', r"pelo"),
                (r'\bpor\s+a\b', r"pela"),
                
                # Este/esta context
                (r'\beste\s+([aeiou])', r"esta \1"),  # Simple heuristic
            ],
            "es": [
                # Contractions
                (r'\bdel\s+el\b', r"del"),
                (r'\bal\s+el\b', r"al"),
                
                # Porque variations
                (r'\bpor\s+que\s+([¿?])', r"por qué \1"),  # Questions
                (r'\bporque\s+([¿?])', r"por qué \1"),
            ]
        }
        
        # Confidence thresholds for corrections
        self.low_confidence_threshold = 0.7
        self.very_low_confidence_threshold = 0.5
        
        logger.info("SimpleContextualCorrector initialized with language-specific patterns")
    
    def apply_simple_corrections(self, text: str, language: str, confidence: float = 1.0) -> str:
        """
        Apply lightweight corrections without heavy processing
        
        Args:
            text (str): Text to correct
            language (str): Language code (pt, en, es)
            confidence (float): Confidence score of the text
            
        Returns:
            str: Corrected text
        """
        if not text or not text.strip():
            return text
        
        start_time = time.time()
        
        try:
            corrected_text = text
            corrections_applied = 0
            
            # Only apply corrections if confidence is low enough
            if confidence <= self.low_confidence_threshold:
                # Apply word-level corrections
                corrected_text, word_corrections = self._apply_word_corrections(corrected_text, language)
                corrections_applied += word_corrections
                
                # Apply context-based patterns for very low confidence
                if confidence <= self.very_low_confidence_threshold:
                    corrected_text, pattern_corrections = self._apply_context_patterns(corrected_text, language)
                    corrections_applied += pattern_corrections
            
            # Apply basic cleanup (always)
            corrected_text = self._apply_basic_cleanup(corrected_text)
            
            processing_time = time.time() - start_time
            
            if corrections_applied > 0:
                logger.debug(f"Applied {corrections_applied} corrections to {language} text (confidence: {confidence:.2f}, time: {processing_time*1000:.1f}ms)")
            
            return corrected_text
            
        except Exception as e:
            logger.warning(f"Correction failed for {language}: {e}")
            return text  # Return original text on error
    
    def _apply_word_corrections(self, text: str, language: str) -> Tuple[str, int]:
        """Apply simple word-level corrections"""
        corrections_applied = 0
        
        # Get language-specific corrections
        correction_dict = self._get_corrections_for_language(language)
        if not correction_dict:
            return text, 0
        
        # Apply corrections (case-insensitive but preserve original case)
        corrected_text = text
        words = text.split()
        
        for i, word in enumerate(words):
            # Clean word for lookup (remove punctuation)
            clean_word = re.sub(r'[^\w]', '', word).lower()
            
            if clean_word in correction_dict:
                # Get the correct form
                corrected_form = correction_dict[clean_word]
                
                # Preserve case pattern from original word
                if word[0].isupper() and len(word) > 1:
                    corrected_form = corrected_form.capitalize()
                elif word.isupper():
                    corrected_form = corrected_form.upper()
                
                # Preserve punctuation
                if word != clean_word:
                    # Extract punctuation pattern
                    punctuation = re.findall(r'[^\w]', word)
                    if punctuation:
                        # Simple punctuation preservation (end of word)
                        if word.endswith(punctuation[-1]):
                            corrected_form += punctuation[-1]
                
                words[i] = corrected_form
                corrections_applied += 1
        
        return ' '.join(words), corrections_applied
    
    def _apply_context_patterns(self, text: str, language: str) -> Tuple[str, int]:
        """Apply context-aware pattern corrections"""
        corrections_applied = 0
        
        patterns = self.context_patterns.get(language, [])
        corrected_text = text
        
        for pattern, replacement in patterns:
            try:
                new_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
                if new_text != corrected_text:
                    corrections_applied += 1
                    corrected_text = new_text
            except re.error as e:
                logger.warning(f"Pattern correction failed for {pattern}: {e}")
        
        return corrected_text, corrections_applied
    
    def _apply_basic_cleanup(self, text: str) -> str:
        """Apply basic text cleanup"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([¿¡])\s+', r'\1', text)  # Spanish punctuation
        
        # Fix spacing after punctuation
        text = re.sub(r'([.!?])\s*([A-ZÁÉÍÓÚÑ])', r'\1 \2', text)
        
        # Fix quote spacing
        text = re.sub(r'\s*"\s*', '"', text)
        text = re.sub(r'\s*\'\s*', "'", text)
        
        return text
    
    def _get_corrections_for_language(self, language: str) -> Dict[str, str]:
        """Get correction dictionary for specific language"""
        language_map = {
            "pt": self.portuguese_corrections,
            "en": self.english_corrections, 
            "es": self.spanish_corrections
        }
        return language_map.get(language, {})
    
    def correct_low_confidence_words(self, transcription_data: List[Dict], language: str) -> List[Dict]:
        """
        Apply corrections only to low-confidence words in transcription data
        
        Args:
            transcription_data (List[Dict]): List of transcription segments
            language (str): Language code
            
        Returns:
            List[Dict]: Corrected transcription data
        """
        if not transcription_data:
            return transcription_data
        
        corrected_data = []
        total_corrections = 0
        
        for segment in transcription_data:
            corrected_segment = segment.copy()
            
            # Get confidence and text
            confidence = segment.get('confidence', 1.0)
            text = segment.get('text', '')
            
            if text and confidence < self.low_confidence_threshold:
                # Apply corrections to this segment
                corrected_text = self.apply_simple_corrections(text, language, confidence)
                
                if corrected_text != text:
                    corrected_segment['text'] = corrected_text
                    corrected_segment['corrected'] = True
                    corrected_segment['original_text'] = text
                    total_corrections += 1
            
            corrected_data.append(corrected_segment)
        
        if total_corrections > 0:
            logger.info(f"Applied corrections to {total_corrections}/{len(transcription_data)} segments in {language}")
        
        return corrected_data
    
    def add_custom_corrections(self, language: str, corrections: Dict[str, str]):
        """
        Add custom corrections for a language
        
        Args:
            language (str): Language code
            corrections (Dict[str, str]): Mapping of incorrect -> correct
        """
        correction_dict = self._get_corrections_for_language(language)
        if correction_dict is not None:
            correction_dict.update(corrections)
            logger.info(f"Added {len(corrections)} custom corrections for {language}")
    
    def get_correction_stats(self) -> Dict[str, int]:
        """Get statistics about available corrections"""
        return {
            "portuguese_corrections": len(self.portuguese_corrections),
            "english_corrections": len(self.english_corrections),
            "spanish_corrections": len(self.spanish_corrections),
            "total_patterns": sum(len(patterns) for patterns in self.context_patterns.values())
        }

class ConfidenceBasedCorrector:
    """Apply corrections only to low-confidence words for better performance"""
    
    def __init__(self, corrector: Optional[SimpleContextualCorrector] = None):
        """
        Initialize confidence-based corrector
        
        Args:
            corrector: SimpleContextualCorrector instance (creates new if None)
        """
        self.corrector = corrector or SimpleContextualCorrector()
        self.correction_threshold = 0.7  # Only correct words below this confidence
        
        logger.info("ConfidenceBasedCorrector initialized")
    
    def correct_low_confidence_words(self, transcription_data: List[Dict], language: str) -> List[Dict]:
        """
        Target corrections to words that need them most
        
        Args:
            transcription_data: List of transcription segments with confidence scores
            language: Language code
            
        Returns:
            List[Dict]: Corrected transcription data preserving high-confidence parts
        """
        return self.corrector.correct_low_confidence_words(transcription_data, language)
    
    def process_segment_with_confidence(self, segment_text: str, confidence: float, language: str) -> str:
        """
        Process single segment based on confidence
        
        Args:
            segment_text: Text to process
            confidence: Confidence score (0.0 - 1.0)
            language: Language code
            
        Returns:
            str: Processed text
        """
        if confidence >= self.correction_threshold:
            # High confidence - minimal processing
            return self.corrector._apply_basic_cleanup(segment_text)
        else:
            # Low confidence - apply corrections
            return self.corrector.apply_simple_corrections(segment_text, language, confidence)

# Global instances for easy access
_global_corrector: Optional[SimpleContextualCorrector] = None
_global_confidence_corrector: Optional[ConfidenceBasedCorrector] = None

def get_contextual_corrector() -> SimpleContextualCorrector:
    """Get or create the global contextual corrector instance"""
    global _global_corrector
    if _global_corrector is None:
        _global_corrector = SimpleContextualCorrector()
    return _global_corrector

def get_confidence_corrector() -> ConfidenceBasedCorrector:
    """Get or create the global confidence-based corrector instance"""
    global _global_confidence_corrector
    if _global_confidence_corrector is None:
        _global_confidence_corrector = ConfidenceBasedCorrector(get_contextual_corrector())
    return _global_confidence_corrector