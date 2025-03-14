import re
import unicodedata
from typing import Dict, List, Optional, Callable

class LanguageProcessor:
    """Base class for language-specific text processing"""
    
    def __init__(self, 
                 language_code: str,
                 language_name: str,
                 default_speaker_id: int = 0):
        self.language_code = language_code
        self.language_name = language_name
        self.default_speaker_id = default_speaker_id
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for the specific language"""
        # Base implementation: normalize unicode, strip extra whitespace
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Apply language-specific preprocessing"""
        return self.normalize_text(text)
    
    def format_speaker_text(self, text: str, speaker_id: Optional[int] = None) -> str:
        """Format text with speaker ID for the model"""
        if speaker_id is None:
            speaker_id = self.default_speaker_id
        return f"[{speaker_id}]{text}"
    
    def get_commonvoice_column_indices(self) -> Dict[str, str]:
        """Return column names for CommonVoice dataset"""
        return {
            "path": "path",           # Path column name
            "text": "sentence",       # Text/sentence column name
            "gender": "gender",       # Gender column name
            "client_id": "client_id", # Client ID column name
            "accent": "accents",      # Accent column name
            "locale": "locale",       # Locale column name
            "segment": "segment"      # Segment column name
        }
    
    @staticmethod
    def get_processor(language_code: str) -> 'LanguageProcessor':
        """Factory method to get language processor by code"""
        if language_code.lower() == "de":
            return GermanProcessor()
        elif language_code.lower() == "en":
            return EnglishProcessor()
        elif language_code.lower() == "fr":
            return FrenchProcessor()
        elif language_code.lower() == "es":
            return SpanishProcessor()
        else:
            # Return default processor if language not specifically supported
            return LanguageProcessor(language_code, f"Unknown ({language_code})")


class GermanProcessor(LanguageProcessor):
    """German language text processor"""
    
    def __init__(self):
        super().__init__("de", "German")
        
        # German-specific replacements
        self.replacements = [
            (r'(\d)\.(\d)', r'\1,\2'),  # Convert decimal points to commas
            (r'ß', 'ss'),               # Replace ß with ss if tokenizer has issues
            (r'(\d)(\s*)€', r'\1 Euro'),# Replace € with Euro
            (r'Hr\.', 'Herr'),          # Expand common abbreviations
            (r'Fr\.', 'Frau'),
            (r'usw\.', 'und so weiter'),
            (r'z\.B\.', 'zum Beispiel'),
        ]
    
    def normalize_text(self, text: str) -> str:
        """German-specific text normalization"""
        text = super().normalize_text(text)
        
        # Apply German-specific replacements
        for pattern, replacement in self.replacements:
            text = re.sub(pattern, replacement, text)
            
        return text


class EnglishProcessor(LanguageProcessor):
    """English language text processor"""
    
    def __init__(self):
        super().__init__("en", "English")
        
        # English-specific replacements
        self.replacements = [
            (r'Mr\.', 'Mister'),        # Expand common abbreviations
            (r'Dr\.', 'Doctor'),
            (r'(\d)\.(\d)', r'\1.\2'),  # Keep decimal points
            (r'(\d)(\s*)£', r'\1 pounds'),
            (r'(\d)(\s*)\$', r'\1 dollars'),
        ]
    
    def normalize_text(self, text: str) -> str:
        """English-specific text normalization"""
        text = super().normalize_text(text)
        
        # Apply English-specific replacements
        for pattern, replacement in self.replacements:
            text = re.sub(pattern, replacement, text)
            
        return text


class FrenchProcessor(LanguageProcessor):
    """French language text processor"""
    
    def __init__(self):
        super().__init__("fr", "French")
        
        # French-specific replacements
        self.replacements = [
            (r'(\d)\.(\d)', r'\1,\2'),  # Convert decimal points to commas
            (r'M\.', 'Monsieur'),       # Expand common abbreviations
            (r'Mme\.', 'Madame'),
            (r'(\d)(\s*)€', r'\1 euros'),
        ]
    
    def normalize_text(self, text: str) -> str:
        """French-specific text normalization"""
        text = super().normalize_text(text)
        
        # Apply French-specific replacements
        for pattern, replacement in self.replacements:
            text = re.sub(pattern, replacement, text)
            
        return text


class SpanishProcessor(LanguageProcessor):
    """Spanish language text processor"""
    
    def __init__(self):
        super().__init__("es", "Spanish")
        
        # Spanish-specific replacements
        self.replacements = [
            (r'(\d)\.(\d)', r'\1,\2'),  # Convert decimal points to commas
            (r'Sr\.', 'Señor'),         # Expand common abbreviations
            (r'Sra\.', 'Señora'),
            (r'(\d)(\s*)€', r'\1 euros'),
        ]
    
    def normalize_text(self, text: str) -> str:
        """Spanish-specific text normalization"""
        text = super().normalize_text(text)
        
        # Apply Spanish-specific replacements
        for pattern, replacement in self.replacements:
            text = re.sub(pattern, replacement, text)
            
        return text
