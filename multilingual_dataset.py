import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Tuple
from language_utils import LanguageProcessor

class MultilingualVoiceDataset(Dataset):
    """Base class for multilingual voice datasets"""
    
    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        mimi_model,
        text_tokenizer,
        language: str,
        max_audio_length: Optional[int] = None,
        language_processor: Optional[LanguageProcessor] = None,
    ):
        """
        Args:
            csv_file (string): Path to the CSV with annotations
            root_dir (string): Directory with all the audio files
            mimi_model: The Mimi codec model for audio tokenization
            text_tokenizer: The text tokenizer
            language (string): Language code (e.g., 'de', 'en')
            max_audio_length (int, optional): Maximum audio length in samples
            language_processor (LanguageProcessor, optional): Custom language processor
        """
        self.voice_frame = pd.read_csv(csv_file, delimiter='\t')
        self.root_dir = root_dir
        self.text_tokenizer = text_tokenizer
        self.mimi = mimi_model
        self.max_audio_length = max_audio_length
        
        # Get appropriate language processor
        self.language_processor = language_processor
        if self.language_processor is None:
            self.language_processor = LanguageProcessor.get_processor(language)
        
        print(f"Initialized dataset for {self.language_processor.language_name} with {len(self.voice_frame)} samples")
        
    def __len__(self):
        return len(self.voice_frame)
        
    def __getitem__(self, idx):
        # Load audio file and resample to 24kHz
        audio_path = os.path.join(self.root_dir, "clips", self.voice_frame.iloc[idx]['path'])
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # Convert to mono
        if sample_rate != 24000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=24000)
        
        # Apply maximum length constraint if specified
        if self.max_audio_length is not None and waveform.size(0) > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        
        # Get text and apply language-specific preprocessing
        text = self.voice_frame.iloc[idx]['sentence']
        processed_text = self.language_processor.preprocess_text(text)
        
        # Format with speaker ID - try to use client_id if available, fallback to gender, otherwise use default
        if 'client_id' in self.voice_frame.columns and not pd.isna(self.voice_frame.iloc[idx]['client_id']):
            # Hash the client_id to an integer to use as speaker ID
            client_id = self.voice_frame.iloc[idx]['client_id']
            speaker_id = hash(client_id) % 100  # Limit to 100 different speaker IDs
        elif 'gender' in self.voice_frame.columns and not pd.isna(self.voice_frame.iloc[idx]['gender']):
            # Map gender to speaker IDs: female=1, male=2, other=3
            gender = self.voice_frame.iloc[idx]['gender'].lower()
            if gender == 'female':
                speaker_id = 1
            elif gender == 'male':
                speaker_id = 2
            else:
                speaker_id = 3
        else:
            speaker_id = self.language_processor.default_speaker_id
            
        formatted_text = self.language_processor.format_speaker_text(processed_text, speaker_id)
        
        # Tokenize text
        text_tokens = self.text_tokenizer.encode(formatted_text)
        
        # Tokenize audio using Mimi
        with torch.no_grad():
            audio_tokens = self.mimi.encode(waveform.unsqueeze(0).unsqueeze(0))[0]
        
        return {
            "text": processed_text,
            "raw_text": text,
            "text_tokens": torch.tensor(text_tokens),
            "audio_tokens": audio_tokens,
            "audio_waveform": waveform,
            "speaker_id": speaker_id,
            "language": self.language_processor.language_code
        }


class GermanVoiceDataset(MultilingualVoiceDataset):
    """German voice dataset implementation"""
    
    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        mimi_model,
        text_tokenizer,
        max_audio_length: Optional[int] = None,
    ):
        # Initialize with German language processor
        super().__init__(
            csv_file=csv_file,
            root_dir=root_dir,
            mimi_model=mimi_model,
            text_tokenizer=text_tokenizer,
            language="de",
            max_audio_length=max_audio_length,
        )
        

def create_dataset_for_language(
    language: str,
    csv_file: str,
    root_dir: str,
    mimi_model,
    text_tokenizer,
    max_audio_length: Optional[int] = None,
) -> MultilingualVoiceDataset:
    """Factory function to create appropriate dataset for a language"""
    
    # Special case handlers
    if language.lower() == "de":
        return GermanVoiceDataset(
            csv_file=csv_file,
            root_dir=root_dir,
            mimi_model=mimi_model,
            text_tokenizer=text_tokenizer,
            max_audio_length=max_audio_length,
        )
    
    # Default case: use the base multilingual dataset with appropriate processor
    return MultilingualVoiceDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        mimi_model=mimi_model,
        text_tokenizer=text_tokenizer,
        language=language,
        max_audio_length=max_audio_length,
    )
