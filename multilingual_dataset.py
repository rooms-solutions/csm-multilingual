import os
import sys

# Try importing pandas with error handling
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"ERROR: Dependency issue: {e}")
    print("Please install the required packages with: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Package compatibility issue: {e}")
    print("This might be due to numpy/pandas version mismatch.")
    print("Try creating a new virtual environment and installing packages from requirements.txt")
    sys.exit(1)

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
        # Ensure the Mimi model is properly loaded
        if mimi_model is None:
            from moshi.models import loaders
            from huggingface_hub import hf_hub_download
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Mimi model not provided, downloading and initializing...")
            mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
            self.mimi = loaders.get_mimi(mimi_weight, device=device)
            self.mimi.set_num_codebooks(32)
        else:
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
        file_path = self.voice_frame.iloc[idx]['path']
        
        # Try multiple possible clip paths
        possible_paths = [
            os.path.join(self.root_dir, "clips", os.path.basename(file_path)),  # Just filename in clips folder
            os.path.join(self.root_dir, "clips", file_path),  # Full path in clips folder
            os.path.join(self.root_dir, file_path),  # Direct path
            file_path if os.path.isabs(file_path) else None,  # Absolute path as is
        ]
        
        # Filter None values
        possible_paths = [p for p in possible_paths if p]
        
        # Print debugging for first file to help diagnose path issues
        if idx == 0:
            print(f"Looking for audio file with base path: {file_path}")
            print(f"Root dir: {self.root_dir}")
            print(f"Trying paths: {possible_paths}")
            
        audio_path = None
        for path in possible_paths:
            if os.path.exists(path):
                audio_path = path
                if idx == 0:
                    print(f"Found audio file at: {path}")
                break
                
        if audio_path is None:
            # Check if clips folder exists
            clips_dir = os.path.join(self.root_dir, "clips")
            if os.path.exists(clips_dir):
                print(f"Clips directory exists at {clips_dir}")
                print(f"Files in clips directory: {os.listdir(clips_dir)[:5]}...")
            else:
                print(f"Clips directory not found at {clips_dir}")
            
            raise FileNotFoundError(f"Audio file not found: {file_path} - tried paths: {possible_paths}")
            
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
        
        # Process audio through Mimi encoder
        with torch.no_grad():
            # Get the device of the Mimi model and ensure tensor is on the same device
            device = next(self.mimi.parameters()).device
            waveform_tensor = waveform.unsqueeze(0).unsqueeze(0).to(device)
            audio_tokens = self.mimi.encode(waveform_tensor)[0]
            
            # Verify token range periodically (helps catch issues early)
            if idx % 100 == 0:
                token_min = audio_tokens.min().item()
                token_max = audio_tokens.max().item()
                token_mean = audio_tokens.float().mean().item()

                # Validate by decoding and re-encoding a sample
                if idx == 0:
                    try:
                        # Decode to audio
                        test_audio = self.mimi.decode(audio_tokens.unsqueeze(0))
                        # Re-encode to tokens
                        test_tokens = self.mimi.encode(test_audio)[0]
                        # Convert to float before computing MSE
                        audio_tokens_float = audio_tokens.float()
                        test_tokens_float = test_tokens.float()
                        # Compare similarity
                        print(f"Token encode-decode test: {torch.nn.functional.mse_loss(audio_tokens_float, test_tokens_float):.4f} MSE")
                        print("✓ Mimi codec is working correctly for encoding/decoding")
                    except Exception as e:
                        print(f"⚠️ Mimi codec test failed: {e}")
        
        # All tensors must be on CPU for DataLoader with pin_memory
        return {
            "text": processed_text,
            "raw_text": text,
            "text_tokens": torch.tensor(text_tokens, device="cpu"),  
            "audio_tokens": audio_tokens.to("cpu"),                 # Move to CPU for DataLoader
            "audio_waveform": waveform.to("cpu"),                  
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
        

def multilingual_collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Pads both text_tokens and audio_tokens to the same length within a batch.
    Keeps audio_waveform as a list since lengths vary significantly.
    """
    # Find the maximum length of text_tokens in the batch
    max_text_length = max(item["text_tokens"].size(0) for item in batch)
    
    # Create padded text_tokens
    for item in batch:
        text_tokens = item["text_tokens"]
        padded_text_tokens = torch.zeros(max_text_length, dtype=text_tokens.dtype, device=text_tokens.device)
        padded_text_tokens[:text_tokens.size(0)] = text_tokens
        item["text_tokens"] = padded_text_tokens
    
    # Find the maximum dimensions for audio_tokens
    # audio_tokens shape is [num_codebooks, sequence_length]
    max_audio_length = max(item["audio_tokens"].size(1) for item in batch)
    num_codebooks = batch[0]["audio_tokens"].size(0)  # Should be the same for all items
    
    # Create padded audio_tokens
    for item in batch:
        audio_tokens = item["audio_tokens"]
        current_length = audio_tokens.size(1)
        if current_length < max_audio_length:
            # Create padded tensor and copy original data
            padded_audio_tokens = torch.zeros(
                (num_codebooks, max_audio_length), 
                dtype=audio_tokens.dtype, 
                device=audio_tokens.device
            )
            padded_audio_tokens[:, :current_length] = audio_tokens
            item["audio_tokens"] = padded_audio_tokens
    
    # Create the batch dictionary, handling each type of data appropriately
    elem = batch[0]
    batch_dict = {}
    
    for key in elem:
        if key == "audio_waveform":
            # Don't stack audio_waveforms - keep as list since lengths vary
            batch_dict[key] = [d[key] for d in batch]
        elif torch.is_tensor(elem[key]):
            # Stack tensors (text_tokens, audio_tokens)
            batch_dict[key] = torch.stack([d[key] for d in batch])
        else:
            # For non-tensor data (strings, etc.)
            batch_dict[key] = [d[key] for d in batch]
    
    return batch_dict


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
