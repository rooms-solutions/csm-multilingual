#!/usr/bin/env python3
"""
Generate audio from text using multilingual CSM 1B models.
"""

import argparse
import os
import torch
import torchaudio
from pathlib import Path

from generator import load_csm_1b, Generator, Segment
from language_utils import LanguageProcessor


def load_multilingual_model(language_code: str, checkpoint_path: str = None, device: str = "cuda") -> Generator:
    """
    Load a language-specific CSM 1B model.
    
    Args:
        language_code: The language code (e.g., "de" for German)
        checkpoint_path: Path to the model checkpoint. If None, looks for language-specific checkpoint
        device: Device to load the model on ("cuda" or "cpu")
        
    Returns:
        Generator object for the specified language
    """
    # If no checkpoint path is provided, look for language-specific checkpoint
    if checkpoint_path is None:
        # Try common paths for language-specific models
        possible_paths = [
            f"./checkpoints/{language_code}/best_model.pt",
            f"./checkpoints/{language_code}/final_model.pt",
            f"./checkpoints/{language_code}/checkpoint-epoch-10.pt",
            f"./checkpoints/{language_code.lower()}/best_model.pt",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                print(f"Found language model checkpoint: {path}")
                break
                
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found for language '{language_code}'. Please specify a checkpoint path.")
    
    # Load model with our custom loader that handles the SimpleDecoderAttention architecture
    print(f"Loading model from {checkpoint_path}...")
    # Replace the load_csm_1b call with our custom loader
    from generator import load_multilingual_model as load_model_with_custom_decoder
    generator = load_model_with_custom_decoder(checkpoint_path, device=device)
    print("Model loaded successfully.")
    
    return generator


def generate_audio(
    generator: Generator,
    text: str,
    language_code: str,
    speaker_id: int = 0,
    output_path: str = None,
    temperature: float = 0.9,
    topk: int = 50,
) -> torch.Tensor:
    """
    Generate audio from text using the specified language model.
    
    Args:
        generator: The Generator model
        text: Input text to synthesize
        language_code: Language code for text processing
        speaker_id: Speaker ID to use for generation
        output_path: If provided, save audio to this path
        temperature: Generation temperature (higher = more diverse)
        topk: Number of top candidates to sample from
        
    Returns:
        Generated audio as torch.Tensor
    """
    # Get language processor for text normalization
    language_processor = LanguageProcessor.get_processor(language_code)
    
    # Process text with language-specific normalization
    processed_text = language_processor.preprocess_text(text)
    print(f"Processing text [{language_code}]: {processed_text}")
    
    # Ensure device consistency
    device = generator.device
    print(f"Using device for generation: {device}")
    
    # Generate audio with maximum length constraint
    audio = generator.generate(
        text=processed_text,
        speaker=speaker_id,
        context=[],  # No context for now
        max_audio_length_ms=10000,  # Limit to 10 seconds max
        temperature=temperature,
        topk=topk,
    )
    
    # Save audio if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        # Move to CPU for saving as torchaudio.save requires CPU tensors
        audio_cpu = audio.cpu()
        torchaudio.save(output_path, audio_cpu.unsqueeze(0), generator.sample_rate)
        print(f"Audio saved to {output_path}")
    
    return audio


def main():
    parser = argparse.ArgumentParser(description="Generate audio using multilingual CSM 1B models")
    
    # Required arguments
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--language", type=str, required=True,
                        help="Language code (e.g., 'de' for German)")
    
    # Optional arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (if not provided, will look for language-specific checkpoint)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output audio file path (defaults to 'output_{language}.wav')")
    parser.add_argument("--speaker", type=int, default=0,
                        help="Speaker ID to use")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature (higher = more diverse)")
    parser.add_argument("--topk", type=int, default=50,
                        help="Number of top candidates to sample from")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        args.output = f"output_{args.language.lower()}.wav"
    
    # Load model
    try:
        generator = load_multilingual_model(
            language_code=args.language,
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        
        # Generate audio
        audio = generate_audio(
            generator=generator,
            text=args.text,
            language_code=args.language,
            speaker_id=args.speaker,
            output_path=args.output,
            temperature=args.temperature,
            topk=args.topk
        )
        
        print(f"Successfully generated {len(audio)/generator.sample_rate:.2f} seconds of audio")
        
    except Exception as e:
        print(f"Error generating audio: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
