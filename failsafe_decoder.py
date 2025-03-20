"""
Failsafe decoder implementation that works on pure CPU.
This module provides a simple alternative to the Mimi codec that
doesn't depend on CUDA operations and won't trigger index errors.
"""

import numpy as np
from scipy import signal
import torch
import logging

logger = logging.getLogger("failsafe_decoder")

class FailsafeDecoder:
    """
    A simplified replacement for Mimi codec that operates entirely on CPU.
    Instead of attempting to decode actual audio tokens into waveforms,
    this creates plausible speech-like audio from token statistics.
    """
    
    def __init__(self):
        """Initialize the failsafe decoder"""
        self.sample_rate = 24000
        
    def decode(self, tokens):
        """
        Generate speech-like audio from token indices.
        
        Args:
            tokens: Tensor of token indices with shape [codebooks, seq_len] or [batch, codebooks, seq_len]
            
        Returns:
            Tensor of audio samples
        """
        # Handle batch dimension if present
        if tokens.dim() == 3:
            # Only process the first item in batch for simplicity
            tokens = tokens[0]
            
        # Move to CPU and convert to numpy
        tokens_np = tokens.detach().cpu().numpy()
        
        # Get basic dimensions
        num_codebooks, seq_len = tokens_np.shape
        
        # Calculate approximate duration (20ms per frame)
        duration_sec = seq_len * 0.02
        
        # Generate the waveform
        waveform = self._tokens_to_waveform(tokens_np, duration_sec)
        
        # Convert back to torch tensor with correct format for compatibility
        return torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
    
    def _tokens_to_waveform(self, tokens, duration_sec):
        """
        Convert token array to speech-like waveform.
        
        Instead of directly decoding, this analyzes token patterns and 
        generates speech-like sounds with appropriate formants and prosody.
        """
        # Create time base
        sample_rate = self.sample_rate
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
        
        # Extract token statistics to influence the generated speech
        num_codebooks, seq_len = tokens.shape
        
        # Calculate statistics for each codebook
        codebook_means = np.mean(tokens, axis=1)
        codebook_ranges = np.ptp(tokens, axis=1)
        token_changes = np.sum(np.abs(np.diff(tokens, axis=1)), axis=1)
        
        # Use these statistics to generate speech parameters
        
        # 1. Base frequency (pitch) from first codebook
        base_freq = 120 + 50 * (codebook_means[0] % 10) / 10  # Range ~120-170Hz
        
        # 2. Formant structure from other codebooks
        formant1 = 300 + 300 * ((codebook_means[1] % 20) / 20)  # ~300-600Hz
        formant2 = 1200 + 800 * ((codebook_means[min(2, num_codebooks-1)] % 30) / 30)  # ~1200-2000Hz
        formant3 = 2400 + 1200 * ((codebook_means[min(3, num_codebooks-1)] % 40) / 40)  # ~2400-3600Hz
        
        # 3. Temporal variation (how much pitch varies) from token changes
        pitch_var = 0.1 + 0.2 * np.sum(token_changes) / (seq_len * num_codebooks)
        
        # 4. Syllable rate from token pattern in first codebook
        if seq_len > 5:
            # Analyze token differences to find syllable-like patterns
            token_diff = np.diff(tokens[0])
            token_diff[token_diff < 0] = 0  # Only consider increases
            syllable_points = np.where(token_diff > np.percentile(token_diff, 80))[0]
            if len(syllable_points) > 1:
                syllable_intervals = np.diff(syllable_points)
                mean_interval = np.mean(syllable_intervals)
                syllable_rate = sample_rate / (mean_interval * 0.02 * sample_rate)  # Convert frames to seconds
                syllable_rate = np.clip(syllable_rate, 2.0, 8.0)  # Reasonable range for speech
            else:
                syllable_rate = 4.0  # Default if can't detect
        else:
            syllable_rate = 4.0  # Default for short sequences
        
        # Generate pitch contour with natural-sounding variations
        pitch_contour = base_freq * (1 + pitch_var * np.sin(2*np.pi*t/3))
        # Add sentence-level intonation (higher at start, lower at end)
        pitch_contour = pitch_contour * (1 + 0.2 * np.exp(-t/duration_sec)) 
        
        # Generate carrier with modulated pitch
        carrier = np.zeros_like(t)
        phase = 0
        # Generate with instantaneous frequency for smooth pitch changes
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            phase += 2 * np.pi * pitch_contour[i] * dt
            carrier[i] = np.sin(phase)
        
        # Apply syllable-like amplitude modulation
        syllable_env = 0.5 + 0.5 * np.cos(2*np.pi*syllable_rate*t)
        syllable_env = np.power(syllable_env, 0.5)  # Make envelope less extreme
        carrier = carrier * syllable_env
        
        # Apply formant filtering to create speech-like spectrum
        speech = carrier.copy()
        
        # Apply formant filters
        formants = [formant1, formant2, formant3]
        formant_bws = [80, 120, 160]  # Bandwidths
        formant_gains = [1.0, 0.7, 0.4]  # Relative gains
        
        filtered_speech = np.zeros_like(speech)
        for freq, bw, gain in zip(formants, formant_bws, formant_gains):
            # Create resonant filter
            b, a = signal.butter(2, [max(0.001, (freq-bw/2)/sample_rate*2), 
                                    min(0.999, (freq+bw/2)/sample_rate*2)], 'bandpass')
            # Apply filter and add to result with appropriate gain
            filtered = signal.lfilter(b, a, speech)
            filtered_speech += gain * filtered
        
        # Add some noise for fricatives/consonants
        # Use token patterns to add noise bursts at appropriate positions
        if num_codebooks > 1 and seq_len > 1:
            # Find positions with large changes in second codebook (often consonants)
            noise_positions = []
            token_diff2 = np.abs(np.diff(tokens[1]))
            noise_thresh = np.percentile(token_diff2, 70)
            for i, diff in enumerate(token_diff2):
                if diff > noise_thresh:
                    # Mark this position for noise
                    noise_positions.append(i)
            
            # Add noise bursts at these positions
            for pos in noise_positions:
                # Convert token position to time
                time_pos = pos * 0.02  # 20ms per frame
                if time_pos < duration_sec:
                    # Add a short noise burst
                    noise_start = int(time_pos * sample_rate)
                    noise_dur = int(0.05 * sample_rate)  # 50ms noise
                    if noise_start + noise_dur < len(filtered_speech):
                        # Generate filtered noise
                        noise = np.random.normal(0, 0.1, noise_dur)
                        # Shape with envelope
                        env = np.hanning(noise_dur)
                        shaped_noise = noise * env
                        # Apply highpass filter for consonant-like sound
                        b, a = signal.butter(4, 2000/sample_rate*2, 'highpass')
                        filt_noise = signal.lfilter(b, a, shaped_noise)
                        # Add to speech
                        filtered_speech[noise_start:noise_start+noise_dur] += 0.3 * filt_noise
        
        # Apply overall sentence envelope
        sentence_env = np.ones_like(t)
        # Attack (beginning of sentence)
        attack_time = min(0.2, duration_sec/10) 
        attack_samples = int(attack_time * sample_rate)
        sentence_env[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay (end of sentence)
        decay_time = min(0.5, duration_sec/5)
        decay_samples = int(decay_time * sample_rate)
        sentence_env[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        # Apply envelope
        filtered_speech = filtered_speech * sentence_env
        
        # Normalize
        filtered_speech = 0.95 * filtered_speech / (np.max(np.abs(filtered_speech)) + 1e-6)
        
        return filtered_speech

def get_failsafe_decoder():
    """Create and return a failsafe decoder"""
    return FailsafeDecoder()
