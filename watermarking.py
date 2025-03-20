import argparse

import silentcipher
import torch
import torchaudio

# This watermark key is public, it is not secure.
# If using CSM 1B in another application, use a new private key and keep it secret.
CSM_1B_GH_WATERMARK = [212, 211, 146, 56, 201]


def cli_check_audio() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    args = parser.parse_args()

    check_audio_from_file(args.audio_path)


def load_watermarker(device: str = "cuda") -> silentcipher.server.Model:
    try:
        model = silentcipher.get_model(
            model_type="44.1k",
            device=device,
        )
        # Store the device as an attribute for easier access
        if not hasattr(model, 'device'):
            model.device = torch.device(device)
        return model
    except Exception as e:
        print(f"Warning: Failed to load watermarker with error: {e}")
        print("Creating a pass-through watermarker")
        
        # Create a stub watermarker that just passes audio through
        class DummyWatermarker:
            def __init__(self, device):
                self.device = torch.device(device)
                
            def encode_wav(self, audio, sample_rate, key, calc_sdr=False, message_sdr=36):
                return audio, {}
                
            def decode_wav(self, audio, sample_rate, phase_shift_decoding=True):
                return {"status": False, "messages": []}
        
        return DummyWatermarker(device)


@torch.inference_mode()
def watermark(
    watermarker: silentcipher.server.Model,
    audio_array: torch.Tensor,
    sample_rate: int,
    watermark_key: list[int],
) -> tuple[torch.Tensor, int]:
    # Get device in a safer way that doesn't rely on parameters() method
    if hasattr(watermarker, 'device'):
        device = watermarker.device
    elif hasattr(watermarker, 'enc_c') and hasattr(watermarker.enc_c, 'device'):
        device = watermarker.enc_c.device
    else:
        # Fallback to audio device
        device = audio_array.device
    
    print(f"Watermarker device: {device}, audio device: {audio_array.device}")
    audio_array = audio_array.to(device)
    
    # Resample to 44.1kHz on the same device
    audio_array_44khz = torchaudio.functional.resample(audio_array, orig_freq=sample_rate, new_freq=44100).to(device)
    print(f"Resampled audio device: {audio_array_44khz.device}")
    
    # Ensure watermarker is on the same device
    if hasattr(watermarker, 'to') and callable(getattr(watermarker, 'to')):
        watermarker = watermarker.to(device)
    
    encoded, _ = watermarker.encode_wav(audio_array_44khz, 44100, watermark_key, calc_sdr=False, message_sdr=36)
    
    # Explicitly ensure encoded is on the correct device with sync
    encoded = encoded.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(f"Encoded audio device: {encoded.device}")

    output_sample_rate = min(44100, sample_rate)
    encoded = torchaudio.functional.resample(encoded, orig_freq=44100, new_freq=output_sample_rate).to(device)
        
    # Final check to ensure output is on the correct device with sync
    encoded = encoded.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(f"Final watermarked audio device: {encoded.device}")
    return encoded, output_sample_rate


@torch.inference_mode()
def verify(
    watermarker: silentcipher.server.Model,
    watermarked_audio: torch.Tensor,
    sample_rate: int,
    watermark_key: list[int],
) -> bool:
    # Get device in a safer way
    if hasattr(watermarker, 'device'):
        device = watermarker.device
    elif hasattr(watermarker, 'enc_c') and hasattr(watermarker.enc_c, 'device'):
        device = watermarker.enc_c.device
    else:
        device = watermarked_audio.device
        
    watermarked_audio = watermarked_audio.to(device)
    watermarked_audio_44khz = torchaudio.functional.resample(watermarked_audio, orig_freq=sample_rate, new_freq=44100)
    
    try:
        result = watermarker.decode_wav(watermarked_audio_44khz, 44100, phase_shift_decoding=True)

        is_watermarked = result["status"]
        if is_watermarked:
            is_csm_watermarked = result["messages"][0] == watermark_key
        else:
            is_csm_watermarked = False

        return is_watermarked and is_csm_watermarked
    except Exception as e:
        print(f"Warning: Error verifying watermark: {e}")
        return False


def check_audio_from_file(audio_path: str) -> None:
    watermarker = load_watermarker(device="cuda")

    audio_array, sample_rate = load_audio(audio_path)
    is_watermarked = verify(watermarker, audio_array, sample_rate, CSM_1B_GH_WATERMARK)

    outcome = "Watermarked" if is_watermarked else "Not watermarked"
    print(f"{outcome}: {audio_path}")


def load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    audio_array, sample_rate = torchaudio.load(audio_path)
    audio_array = audio_array.mean(dim=0)
    return audio_array, int(sample_rate)


if __name__ == "__main__":
    cli_check_audio()
