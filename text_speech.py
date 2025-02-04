from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
import os

# Initialize processor, model, and vocoder
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Folder to store generated audio files
AUDIO_FOLDER = 'audio'

# Ensure the audio folder exists
os.makedirs(AUDIO_FOLDER, exist_ok=True)

def generate_audio_from_text(text):
    """
    Generates audio from the provided text using the SpeechT5 model.
    
    Args:
    text (str): The text input to convert to speech.

    Returns:
    str: The filename of the generated audio.
    """
    # Prepare input text
    inputs = processor(text=text, return_tensors="pt")

    # Load xvector containing speaker's voice characteristics from a dataset
    # (For simplicity, using a random speaker embedding)
    speaker_embeddings = torch.zeros((1, 512))  # Example speaker embedding
    
    # Generate speech
    with torch.no_grad():  # Ensure no gradients are computed during inference
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # Apply sample rate of 16000 Hz (adjust based on model specifications)
    sample_rate = 16000

    # Generate a filename for the audio file
    audio_filename = f"{text[:10]}_speech.wav"  
    audio_path = os.path.join(AUDIO_FOLDER, audio_filename)

    # Ensure speech is within the expected sample rate range (16kHz typically)
    speech = speech.squeeze().cpu().numpy()

    # Save the generated speech to a .wav file with the correct sample rate
    sf.write(audio_path, speech, samplerate=sample_rate)
    
    return audio_filename
