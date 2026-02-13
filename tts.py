import sounddevice as sd
import torch
from pocket_tts import TTSModel
from threading import Lock

class TTSEngine:
    def __init__(self, voice="azelma"):
        self.tts = TTSModel.load_model()
        self.voice_state = self.tts.get_state_for_audio_prompt(voice)
        self.sample_rate = self.tts.sample_rate
        self.lock = Lock()

    def speak(self, text: str):
        if not text.strip():
            return

        with self.lock:
            audio = self.tts.generate_audio(self.voice_state, text)
            samples = audio.detach().cpu().numpy()
            sd.play(samples, samplerate=self.sample_rate)
            sd.wait()
