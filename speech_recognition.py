import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import time

SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds

audio_queue = queue.Queue()

model = whisper.load_model("small.en")

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def record_audio():
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    ):
        while True:
            time.sleep(0.1)

def transcribe():
    buffer = np.zeros((0, 1), dtype=np.float32)

    while True:
        while not audio_queue.empty():
            buffer = np.concatenate((buffer, audio_queue.get()))

        if len(buffer) >= SAMPLE_RATE * CHUNK_DURATION:
            audio = buffer[: SAMPLE_RATE * CHUNK_DURATION]
            buffer = buffer[SAMPLE_RATE * CHUNK_DURATION :]

            audio = audio.flatten()

            result = model.transcribe(
                audio,
                fp16=False,
                language="en"
            )

            text = result["text"].strip()
            if text:
                print("🗣️", text)

        time.sleep(0.05)

threading.Thread(target=record_audio, daemon=True).start()
threading.Thread(target=transcribe, daemon=True).start()

print("🎙 Listening... Press Ctrl+C to stop.")
while True:
    time.sleep(1)
