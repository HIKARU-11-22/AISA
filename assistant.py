import os
import subprocess
import json
import numpy as np
import time
import datetime
import webbrowser
from threading import Thread
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from music import music_player
from vision import identify_person
last_interaction_time = time.time()
from threading import Thread, Lock
from tts import TTSModel
import sounddevice as sd

# =========================
# TTS SETUP
# =========================
class TTSEngine:
    def __init__(self, voice="azelma"):
        self.tts = TTSModel.load_model()
        self.voice_state = self.tts.get_state_for_audio_prompt(voice)
        self.sample_rate = self.tts.sample_rate
        self.lock = Lock()

    def speak(self, text: str):
        if not text.strip(): return
        with self.lock:
            # Clean text of ACTION codes so she doesn't read them aloud
            clean_text = text.split("ACTION_")[0].strip()
            if not clean_text: return
            
            audio = self.tts.generate_audio(self.voice_state, clean_text)
            samples = audio.detach().cpu().numpy()
            sd.play(samples, samplerate=self.sample_rate)
            sd.wait()
speech = TTSEngine()

def say(text):
    """Aggressive phonetic correction for A.I.S.A."""
    clean_text = text.split("ACTION_")[0].strip()
    # If 'aisa' sounds like 'eye-sa', try 'aysa' or 'aysa'.
    phonetic_map = {
        "A.I.S.A.": "eye-sa",
        "A.I.S.A": "eye-sa",
        "AISA": "eye-sa",
        "Aisa": "eye-sa"
    }
    
    for target, replacement in phonetic_map.items():
        clean_text = clean_text.replace(target, replacement)

    if not clean_text: return
    
    # Send the modified lowercase text to the engine
    Thread(target=speech.speak, args=(clean_text,), daemon=True).start()
# =========================
# Paths & Models
# =========================
history_path = "history.jsonl"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Set verbose=False to keep the terminal clean from Llama logs
llm = Llama(
    model_path="/home/hikaru/AI_project/mistral-7b-v0.1.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=40,
    f16_kv=True,
    verbose=False 
)

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are A.I.S.A., a local AI assistant.
IDENTITY: Created by Roshan (Hikaru).
BEHAVIOR: Be clear, calm, and helpful. 

TOOL USE:
If the user wants you to perform a system action, output the specific ACTION code.
- To shutdown the PC: ACTION_SHUTDOWN
- To open a website: ACTION_OPEN_WEB [URL]
- If you use an action, do not say anything else.
"""

KNOWN_FACTS = "KNOWN FACTS: The user is Roshan (Hikaru), the creator."

# =========================
# Robust Memory Utilities
# =========================
def load_history():
    if not os.path.exists(history_path): return [], []
    messages, embeddings = [], []
    with open(history_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                messages.append(data["message"])
                embeddings.append(np.array(data["embedding"]))
            except json.JSONDecodeError:
                continue # Skip corrupted lines
    return messages, embeddings

def save_to_history(message, embedding):
    with open(history_path, "a") as f:
        json.dump({"message": message, "embedding": embedding.tolist()}, f)
        f.write("\n")

def retrieve_relevant_history(user_input, messages, embeddings, top_k=3, threshold=0.6):
    if not messages: return []
    input_embedding = embedding_model.encode([user_input])[0]
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [messages[i] for i in top_indices if similarities[i] >= threshold]

# =========================
# System Actions (Silent & Clean)
# =========================
def execute_system_action(response):
    import sys
    from urllib.parse import urlparse
    if "ACTION_" in response:
        print("\n[A.I.S.A.]: Verifying identity for system command...")
        user = identify_person()
        if user != "Roshan":
            print(f"SECURITY ALERT: Unauthorized user ({user}) attempted a system action.")
            print("A.I.S.A.: I'm sorry, but I am only authorized to perform system tasks for Roshan.")
            return False 

    if "ACTION_SHUTDOWN" in response:
        print("\nA.I.S.A.: Systems closing. Goodbye, Roshan.")
        # os.system("shutdown /s /t 1") 
        os.system("sudo shutdown now")
        return True 

    if "ACTION_OPEN_WEB" in response:
        parts = response.split("ACTION_OPEN_WEB")
        raw_url = parts[1].strip() if len(parts) > 1 else ""
        if raw_url:
            full_url = raw_url if raw_url.startswith("http") else "https://" + raw_url
            clean_name = urlparse(full_url).netloc.replace("www.", "")

            # Silence browser stderr logs
            save_stderr = os.dup(sys.stderr.fileno())
            try:
                null_out = os.open(os.devnull, os.O_RDWR)
                os.dup2(null_out, sys.stderr.fileno())
                webbrowser.open(full_url)
                time.sleep(1) # Give browser time to launch
                os.close(null_out)
            finally:
                os.dup2(save_stderr, sys.stderr.fileno())
                os.close(save_stderr)
            try:
                import termios
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except: pass

            print(f"\nA.I.S.A.: I have opened {clean_name} for you.")
    return False

# =========================
# Autonomous Features
# =========================
def generate_memory_aware_greeting(user_name):
    identity_context = f"You are currently looking at {user_name} via the camera."
    
    prompt = f"""### Instruction:
You are A.I.S.A. 
FACT: {identity_context}
STATUS: System just booted up.

TASK: Greet the user by their name ({user_name}) and ask how you can help. 
Do not ask for their name, because you already know it from the vision sensor.

### Response:
A.I.S.A.:"""

    output = llm(prompt, max_tokens=50, stop=["###", "User:", "Name?"])
    return output["choices"][0]["text"].strip()


def proactive_monitor():
    global last_interaction_time
    while True:
        try:
            time.sleep(60)
            if time.time() - last_interaction_time > 3600: # 1 hour idle
                print("\n\n[A.I.S.A.]: Roshan, you've been working hard. Should we take a break?")
                print("User: ", end="", flush=True)
                last_interaction_time = time.time()
        except Exception as e:
            print(f"Monitor Error: {e}")

# =========================
# Main Loop
# =========================
if __name__ == "__main__":
    print("\n[A.I.S.A. Booting...]")
    print("[A.I.S.A.]: Checking user identity...")
    current_user = identify_person()
    print(f"[A.I.S.A.]: Subject identified as {current_user}")
    Thread(target=proactive_monitor, daemon=True).start()
    greeting = generate_memory_aware_greeting(current_user)
    say(greeting)
    
    print(f"\n{'='*40}\nONLINE | {datetime.datetime.now().strftime('%H:%M')}\nA.I.S.A.: {greeting}\n{'='*40}")

    while True:
        user_input = input("User: ").strip()
        if not user_input: continue
        msgs, embs = load_history()
        relevant = retrieve_relevant_history(user_input, msgs, embs)
        prompt = f"""### Instruction:
    {SYSTEM_PROMPT}
    {KNOWN_FACTS}
    CURRENT SESSION: The person in front of the camera is identified as: {current_user}.
    If the identity is 'Stranger', do not address them as Roshan.

    RELEVANT MEMORY:
    """
        for m in relevant: prompt += f"- {m}\n"
        prompt += f"\nUSER: {user_input}\nA.I.S.A.:"

        output = llm(prompt, max_tokens=300, temperature=0.4, stop=["USER:", "A.I.S.A.:"])
            
        last_interaction_time = time.time()
        user_embedding = embedding_model.encode([user_input])[0]

        # ---- Music Commands ----
        if "play playlist" in user_input:
            p_name = user_input.split("play playlist")[-1].strip()
            if music_player.load_playlist(p_name):
                Thread(target=music_player.play).start()
                print(f"A.I.S.A.: Playing playlist: {p_name}")
            continue
        elif "pause music" in user_input:
            music_player.pause(); print("A.I.S.A.: Music paused."); continue

        # ---- Memory & Prompt ----
        msgs, embs = load_history()
        relevant = retrieve_relevant_history(user_input, msgs, embs)
        
        prompt = f"SYSTEM:\n{SYSTEM_PROMPT}\n{KNOWN_FACTS}\nRELEVANT CONTEXT:\n"
        for m in relevant: prompt += f"- {m}\n"
        prompt += f"\nUSER: {user_input}\nA.I.S.A.:"

        # ---- Run AI ----
        output = llm(prompt, max_tokens=300, temperature=0.4, stop=["USER:", "A.I.S.A.:"])
        response = output["choices"][0]["text"].strip()

        # ---- Actions ----
        should_exit = execute_system_action(response)
        if "ACTION_" not in response:
            print("A.I.S.A.:", response)
            say(response)

        if should_exit: break

        # Save conversation
        save_to_history(f"User: {user_input}\nA.I.S.A.: {response}", user_embedding)