import os
import time
import json
import numpy as np
from threading import Thread
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from music import music_player

# ==== Paths ====
history_path = "history.jsonl"

# ==== Models ====
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = Llama(
    model_path="/home/hikaru/AI project/mistral-7b-v0.1.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=40,
    f16_kv=True
)

# ==== Load history and embeddings ====
def load_history():
    if not os.path.exists(history_path):
        return [], []
    messages, embeddings = [], []
    with open(history_path, "r") as f:
        for line in f:
            data = json.loads(line)
            messages.append(data["message"])
            embeddings.append(np.array(data["embedding"]))
    return messages, embeddings

def save_to_history(message, embedding):
    with open(history_path, "a") as f:
        json.dump({"message": message, "embedding": embedding.tolist()}, f)
        f.write("\n")

# ==== Get similar past messages ====
def retrieve_relevant_history(user_input, messages, embeddings, top_k=3):
    if not messages:
        return []
    input_embedding = embedding_model.encode([user_input])[0]
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [messages[i] for i in top_indices]


# ==== Main loop ====
while True:
    user_input = input("User: ")
    user_embedding = embedding_model.encode([user_input])[0]
    if "play playlist" in user_input:
        playlist_name = user_input.split("play playlist")[-1].strip()  # Get playlist name from user input
        if music_player.load_playlist(playlist_name):  # Only start playing if playlist exists
            Thread(target=music_player.play).start()
            print(f"A.I.S.A.: Playing playlist: {playlist_name}...")
        continue  # Skip further processing for this command
    
    elif "play song" in user_input:
        song_name = user_input.split("play song")[-1].strip()  # Get song name from user input
        music_player.play_song(song_name)
        print(f"A.I.S.A.: Playing song: {song_name}...")
        continue

    elif "pause music" in user_input:
        music_player.pause()
        print("A.I.S.A.: Music paused.")
        continue
    
    elif "resume music" in user_input:
        music_player.resume()
        print("A.I.S.A.: Music resumed.")
        continue
    
    elif "next track" in user_input:
        music_player.next_track()
        print("A.I.S.A.: Playing next track...")
        continue

    # Load history
    messages, embeddings = load_history()

    # Retrieve relevant past
    relevant_context = retrieve_relevant_history(user_input, messages, embeddings)

    # Build prompt
    prompt = ""
    for msg in relevant_context:
        prompt += msg + "\n"
    prompt += f"User: {user_input}\nA.I.S.A.:"

    # Run model
    output = llm(prompt, max_tokens=500, temperature=0.7, stop=["User:", "A.I.S.A.:"])
    response = output["choices"][0]["text"].strip()

    print("A.I.S.A.:", response)

    # Save to history
    save_to_history(f"User: {user_input}\nA.I.S.A.: {response}", user_embedding)
