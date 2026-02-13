
# A.I.S.A. – Local AI Smart Assistant

A.I.S.A. (Artificial Intelligent Simulated Assistant) is a lightweight, local AI assistant designed to run on **low-spec computers** using an optimized open-source language model.

This project is focused on building a **fully offline, privacy-friendly AI assistant** capable of conversation, contextual memory, and local media control — with more features planned in future updates.

---

## Features (Current)

* Local conversational AI (powered by Mistral 7B)
* Semantic memory with vector embeddings
* Persistent conversation history (`history.jsonl`)
* Local music control system:

  * Play playlist
  * Play specific song
  * Pause music
  * Resume music
  * Next track
* Context retrieval using cosine similarity
* Designed for low-spec machines

---

## Model Information

This project uses:

* **Mistral 7B (Quantized GGUF)**

  * Optimized version for low VRAM usage
  * Runs locally using `llama-cpp-python`
* **Sentence Transformers (all-MiniLM-L6-v2)**

  * Used for generating embeddings
  * Enables semantic search of past conversations

Everything runs **100% locally** — no API calls, no cloud dependency.

---

## Tech Stack

* Python 3.10+
* llama-cpp-python
* sentence-transformers
* scikit-learn
* numpy
* threading
* Local music module (`music_player`)

---

## Project Structure

```
AISA/
│
├── assistant.py
├── history.jsonl
├── music.py
├── models/
│   └── mistral-7b-v0.1.Q4_K_M.gguf
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/HIKARU-11-22/AISA
cd AISA
```

### Install Dependencies

```bash
pip install llama-cpp-python
pip install sentence-transformers
pip install scikit-learn
pip install numpy
```

### Download the Model

Download a quantized GGUF version of:

```
mistral-7b-v0.1.Q4_K_M.gguf
```

Place it inside your project folder and update:

```python
model_path="your/model/path/mistral-7b-v0.1.Q4_K_M.gguf"
```

---

## Running the Assistant

```bash
python assistant.py
```

Example commands:

```
play playlist chill
play song believer
pause music
resume music
next track
```

---

## How Memory Works

* User inputs are converted into embeddings.
* Stored in `history.jsonl`.
* Cosine similarity retrieves the most relevant past messages.
* Relevant context is injected into the model prompt.

This enables:

* Context-aware responses
* Lightweight long-term memory
* Efficient local semantic recall

---

## Designed for Low-Spec PCs

The assistant is optimized to:

* Run on quantized 7B models
* Use reduced VRAM via `n_gpu_layers`
* Operate within limited system RAM
* Avoid cloud processing

Tested on:

* 8–16GB RAM systems
* Mid-range GPUs
* CPU-only setups (with lower performance)

---

## Planned Features

This is an ongoing project. Upcoming features may include:

* Voice input/output
* File system interaction
* Optional offline web search
* GUI interface
* Encrypted memory storage
* Smarter memory indexing
* Plugin system
* Smart home control
* Task and reminder system

---

## Project Goal

To create a **fully private, offline AI assistant** that anyone can run on affordable hardware — without relying on big tech APIs or cloud services.

---

## Disclaimer

This project is experimental and under active development.
Performance depends on hardware and model quantization.

---

## License

This project uses open-source models and libraries.
Ensure you follow the license terms of:

* Mistral AI
* llama.cpp
* Sentence Transformers

---

## Contributing

Contributions are welcome!

If you'd like to:

* Improve performance
* Add features
* Optimize memory
* Improve modularity

Feel free to fork and submit a pull request.

---

## Support the Project

If you like the idea of a fully local AI assistant for low-end hardware:

Give the repo a ⭐ and follow development!

---
