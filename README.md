# Hifzapp — Live Quran Recitation Tracker

A free, open-source Quran memorization companion with live word-by-word
highlighting (Tarteel-style UX). Recite into your microphone and watch
the mushaf highlight each word as you say it.

**Try it now:** https://suzuki-microwave-instrumentation-registration.trycloudflare.com/

## What's in this repo

This is a monorepo (as of 2026-07-12) that holds both the frontend
and the backend. Previously they were split across two repos
(`Hifzapp` + `HafizAssistant`); they've been merged for simpler
maintenance.

```
Hifzapp/
├── README.md                    # This file
├── .gitignore
│
├── hafizAssist_streaming.html   # ← Production frontend (v1.5.6)
├── HafizAssist.html             # ← webkitSpeechRecognition fallback (v3.0)
├── index.html                   # ← Entry point (alias for streaming)
├── phonemes-data.js             # ← 5.3MB Quran + Tajweed data
│
├── backend/                     # ← FastConformer ASR backend
│   ├── app.py                       (FastAPI + WebSocket)
│   ├── Dockerfile                   (Python 3.10 + NeMo)
│   ├── docker-compose.yml           (local dev)
│   ├── requirements.txt
│   ├── .dockerignore
│   ├── .gitignore
│   └── README.md                    (backend-specific docs)
│
└── deploy/                      # ← One-shot deploy scripts
    ├── deploy-frontend.sh            (ship HTML/JS to droplet)
    ├── deploy-backend.sh             (rebuild Docker on droplet)
    ├── setup-droplet.sh              (first-time setup for a new droplet)
    └── nginx.conf                    (HTTP+HTTPS proxy + WS)
```

## Quick start

### Just want to use it?

Visit the live URL — no setup needed:
**https://suzuki-microwave-instrumentation-registration.trycloudflare.com/**

(Or use the self-hosted version at `http://167.99.116.58/`)

### Want to self-host?

```bash
# 1. Clone this repo
git clone https://github.com/gufranmohammed48-git/Hifzapp.git
cd Hifzapp

# 2. On a fresh Ubuntu 22.04 droplet (as root)
./deploy/setup-droplet.sh

# 3. From your local machine, deploy the backend
./deploy/deploy-backend.sh

# 4. Deploy the frontend
./deploy/deploy-frontend.sh

# 5. (Optional) Get a real HTTPS cert via cloudflared
ssh root@your-droplet 'cloudflared tunnel --url http://localhost:80'
```

The page should now be live at `http://your-droplet-ip/`.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Browser (hafizAssist_streaming.html)               │
│  - Web Audio API: getUserMedia (no noise suppress)  │
│  - 1-second chunks, software gain, 100Hz HPF        │
│  - WebSocket → ws://droplet/ws                      │
└────────────────────┬────────────────────────────────┘
                     │ 16kHz mono int16 PCM
                     ▼
┌─────────────────────────────────────────────────────┐
│  nginx (port 80/443 on DO droplet)                  │
│  - /ws        → proxy to FastConformer container    │
│  - /api/      → proxy to FastConformer container    │
│  - /*         → serve static HTML + phonemes-data.js│
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  FastConformer container (port 8080)                │
│  - FastAPI + WebSocket                              │
│  - shahabazkc10/fastconformer-quran-bucket model    │
│    (459MB, 4.13% WER, 0.93% on unseen reciters)     │
│  - 2-second rolling buffer                          │
│  - Returns partial + final transcriptions with      │
│    full diacritics                                  │
└─────────────────────────────────────────────────────┘
```

## How the match algorithm works

The browser receives the model's output (Arabic text WITH diacritics),
normalizes it (strip diacritics), then:

1. **Exact match** against the expected Quran text at the current pointer
2. **Substring match** (the model sometimes merges ال with the next word)
3. **Levenshtein ratio ≥ 0.70** (handles "ya-siruna" vs "yasiruna")
4. **Phoneme fallback ≥ 0.70** (when text ratio is 0.40-0.70)
5. **First-word bias** — when pointer=0, the first interim result is
   often a partial word, so we use a lenient threshold of 0.40
6. **Delta tracking** — only process NEW words from each model response
   (the model returns a rolling buffer, so the same word can appear
   multiple times as the buffer grows)

## Production model

We use [`shahabazkc10/fastconformer-quran-bucket`](https://huggingface.co/shahabazkc10/fastconformer-quran-bucket):
- 4.13% WER overall
- **0.93% WER on held-out unseen reciters** ← the key metric for production
- Full diacritics in output
- Tested with Alafasy, AbdulBasit, Minshawy, Hudhaify

The previous model (`mohammed/fastconformer-quran-ar`, 6.95% WER) was
biased to Alafasy's voice and didn't work for other reciters.

## License

Open source — make it yours. Phoneme data from the Qur'anic Phonemizer
(MIT, MIT, 71-symbol inventory).
