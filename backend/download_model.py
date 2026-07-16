"""Download the Zipformer Quran model from HuggingFace.

Used when you DON'T have the model files locally and want the Docker
container to download them at build time (instead of mounting from
the host). Set SKIP_DOWNLOAD=0 in the build args to enable this.

Repo: https://huggingface.co/Muno459/zipformer_p-quran
"""
import os
import sys


REPO_ID = "Muno459/zipformer_p-quran"
MODEL_FILE = "quran_phoneme_zipformer.int8.onnx"
TOKENS_FILE = "tokens.txt"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data")
HF_TOKEN = os.environ.get("HF_TOKEN")  # only if model is gated


def main():
    from huggingface_hub import hf_hub_download

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Repo: {REPO_ID}")
    if HF_TOKEN:
        print("(using HF_TOKEN from env)")

    print(f"\nDownloading {MODEL_FILE}...")
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE,
        local_dir=OUTPUT_DIR,
        token=HF_TOKEN,
    )
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"  OK: {model_path} ({size_mb:.1f} MB)")

    print(f"\nDownloading {TOKENS_FILE}...")
    tokens_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=TOKENS_FILE,
        local_dir=OUTPUT_DIR,
        token=HF_TOKEN,
    )
    print(f"  OK: {tokens_path}")

    print("\nDone. Files in", OUTPUT_DIR, ":")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        full = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(full):
            print(f"  {os.path.getsize(full) / 1024 / 1024:7.2f} MB  {f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        if "gated" in str(e).lower() or "401" in str(e):
            print(
                "\nThis model is GATED. You need to:"
                "\n  1. Go to https://huggingface.co/Muno459/zipformer_p-quran"
                "\n  2. Accept the terms (click the button)"
                "\n  3. Get a token at https://huggingface.co/settings/tokens"
                "\n  4. Set HF_TOKEN env var or save it to backend/hf_token.txt"
            )
        sys.exit(1)
