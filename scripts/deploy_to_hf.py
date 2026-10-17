#!/usr/bin/env python3
"""
Deploy TarjimaAI to Hugging Face Spaces (Docker SDK).
Uses huggingface_hub to create the Space and upload application files,
including the 760MB Attention and 370MB Baseline PyTorch models via Git LFS.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, login
except ImportError:
    print("❌ huggingface_hub is required. Install it using: pip install huggingface_hub")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy TarjimaAI to Hugging Face Spaces")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face Write Token (hf_...)")
    parser.add_argument("--space_name", type=str, default="TarjimaAI-English-Uzbek-Translator", help="Target Space name")
    parser.add_argument("--private", action="store_true", help="Make Space private (default is public)")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    print("=" * 70)
    print("  🌟 TARJIMA-AI: HUGGING FACE SPACES DEPLOYMENT")
    print("=" * 70)

    # 1. Obtain Token
    token = args.token
    if not token:
        token = input("🔑 Enter your Hugging Face Write Token (from https://huggingface.co/settings/tokens): ").strip()
    
    if not token:
        print("❌ Error: A valid Hugging Face Write Token is required.")
        sys.exit(1)

    # 2. Authenticate
    try:
        login(token=token, add_to_git_credential=True)
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info["name"]
        print(f"✓ Authenticated as: {username} ({user_info.get('email', '')})")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)

    repo_id = f"{username}/{args.space_name}"
    print(f"Target Hugging Face Space: https://huggingface.co/spaces/{repo_id}")

    # 3. Create Space if it doesn't exist
    try:
        print("\n[1/3] Ensuring Hugging Face Docker Space exists...")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
            token=token
        )
        print("✓ Space repository is ready.")
    except Exception as e:
        print(f"❌ Failed to create/verify space: {e}")
        sys.exit(1)

    # 4. Verify checkpoints before upload
    checkpoints_dir = project_root / "checkpoints"
    attn_pth = checkpoints_dir / "seq2seq_attention.pth"
    base_pth = checkpoints_dir / "seq2seq_baseline.pth"
    vocab_eng = checkpoints_dir / "vocab_eng.json"
    vocab_uzb = checkpoints_dir / "vocab_uzb.json"

    missing = []
    for f in [attn_pth, base_pth, vocab_eng, vocab_uzb]:
        if not f.exists():
            missing.append(f.name)
    
    if missing:
        print(f"⚠️ Warning: Missing checkpoint files in checkpoints/: {missing}")
        confirm = input("Proceed anyway? (y/n): ").strip().lower()
        if confirm != "y":
            sys.exit(1)
    else:
        print("\n[2/3] Verified all checkpoint weights and vocabularies are present:")
        print(f"  • seq2seq_attention.pth ({round(attn_pth.stat().st_size / (1024*1024), 1)} MB)")
        print(f"  • seq2seq_baseline.pth  ({round(base_pth.stat().st_size / (1024*1024), 1)} MB)")
        print(f"  • vocab_eng.json        ({round(vocab_eng.stat().st_size / 1024, 1)} KB)")
        print(f"  • vocab_uzb.json        ({round(vocab_uzb.stat().st_size / 1024, 1)} KB)")

    # 5. Upload folder directly to Hugging Face Space
    print("\n[3/3] Uploading project files and model weights via Hugging Face LFS...")
    print("      (This may take a couple of minutes depending on your internet upload speed)")

    ignore_patterns = [
        ".git/*",
        ".git",
        "*.pyc",
        "__pycache__/*",
        "__pycache__",
        ".DS_Store",
        "*.zip",
        ".env",
        ".pytest_cache/*",
        "data/Translations.jsonl"  # Training dataset not needed for web inference
    ]

    try:
        api.upload_folder(
            folder_path=str(project_root),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            commit_message="Deploy TarjimaAI: FastAPI with Bahdanau Attention & Baseline models"
        )
        print("\n" + "=" * 70)
        print("🎉 DEPLOYMENT UPLOAD COMPLETE!")
        print(f"Your public Space is now building and will be live in 1-2 minutes at:")
        print(f"👉 https://huggingface.co/spaces/{repo_id}")
        print("=" * 70)
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
