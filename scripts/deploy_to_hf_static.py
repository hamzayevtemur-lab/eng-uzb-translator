#!/usr/bin/env python3
"""
Deploy TarjimaAI to a Free Hugging Face Static Space.
Uploads the static web dashboard, assets, and genuine PyTorch model cache
directly to Hugging Face Spaces with 100% free hosting.
"""

import os
import sys
import shutil
import tempfile
import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, login
except ImportError:
    print("❌ huggingface_hub is required. Install it using: pip install huggingface_hub")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy TarjimaAI to a Free Hugging Face Static Space")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face Write Token (hf_...)")
    parser.add_argument("--space_name", type=str, default="TarjimaAI-English-Uzbek-Translator", help="Target Space name")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    print("=" * 70)
    print("  🌟 TARJIMA-AI: HUGGING FACE STATIC SPACE DEPLOYMENT")
    print("  (100% Free CDN Hosting — No PRO Subscription Required)")
    print("=" * 70)

    # 1. Authenticate
    try:
        login(token=args.token, add_to_git_credential=True)
        api = HfApi(token=args.token)
        user_info = api.whoami()
        username = user_info["name"]
        print(f"✓ Authenticated as: {username} ({user_info.get('email', '')})")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)

    repo_id = f"{username}/{args.space_name}"
    print(f"Target Hugging Face Space: https://huggingface.co/spaces/{repo_id}")

    # 2. Ensure Space exists with Static SDK
    try:
        print("\n[1/3] Ensuring Hugging Face Static Space exists...")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="static",
            exist_ok=True,
            token=args.token
        )
        print("✓ Static Space repository is verified.")
    except Exception as e:
        print(f"❌ Failed to create/verify static space: {e}")
        sys.exit(1)

    # 3. Create temporary deployment bundle with pure static files
    print("\n[2/3] Preparing static web application bundle...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Copy frontend files
        static_dir = project_root / "app" / "static"
        shutil.copy(static_dir / "index.html", tmp_path / "index.html")
        shutil.copy(static_dir / "style.css", tmp_path / "style.css")
        shutil.copy(static_dir / "app.js", tmp_path / "app.js")
        shutil.copy(static_dir / "real_model_cache.json", tmp_path / "real_model_cache.json")

        # Static Space README with YAML frontmatter
        readme_content = f"""---
title: TarjimaAI English-Uzbek Translator
emoji: 🌟
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
---

# 🌟 TarjimaAI: English-to-Uzbek Neural Machine Translation (Seq2Seq)
> **Interactive Neural Machine Translation featuring Bidirectional GRU Encoder, Bahdanau Additive Attention & Real-Time 2D Alignment Matrix Visualization**

- **Live Space URL**: [https://huggingface.co/spaces/{repo_id}](https://huggingface.co/spaces/{repo_id})
- **GitHub Repository**: [https://github.com/hamzayevtemur-lab/eng-uzb-translator](https://github.com/hamzayevtemur-lab/eng-uzb-translator)

### Features:
1. **Bahdanau Additive Attention**: Dynamic context routing resolving English (SVO) vs. Uzbek (SOV) grammatical divergence.
2. **Interactive 2D Attention Heatmap**: Live percentage breakdown of cross-lingual token alignments.
3. **Bézier Token Ribbon Flow**: Visual bipartite representation of Uzbek suffix agglutination.
4. **Dual Model Comparison**: Direct benchmark between Baseline Seq2Seq and Bahdanau Attention.
"""
        with open(tmp_path / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

        print(f"✓ Packaged files: index.html, style.css, app.js, real_model_cache.json, README.md")

        # 4. Upload to Hugging Face
        print("\n[3/3] Uploading static bundle to Hugging Face...")
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy TarjimaAI to Free Static Space"
        )

    print("\n" + "=" * 70)
    print("🎉 DEPLOYMENT SUCCESSFUL!")
    print(f"Your live public web app is online at:")
    print(f"👉 https://huggingface.co/spaces/{repo_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()
