"""
Master Training Orchestrator.
Allows training either model via CLI flag:
    python3 src/train.py --model attention  (default)
    python3 src/train.py --model baseline
    python3 src/train.py --model all        (trains both models sequentially)
"""

import argparse
from train_baseline import main as train_baseline_main
from train_attention import main as train_attention_main


def main():
    parser = argparse.ArgumentParser(description="Train English-to-Uzbek Seq2Seq Models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["attention", "baseline", "all"],
        default="attention",
        help="Which model to train: 'attention' (default), 'baseline', or 'all'"
    )
    args = parser.parse_args()

    if args.model in ("baseline", "all"):
        print("\n" + "=" * 70)
        print("  STARTING BASELINE SEQ2SEQ TRAINING (WITHOUT ATTENTION)")
        print("=" * 70)
        train_baseline_main()

    if args.model in ("attention", "all"):
        print("\n" + "=" * 70)
        print("  STARTING BAHDANAU ATTENTION SEQ2SEQ TRAINING")
        print("=" * 70)
        train_attention_main()


if __name__ == "__main__":
    main()
