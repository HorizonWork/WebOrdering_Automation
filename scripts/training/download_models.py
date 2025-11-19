"""
Download all required models for WOA Agent
- PhoBERT (vinai/phobert-base-v2)
- ViT5 (VietAI/vit5-base)
"""

import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def download_phobert():
    """Download PhoBERT model and tokenizer"""
    print("=" * 70)
    print("📥 Downloading PhoBERT (vinai/phobert-base-v2)...")
    print("=" * 70)

    model_name = "vinai/phobert-base-v2"
    cache_dir = ROOT_DIR / "checkpoints" / "phobert_base"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
        model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir))

        print(f"✅ PhoBERT downloaded to: {cache_dir}")
        print(f"   - Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"❌ Failed to download PhoBERT: {e}")
        return False


def download_vit5():
    """Download ViT5 model and tokenizer"""
    print("\n" + "=" * 70)
    print("📥 Downloading ViT5 (VietAI/vit5-base)...")
    print("=" * 70)

    model_name = "VietAI/vit5-base"
    cache_dir = ROOT_DIR / "checkpoints" / "vit5_base"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=str(cache_dir)
        )

        print(f"✅ ViT5 downloaded to: {cache_dir}")
        print(f"   - Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"❌ Failed to download ViT5: {e}")
        return False


def main():
    print("\n🚀 WOA Agent - Model Downloader")
    print("=" * 70)
    print("This script will download:")
    print("  1. PhoBERT (vinai/phobert-base-v2) - ~500 MB")
    print("  2. ViT5 (VietAI/vit5-base) - ~900 MB")
    print("  Total: ~1.4 GB")
    print("=" * 70 + "\n")

    results = []

    # Download PhoBERT
    results.append(("PhoBERT", download_phobert()))

    # Download ViT5
    results.append(("ViT5", download_vit5()))

    # Summary
    print("\n" + "=" * 70)
    print("📊 Download Summary")
    print("=" * 70)
    for model_name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{model_name:20s}: {status}")

    all_success = all(success for _, success in results)
    if all_success:
        print("\n🎉 All models downloaded successfully!")
        print("\nNext steps:")
        print("  1. Run tests: python -m pytest tests/unit/")
        print("  2. Train models (optional): python scripts/training/train_controller.py")
        print("  3. Run agent: python run_agent.py")
    else:
        print("\n⚠️  Some models failed to download. Please check errors above.")
        print("   You may need to:")
        print("   - Check internet connection")
        print("   - Install missing dependencies: pip install transformers torch")
        print("   - Manually download from HuggingFace Hub")


if __name__ == "__main__":
    main()
