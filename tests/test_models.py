import torch

import path_setup  # noqa: F401

from src.models.phobert_encoder import PhoBERTEncoder
from src.models.vit5_planner import ViT5Planner


def test_models():
    print("Testing Core AI Models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # 1. Test PhoBERT
        print("Testing PhoBERT...")
        encoder = PhoBERTEncoder(device=device)
        embedding = encoder.encode_text("ki?m tra phobert")
        print(f"PhoBERT embedding shape: {embedding.shape}")
        assert embedding.shape == (1, 768)
        print("PhoBERT OK!")

        # 2. Test ViT5
        print("Testing ViT5...")
        planner = ViT5Planner(device=device)
        # D�y l� test generation don gi?n, kh�ng ph?i test logic
        output = planner.model.generate(
            planner.tokenizer("Vietnamese text", return_tensors="pt").input_ids.to(
                device
            )
        )
        decoded = planner.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"ViT5 generated: {decoded}")
        assert len(decoded) > 0
        print("ViT5 OK!")

        print("Models Test Successful!")

    except Exception as e:
        print(f"MODELS TEST FAILED: {e}")


if __name__ == "__main__":
    test_models()
