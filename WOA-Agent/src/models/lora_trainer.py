"""
LoRA Trainer - Low-Rank Adaptation for Fine-tuning
Fine-tune ViT5 efficiently for domain-specific tasks (Shopee, Lazada)
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import Dataset
from typing import List, Dict, Optional
import json
from pathlib import Path
import yaml

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LoRATrainer:
    """
    LoRA Fine-tuning for ViT5.
    
    **LoRA** (Low-Rank Adaptation):
        - Freeze base model weights
        - Add trainable rank decomposition matrices
        - 10-100x fewer trainable parameters
        - 3-5x faster training
        - Same inference performance
    
    **Training Data Format**:
        ```
        [
            {
                "input": "Trạng thái: Có search box. Tác vụ: tìm áo khoác",
                "output": "type(#search, 'áo khoác')"
            },
            ...
        ]
        ```
    
    **GPU Requirements**:
        - ViT5-base with LoRA: ~4GB VRAM
        - Training time: 7-8 hours (V100/A100)
        - Batch size 4 recommended
    """
    
    def __init__(
        self,
        model_name: str = "VietAI/vit5-base",
        device: Optional[str] = None
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            model_name: Base ViT5 model
            device: cuda/cpu
        """
        self.model_name = model_name
        self.device = device or settings.device
        
        # Load config
        config_path = Path("config/models.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                self.lora_config_dict = config.get('vit5', {}).get('lora', {})
        else:
            self.lora_config_dict = {}
        
        # LoRA hyperparameters
        self.lora_r = self.lora_config_dict.get('r', 8)
        self.lora_alpha = self.lora_config_dict.get('lora_alpha', 16)
        self.lora_dropout = self.lora_config_dict.get('lora_dropout', 0.1)
        self.target_modules = self.lora_config_dict.get('target_modules', ['q', 'v'])
        
        logger.info(f"🚀 LoRA Trainer initialized")
        logger.info(f"📍 Base model: {self.model_name}")
        logger.info(f"📍 Device: {self.device}")
        logger.info(f"⚙️  LoRA config: r={self.lora_r}, alpha={self.lora_alpha}")
        
        self.tokenizer = None
        self.model = None
        self.peft_model = None
    
    def load_model(self):
        """Load base model and tokenizer"""
        logger.info("Loading base model and tokenizer...")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info("✓ Tokenizer loaded")
        
        # Model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        logger.info("✓ Base model loaded")
        
        # Print model size
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"📊 Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    def apply_lora(self):
        """Apply LoRA to model"""
        logger.info("Applying LoRA configuration...")
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        logger.info("✓ LoRA applied successfully")
        return self.peft_model
    
    def prepare_dataset(
        self,
        data: List[Dict[str, str]],
        max_input_length: int = 512,
        max_output_length: int = 256
    ) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            data: List of {"input": str, "output": str}
            max_input_length: Max input tokens
            max_output_length: Max output tokens
            
        Returns:
            HuggingFace Dataset
        """
        logger.info(f"Preparing dataset ({len(data)} examples)...")
        
        def preprocess_function(examples):
            """Tokenize inputs and outputs"""
            inputs = [ex["input"] for ex in examples]
            targets = [ex["output"] for ex in examples]
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=max_input_length,
                truncation=True,
                padding="max_length"
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                targets,
                max_length=max_output_length,
                truncation=True,
                padding="max_length"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Create dataset
        dataset = Dataset.from_list(data)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function([examples]),
            remove_columns=dataset.column_names
        )
        
        logger.info(f"✓ Dataset prepared: {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def train(
        self,
        train_data: List[Dict[str, str]],
        output_dir: str = "./checkpoints/vit5-lora",
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.1,
        eval_data: Optional[List[Dict[str, str]]] = None,
        save_steps: int = 500,
        logging_steps: int = 100
    ):
        """
        Fine-tune model with LoRA.
        
        Args:
            train_data: Training data
            output_dir: Output directory for checkpoints
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Per-device batch size
            gradient_accumulation_steps: Gradient accumulation
            warmup_ratio: Warmup ratio
            eval_data: Evaluation data (optional)
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
        """
        logger.info("=" * 60)
        logger.info("Starting LoRA Fine-tuning")
        logger.info("=" * 60)
        
        # Load model if not loaded
        if self.model is None:
            self.load_model()
        
        # Apply LoRA if not applied
        if self.peft_model is None:
            self.apply_lora()
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data)
        eval_dataset = self.prepare_dataset(eval_data) if eval_data else None
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=save_steps if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False if eval_dataset else None,
            fp16=True if self.device == "cuda" else False,
            report_to="none",
            remove_unused_columns=False,
            predict_with_generate=True
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.peft_model,
            padding=True
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        # Train
        logger.info("🔥 Starting training...")
        logger.info(f"📊 Training examples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"📊 Eval examples: {len(eval_dataset)}")
        logger.info(f"📊 Epochs: {num_epochs}")
        logger.info(f"📊 Batch size: {batch_size} × {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps}")
        logger.info(f"📊 Learning rate: {learning_rate}")
        
        train_result = trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        metrics_path = Path(output_dir) / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("✅ Training completed!")
        logger.info(f"📂 Model saved to: {output_dir}")
        logger.info(f"📊 Final loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
        logger.info("=" * 60)
        
        return train_result
    
    def load_finetuned_model(self, checkpoint_dir: str):
        """
        Load fine-tuned LoRA model.
        
        Args:
            checkpoint_dir: Directory with LoRA checkpoint
        """
        logger.info(f"Loading fine-tuned model from {checkpoint_dir}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        
        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Load LoRA weights
        self.peft_model = PeftModel.from_pretrained(
            base_model,
            checkpoint_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        logger.info("✓ Fine-tuned model loaded")
        return self.peft_model
    
    def generate(self, prompt: str, max_length: int = 256) -> str:
        """
        Generate output from fine-tuned model.
        
        Args:
            prompt: Input prompt
            max_length: Max output length
            
        Returns:
            Generated text
        """
        if self.peft_model is None:
            raise ValueError("Model not loaded. Call load_model() or load_finetuned_model() first.")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated


# Example Usage & Test
if __name__ == "__main__":
    print("=" * 60)
    print("LoRA Trainer - Example & Test")
    print("=" * 60 + "\n")
    
    # Sample training data (Vietnamese Shopee actions)
    sample_data = [
        {
            "input": "Trạng thái: Trang Shopee. Tác vụ: Tìm áo khoác",
            "output": "click(#search-box)"
        },
        {
            "input": "Trạng thái: Search box focused. Tác vụ: Nhập từ khóa 'áo khoác'",
            "output": "type(#search-box, 'áo khoác')"
        },
        {
            "input": "Trạng thái: Đã nhập text. Tác vụ: Tìm kiếm",
            "output": "click(#search-button)"
        },
        {
            "input": "Trạng thái: Danh sách sản phẩm. Tác vụ: Lọc giá dưới 500k",
            "output": "select(#price-filter, '<500000')"
        },
        {
            "input": "Trạng thái: Đã lọc sản phẩm. Tác vụ: Chọn sản phẩm đầu tiên",
            "output": "click(.product-item:nth-of-type(1))"
        },
        {
            "input": "Trạng thái: Chi tiết sản phẩm. Tác vụ: Thêm vào giỏ hàng",
            "output": "click(.add-to-cart-button)"
        },
        {
            "input": "Trạng thái: Đã thêm vào giỏ. Tác vụ: Hoàn thành",
            "output": "complete()"
        }
    ]
    
    # Duplicate data to make it larger (for demo)
    training_data = sample_data * 10  # 70 examples
    eval_data = sample_data[:3]  # 3 examples for eval
    
    print(f"📊 Training data: {len(training_data)} examples")
    print(f"📊 Eval data: {len(eval_data)} examples\n")
    
    # Initialize trainer
    trainer = LoRATrainer()
    
    # Load model
    trainer.load_model()
    
    # Apply LoRA
    trainer.apply_lora()
    
    print("\n" + "=" * 60)
    print("Ready to train!")
    print("=" * 60)
    print("\nTo start training, run:")
    print("```
    print("trainer.train(")
    print("    train_data=training_data,")
    print("    eval_data=eval_data,")
    print("    output_dir='./checkpoints/vit5-shopee-lora',")
    print("    num_epochs=3,")
    print("    learning_rate=5e-5,")
    print("    batch_size=4")
    print(")")
    print("```")
    print("\n⚠️  Note: Actual training takes 7-8 hours on GPU")
    print("💡 Tip: Use smaller dataset for quick testing\n")
    
    # Demonstrate generation (before training)
    print("=" * 60)
    print("Test Generation (Before Fine-tuning)")
    print("=" * 60)
    test_prompt = "Trạng thái: Trang chủ Shopee. Tác vụ: Tìm giày thể thao"
    print(f"\nPrompt: {test_prompt}")
    
    try:
        output = trainer.generate(test_prompt)
        print(f"Output: {output}")
    except Exception as e:
        print(f"Generation skipped: {e}")
    
    print("\n✅ LoRA Trainer setup complete!")
