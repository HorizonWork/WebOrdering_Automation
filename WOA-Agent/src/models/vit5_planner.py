"""
ViT5 Planner - Vietnamese Action Generation
Uses VietAI/vit5-base for generating action sequences
Encoder-Decoder architecture (like T5) for text-to-text tasks
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from typing import List, Dict, Optional, Union
import json
from dataclasses import dataclass
from pathlib import Path
import yaml

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Action:
    """Action data structure"""
    skill: str
    params: Dict[str, any]
    confidence: float = 0.8
    reason: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'skill': self.skill,
            'params': self.params,
            'confidence': self.confidence,
            'reason': self.reason
        }


class ViT5Planner:
    """
    ViT5 wrapper for Vietnamese action planning and generation.
    
    **Architecture**: Encoder-Decoder (like T5)
    **Model**: VietAI/vit5-base (310M parameters)
    **Use Cases**:
        - Generate action sequences from observations
        - Multi-step workflow planning
        - ReAct reasoning (Thought → Action)
        - Domain adaptation via LoRA fine-tuning
    
    **NOT for**:
        - Text embedding (use PhoBERT instead)
    
    **Performance**:
        - Inference: ~2 seconds per action (GPU)
        - Fine-tuning: 7-8 hours with LoRA (V100/A100)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize ViT5 planner.
        
        Args:
            model_name: ViT5 model identifier (default: VietAI/vit5-base)
            device: cuda/cpu/mps
            cache_dir: Model cache directory
        """
        # Load config from YAML
        config_path = Path("config/models.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                vit5_config = config.get('vit5', {})
        else:
            vit5_config = {}
        
        # Settings
        self.model_name = model_name or vit5_config.get('model_name', 'VietAI/vit5-base')
        self.device = device or settings.device
        self.max_input_length = vit5_config.get('max_input_length', 512)
        self.max_output_length = vit5_config.get('max_output_length', 256)
        self.num_beams = vit5_config.get('num_beams', 4)
        self.temperature = vit5_config.get('temperature', 0.7)
        self.top_p = vit5_config.get('top_p', 0.9)
        self.cache_dir = cache_dir or vit5_config.get('cache_dir', './checkpoints/vit5')
        
        logger.info(f"🚀 Loading ViT5 from {self.model_name}")
        logger.info(f"📍 Device: {self.device}")
        logger.info(f"📏 Input max length: {self.max_input_length}")
        logger.info(f"📏 Output max length: {self.max_output_length}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        logger.info("✓ Tokenizer loaded")
        
        # Load model
        logger.info("Loading model...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
        logger.info("✓ Model loaded")
        
        # Generation config
        self.generation_config = GenerationConfig(
            max_length=self.max_output_length,
            num_beams=self.num_beams,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            early_stopping=True
        )
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ ViT5 ready! Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt in Vietnamese
            max_length: Override max output length
            num_beams: Override beam search width
            temperature: Override sampling temperature
            top_p: Override nucleus sampling
            
        Returns:
            Generated text
            
        Example:
            >>> planner = ViT5Planner()
            >>> output = planner.generate("Nhiệm vụ: Tìm áo khoác")
            >>> print(output)
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Create generation config
        gen_config = GenerationConfig(
            max_length=max_length or self.max_output_length,
            num_beams=num_beams or self.num_beams,
            temperature=temperature or self.temperature,
            top_p=top_p or self.top_p,
            do_sample=True,
            early_stopping=True
        )
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            generation_config=gen_config
        )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated
    
    def generate_thought(
        self,
        query: str,
        observation: Dict,
        history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate reasoning thought (ReAct pattern).
        
        Args:
            query: Original user query
            observation: Current observation {url, dom, elements}
            history: Previous steps
            
        Returns:
            Thought/reasoning text
            
        Example:
            >>> thought = planner.generate_thought(
            ...     query="Tìm áo khoác",
            ...     observation={'url': 'shopee.vn', 'dom': '...'},
            ...     history=[]
            ... )
            >>> print(thought)
            "Cần tìm kiếm sản phẩm trên Shopee"
        """
        # Build context
        context = f"🎯 Nhiệm vụ: {query}\n\n"
        
        # Add history (last 3 steps to keep context manageable)
        if history:
            context += "📜 Lịch sử hành động:\n"
            for i, step in enumerate(history[-3:], 1):
                thought = step.get('thought', '')
                action = step.get('action', {})
                obs_summary = str(step.get('observation', ''))[:100]
                
                context += f"{i}. 💭 Suy nghĩ: {thought}\n"
                context += f"   ⚡ Hành động: {action.get('skill', '')}({action.get('params', {})})\n"
                context += f"   👁️  Quan sát: {obs_summary}...\n\n"
        
        # Add current observation
        dom_snippet = observation.get('dom', '')[:400]  # Truncate DOM
        url = observation.get('url', '')
        elements = observation.get('elements', [])
        
        context += f"🌐 Trạng thái hiện tại:\n"
        context += f"  • URL: {url}\n"
        context += f"  • DOM: {dom_snippet}...\n"
        context += f"  • Số phần tử tương tác: {len(elements)}\n\n"
        context += "💡 Hãy suy nghĩ xem bước tiếp theo cần làm gì:"
        
        # Generate thought
        thought = self.generate(
            context,
            max_length=128,
            num_beams=4,
            temperature=0.8
        )
        
        logger.debug(f"💭 Thought: {thought}")
        return thought
    
    def generate_action(
        self,
        thought: str,
        observation: Dict,
        available_skills: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate next action from thought and observation.
        
        Args:
            thought: Current reasoning/thought
            observation: Current state observation
            available_skills: List of available skill names
            
        Returns:
            Action dictionary: {skill, params, confidence, reason}
            
        Example:
            >>> action = planner.generate_action(
            ...     thought="Cần click vào search box",
            ...     observation={'dom': '...'},
            ...     available_skills=['goto', 'click', 'type']
            ... )
            >>> print(action)
            {'skill': 'click', 'params': {'selector': '#search'}, ...}
        """
        if available_skills is None:
            available_skills = [
                'goto', 'click', 'type', 'select', 'scroll', 
                'wait_for', 'screenshot', 'complete'
            ]
        
        # Get observation details
        dom = observation.get('dom', '')[:600]
        elements = observation.get('elements', [])[:10]  # Top 10
        url = observation.get('url', '')
        
        # Format interactive elements
        elements_str = ""
        for elem in elements:
            selector = elem.get('selector', '')
            text = elem.get('text', '')[:50]
            tag = elem.get('tag', '')
            elements_str += f"  • {selector} ({tag}): \"{text}\"\n"
        
        # Build prompt
        prompt = f"""Dựa trên tình huống, hãy quyết định hành động tiếp theo.

💭 Suy nghĩ: {thought}

🛠️  Kỹ năng có sẵn: {', '.join(available_skills)}

🌐 URL hiện tại: {url}

📄 Trạng thái DOM:
{dom}

🔘 Các phần tử tương tác:
{elements_str}

Trả lời theo định dạng JSON:
{{
  "skill": "tên_kỹ_năng",
  "params": {{"param1": "value1", "param2": "value2"}},
  "reason": "giải thích ngắn gọn"
}}

Hành động:"""
        
        # Generate action JSON
        action_json = self.generate(
            prompt,
            max_length=256,
            num_beams=1,  # Greedy for consistency
            temperature=0.3  # Low temp for deterministic output
        )
        
        # Parse JSON
        try:
            # Clean markdown formatting if present
            if "```
                action_json = action_json.split("```json").split("```
            elif "```" in action_json:
                action_json = action_json.split("``````")[0]
            
            action = json.loads(action_json.strip())
            
            # Validate skill
            if action.get('skill') not in available_skills:
                logger.warning(f"⚠️  Invalid skill: {action.get('skill')}, defaulting to 'wait_for'")
                action['skill'] = 'wait_for'
                action['params'] = {'selector': 'body'}
            
            # Ensure params is dict
            if not isinstance(action.get('params'), dict):
                action['params'] = {}
            
            # Add confidence
            action['confidence'] = 0.85
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.error(f"❌ Failed to parse action JSON: {e}")
            logger.error(f"Raw output: {action_json}")
            
            # Fallback action
            action = {
                'skill': 'wait_for',
                'params': {'selector': 'body'},
                'reason': f'JSON parse error: {str(e)}',
                'confidence': 0.3
            }
        
        logger.debug(f"⚡ Action: {action['skill']}({action['params']})")
        return action
    
    def generate_plan(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate high-level multi-step plan from query.
        
        Args:
            query: User query in Vietnamese
            context: Additional context (url, dom, etc.)
            
        Returns:
            Plan as text with numbered steps
            
        Example:
            >>> plan = planner.generate_plan("Mua áo khoác trên Shopee")
            >>> print(plan)
            1. Vào trang Shopee
            2. Tìm kiếm "áo khoác"
            3. Lọc kết quả...
        """
        prompt = f"🎯 Nhiệm vụ: {query}\n\n"
        
        if context:
            if 'url' in context:
                prompt += f"🌐 Trang web: {context['url']}\n"
            if 'constraints' in context:
                prompt += f"⚠️  Điều kiện: {context['constraints']}\n"
        
        prompt += "\n📋 Hãy lập kế hoạch chi tiết từng bước để hoàn thành nhiệm vụ:"
        
        plan = self.generate(
            prompt,
            max_length=512,
            num_beams=4,
            temperature=0.7
        )
        
        logger.debug(f"📋 Plan generated: {len(plan)} chars")
        return plan
    
    def explain_action(
        self,
        action: Dict,
        result: Dict
    ) -> str:
        """
        Generate explanation for why action was taken and its result.
        
        Args:
            action: Action that was executed
            result: Result of the action
            
        Returns:
            Explanation text
        """
        prompt = f"""Hành động đã thực hiện:
Kỹ năng: {action.get('skill')}
Tham số: {action.get('params')}

Kết quả:
Trạng thái: {result.get('status')}
Thông điệp: {result.get('message')}

Giải thích ngắn gọn tại sao hành động này được chọn và kết quả của nó:"""
        
        explanation = self.generate(
            prompt,
            max_length=128,
            temperature=0.7
        )
        
        return explanation


# Test & Usage Examples
if __name__ == "__main__":
    print("=" * 70)
    print("ViT5 Planner - Test & Examples")
    print("=" * 70 + "\n")
    
    # Initialize planner
    planner = ViT5Planner()
    
    # Mock observation
    observation = {
        'url': 'https://shopee.vn',
        'dom': '<div class="search-container"><input id="search-box" placeholder="Tìm kiếm"/><button id="search-btn">Tìm</button></div>',
        'elements': [
            {'selector': '#search-box', 'tag': 'input', 'text': '', 'attributes': {'placeholder': 'Tìm kiếm'}},
            {'selector': '#search-btn', 'tag': 'button', 'text': 'Tìm', 'attributes': {}},
            {'selector': '.category-link', 'tag': 'a', 'text': 'Thời trang nam', 'attributes': {'href': '/fashion'}}
        ]
    }
    
    # Test 1: Generate Thought
    print("=" * 70)
    print("Test 1: Generate Thought (ReAct Pattern)")
    print("=" * 70)
    query = "Tìm áo khoác nam màu đen giá dưới 500k"
    thought = planner.generate_thought(
        query=query,
        observation=observation,
        history=[]
    )
    print(f"\n🎯 Query: {query}")
    print(f"💭 Thought: {thought}\n")
    
    # Test 2: Generate Action
    print("=" * 70)
    print("Test 2: Generate Action")
    print("=" * 70)
    action = planner.generate_action(
        thought=thought,
        observation=observation,
        available_skills=['goto', 'click', 'type', 'select', 'complete']
    )
    print(f"\n💭 Thought: {thought}")
    print(f"⚡ Action: {json.dumps(action, ensure_ascii=False, indent=2)}\n")
    
    # Test 3: Generate Plan
    print("=" * 70)
    print("Test 3: Generate High-Level Plan")
    print("=" * 70)
    plan = planner.generate_plan(
        query="Mua áo khoác trên Shopee với giá tốt nhất",
        context={'url': 'https://shopee.vn'}
    )
    print(f"\n🎯 Query: Mua áo khoác trên Shopee với giá tốt nhất")
    print(f"📋 Plan:\n{plan}\n")
    
    # Test 4: Explain Action
    print("=" * 70)
    print("Test 4: Explain Action Result")
    print("=" * 70)
    mock_action = {'skill': 'click', 'params': {'selector': '#search-box'}}
    mock_result = {'status': 'success', 'message': 'Search box focused'}
    explanation = planner.explain_action(mock_action, mock_result)
    print(f"\n⚡ Action: click(#search-box)")
    print(f"✅ Result: success")
    print(f"💡 Explanation: {explanation}\n")
    
    print("=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)
