"""
ViT5 Planner - Vietnamese Action Generation
FINAL VERSION with all fixes
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional
import json
import re

# Add project root
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ViT5Planner:
    """
    ViT5 wrapper for action planning and generation.
    FINAL VERSION with robust JSON parsing and fallback.
    """
    
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/vit5",
        device: Optional[str] = None,
        input_max_length: int = 512,
        output_max_length: int = 256
    ):
        """Initialize ViT5 planner"""
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.checkpoint_path = checkpoint_path
        
        logger.info(f"🚀 Loading ViT5 from {checkpoint_path}")
        logger.info(f"📍 Device: {self.device}")
        logger.info(f"📏 Input max length: {input_max_length}")
        logger.info(f"📏 Output max length: {output_max_length}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        logger.info("✓ Tokenizer loaded")
        
        # Load model
        logger.info("Loading model...")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype
        ).to(self.device)
        
        self.model.eval()
        logger.info("✓ Model loaded")
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ ViT5 ready! Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    def generate_action(
        self,
        query: str,
        observation: Dict,
        history: Optional[List[Dict]] = None
    ) -> Dict:
        """Generate action matching training format"""
        # Extract platform from URL
        url = observation.get('url', '')
        platform = self._extract_platform(url)
        
        # Format elements (MATCH TRAINING FORMAT)
        elements = observation.get('interactive_elements', [])
        elements_str = self._format_elements(elements)
        
        # Format history (MATCH TRAINING FORMAT)
        history_str = self._format_history(history)
        
        # CREATE INPUT MATCHING TRAINING FORMAT EXACTLY
        input_text = f"""[INSTRUCTION] {query}
[PLATFORM] {platform}
[UI_ELEMENTS] {elements_str}
[HISTORY] {history_str}"""
        
        logger.debug(f"ViT5 input:\n{input_text[:300]}...")
        
        # Generate
        output_text = self._generate(input_text)
        
        logger.debug(f"ViT5 raw output: {output_text[:200]}...")
        
        # Parse output (training format → runtime format)
        action = self._parse_action_robust(output_text, query, observation)
        
        return action
    
    def _extract_platform(self, url: str) -> str:
        """Extract platform name from URL"""
        url_lower = url.lower()
        
        if 'shopee' in url_lower:
            return 'shopee'
        elif 'lazada' in url_lower:
            return 'lazada'
        elif 'tiki' in url_lower:
            return 'tiki'
        elif 'sendo' in url_lower:
            return 'sendo'
        else:
            return 'unknown'
    
    def _format_elements(self, elements: List[Dict], max_elements: int = 10) -> str:
        """Format elements matching training format"""
        if not elements:
            return "SELECTOR:body"
        
        lines = []
        for elem in elements[:max_elements]:
            selector = elem.get('selector', 'unknown')
            text = elem.get('text', '')
            
            line = f"SELECTOR:{selector}"
            if text:
                text_clean = text.strip()[:50]
                line += f" TEXT:{text_clean}"
            
            lines.append(line)
        
        return '\n'.join(lines) if lines else "SELECTOR:body"
    
    def _format_history(self, history: Optional[List[Dict]]) -> str:
        """Format history matching training format"""
        if not history:
            return "[]"
        
        recent_history = history[-5:] if len(history) > 5 else history
        
        history_list = []
        for step in recent_history:
            action = step.get('action', {})
            
            if isinstance(action, dict):
                skill = action.get('skill', 'unknown')
                params = action.get('params', {})
                selector = params.get('selector', 'body')
                
                history_list.append({
                    'action': skill,
                    'selector': selector
                })
        
        return json.dumps(history_list)
    
    @torch.no_grad()
    def _generate(self, input_text: str) -> str:
        """Generate text from input (FIXED warnings)"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=self.input_max_length,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate (FIXED config to avoid warnings)
            outputs = self.model.generate(
                **inputs,
                max_length=self.output_max_length,
                num_beams=4,  # FIX: Set >1 for early_stopping
                do_sample=False,  # FIX: Greedy decoding
                early_stopping=True,
                no_repeat_ngram_size=2,  # FIX: Prevent repetition
                repetition_penalty=1.2  # FIX: Discourage repeats
            )
            
            # Decode
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return output_text
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""
    
    def _repair_json(self, text: str) -> str:
        """Attempt to repair malformed JSON"""
        text = text.strip()
        
        # Remove markdown fences (```json ... ```)
        if "```" in text:
            if "```json" in text:
                text = text.split("```json", 1)[1]
            text = text.split("```", 1)[0]
            text = text.strip()
        
        # Remove leading/trailing junk
        # Look for first { and last }
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]
        else:
            # Add missing braces
            if not text.startswith('{'):
                text = '{' + text
            if not text.endswith('}'):
                text = text + '}'
        
        # Fix common issues
        # Replace single quotes with double quotes
        text = text.replace("'", '"')
        
        # Fix unquoted keys (heuristic)
        text = re.sub(r'(\w+):\s*', r'"\1": ', text)
        
        return text
    
    def _parse_action_robust(
        self,
        output_text: str,
        query: str,
        observation: Dict
    ) -> Dict:
        """
        Parse action with robust fallback.
        Training format → Runtime format
        """
        # Try direct JSON parse
        try:
            parsed = json.loads(output_text.strip())
            return self._convert_format(parsed)
        except json.JSONDecodeError:
            pass
        
        # Try JSON repair
        try:
            repaired = self._repair_json(output_text)
            parsed = json.loads(repaired)
            return self._convert_format(parsed)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse action JSON: {e}")
            logger.error(f"Raw output: {output_text[:200]}")
            
            # FALLBACK: Keyword-based extraction
            return self._fallback_parse(output_text, query, observation)
    
    def _convert_format(self, parsed: Dict) -> Dict:
        """Convert training format → runtime format"""
        skill = parsed.get('action', 'wait')
        target = parsed.get('target', {})
        args = parsed.get('args', {})
        meta = parsed.get('meta', {})
        
        action = {
            'skill': skill,
            'params': {},
            'confidence': meta.get('confidence', 0.5),
            'reasoning': meta.get('reason', '')
        }
        
        # Extract selector from target
        if isinstance(target, dict) and target.get('ref_type') == 'selector':
            action['params']['selector'] = target.get('ref', 'body')
        
        # Add args to params
        if isinstance(args, dict):
            action['params'].update(args)
        
        logger.info(f"✅ Parsed action: {skill}({action['params']})")
        
        return action
    
    def _fallback_parse(
        self,
        text: str,
        query: str,
        observation: Dict
    ) -> Dict:
        """Fallback parser when JSON fails"""
        text_lower = text.lower()
        
        # Search intent
        if any(kw in text_lower for kw in ['search', 'tìm', 'find', 'look']):
            elements = observation.get('interactive_elements', [])
            search_input = None
            
            for elem in elements:
                selector = elem.get('selector', '').lower()
                if 'search' in selector or 'input' in elem.get('tag', '').lower():
                    search_input = elem.get('selector')
                    break
            
            if not search_input:
                search_input = 'input[type="search"], input[type="text"]'
            
            return {
                'skill': 'type',
                'params': {
                    'selector': search_input,
                    'text': query,
                    'clear': True
                },
                'confidence': 0.7,
                'reasoning': 'Detected search intent (fallback)'
            }
        
        # Click intent
        if any(kw in text_lower for kw in ['click', 'press', 'tap', 'nhấn']):
            selector = 'button, a'
            
            if 'button' in text_lower:
                selector = 'button'
            elif 'link' in text_lower:
                selector = 'a'
            
            return {
                'skill': 'click',
                'params': {'selector': selector},
                'confidence': 0.6,
                'reasoning': 'Detected click intent (fallback)'
            }
        
        # Type intent
        if any(kw in text_lower for kw in ['type', 'enter', 'fill', 'nhập']):
            return {
                'skill': 'type',
                'params': {
                    'selector': 'input',
                    'text': query
                },
                'confidence': 0.6,
                'reasoning': 'Detected input intent (fallback)'
            }
        
        # Submit intent
        if 'enter' in text_lower or 'submit' in text_lower:
            return {
                'skill': 'press',
                'params': {'key': 'Enter'},
                'confidence': 0.7,
                'reasoning': 'Detected submit intent (fallback)'
            }
        
        # Default: wait
        logger.warning(f"Using default wait action (fallback)")
        return {
            'skill': 'wait_for',
            'params': {'selector': 'body', 'timeout': 1000},
            'confidence': 0.3,
            'reasoning': 'Default fallback action'
        }


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("ViT5 Planner - Test (Fixed)")
    print("=" * 70 + "\n")
    
    # Initialize
    try:
        planner = ViT5Planner(checkpoint_path="checkpoints/vit5")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Mock observation
    observation = {
        'url': 'https://shopee.vn',
        'interactive_elements': [
            {'selector': 'input#search', 'tag': 'input', 'text': ''},
            {'selector': 'button.search-btn', 'tag': 'button', 'text': 'Tìm kiếm'},
            {'selector': 'a.category', 'tag': 'a', 'text': 'Thời trang'}
        ]
    }
    
    # Test
    print("Test: Generate Action")
    print("-" * 40)
    
    query = "Tìm áo khoác"
    action = planner.generate_action(
        query=query,
        observation=observation,
        history=[]
    )
    
    print(f"Query: {query}")
    print(f"Action: {json.dumps(action, ensure_ascii=False, indent=2)}")
    print(f"Confidence: {action['confidence']:.2f}")
    print(f"Reasoning: {action['reasoning']}\n")
    
    print("=" * 70)
    print("✅ Test completed!")
    print("=" * 70)
