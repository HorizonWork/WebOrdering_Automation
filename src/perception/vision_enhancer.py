"""
Vision Enhancer - optional multimodal perception (YOLO + Florence captioning).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import torch

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class VisionResult:
    caption: Optional[str]
    detections: List[Dict[str, Any]]


class VisionEnhancer:
    """
    Combines YOLO object detection + Florence captioning to enrich perception.

    All components are optional: if a model or dependency is missing we simply
    log a warning and continue without that signal.
    """

    def __init__(
        self,
        yolo_model_path: str | None = None,
        florence_base_model: str | None = None,
        florence_adapter_path: str | None = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or settings.device
        self.yolo_path = yolo_model_path or settings.yolo_model_path
        self.florence_base = florence_base_model or settings.florence_base_model
        self.florence_adapter = florence_adapter_path or settings.florence_adapter_path

        self._yolo = self._load_yolo()
        self._processor = None
        self._florence = self._load_florence()

    def _load_yolo(self):
        if not self.yolo_path or not Path(self.yolo_path).exists():
            logger.warning(f"YOLO model not found at {self.yolo_path}, detection disabled")
            return None
        try:
            from ultralytics import YOLO as UltralyticsYOLO  # type: ignore
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Ultralytics not available ({exc}); YOLO disabled")
            return None
        try:
            model = UltralyticsYOLO(self.yolo_path)
            logger.info(f"YOLO loaded from {self.yolo_path}")
            return model
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Failed to load YOLO model: %s. Ensure ultralytics>=8.2.0 "
                "matches the checkpoint architecture.", exc
            )
            return None

    def _load_florence(self):
        if not self.florence_base:
            logger.info("No Florence base model configured, captioning disabled")
            return None
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM  # type: ignore
            from peft import PeftModel  # type: ignore
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Transformers/Peft not available ({exc}); Florence disabled")
            return None

        local_only = Path(self.florence_base).exists()
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_only,
        }

        try:
            processor = AutoProcessor.from_pretrained(self.florence_base, **load_kwargs)
            model = AutoModelForCausalLM.from_pretrained(
                self.florence_base,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                **load_kwargs,
            )
            if self.florence_adapter and Path(self.florence_adapter).exists():
                try:
                    model = PeftModel.from_pretrained(model, self.florence_adapter)
                    logger.info(f"Loaded Florence adapter from {self.florence_adapter}")
                except Exception as exc:
                    logger.warning(f"Failed to load Florence adapter: {exc}")
            self._processor = processor
            model.to(self.device)
            model.eval()
            logger.info(f"Florence model ready on {self.device}")
            return model
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to load Florence model: {exc}")
            return None

    async def analyze_async(self, screenshot: Optional[bytes]) -> Optional[VisionResult]:
        if not screenshot:
            return None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze, screenshot)

    def analyze(self, screenshot: bytes) -> Optional[VisionResult]:
        image = Image.open(BytesIO(screenshot)).convert("RGB")
        detections = self._run_yolo(image)
        caption = self._run_florence(image)
        if not caption and not detections:
            return None
        return VisionResult(caption=caption, detections=detections)

    def _run_yolo(self, image: Image.Image) -> List[Dict[str, Any]]:
        if not self._yolo:
            return []
        try:
            results = self._yolo(image, verbose=False)
            detections: List[Dict[str, Any]] = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = result.names.get(cls_id, f"class_{cls_id}")
                    detections.append(
                        {
                            "label": name,
                            "confidence": conf,
                            "box": [float(v) for v in xyxy],
                        }
                    )
            return detections
        except Exception as exc:
            logger.warning(f"YOLO detection failed: {exc}")
            return []

    def _run_florence(self, image: Image.Image) -> Optional[str]:
        if not self._florence or not self._processor:
            return None
        prompt = "<OD>Describe the main UI elements and any prominent text.</OD>"
        try:
            inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self._florence.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )
            caption = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return caption.strip()
        except Exception as exc:
            logger.warning(f"Florence caption failed: {exc}")
            return None
