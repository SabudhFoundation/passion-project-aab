"""
perception_module.py
--------------------
Single-frame multi-modal perception.

Models (same as old pipeline):
  BLIP   -> scene caption + 768-D visual embedding
  YOLO   -> object detection (name, confidence, bbox, direction, range)
  MiDaS  -> monocular depth estimation

Input : RGB numpy array (H, W, 3)   <- from frame_source.py (BGR→RGB converted in pipeline)
Output: Dict (same schema as old perception_module.process_scene())
  {
    'caption'          : str,
    'caption_embedding': np.ndarray | None,   # 768-D BLIP visual embedding
    'objects'          : [{'name', 'confidence', 'bbox', 'center',
                           'direction', 'range'}],
    'depth'            : np.ndarray | None,   # normalised depth map (H,W)
    'num_objects'      : int,
    'has_depth'        : bool,
    'frame_id'         : int,
    'timestamp'        : float
  }

This is a pure-perception module — no navigation / decision logic here.
"""

import time
import warnings
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, List, Optional

warnings.filterwarnings("ignore")


class PerceptionModule:
    """
    Multi-modal perception for a single RGB frame.

    Unchanged API from old pipeline — only the *source* of frames changes
    (live webcam instead of pre-saved images in a database folder).
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        device: str = "cpu",
        blip_model_name: str = "Salesforce/blip-image-captioning-base",
        yolo_model_name: str = "yolov8n.pt",
        midas_variant: str = "DPT_Hybrid",
        yolo_conf_threshold: float = 0.25,
    ):
        self.device = torch.device(
            device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self.yolo_conf = yolo_conf_threshold
        print(f"[PerceptionModule] device={self.device}")

        self._load_blip(blip_model_name)
        self._load_yolo(yolo_model_name)
        self._load_midas(midas_variant)
        print("[PerceptionModule] all models loaded\n")

    # ---------------------------------------------------------------- loaders

    def _load_blip(self, name: str):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print(f"  loading BLIP: {name}")
        self.blip_processor = BlipProcessor.from_pretrained(name)
        self.blip_model = (
            BlipForConditionalGeneration.from_pretrained(name)
            .to(self.device)
            .eval()
        )
        print("  BLIP ready")

    def _load_yolo(self, name: str):
        from ultralytics import YOLO
        print(f"  loading YOLO: {name}")
        self.yolo_model = YOLO(name)
        print("  YOLO ready")

    def _load_midas(self, variant: str):
        print(f"  loading MiDaS: {variant}")
        self.midas_model = (
            torch.hub.load("intel-isl/MiDaS", variant, verbose=False)
            .to(self.device)
            .eval()
        )
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
        if variant in ("DPT_Large", "DPT_Hybrid"):
            self.midas_transform = transforms.dpt_transform
        else:
            self.midas_transform = transforms.small_transform
        print("  MiDaS ready")

    # --------------------------------------------------------- individual runs

    def generate_caption(self, image: np.ndarray) -> str:
        try:
            pil = Image.fromarray(image.astype(np.uint8))
            inputs = self.blip_processor(images=pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_new_tokens=50)
            return self.blip_processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            return f"caption error: {e}"

    def get_caption_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """768-D BLIP visual embedding — used for Semantic Alignment Score."""
        try:
            pil = Image.fromarray(image.astype(np.uint8))
            inputs = self.blip_processor(images=pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                vision_out = self.blip_model.vision_model(
                    pixel_values=inputs["pixel_values"]
                )
                feat = vision_out.last_hidden_state          # (1, seq, 768)
                emb = torch.mean(feat, dim=1).squeeze()     # (768,)
            return emb.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"  [embed] {e}")
            return None

    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            results = self.yolo_model(image, conf=self.yolo_conf, verbose=False)
            objs = []
            for box in results[0].boxes:
                label = self.yolo_model.names[int(box.cls)]
                conf  = float(box.conf)
                bbox  = box.xyxy[0].tolist()
                cx    = (bbox[0] + bbox[2]) / 2
                cy    = (bbox[1] + bbox[3]) / 2
                objs.append({
                    "name":       label,
                    "confidence": conf,
                    "bbox":       bbox,
                    "center":     [cx, cy],
                })
            return objs
        except Exception as e:
            print(f"  [yolo] {e}")
            return []

    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            batch = self.midas_transform(image).to(self.device)
            with torch.no_grad():
                pred = self.midas_model(batch)
            d = pred.squeeze().cpu().numpy()
            return (d - d.min()) / (d.max() - d.min() + 1e-8)
        except Exception as e:
            print(f"  [midas] {e}")
            return None

    # --------------------------------------------------------- spatial helpers

    @staticmethod
    def _direction(cx: float, w: int) -> str:
        r = cx / w
        return "left" if r < 0.33 else ("center" if r < 0.67 else "right")

    @staticmethod
    def _range_from_depth(bbox: List[float], depth: Optional[np.ndarray]) -> str:
        if depth is None:
            return "unknown"
        x1, y1, x2, y2 = (int(c) for c in bbox)
        h, w = depth.shape
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        region = depth[y1:y2, x1:x2]
        if region.size == 0:
            return "unknown"
        d = float(np.mean(region))
        return "near" if d > 0.7 else ("mid" if d > 0.3 else "far")

    # ------------------------------------------------------------ main method

    def process_scene(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        compute_embedding: bool = False,
    ) -> Dict[str, Any]:
        """
        Full perception pipeline on one RGB frame.

        Args:
            image            : RGB numpy array (H, W, 3)
            frame_id         : monotonic frame counter from the live feed
            compute_embedding: set True to also return 768-D BLIP embedding
                               (slower; only needed for SAS metric)

        Returns:
            Same schema as old pipeline's process_scene() plus
            'frame_id' and 'timestamp' for live-feed bookkeeping.
        """
        h, w = image.shape[:2]
        t0 = time.time()

        caption    = self.generate_caption(image)
        objects    = self.detect_objects(image)
        depth      = self.estimate_depth(image)
        embedding  = self.get_caption_embedding(image) if compute_embedding else None

        # enrich objects with spatial info
        for obj in objects:
            obj["direction"] = self._direction(obj["center"][0], w)
            obj["range"]     = self._range_from_depth(obj["bbox"], depth)

        return {
            "caption":           caption,
            "caption_embedding": embedding,
            "objects":           objects,
            "depth":             depth,
            "num_objects":       len(objects),
            "has_depth":         depth is not None,
            "frame_id":          frame_id,
            "timestamp":         time.time(),
            "inference_ms":      round((time.time() - t0) * 1000, 1),
        }
