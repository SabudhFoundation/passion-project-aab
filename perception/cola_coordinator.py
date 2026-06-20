"""
cola_coordinator.py
--------------------
Implements the Cola paradigm (NeurIPS 2023) adapted for VLN active perception.

Original Cola: LLM coordinates two VLMs (OFA + BLIP) for visual question answering.
Our adaptation: LLM coordinates the BLIP caption (VLM-1) and YOLO object list (VLM-2)
to reason about the scene and produce a navigation-relevant scene understanding.

Inspired by AP-VLM (active perception paper): the coordinator also decides
whether the current frame is *conclusive* enough or if the robot/camera
should gather more information (active perception signal).

Cola prompt template (Table 1 from paper):
  VLM-1 (BLIP) description: <caption>
  VLM-2 (YOLO) description: <object list as natural language>
  Q: <navigation instruction>
  VLM-1 answer: <what BLIP sees>
  VLM-2 answer: <what YOLO sees>
  A:

Input : perception output dict from perception_module.process_scene()
        + current navigation instruction (text)
Output: coordination_result dict
  {
    'scene_summary'  : str,   # LLM-coordinated scene description
    'relevant_objects': [...], # objects relevant to instruction
    'is_conclusive'  : bool,  # AP-VLM signal: enough info to navigate?
    'confidence'     : float, # 0-1
    'action_hint'    : str,   # "move_forward" / "turn_left" / "explore_more"
    'reasoning'      : str    # LLM reasoning chain
  }
"""

import json
import re
from typing import Dict, Any, List, Optional


# ── Cola prompt template (adapted from Table 1, NeurIPS 2023) ────────────────
COLA_PROMPT_TEMPLATE = """You are an intelligent robot perception coordinator for Vision-Language Navigation.

You have two vision modules providing clues about the current scene:
VLM-1 (BLIP caption): {blip_caption}
VLM-2 (YOLO objects): {yolo_description}

Navigation instruction: {instruction}

VLM-1 answer (scene understanding): {vlm1_answer}
VLM-2 answer (objects detected): {vlm2_answer}

Based on both VLMs, answer the following in JSON:
{{
  "scene_summary": "<2-sentence coordinated scene description>",
  "relevant_objects": ["<objects relevant to the instruction>"],
  "is_conclusive": <true if current view is enough to take a navigation action, false if more exploration needed>,
  "confidence": <0.0-1.0>,
  "action_hint": "<one of: move_forward, turn_left, turn_right, stop, explore_more>",
  "reasoning": "<1-2 sentences explaining your coordination decision>"
}}

Respond ONLY with the JSON object, no preamble."""


# ── VLM-2 text converter (YOLO → natural language, mirrors Cola's VLM-2) ─────
def yolo_to_natural_language(objects: List[Dict[str, Any]]) -> str:
    """
    Convert YOLO detections to natural language description.
    This is VLM-2's 'answer' in the Cola framework.
    """
    if not objects:
        return "No objects detected in the current view."

    parts = []
    for obj in objects[:10]:   # cap at 10 for prompt length
        name = obj.get("name", "object")
        direction = obj.get("direction", "center")
        dist = obj.get("range", "unknown")
        conf = obj.get("confidence", 0.0)
        parts.append(f"a {name} ({dist}, {direction}, conf={conf:.2f})")

    return "Detected: " + ", ".join(parts) + "."


def extract_navigation_relevant(
    objects: List[Dict[str, Any]],
    instruction: str,
) -> List[str]:
    """
    Heuristic pre-filter: keep objects whose name appears in instruction.
    The LLM coordinator overrides this with its own reasoning.
    """
    inst_lower = instruction.lower()
    relevant = []
    for obj in objects:
        name = obj.get("name", "").lower()
        if any(word in inst_lower for word in name.split()):
            relevant.append(name)
    return relevant or [o.get("name", "") for o in objects[:3]]


# ── Main coordinator class ────────────────────────────────────────────────────
class ColaCoordinator:
    """
    Cola-style multi-VLM coordinator for VLN active perception.

    Uses an LLM (via Hugging Face pipeline or OpenAI API) to coordinate
    BLIP (VLM-1) and YOLO (VLM-2) outputs into a navigation decision.

    AP-VLM integration: adds `is_conclusive` flag — if False, the
    pipeline signals that the robot/camera should seek a better viewpoint
    before acting (mirrors AP-VLM's iterative exploration loop).
    """

    def __init__(
        self,
        backend: str = "hf",          # "hf" | "openai"
        hf_model: str = "google/flan-t5-base",   # small, runs on CPU
        openai_model: str = "gpt-4o-mini",
        openai_api_key: str = "",
        max_new_tokens: int = 512,
        device: str = "cpu",
    ):
        self.backend = backend
        self.max_new_tokens = max_new_tokens

        if backend == "hf":
            self._init_hf(hf_model, device)
        elif backend == "openai":
            self._init_openai(openai_model, openai_api_key)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'hf' or 'openai'.")

    # ── backend init ──────────────────────────────────────────────────────────

    def _init_hf(self, model_name: str, device: str):
        from transformers import pipeline as hf_pipeline
        print(f"[ColaCoordinator] HF model: {model_name} on {device}")
        self._pipe = hf_pipeline(
            "text2text-generation",
            model=model_name,
            device=0 if device == "cuda" else -1,
            max_new_tokens=self.max_new_tokens,
        )
        print("[ColaCoordinator] LLM rea
    # def _init_hf(self, model_name: str, device: str):
    #     from transformers import pipeline as hf_pipeline
    #     self._pipe = hf_pipeline(
    #         "text-generation",           # was "text2text-generation"
    #         model=model_name,
    #         device=0 if device == "cuda" else -1,
    #         max_new_tokens=self.max_new_tokens,
    #     )

    def _init_openai(self, model: str, api_key: str):
        try:
            from openai import OpenAI
            self._oai_client = OpenAI(api_key=api_key)
            self._oai_model = model
            print(f"[ColaCoordinator] OpenAI model: {model}")
        except ImportError:
            raise ImportError("pip install openai")

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "hf":
            out = self._pipe(prompt, do_sample=False)
            return out[0]["generated_text"]
        else:
            resp = self._oai_client.chat.completions.create(
                model=self._oai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.max_new_tokens,
            )
            return resp.choices[0].message.content

    # ── JSON parse ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract first JSON object from LLM output."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    # ── fallback (no LLM / parse failure) ────────────────────────────────────

    @staticmethod
    def _fallback(
        perception: Dict[str, Any],
        instruction: str,
    ) -> Dict[str, Any]:
        """Return a safe heuristic result when LLM unavailable."""
        objs = perception.get("objects", [])
        relevant = extract_navigation_relevant(objs, instruction)
        return {
            "scene_summary":   perception.get("caption", "Scene unavailable."),
            "relevant_objects": relevant,
            "is_conclusive":   len(objs) > 0,
            "confidence":      0.5,
            "action_hint":     "explore_more" if not objs else "move_forward",
            "reasoning":       "Fallback heuristic (LLM unavailable).",
        }

    # ── main coordination method ──────────────────────────────────────────────

    def coordinate(
        self,
        perception: Dict[str, Any],
        instruction: str,
    ) -> Dict[str, Any]:
        """
        Cola coordination: BLIP + YOLO → LLM → navigation decision.

        Args:
            perception : output of PerceptionModule.process_scene()
            instruction: current navigation instruction text

        Returns:
            coordination_result dict (see module docstring)
        """
        caption      = perception.get("caption", "")
        objects      = perception.get("objects", [])
        yolo_desc    = yolo_to_natural_language(objects)

        # Build Cola prompt (Table 1 from paper)
        vlm1_answer = f"The scene shows: {caption}"
        vlm2_answer = yolo_desc

        prompt = COLA_PROMPT_TEMPLATE.format(
            blip_caption  = caption,
            yolo_description = yolo_desc,
            instruction   = instruction,
            vlm1_answer   = vlm1_answer,
            vlm2_answer   = vlm2_answer,
        )

        try:
            raw = self._call_llm(prompt)
            result = self._parse_json(raw)
            if result is None:
                print("[ColaCoordinator] JSON parse failed, using fallback")
                result = self._fallback(perception, instruction)
        except Exception as e:
            print(f"[ColaCoordinator] LLM error: {e}")
            result = self._fallback(perception, instruction)

        # Ensure required keys exist
        result.setdefault("scene_summary",    caption)
        result.setdefault("relevant_objects", [])
        result.setdefault("is_conclusive",    False)
        result.setdefault("confidence",       0.0)
        result.setdefault("action_hint",      "explore_more")
        result.setdefault("reasoning",        "")

        # Attach frame metadata passthrough
        result["frame_id"]     = perception.get("frame_id", -1)
        result["timestamp"]    = perception.get("timestamp", 0.0)
        result["num_objects"]  = perception.get("num_objects", 0)

        return result
