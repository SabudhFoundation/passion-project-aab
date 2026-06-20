"""
active_perception_policy.py
-----------------------------
AP-VLM active perception policy adapted for webcam / live feed.

Original AP-VLM (Sripada et al., 2024):
  - Robot with in-hand camera moves to new viewpoints
  - VLM suggests next grid vertex when current view is inconclusive
  - Terminates when VLM answers confidently or max iterations reached

Our adaptation (no robot arm yet):
  - "Viewpoints" = webcam orientation hints OR video frame skip
  - `is_conclusive` flag from ColaCoordinator triggers exploration signal
  - Outputs an `action_signal` dict the main loop / future robot can act on
  - Maintains knowledge history (AP-VLM context κ) across iterations

When integrated with a real robot later:
  - Swap `_suggest_viewpoint_adjustment()` with actual Nav2 / arm commands
  - The rest of the policy logic stays identical
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


# ─── AP-VLM knowledge accumulator (context κ in the paper) ───────────────────
@dataclass
class PerceptionKnowledge:
    """
    Stores cumulative knowledge across active perception iterations.
    Mirrors κ (accumulated context) from AP-VLM framework.
    """
    visited_frame_ids: List[int]          = field(default_factory=list)
    seen_objects:      List[str]          = field(default_factory=list)
    scene_summaries:   List[str]          = field(default_factory=list)
    iteration:         int                = 0
    is_terminated:     bool               = False
    final_answer:      Optional[str]      = None

    def update(self, coord_result: Dict[str, Any]):
        self.visited_frame_ids.append(coord_result.get("frame_id", -1))
        self.seen_objects.extend(coord_result.get("relevant_objects", []))
        self.scene_summaries.append(coord_result.get("scene_summary", ""))
        self.iteration += 1

    def to_context_string(self) -> str:
        """Serialise knowledge for LLM context (mirrors κ in AP-VLM)."""
        lines = [
            f"Iteration: {self.iteration}",
            f"Frames seen: {self.visited_frame_ids[-5:]}",
            f"Objects accumulated: {list(set(self.seen_objects))[:10]}",
            f"Last summary: {self.scene_summaries[-1] if self.scene_summaries else 'none'}",
        ]
        return " | ".join(lines)


# ─── Active Perception Policy ─────────────────────────────────────────────────
class ActivePerceptionPolicy:
    """
    AP-VLM active perception policy for live webcam feed.

    Decision loop (mirrors AP-VLM iterative exploration):
      1. Receive coordination result from ColaCoordinator
      2. If `is_conclusive` → terminate, output final answer
      3. If not → suggest viewpoint adjustment + continue
      4. If max_iterations reached → terminate with best answer

    Action signals (future robot integration):
      move_forward / turn_left / turn_right / look_up / look_down / stop
      These map directly to robot commands when integrated with Nav2 / ROS2.
    """

    # AP-VLM action space (mirrors the paper's 8 action spaces simplified)
    ACTION_SPACE = [
        "move_forward",
        "turn_left",
        "turn_right",
        "look_up",
        "look_down",
        "stop",
        "explore_more",
    ]

    def __init__(
        self,
        max_iterations: int = 10,         # AP-VLM default: 10
        confidence_threshold: float = 0.6, # terminate if confidence >= this
    ):
        self.max_iterations      = max_iterations
        self.confidence_threshold = confidence_threshold
        self.knowledge           = PerceptionKnowledge()
        print(
            f"[ActivePerceptionPolicy] max_iter={max_iterations}, "
            f"conf_thresh={confidence_threshold}"
        )

    def reset(self):
        """Reset policy for a new navigation episode."""
        self.knowledge = PerceptionKnowledge()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _suggest_viewpoint_adjustment(
        self,
        coord_result: Dict[str, Any],
        instruction: str,
    ) -> str:
        """
        Heuristic viewpoint suggestion (AP-VLM App → grid vertex).

        On robot: replace this with LLM-suggested grid vertex → robot command.
        Now: returns a camera orientation hint for webcam or frame navigation.
        """
        hint = coord_result.get("action_hint", "explore_more")

        # Map Cola coordinator hint → AP-VLM action
        mapping = {
            "move_forward": "move_forward",
            "turn_left":    "turn_left",
            "turn_right":   "turn_right",
            "stop":         "stop",
            "explore_more": "turn_left",   # default exploration: rotate
        }
        return mapping.get(hint, "turn_left")

    def _build_action_signal(
        self,
        action: str,
        coord_result: Dict[str, Any],
        terminated: bool,
    ) -> Dict[str, Any]:
        """Package action and metadata for the main pipeline / robot."""
        return {
            "action":           action,
            "terminated":       terminated,
            "iteration":        self.knowledge.iteration,
            "confidence":       coord_result.get("confidence", 0.0),
            "is_conclusive":    coord_result.get("is_conclusive", False),
            "scene_summary":    coord_result.get("scene_summary", ""),
            "relevant_objects": coord_result.get("relevant_objects", []),
            "reasoning":        coord_result.get("reasoning", ""),
            "knowledge_context": self.knowledge.to_context_string(),
            "frame_id":         coord_result.get("frame_id", -1),
            "timestamp":        coord_result.get("timestamp", time.time()),
        }

    # ── main method ───────────────────────────────────────────────────────────

    def step(
        self,
        coord_result: Dict[str, Any],
        instruction: str,
    ) -> Dict[str, Any]:
        """
        One AP-VLM policy step.

        Args:
            coord_result : output of ColaCoordinator.coordinate()
            instruction  : current navigation instruction

        Returns:
            action_signal dict with action + termination flag + metadata
        """
        # Update accumulated knowledge (κ)
        self.knowledge.update(coord_result)

        confidence   = float(coord_result.get("confidence", 0.0))
        is_conclusive = bool(coord_result.get("is_conclusive", False))

        # Termination condition 1: conclusive + confident (AP-VLM criterion)
        if is_conclusive and confidence >= self.confidence_threshold:
            self.knowledge.is_terminated = True
            self.knowledge.final_answer  = coord_result.get("scene_summary", "")
            return self._build_action_signal("stop", coord_result, terminated=True)

        # Termination condition 2: max iterations reached
        if self.knowledge.iteration >= self.max_iterations:
            self.knowledge.is_terminated = True
            self.knowledge.final_answer  = (
                self.knowledge.scene_summaries[-1]
                if self.knowledge.scene_summaries else "Max iterations reached."
            )
            return self._build_action_signal("stop", coord_result, terminated=True)

        # Continue exploring: suggest next viewpoint
        action = self._suggest_viewpoint_adjustment(coord_result, instruction)
        return self._build_action_signal(action, coord_result, terminated=False)

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return full episode summary after termination."""
        return {
            "total_iterations":   self.knowledge.iteration,
            "final_answer":       self.knowledge.final_answer,
            "unique_objects_seen": list(set(self.knowledge.seen_objects)),
            "frames_processed":   self.knowledge.visited_frame_ids,
            "terminated":         self.knowledge.is_terminated,
        }
