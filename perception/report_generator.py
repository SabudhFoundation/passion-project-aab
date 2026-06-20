"""
report_generator.py
--------------------
Generates the perception pipeline report.

Output format mirrors the bugdetection repo:
  results/   -> JSON + TXT metrics files
  figures/   -> matplotlib plots (.png)

Report contents:
  1. Per-frame perception log (caption, objects, depth stats, inference time)
  2. Coordination results (Cola VLM-1+VLM-2 scene summaries)
  3. Active perception trace (AP-VLM iterations, action sequence)
  4. Figures:
     - object_detection_frequency.png
     - inference_time_distribution.png
     - confidence_over_frames.png
     - action_sequence.png
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class ReportGenerator:
    """
    Collects per-frame data during a live session and produces
    a full report (txt + json + png figures) at the end.
    """

    def __init__(
        self,
        results_dir: str = "results",
        figures_dir: str = "figures",
        session_name: str = "",
    ):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.session_name = session_name or time.strftime("%Y%m%d_%H%M%S")

        # Accumulated data
        self._perception_log: List[Dict[str, Any]] = []
        self._coord_log:      List[Dict[str, Any]] = []
        self._action_log:     List[Dict[str, Any]] = []
        self._episode_summary: Optional[Dict[str, Any]] = None

        print(f"[ReportGenerator] session={self.session_name}")
        print(f"  results → {self.results_dir}")
        print(f"  figures → {self.figures_dir}")

    # ── data collection ───────────────────────────────────────────────────────

    def log_perception(self, perception: Dict[str, Any]):
        """Add one frame's perception output to the log."""
        entry = {
            "frame_id":     perception.get("frame_id", -1),
            "timestamp":    perception.get("timestamp", 0.0),
            "caption":      perception.get("caption", ""),
            "num_objects":  perception.get("num_objects", 0),
            "objects":      [
                {k: v for k, v in o.items() if k != "bbox"}   # drop bbox for readability
                for o in perception.get("objects", [])
            ],
            "has_depth":    perception.get("has_depth", False),
            "inference_ms": perception.get("inference_ms", 0.0),
        }
        self._perception_log.append(entry)

    def log_coordination(self, coord: Dict[str, Any]):
        """Add one frame's coordination result to the log."""
        self._coord_log.append({
            "frame_id":        coord.get("frame_id", -1),
            "scene_summary":   coord.get("scene_summary", ""),
            "is_conclusive":   coord.get("is_conclusive", False),
            "confidence":      coord.get("confidence", 0.0),
            "action_hint":     coord.get("action_hint", ""),
            "relevant_objects": coord.get("relevant_objects", []),
        })

    def log_action(self, action_signal: Dict[str, Any]):
        """Add one AP-VLM action step to the log."""
        self._action_log.append({
            "iteration":    action_signal.get("iteration", 0),
            "frame_id":     action_signal.get("frame_id", -1),
            "action":       action_signal.get("action", ""),
            "confidence":   action_signal.get("confidence", 0.0),
            "terminated":   action_signal.get("terminated", False),
        })

    def set_episode_summary(self, summary: Dict[str, Any]):
        self._episode_summary = summary

    # ── figure helpers ────────────────────────────────────────────────────────

    def _fig_object_frequency(self):
        """Bar chart: most frequently detected objects."""
        counter: Dict[str, int] = {}
        for entry in self._perception_log:
            for obj in entry.get("objects", []):
                name = obj.get("name", "unknown")
                counter[name] = counter.get(name, 0) + 1

        if not counter:
            return

        names  = sorted(counter, key=counter.get, reverse=True)[:15]
        counts = [counter[n] for n in names]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, counts, color="#4A90D9", edgecolor="white")
        ax.set_xlabel("Object Class")
        ax.set_ylabel("Detection Count")
        ax.set_title("Object Detection Frequency (Live Feed)")
        ax.set_xticklabels(names, rotation=45, ha="right")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(cnt), ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        path = self.figures_dir / f"{self.session_name}_object_detection_frequency.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  saved: {path}")

    def _fig_inference_time(self):
        """Histogram of per-frame inference times (ms)."""
        times = [e.get("inference_ms", 0) for e in self._perception_log]
        if not times:
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(times, bins=20, color="#E87D4A", edgecolor="white")
        ax.axvline(np.mean(times), color="black", linestyle="--",
                   label=f"mean={np.mean(times):.0f} ms")
        ax.set_xlabel("Inference Time (ms)")
        ax.set_ylabel("Frame Count")
        ax.set_title("Perception Inference Time Distribution")
        ax.legend()
        plt.tight_layout()
        path = self.figures_dir / f"{self.session_name}_inference_time_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  saved: {path}")

    def _fig_confidence_over_frames(self):
        """Line plot: Cola coordinator confidence across frames."""
        if not self._coord_log:
            return

        frames = [e["frame_id"]  for e in self._coord_log]
        confs  = [e["confidence"] for e in self._coord_log]
        conclusive = [e["is_conclusive"] for e in self._coord_log]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames, confs, color="#4A90D9", linewidth=1.5, label="confidence")
        # Mark conclusive frames
        cx = [frames[i] for i, c in enumerate(conclusive) if c]
        cy = [confs[i]   for i, c in enumerate(conclusive) if c]
        ax.scatter(cx, cy, color="green", zorder=5, s=50, label="conclusive")
        ax.axhline(0.6, color="red", linestyle="--", linewidth=0.8, label="threshold=0.6")
        ax.set_xlabel("Frame ID")
        ax.set_ylabel("Confidence")
        ax.set_title("Cola Coordinator Confidence Over Frames (AP-VLM)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()
        path = self.figures_dir / f"{self.session_name}_confidence_over_frames.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  saved: {path}")

    def _fig_action_sequence(self):
        """Horizontal bar chart: action sequence over AP-VLM iterations."""
        if not self._action_log:
            return

        iterations = [e["iteration"] for e in self._action_log]
        actions    = [e["action"]    for e in self._action_log]
        action_set = sorted(set(actions))
        color_map  = {
            "move_forward": "#2ECC71",
            "turn_left":    "#3498DB",
            "turn_right":   "#9B59B6",
            "look_up":      "#F39C12",
            "look_down":    "#E67E22",
            "stop":         "#E74C3C",
            "explore_more": "#95A5A6",
        }

        fig, ax = plt.subplots(figsize=(10, 4))
        for i, (it, act) in enumerate(zip(iterations, actions)):
            color = color_map.get(act, "#BDC3C7")
            ax.barh(it, 1, left=i, color=color, edgecolor="white", height=0.8)

        patches = [mpatches.Patch(color=color_map.get(a, "#BDC3C7"), label=a)
                   for a in action_set]
        ax.legend(handles=patches, loc="upper right", fontsize=8)
        ax.set_xlabel("Step")
        ax.set_ylabel("AP-VLM Iteration")
        ax.set_title("Active Perception Action Sequence")
        plt.tight_layout()
        path = self.figures_dir / f"{self.session_name}_action_sequence.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  saved: {path}")

    # ── main save methods ─────────────────────────────────────────────────────

    def _save_json(self):
        data = {
            "session":         self.session_name,
            "total_frames":    len(self._perception_log),
            "perception_log":  self._perception_log,
            "coordination_log": self._coord_log,
            "action_log":      self._action_log,
            "episode_summary": self._episode_summary,
        }
        path = self.results_dir / f"{self.session_name}_full_log.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  saved: {path}")
        return path

    def _save_txt_report(self):
        """Human-readable text report — mirrors bugdetection results/*.txt format."""
        lines = []
        lines.append("=" * 70)
        lines.append("VLN LIVE PERCEPTION PIPELINE — SESSION REPORT")
        lines.append(f"Session : {self.session_name}")
        lines.append(f"Frames  : {len(self._perception_log)}")
        lines.append("=" * 70)

        # --- Perception Summary ---
        lines.append("\n[1] PERCEPTION SUMMARY (BLIP + YOLO + MiDaS)")
        lines.append("-" * 50)
        if self._perception_log:
            inf_times = [e["inference_ms"] for e in self._perception_log]
            obj_counts = [e["num_objects"]  for e in self._perception_log]
            lines.append(f"  Mean inference time : {np.mean(inf_times):.1f} ms")
            lines.append(f"  Max  inference time : {np.max(inf_times):.1f} ms")
            lines.append(f"  Mean objects/frame  : {np.mean(obj_counts):.2f}")
            lines.append(f"  Frames with depth   : "
                         f"{sum(1 for e in self._perception_log if e['has_depth'])}")

            # Object frequency
            counter: Dict[str, int] = {}
            for entry in self._perception_log:
                for obj in entry.get("objects", []):
                    n = obj.get("name", "?")
                    counter[n] = counter.get(n, 0) + 1
            top = sorted(counter, key=counter.get, reverse=True)[:10]
            lines.append(f"  Top objects         : {', '.join(f'{n}({counter[n]})' for n in top)}")

        # --- Coordination Summary ---
        lines.append("\n[2] COLA COORDINATION SUMMARY (VLM-1 + VLM-2 → LLM)")
        lines.append("-" * 50)
        if self._coord_log:
            confs = [e["confidence"] for e in self._coord_log]
            n_conclusive = sum(1 for e in self._coord_log if e["is_conclusive"])
            lines.append(f"  Mean confidence     : {np.mean(confs):.3f}")
            lines.append(f"  Conclusive frames   : {n_conclusive} / {len(self._coord_log)}")
            lines.append(f"  Last scene summary  : {self._coord_log[-1].get('scene_summary', '')}")
            action_counts: Dict[str, int] = {}
            for e in self._coord_log:
                h = e.get("action_hint", "?")
                action_counts[h] = action_counts.get(h, 0) + 1
            lines.append(f"  Action hints        : {action_counts}")

        # --- AP-VLM Active Perception Trace ---
        lines.append("\n[3] AP-VLM ACTIVE PERCEPTION TRACE")
        lines.append("-" * 50)
        for step in self._action_log:
            terminated_str = " [TERMINATED]" if step["terminated"] else ""
            lines.append(
                f"  iter={step['iteration']:2d} | frame={step['frame_id']:4d} | "
                f"action={step['action']:14s} | conf={step['confidence']:.2f}"
                + terminated_str
            )

        # --- Episode Summary ---
        if self._episode_summary:
            lines.append("\n[4] EPISODE SUMMARY")
            lines.append("-" * 50)
            s = self._episode_summary
            lines.append(f"  Total iterations   : {s.get('total_iterations', 0)}")
            lines.append(f"  Final answer       : {s.get('final_answer', 'N/A')}")
            lines.append(f"  Unique objects seen: {s.get('unique_objects_seen', [])}")
            lines.append(f"  Terminated         : {s.get('terminated', False)}")

        lines.append("\n" + "=" * 70)
        lines.append("FIGURES GENERATED:")
        for png in sorted(self.figures_dir.glob(f"{self.session_name}_*.png")):
            lines.append(f"  {png}")
        lines.append("=" * 70)

        path = self.results_dir / f"{self.session_name}_report.txt"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"  saved: {path}")
        return path

    # ── public entry point ────────────────────────────────────────────────────

    def generate(self) -> Dict[str, str]:
        """
        Generate all outputs.

        Returns:
            Dict of {label: path} for all saved files.
        """
        print("\n[ReportGenerator] generating report ...")
        self._fig_object_frequency()
        self._fig_inference_time()
        self._fig_confidence_over_frames()
        self._fig_action_sequence()
        json_path = self._save_json()
        txt_path  = self._save_txt_report()
        print("[ReportGenerator] done.\n")
        return {
            "json":   str(json_path),
            "report": str(txt_path),
            "figures_dir": str(self.figures_dir),
        }
