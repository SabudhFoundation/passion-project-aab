"""
main.py
-------
VLN Live Perception Pipeline — entry point.

Combines:
  - AP-VLM (Sripada et al., 2024): active perception + iterative exploration
  - Cola   (Chen et al., NeurIPS 2023): dual-VLM LLM coordination

Usage:
  python main.py                        # uses config.yaml defaults (webcam)
  python main.py --source video_file --video_path demo.mp4
  python main.py --source ros_topic     # when robot is available
  python main.py --instruction "Go to the kitchen"
  python main.py --max_frames 50 --no_coordinator   # perception only

Output (same layout as sukhjitsehra/bugdetection repo):
  results/<session>_report.txt
  results/<session>_full_log.json
  figures/<session>_object_detection_frequency.png
  figures/<session>_inference_time_distribution.png
  figures/<session>_confidence_over_frames.png
  figures/<session>_action_sequence.png
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from perception import (
    build_source,
    PerceptionModule,
    ColaCoordinator,
    ActivePerceptionPolicy,
    ReportGenerator,
)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="VLN Live Perception Pipeline")
    p.add_argument("--config",       default="config.yaml")
    p.add_argument("--source",       default=None,  help="webcam|video_file|ros_topic")
    p.add_argument("--video_path",   default=None)
    p.add_argument("--instruction",  default=None)
    p.add_argument("--max_frames",   type=int, default=None)
    p.add_argument("--device",       default=None,  help="cpu|cuda")
    p.add_argument("--backend",      default=None,  help="hf|openai")
    p.add_argument("--no_coordinator", action="store_true",
                   help="run perception only (no LLM)")
    p.add_argument("--show_preview", action="store_true",
                   help="show live cv2 window")
    return p.parse_args()


# ── Config loader ─────────────────────────────────────────────────────────────
def load_config(path: str, args) -> dict:
    cfg = {}
    if Path(path).exists():
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}

    # CLI overrides
    if args.source:
        cfg.setdefault("input", {})["source"] = args.source
    if args.video_path:
        cfg.setdefault("input", {})["video_path"] = args.video_path
    if args.instruction:
        cfg.setdefault("navigation", {})["instruction"] = args.instruction
    if args.max_frames is not None:
        cfg.setdefault("pipeline", {})["max_frames"] = args.max_frames
    if args.device:
        cfg.setdefault("perception", {})["device"] = args.device
    if args.backend:
        cfg.setdefault("coordinator", {})["backend"] = args.backend

    # Defaults
    cfg.setdefault("input",            {})
    cfg.setdefault("perception",       {})
    cfg.setdefault("coordinator",      {})
    cfg.setdefault("active_perception",{})
    cfg.setdefault("navigation",       {})
    cfg.setdefault("pipeline",         {})
    cfg.setdefault("output",           {})

    cfg["input"].setdefault("source",    "webcam")
    cfg["input"].setdefault("device_id", 0)
    cfg["input"].setdefault("width",     640)
    cfg["input"].setdefault("height",    480)

    cfg["perception"].setdefault("device",              "cpu")
    cfg["perception"].setdefault("blip_model",          "Salesforce/blip-image-captioning-base")
    cfg["perception"].setdefault("yolo_model",          "yolov8n.pt")
    cfg["perception"].setdefault("midas_variant",       "DPT_Hybrid")
    cfg["perception"].setdefault("yolo_conf_threshold", 0.25)
    cfg["perception"].setdefault("compute_embedding",   False)

    cfg["coordinator"].setdefault("backend",       "hf")
    cfg["coordinator"].setdefault("hf_model",      "google/flan-t5-base")
    cfg["coordinator"].setdefault("openai_model",  "gpt-4o-mini")
    cfg["coordinator"].setdefault("openai_api_key","")

    cfg["active_perception"].setdefault("max_iterations",       10)
    cfg["active_perception"].setdefault("confidence_threshold", 0.6)

    cfg["navigation"].setdefault("instruction",
        "Navigate to the exit door and avoid obstacles.")

    cfg["pipeline"].setdefault("frame_interval", 5)
    cfg["pipeline"].setdefault("max_frames",     100)
    cfg["pipeline"].setdefault("show_preview",   False)

    cfg["output"].setdefault("results_dir",  "results")
    cfg["output"].setdefault("figures_dir",  "figures")
    cfg["output"].setdefault("session_name", "")

    if args.show_preview:
        cfg["pipeline"]["show_preview"] = True

    return cfg


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(cfg: dict, use_coordinator: bool = True):

    instruction   = cfg["navigation"]["instruction"]
    frame_interval = cfg["pipeline"]["frame_interval"]
    max_frames    = cfg["pipeline"]["max_frames"]
    show_preview  = cfg["pipeline"]["show_preview"]

    print("\n" + "=" * 60)
    print("  VLN LIVE PERCEPTION PIPELINE")
    print("  Papers: AP-VLM (2024) + Cola (NeurIPS 2023)")
    print("=" * 60)
    print(f"  source      : {cfg['input']['source']}")
    print(f"  instruction : {instruction}")
    print(f"  coordinator : {'enabled (' + cfg['coordinator']['backend'] + ')' if use_coordinator else 'disabled'}")
    print(f"  max_frames  : {max_frames}")
    print("=" * 60 + "\n")

    # ── Build components ──────────────────────────────────────────────────────
    frame_source = build_source(cfg["input"])

    perception = PerceptionModule(
        device              = cfg["perception"]["device"],
        blip_model_name     = cfg["perception"]["blip_model"],
        yolo_model_name     = cfg["perception"]["yolo_model"],
        midas_variant       = cfg["perception"]["midas_variant"],
        yolo_conf_threshold = cfg["perception"]["yolo_conf_threshold"],
    )

    coordinator = None
    if use_coordinator:
        api_key = (
            cfg["coordinator"]["openai_api_key"]
            or os.environ.get("OPENAI_API_KEY", "")
        )
        coordinator = ColaCoordinator(
            backend       = cfg["coordinator"]["backend"],
            hf_model      = cfg["coordinator"]["hf_model"],
            openai_model  = cfg["coordinator"]["openai_model"],
            openai_api_key= api_key,
            device        = cfg["perception"]["device"],
        )

    policy = ActivePerceptionPolicy(
        max_iterations      = cfg["active_perception"]["max_iterations"],
        confidence_threshold= cfg["active_perception"]["confidence_threshold"],
    )

    report = ReportGenerator(
        results_dir  = cfg["output"]["results_dir"],
        figures_dir  = cfg["output"]["figures_dir"],
        session_name = cfg["output"]["session_name"],
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_count   = 0
    process_count = 0
    terminated    = False

    print("Starting live feed... (Ctrl+C to stop)\n")
    try:
        while frame_source.is_opened():
            ret, bgr_frame = frame_source.read()
            if not ret:
                print("[main] frame read failed, retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1

            # Only process every N-th frame
            if frame_count % frame_interval != 0:
                continue

            process_count += 1

            # BGR → RGB for models
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            # ── Perception (BLIP + YOLO + MiDaS) ────────────────────────────
            perc = perception.process_scene(
                rgb_frame,
                frame_id          = frame_count,
                compute_embedding = cfg["perception"]["compute_embedding"],
            )
            report.log_perception(perc)

            print(f"[frame {frame_count:4d}] "
                  f"caption='{perc['caption'][:60]}...' | "
                  f"objects={perc['num_objects']} | "
                  f"{perc['inference_ms']} ms")

            # ── Cola Coordination (VLM-1 + VLM-2 → LLM) ────────────────────
            if coordinator:
                coord = coordinator.coordinate(perc, instruction)
                report.log_coordination(coord)

                # ── AP-VLM Policy step ───────────────────────────────────────
                action_signal = policy.step(coord, instruction)
                report.log_action(action_signal)

                print(f"          [cola] summary='{coord.get('scene_summary','')[:50]}' | "
                      f"conf={coord.get('confidence', 0):.2f} | "
                      f"conclusive={coord.get('is_conclusive', False)}")
                print(f"          [apvlm] iter={action_signal['iteration']} "
                      f"action={action_signal['action']} "
                      f"terminated={action_signal['terminated']}")

                if action_signal["terminated"]:
                    terminated = True
                    print("\n[AP-VLM] Episode terminated — perception conclusive.")
                    break

            # ── Preview window ────────────────────────────────────────────────
            if show_preview:
                # Annotate frame
                ann = bgr_frame.copy()
                for obj in perc.get("objects", []):
                    bbox = obj.get("bbox", [])
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = (int(v) for v in bbox)
                        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            ann,
                            f"{obj['name']} {obj['range']}",
                            (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1,
                        )
                cv2.putText(
                    ann,
                    perc["caption"][:60],
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 0), 1,
                )
                cv2.imshow("VLN Live Perception", ann)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[main] quit key pressed")
                    break

            # ── Max frames check ──────────────────────────────────────────────
            if max_frames > 0 and process_count >= max_frames:
                print(f"[main] reached max_frames={max_frames}")
                break

    except KeyboardInterrupt:
        print("\n[main] interrupted by user")

    finally:
        frame_source.release()
        if show_preview:
            cv2.destroyAllWindows()

    # ── Generate report ───────────────────────────────────────────────────────
    if coordinator:
        report.set_episode_summary(policy.get_episode_summary())

    output_paths = report.generate()

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Frames captured  : {frame_count}")
    print(f"  Frames processed : {process_count}")
    print(f"  AP-VLM terminated: {terminated}")
    print(f"  Report  → {output_paths['report']}")
    print(f"  JSON    → {output_paths['json']}")
    print(f"  Figures → {output_paths['figures_dir']}/")
    print("=" * 60 + "\n")

    return output_paths


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config, args)
    run(cfg, use_coordinator=not args.no_coordinator)
