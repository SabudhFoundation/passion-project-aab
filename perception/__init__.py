"""
perception/ — modular live perception pipeline for VLN.

Modules:
  frame_source           : webcam / video / ROS2 input abstraction
  perception_module      : BLIP + YOLO + MiDaS per-frame inference
  cola_coordinator       : Cola-style dual-VLM LLM coordination
  active_perception_policy: AP-VLM iterative exploration policy
  report_generator       : results/ + figures/ output
"""

from .frame_source import build_source, WebcamSource, VideoFileSource, ROSTopicSource
from .perception_module import PerceptionModule
from .cola_coordinator import ColaCoordinator
from .active_perception_policy import ActivePerceptionPolicy
from .report_generator import ReportGenerator

__all__ = [
    "build_source",
    "WebcamSource",
    "VideoFileSource",
    "ROSTopicSource",
    "PerceptionModule",
    "ColaCoordinator",
    "ActivePerceptionPolicy",
    "ReportGenerator",
]
