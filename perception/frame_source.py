"""
frame_source.py
---------------
Abstraction layer for live frame acquisition.

Swap `source` in config.yaml to change input:
  webcam     -> cv2.VideoCapture(0)   [current, no robot needed]
  ros_topic  -> ROS2 /camera/image_raw [future, on robot]
  video_file -> path to mp4 / bag replay

All sources expose the same interface:
  .read()  -> (success: bool, frame: np.ndarray [BGR])
  .release()
  .is_opened() -> bool
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class WebcamSource:
    """
    Live webcam feed via OpenCV.
    Default source — works on any laptop, no robot required.
    """

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam device {device_id}")
        print(f"[FrameSource] Webcam {device_id} opened ({width}x{height})")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def release(self):
        self.cap.release()

    def is_opened(self) -> bool:
        return self.cap.isOpened()


class VideoFileSource:
    """
    Replay a video file — useful for demos without webcam / robot.
    Loops the file indefinitely so the pipeline never starves for frames.
    """

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {path}")
        print(f"[FrameSource] Video file: {path}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if not ret:                     # loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()

    def is_opened(self) -> bool:
        return self.cap.isOpened()


class ROSTopicSource:
    """
    ROS2 camera topic source — for future robot integration.

    NOT active now (no robot / ROS needed for development).
    Swap config.yaml  source: webcam  →  source: ros_topic
    and install: pip install rclpy cv_bridge

    The rest of the pipeline (perception modules, coordinator, report)
    stays identical — only this class changes.
    """

    def __init__(self, topic: str = "/camera/image_raw"):
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image as ROSImage
            from cv_bridge import CvBridge
        except ImportError:
            raise ImportError(
                "rclpy / cv_bridge not found. "
                "Install ROS2 and run: pip install cv_bridge\n"
                "For now use source: webcam in config.yaml"
            )

        self._bridge = CvBridge()
        self._latest_frame: Optional[np.ndarray] = None
        self._topic = topic

        rclpy.init()
        self._node = rclpy.create_node("vln_perception_node")
        self._sub = self._node.create_subscription(
            ROSImage, topic, self._callback, 10
        )
        print(f"[FrameSource] ROS2 topic: {topic}")

    def _callback(self, msg):
        import rclpy
        self._latest_frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        import rclpy
        rclpy.spin_once(self._node, timeout_sec=0.1)
        if self._latest_frame is not None:
            return True, self._latest_frame.copy()
        return False, None

    def release(self):
        import rclpy
        self._node.destroy_node()
        rclpy.shutdown()

    def is_opened(self) -> bool:
        return True


def build_source(cfg: dict):
    """
    Factory: build the right FrameSource from config dict.

    cfg keys:
      source: webcam | ros_topic | video_file
      device_id: 0          (webcam)
      width / height: 640/480
      video_path: path      (video_file)
      ros_topic: /camera/image_raw
    """
    src = cfg.get("source", "webcam")

    if src == "webcam":
        return WebcamSource(
            device_id=cfg.get("device_id", 0),
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
        )
    elif src == "video_file":
        return VideoFileSource(path=cfg["video_path"])
    elif src == "ros_topic":
        return ROSTopicSource(topic=cfg.get("ros_topic", "/camera/image_raw"))
    else:
        raise ValueError(f"Unknown source type: {src}")
