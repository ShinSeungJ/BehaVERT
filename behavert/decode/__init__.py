"""From videos or keypoints to behavior, embeddings, attention and synchronized videos.

>>> from behavert.decode import decode
>>> decode("oft-behavior", video="demo/video/oft.mp4", overlay=True)              # video -> pose -> behavior
>>> decode("calms21-task1", keypoints="session.csv", video="session.mp4", sync=True)
"""

from .analysis import Analysis, analyze
from .layout import KeypointLayout
from .pipeline import decode

__all__ = ["Analysis", "KeypointLayout", "analyze", "decode"]
