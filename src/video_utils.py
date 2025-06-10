from config import *
from utils import get_logger
import cv2
import numpy as np
from typing import Generator

logger = get_logger("VideoUtils")

class VideoUtils:
    """ ======= Class to handle video processing tasks ======= """
    
    @staticmethod
    def load_video(video_path: str) -> cv2.VideoCapture:
        """Load a video file."""
        logger.trace(f"Loading video from {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Error opening video file {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")
        
        logger.debug(f"Video {video_path} loaded successfully")
        return cap
    
    @staticmethod
    def read_video_frames(video_path: str) -> Generator[np.ndarray, None, None]:
        """Read frames from a video file."""
        logger.trace(f"Reading video frames from {video_path}")
        cap = VideoUtils.load_video(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug(f"Total frames in video: {n_frames}")
        
        for frame_idx in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Frame {frame_idx} could not be read. Stopping.")
                break
            
            logger.trace(f"Yielding frame {frame_idx}")
            yield frame
        cap.release()
        logger.debug("All frames read from video successfully")
        logger.trace("Video capture released")
        return
