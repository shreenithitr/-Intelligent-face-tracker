"""
logger.py - Structured event logging: file log + cropped image storage + DB.

Features:
- Thread-safe, standardized event logging (SYSTEM, NEW_FACE_REGISTERED, FACE_ENTRY, FACE_TRACKING, FACE_EXIT, FACE_RECOGNITION)
- Normalized image paths saved on disk (logs/entries/YYYY-MM-DD/ and logs/exits/YYYY-MM-DD/)
- Direct linkage to SQLite database for auditability and verification
"""
import os
import logging
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, List
import database


def setup_file_logger(logs_dir: str) -> logging.Logger:
    """
    Sets up the application logger with:
    - FileHandler writing to logs/events.log at DEBUG level (captures all events + tracking)
    - StreamHandler writing to Console at INFO level (clean, concise output)
    - Formats timestamps consistently as YYYY-MM-DD HH:MM:SS
    """
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger("face_tracker")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to prevent duplicate lines when reinitialized
    if logger.handlers:
        logger.handlers.clear()

    log_path = os.path.join(logs_dir, "events.log")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # console shows INFO+ (suppresses verbose FACE_TRACKING)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _image_dir(logs_dir: str, event_type: str) -> str:
    """Creates and returns the daily directory for event face crops."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder = "entries" if event_type == "entry" else "exits"
    path = os.path.join(logs_dir, folder, date_str).replace("\\", "/")
    os.makedirs(path, exist_ok=True)
    return path


def save_face_image(logs_dir: str, event_type: str, face_id: str, frame: np.ndarray, bbox: list) -> str:
    """
    Crops the face bounding box from the video frame and saves it as a JPEG image.
    Returns normalized relative path string.
    """
    if frame is None or frame.size == 0 or not bbox or len(bbox) < 4:
        return ""

    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Boundary clamp
    x1 = max(0, min(x1, fw - 1))
    y1 = max(0, min(y1, fh - 1))
    x2 = max(x1 + 1, min(x2, fw))
    y2 = max(y1 + 1, min(y2, fh))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ""

    img_dir = _image_dir(logs_dir, event_type)
    ts = datetime.now().strftime("%H%M%S_%f")
    filename = f"{face_id}_{ts}.jpg"
    filepath = os.path.join(img_dir, filename).replace("\\", "/")
    cv2.imwrite(filepath, crop)
    return filepath


def log_entry(
    logger: logging.Logger,
    db_path: str,
    logs_dir: str,
    face_id: str,
    frame: np.ndarray,
    bbox: list,
    is_new: bool,
):
    """
    Logs visitor entry:
    - Saves cropped face thumbnail
    - Updates SQLite visitors table & events table
    - Emits structured log line: NEW_FACE_REGISTERED (first visit) or FACE_ENTRY (re-entry)
    """
    ts = datetime.now().isoformat(timespec="seconds")
    img_path = save_face_image(logs_dir, "entry", face_id, frame, bbox)

    if is_new:
        database.register_visitor(db_path, face_id, ts)
        logger.info(f"NEW_FACE_REGISTERED | face_id={face_id} | image={img_path}")
    else:
        database.update_visitor_last_seen(db_path, face_id, ts)
        logger.info(f"FACE_ENTRY | face_id={face_id} | image={img_path}")

    database.log_event(db_path, face_id, "entry", ts, img_path)


def log_recognition(logger: logging.Logger, face_id: str):
    """Log when an existing registered face is re-identified via embedding match."""
    logger.info(f"FACE_RECOGNITION | face_id={face_id}")


def log_tracking(logger: logging.Logger, face_id: str, bbox: list):
    """DEBUG-level tracking log — written to events.log file only, suppressed on console."""
    logger.debug(f"FACE_TRACKING | face_id={face_id} | bbox={bbox}")


def log_exit(
    logger: logging.Logger,
    db_path: str,
    logs_dir: str,
    face_id: str,
    frame: np.ndarray,
    bbox: list,
):
    """
    Logs visitor exit:
    - Saves cropped face thumbnail at departure
    - Records exit event in database
    - Emits structured log line: FACE_EXIT
    """
    ts = datetime.now().isoformat(timespec="seconds")
    img_path = save_face_image(logs_dir, "exit", face_id, frame, bbox)
    database.log_event(db_path, face_id, "exit", ts, img_path)
    logger.info(f"FACE_EXIT | face_id={face_id} | image={img_path}")


def log_system(logger: logging.Logger, message: str):
    """Logs system lifecycle event (e.g. Pipeline started, Pipeline ended)."""
    logger.info(f"SYSTEM | {message}")
