"""
run_all.py - Batch process all video files in the videos/ folder.

Key Multi-Video Architecture:
- Shares a single FaceEngine instance (avoiding redundant model reload overhead)
- Shares a single FaceTracker with persistent ArcFace identity gallery
- Calls tracker.reset_active_tracks() between video clips to ensure spatial IoU
  continuity does not mistakenly link disparate video scenes, while cross-video
  face re-identification via gallery embedding continues to match the same person!
- Maintains database consistency across all processed videos
"""
import glob
import os
from main import run, load_config
from face_engine import FaceEngine
from tracker import FaceTracker
import database

def main():
    config = load_config("config.json")
    config["display_window"] = False  # Headless execution for batch processing

    print("🚀 Initializing shared FaceEngine & FaceTracker...")
    shared_engine = FaceEngine(
        yolo_model=config.get("yolo_model", "yolov8n-face.pt"),
        embedding_model=config.get("embedding_model", "buffalo_l"),
        yolo_conf=config.get("yolo_confidence", 0.35),
        min_face_size=config.get("min_face_size", 40),
    )
    shared_tracker = FaceTracker(
        max_age=config.get("tracker_max_age", 60),
        min_hits=config.get("tracker_min_hits", 3),
        recognition_threshold=config.get("recognition_threshold", 0.55),
        max_missed_frames=config.get("max_missed_frames", 90),
    )

    db_path = config.get("database_path", "data/visitors.db")
    database.init_db(db_path)

    # Pre-populate registered identities from database
    shared_registered: set[str] = set(
        v["face_id"] for v in database.get_all_visitors(db_path)
    )

    video_files = sorted(glob.glob("videos/*.mp4"))
    if not video_files:
        print("⚠️ No .mp4 files found in videos/ folder.")
        return

    print(f"\n🎬 Found {len(video_files)} videos to process sequentially:\n")
    for v in video_files:
        print(f"  • {v}")
    print()

    for i, video in enumerate(video_files, 1):
        print(f"\n{'='*65}")
        print(f"[{i}/{len(video_files)}] Processing: {video}")
        print(f"{'='*65}")

        # Reset active frame-to-frame tracks before each new video
        # (while retaining the persistent face identity gallery)
        shared_tracker.reset_active_tracks()

        run(
            config,
            source=video,
            shared_engine=shared_engine,
            shared_tracker=shared_tracker,
            shared_registered=shared_registered,
            headless=True,
        )

    print("\n" + "="*65)
    print("✅ All videos processed successfully!")
    print(f"   Total Unique Visitors in Database : {database.get_unique_visitor_count(db_path)}")
    print(f"   Unique Gallery Identities        : {len(shared_tracker.gallery)}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
