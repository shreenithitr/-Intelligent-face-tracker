"""
main.py — Intelligent Face Tracker pipeline entry point.

Usage:
    python main.py --source videos/sample.mp4
    python main.py --source rtsp://...
    python main.py --source 0              # webcam
"""
import argparse
import json
import cv2
import numpy as np
import sys
import os
import time

import database
import logger as log_module
from face_engine import FaceEngine
from tracker import FaceTracker

# Embedding is regenerated only after this many frames for a stable confirmed track
EMBED_REFRESH_FRAMES = 40


def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def draw_overlay(frame: np.ndarray, confirmed: list,
                 session_unique: int, currently_inside: int,
                 session_entries: int, session_exits: int,
                 registered_identities: int,
                 line_cfg: dict = None):
    """Draw bounding boxes, counting line, and session dashboard HUD."""
    fh, fw = frame.shape[:2]

    # ── counting line ─────────────────────────────────────────────────────────
    if line_cfg and line_cfg.get("enabled", True):
        orientation = line_cfg.get("orientation", "horizontal")
        rel_pos = float(line_cfg.get("position", 0.5))
        buf = int(line_cfg.get("buffer_pixels", 25))
        if orientation == "horizontal":
            ly = int(fh * rel_pos) if rel_pos <= 1.0 else int(rel_pos)
            cv2.line(frame, (0, ly), (fw, ly), (0, 215, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (0, ly - buf), (fw, ly + buf), (0, 165, 255), 1)
            cv2.putText(frame, "OUTSIDE", (10, max(20, ly - buf - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "INSIDE", (10, min(fh - 6, ly + buf + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)
        else:
            lx = int(fw * rel_pos) if rel_pos <= 1.0 else int(rel_pos)
            cv2.line(frame, (lx, 0), (lx, fh), (0, 215, 255), 2, cv2.LINE_AA)

    # ── face bounding boxes ───────────────────────────────────────────────────
    for t in confirmed:
        if not t.get("face_id"):
            continue
        x1, y1, x2, y2 = t["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID:{t['face_id'][:6]}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 4, ly), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Session dashboard — left side
    cv2.putText(frame, f"Unique Visitors : {session_unique}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)
    cv2.putText(frame, f"Currently Inside: {currently_inside}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    cv2.putText(frame, f"Entries         : {session_entries}",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.putText(frame, f"Exits           : {session_exits}",
                (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.putText(frame, f"Registered IDs  : {registered_identities}",
                (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)


def run(config: dict,
        source: str | None = None,
        shared_engine: FaceEngine | None = None,
        shared_tracker: FaceTracker | None = None,
        shared_registered: set | None = None):

    src      = source or config["video_source"]
    db_path  = config["database_path"]
    logs_dir = config["logs_dir"]
    skip     = config.get("detection_skip_frames", 0)

    database.init_db(db_path)
    app_logger = log_module.setup_file_logger(logs_dir)
    log_module.log_system(app_logger, f"Pipeline started | source={src}")

    engine = shared_engine or FaceEngine(
        yolo_model=config["yolo_model"],
        embedding_model=config["embedding_model"],
        yolo_conf=config["yolo_confidence"],
        min_face_size=config["min_face_size"],
    )
    tracker = shared_tracker or FaceTracker(
        max_age=config["tracker_max_age"],
        min_hits=config["tracker_min_hits"],
        recognition_threshold=config["recognition_threshold"],
        max_missed_frames=config.get("max_missed_frames", 90),
    )

    try:
        src = int(src)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        app_logger.error(f"Cannot open source: {src}")
        sys.exit(1)

    log_module.log_system(app_logger, "Video capture opened successfully")

    # Counting line config
    line_cfg = config.get("counting_line", {
        "enabled": True,
        "orientation": "horizontal",
        "position": 0.5,
        "entry_direction": "top_to_bottom",
        "buffer_pixels": 25,
        "debounce_frames": 30,
    })

    def _zone(bbox, fh, fw):
        """Returns 'outside', 'inside', or 'neutral' for a bbox."""
        if not line_cfg.get("enabled", True):
            return "inside"
        x1, y1, x2, y2 = bbox
        cy = float(y2)  # use bottom of bbox (feet)
        cx = (x1 + x2) / 2.0
        orientation = line_cfg.get("orientation", "horizontal")
        rel_pos = float(line_cfg.get("position", 0.5))
        buf = float(line_cfg.get("buffer_pixels", 25))
        direction = line_cfg.get("entry_direction", "top_to_bottom")
        if orientation == "horizontal":
            line_coord = rel_pos if rel_pos > 1.0 else fh * rel_pos
            val = cy
        else:
            line_coord = rel_pos if rel_pos > 1.0 else fw * rel_pos
            val = cx
        if direction in ("top_to_bottom", "left_to_right"):
            if val < line_coord - buf:   return "outside"
            elif val > line_coord + buf: return "inside"
            else:                        return "neutral"
        else:
            if val > line_coord + buf:   return "outside"
            elif val < line_coord - buf: return "inside"
            else:                        return "neutral"

    frame_idx    = 0
    inside_faces: set[str] = set()   # currently inside this session
    # track last confirmed zone per face_id for crossing detection
    face_last_zone: dict[str, str] = {}
    face_last_event_frame: dict[str, int] = {}
    debounce = int(line_cfg.get("debounce_frames", 30))

    # Session-specific counters — never mixed with DB totals
    session_face_ids: set[str] = set()   # unique confirmed face_ids this session
    session_entries  = 0
    session_exits    = 0

    # Persistent gallery — loaded from DB for cross-session re-ID
    # but does NOT affect session_face_ids count
    registered_faces: set[str] = (shared_registered
                                   if shared_registered is not None else set())
    if not registered_faces:
        for v in database.get_all_visitors(db_path):
            registered_faces.add(v["face_id"])

    frame_times:     list[float] = []
    detection_times: list[float] = []
    embedding_times: list[float] = []
    tracking_times:  list[float] = []
    embed_calls_total = 0
    embed_skipped_total = 0

    diag_limit = int(os.environ.get("DIAG_FRAMES", 0))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log_module.log_system(app_logger, "End of stream reached")
                break

            t0 = time.perf_counter()
            run_detection = (frame_idx % (skip + 1) == 0)
            detections: list[dict] = []

            if run_detection:
                t_det = time.perf_counter()
                raw_faces = engine.detect_faces(frame)
                detection_times.append(time.perf_counter() - t_det)

                t_emb = time.perf_counter()
                for f in raw_faces:
                    # find best IoU match among active tracks
                    best_iou, best_track = 0.0, None
                    for t in tracker.tracks:
                        if not t.active:
                            continue
                        iou = engine._iou(t.bbox, f["bbox"])
                        if iou > best_iou:
                            best_iou, best_track = iou, t

                    emb = None
                    reason = None

                    # always check cooldown first regardless of IoU match
                    if best_track is not None and best_track.embed_cooldown > 0:
                        emb = best_track.embedding  # reuse last known embedding
                        embed_skipped_total += 1
                    elif best_track is None or best_iou < 0.25:
                        # No IoU match — new detection or re-entry after crossing
                        reason = "new_track" if best_track is None else "face_reappeared"
                        emb = engine.get_embedding(frame, f["bbox"])
                        embed_calls_total += 1
                        if emb is None and best_track is not None:
                            tracker.set_embed_cooldown(best_track.track_id, frames=8)
                    elif not tracker.needs_embedding(best_track.track_id):
                        # Stable confirmed track within refresh window — reuse embedding
                        emb = best_track.embedding
                        embed_skipped_total += 1
                    else:
                        # Needs embedding: unconfirmed, no identity, or refresh due
                        if not best_track.confirmed:
                            reason = "unknown_identity"
                        elif best_track.embedding is None:
                            reason = "identity_uncertain"
                        elif best_track.frames_since_embedding >= EMBED_REFRESH_FRAMES:
                            reason = "periodic_refresh"
                        else:
                            reason = "reidentification"
                        emb = engine.get_embedding(frame, f["bbox"])
                        embed_calls_total += 1
                        if emb is None and best_track is not None:
                            tracker.set_embed_cooldown(best_track.track_id, frames=8)

                    if reason:
                        print(f"[EMBED_CALL] frame={frame_idx}"
                              f" track_id={best_track.track_id if best_track else 'new'}"
                              f" face_id={best_track.face_id if best_track else 'none'}"
                              f" reason={reason}")

                    detections.append({"bbox": f["bbox"], "embedding": emb,
                                       "conf": f["conf"]})
                embedding_times.append(time.perf_counter() - t_emb)

            # tracker update
            t_trk = time.perf_counter()
            confirmed, exited_ids = tracker.update(detections, frame_idx=frame_idx)
            tracking_times.append(time.perf_counter() - t_trk)

            fh_cur, fw_cur = frame.shape[:2]

            # ── session entry/exit via zone crossing ──────────────────────────
            for t in confirmed:
                fid = t["face_id"]
                if not fid:
                    continue

                # unique visitor count — one increment per face_id ever
                if fid not in session_face_ids:
                    session_face_ids.add(fid)

                zone = _zone(t["bbox"], fh_cur, fw_cur)

                if zone == "neutral":
                    # still in buffer — update tracking log only
                    if run_detection:
                        log_module.log_tracking(app_logger, fid, t["bbox"])
                    continue

                prev_zone = face_last_zone.get(fid)
                last_ev_frame = face_last_event_frame.get(fid, -debounce)
                frames_since = frame_idx - last_ev_frame

                # ENTRY: outside → inside crossing
                if zone == "inside" and prev_zone == "outside":
                    if frames_since >= debounce and fid not in inside_faces:
                        inside_faces.add(fid)
                        session_entries += 1
                        face_last_event_frame[fid] = frame_idx
                        is_new = fid not in registered_faces
                        if is_new:
                            registered_faces.add(fid)
                        log_module.log_entry(app_logger, db_path, logs_dir,
                                             fid, frame, t["bbox"], is_new=is_new)
                        if not is_new:
                            log_module.log_recognition(app_logger, fid)

                # EXIT: inside → outside crossing
                elif zone == "outside" and prev_zone == "inside":
                    if frames_since >= debounce and fid in inside_faces:
                        inside_faces.discard(fid)
                        session_exits += 1
                        face_last_event_frame[fid] = frame_idx
                        log_module.log_exit(app_logger, db_path, logs_dir,
                                            fid, frame, t["bbox"])

                elif prev_zone is None:
                    # first time we see this face — initialise zone, no event yet
                    if zone == "inside" and fid not in inside_faces:
                        # already inside when first detected (e.g. video starts mid-scene)
                        inside_faces.add(fid)
                        session_entries += 1
                        face_last_event_frame[fid] = frame_idx
                        is_new = fid not in registered_faces
                        if is_new:
                            registered_faces.add(fid)
                        log_module.log_entry(app_logger, db_path, logs_dir,
                                             fid, frame, t["bbox"], is_new=is_new)

                elif run_detection:
                    log_module.log_tracking(app_logger, fid, t["bbox"])

                # always update last confirmed zone (skip neutral)
                face_last_zone[fid] = zone

            # ── track-lost exits ─────────────────────────────────────────────
            for fid in exited_ids:
                if fid in inside_faces:
                    inside_faces.discard(fid)
                    session_exits += 1
                    trk = next((t for t in tracker.tracks if t.face_id == fid), None)
                    bbox = trk.bbox if trk else [0, 0, 50, 50]
                    log_module.log_exit(app_logger, db_path, logs_dir,
                                        fid, frame, bbox)

            # ── display ───────────────────────────────────────────────────────
            if config.get("display_window", True):
                display = frame.copy()
                draw_overlay(
                    display, confirmed,
                    session_unique=len(session_face_ids),
                    currently_inside=len(inside_faces),
                    session_entries=session_entries,
                    session_exits=session_exits,
                    registered_identities=len(registered_faces),
                    line_cfg=line_cfg,
                )
                cv2.imshow("Face Tracker", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log_module.log_system(app_logger, "User quit")
                    break

            frame_times.append(time.perf_counter() - t0)

            if frame_idx > 0 and frame_idx % 100 == 0:
                w = min(100, len(frame_times))
                fps = 1.0 / np.mean(frame_times[-w:])
                avg_det = np.mean(detection_times[-w:]) * 1000 if detection_times else 0
                avg_emb = np.mean(embedding_times[-w:]) * 1000 if embedding_times else 0
                print(f"[PERF] frame={frame_idx} FPS={fps:.1f}"
                      f" det={avg_det:.0f}ms emb={avg_emb:.0f}ms"
                      f" session_unique={len(session_face_ids)}"
                      f" embed_calls={embed_calls_total} skipped={embed_skipped_total}")

            frame_idx += 1
            if diag_limit and frame_idx >= diag_limit:
                break

    except KeyboardInterrupt:
        log_module.log_system(app_logger, "Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        db_registered = database.get_unique_visitor_count(db_path)
        log_module.log_system(
            app_logger,
            f"Pipeline ended | session_unique={len(session_face_ids)}"
            f" | db_registered={db_registered} | frames_processed={frame_idx}")

        if frame_times:
            avg_fps = 1.0 / np.mean(frame_times)
            avg_det = np.mean(detection_times) * 1000 if detection_times else 0
            avg_emb = np.mean(embedding_times) * 1000 if embedding_times else 0
            avg_trk = np.mean(tracking_times)  * 1000 if tracking_times else 0
            print(f"\n{'='*52}")
            print(f"  Frames processed         : {frame_idx}")
            print(f"  Average FPS              : {avg_fps:.1f}")
            print(f"  --- Session Metrics ---")
            print(f"  Session unique visitors  : {len(session_face_ids)}")
            print(f"  Currently inside         : {len(inside_faces)}")
            print(f"  Session entries          : {session_entries}")
            print(f"  Session exits            : {session_exits}")
            print(f"  New IDs created          : {tracker.stat_new_ids}")
            print(f"  Reconnections            : {tracker.stat_reconnected}")
            print(f"  --- DB Metrics ---")
            print(f"  DB registered identities : {db_registered}")
            print(f"  --- Performance ---")
            print(f"  Avg detection time       : {avg_det:.1f}ms")
            print(f"  Avg embedding time       : {avg_emb:.1f}ms")
            print(f"  Avg tracking time        : {avg_trk:.1f}ms")
            print(f"  Embed calls (actual)     : {embed_calls_total}")
            print(f"  Embed calls (skipped)    : {embed_skipped_total}")
            print(f"{'='*52}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    run(load_config(args.config), args.source)
