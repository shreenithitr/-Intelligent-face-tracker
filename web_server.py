"""
web_server.py — High-performance Web Dashboard & Real-Time Streaming Server.

Provides a modern web interface for:
- Live MJPEG video stream with bounding boxes, visitor IDs, and HUD
- Real-time KPI metrics (unique visitors, current inside, total entries/exits, FPS)
- Interactive Visitor Catalog with face crop photos and complete history
- Live Event Activity Feed with instant badge highlights
- Video source selector, pipeline controls, parameter tuning, and CSV/JSON export
"""
import os
import glob
import json
import time
import threading
import io
import csv
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from fastapi import FastAPI, Response, Query, Body, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import database
import logger as log_module
from face_engine import FaceEngine
from tracker import FaceTracker

app = FastAPI(title="Intelligent Face Tracker & Visitor Counting")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Pipeline Manager State
class PipelineManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        self.paused = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        self.db_path = self.config.get("database_path", "data/visitors.db")
        self.logs_dir = self.config.get("logs_dir", "logs")
        database.init_db(self.db_path)
        self.app_logger = log_module.setup_file_logger(self.logs_dir)

        # Shared Engine & Tracker
        self.engine: Optional[FaceEngine] = None
        self.tracker: Optional[FaceTracker] = None
        self.registered_faces: set = set()
        self.inside_faces: set = set()

        # Streaming frame buffer
        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_metrics: Dict[str, Any] = {
            "fps": 0.0,
            "unique_visitors": 0,
            "currently_inside": 0,
            "total_entries": 0,
            "total_exits": 0,
            "frames_processed": 0,
            "det_time_ms": 0.0,
            "emb_time_ms": 0.0,
            "trk_time_ms": 0.0,
            "source": "",
            "is_running": False,
            "is_paused": False,
        }

    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "video_source": "videos/video_sample1.mp4",
            "detection_skip_frames": 0,
            "recognition_threshold": 0.55,
            "min_face_size": 40,
            "tracker_max_age": 60,
            "tracker_min_hits": 3,
            "max_missed_frames": 90,
            "embedding_model": "buffalo_l",
            "yolo_model": "yolov8n-face.pt",
            "yolo_confidence": 0.35,
            "database_path": "data/visitors.db",
            "logs_dir": "logs",
        }

    def _save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    def init_engine(self):
        if self.engine is None:
            self.engine = FaceEngine(
                yolo_model=self.config.get("yolo_model", "yolov8n-face.pt"),
                embedding_model=self.config.get("embedding_model", "buffalo_l"),
                yolo_conf=float(self.config.get("yolo_confidence", 0.35)),
                min_face_size=int(self.config.get("min_face_size", 40)),
            )
        if self.tracker is None:
            self.tracker = FaceTracker(
                max_age=int(self.config.get("tracker_max_age", 60)),
                min_hits=int(self.config.get("tracker_min_hits", 3)),
                recognition_threshold=float(self.config.get("recognition_threshold", 0.55)),
                max_missed_frames=int(self.config.get("max_missed_frames", 90)),
                counting_line=self.config.get("counting_line", {"type": "horizontal", "position": 0.5, "buffer": 25}),
            )
        # Load known visitors
        self.registered_faces = set(
            v["face_id"] for v in database.get_all_visitors(self.db_path)
        )

    def start_pipeline(self, source: Optional[str] = None):
        with self.lock:
            if self.running:
                self.stop_pipeline()

            if source:
                self.config["video_source"] = source
                self._save_config()

            self.init_engine()
            self.tracker.reset_active_tracks()
            self.inside_faces = self.tracker.inside_ids
            self.running = True
            self.paused = False
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()

    def stop_pipeline(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.thread = None
        self.latest_metrics["is_running"] = False

    def toggle_pause(self):
        self.paused = not self.paused
        self.latest_metrics["is_paused"] = self.paused

    def _worker_loop(self):
        src = self.config.get("video_source", "videos/video_sample1.mp4")
        skip = int(self.config.get("detection_skip_frames", 0))

        try:
            source_val = int(src)
        except (ValueError, TypeError):
            source_val = src

        cap = cv2.VideoCapture(source_val)
        if not cap.isOpened():
            log_module.log_system(self.app_logger, f"Failed to open video source: {src}")
            self.running = False
            return

        log_module.log_system(self.app_logger, f"Web pipeline started | source={src}")
        frame_idx = 0
        frame_times = []
        det_times = []
        emb_times = []
        trk_times = []
        last_frame = None

        try:
            while self.running:
                if self.paused:
                    time.sleep(0.05)
                    continue

                ret, frame = cap.read()
                if not ret:
                    log_module.log_system(self.app_logger, "End of stream reached")
                    break

                last_frame = frame
                t0 = time.perf_counter()
                run_detection = (frame_idx % (skip + 1) == 0)
                detections = []

                if run_detection:
                    # Detection
                    t_det = time.perf_counter()
                    raw_faces = self.engine.detect_faces(frame)
                    det_times.append(time.perf_counter() - t_det)

                    # Selective ArcFace Embedding
                    t_emb = time.perf_counter()
                    for f in raw_faces:
                        best_iou = 0.0
                        best_track = None
                        for t in self.tracker.tracks:
                            if not t.active:
                                continue
                            iou = self.engine._iou(t.bbox, f["bbox"])
                            if iou > best_iou:
                                best_iou, best_track = iou, t

                        need_emb = True
                        if best_track is not None and best_iou >= 0.25:
                            need_emb = self.tracker.needs_embedding(best_track.track_id)

                        if need_emb:
                            emb = self.engine.get_embedding(frame, f["bbox"])
                        else:
                            emb = best_track.embedding

                        detections.append({
                            "bbox": f["bbox"],
                            "embedding": emb,
                            "conf": f["conf"]
                        })
                    emb_times.append(time.perf_counter() - t_emb)

                # Tracker Update with Zone Crossing Detection
                t_trk = time.perf_counter()
                confirmed, events_triggered = self.tracker.update(detections, frame_idx=frame_idx)
                trk_times.append(time.perf_counter() - t_trk)

                # Event Logging based on Directional Zone Crossings
                for ev in events_triggered:
                    fid = ev["face_id"]
                    if fid is None:
                        continue
                    if ev["type"] == "ENTRY":
                        is_new = fid not in self.registered_faces
                        if is_new:
                            self.registered_faces.add(fid)
                        log_module.log_entry(
                            self.app_logger, self.db_path, self.logs_dir,
                            fid, frame, ev["bbox"], is_new=is_new
                        )
                        if not is_new and ev.get("is_recognized"):
                            log_module.log_recognition(self.app_logger, fid)
                    elif ev["type"] == "EXIT":
                        log_module.log_exit(
                            self.app_logger, self.db_path, self.logs_dir,
                            fid, frame, ev["bbox"]
                        )

                # Track position log
                if run_detection:
                    for t in confirmed:
                        fid = t.get("face_id")
                        if fid:
                            log_module.log_tracking(self.app_logger, fid, t["bbox"])

                # Draw Overlay on Display Copy
                display = frame.copy()
                fps_calc = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0.0
                self._draw_overlay(display, confirmed, self.tracker.unique_count, fps=fps_calc)

                # Encode frame for Web Streaming (resized for smooth HTTP streaming)
                h, w = display.shape[:2]
                max_w = 960
                if w > max_w:
                    scale = max_w / w
                    display = cv2.resize(display, (int(w * scale), int(h * scale)))
                
                _, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 75])
                self.latest_frame_jpeg = jpeg.tobytes()

                elapsed = time.perf_counter() - t0
                frame_times.append(elapsed)

                # Update live metrics
                w_avg = min(30, len(frame_times))
                self.latest_metrics = {
                    "fps": round(1.0 / np.mean(frame_times[-w_avg:]), 1) if frame_times else 0.0,
                    "unique_visitors": self.tracker.unique_count,
                    "currently_inside": len(self.tracker.inside_ids),
                    "total_entries": self.tracker.total_entries,
                    "total_exits": self.tracker.total_exits,
                    "frames_processed": frame_idx,
                    "det_time_ms": round(np.mean(det_times[-w_avg:]) * 1000, 1) if det_times else 0.0,
                    "emb_time_ms": round(np.mean(emb_times[-w_avg:]) * 1000, 1) if emb_times else 0.0,
                    "trk_time_ms": round(np.mean(trk_times[-w_avg:]) * 1000, 1) if trk_times else 0.0,
                    "source": str(src),
                    "is_running": True,
                    "is_paused": self.paused,
                }

                frame_idx += 1
                # Small yield for I/O
                time.sleep(0.001)

        finally:
            cap.release()
            self.running = False
            self.latest_metrics["is_running"] = False
            unique = self.tracker.unique_count
            log_module.log_system(
                self.app_logger,
                f"Web pipeline stopped | unique_visitors={unique} | inside={len(self.tracker.inside_ids)} | entries={self.tracker.total_entries} | exits={self.tracker.total_exits} | frames={frame_idx}"
            )

    def _draw_overlay(self, frame: np.ndarray, confirmed: list, unique_count: int, fps: float = 0.0):
        fh, fw = frame.shape[:2]
        
        # Draw virtual counting line
        line_cfg = self.tracker.counting_line
        if line_cfg["type"] == "horizontal":
            line_y = int(line_cfg["position"] * fh) if line_cfg["position"] <= 1.0 else int(line_cfg["position"])
            cv2.line(frame, (0, line_y), (fw, line_y), (0, 215, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "▲ OUTSIDE", (20, max(20, line_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "▼ INSIDE (MONITORED AREA)", (20, min(fh - 10, line_y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)
        else:
            line_x = int(line_cfg["position"] * fw) if line_cfg["position"] <= 1.0 else int(line_cfg["position"])
            cv2.line(frame, (line_x, 0), (line_x, fh), (0, 215, 255), 2, cv2.LINE_AA)

        for t in confirmed:
            if t.get("face_id") is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            is_new = t.get("is_new", False)
            color = (0, 165, 255) if is_new else (0, 230, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{t['face_id']}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + lw + 6, max(0, y1)), color, -1)
            cv2.putText(frame, label, (x1 + 3, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # HUD
        hud_bg = frame[:60, :420]
        if hud_bg.shape[0] >= 60 and hud_bg.shape[1] >= 420:
            overlay = hud_bg.copy()
            cv2.rectangle(overlay, (0, 0), (420, 60), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, hud_bg, 0.25, 0, hud_bg)
            cv2.putText(frame, f"Unique: {unique_count} | Inside: {len(self.tracker.inside_ids)}",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Entries: {self.tracker.total_entries} | Exits: {self.tracker.total_exits} | FPS: {fps:.1f}",
                        (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


pipeline = PipelineManager()

# Mount logs static directory so cropped images can be served directly
if os.path.exists("logs"):
    app.mount("/logs", StaticFiles(directory="logs"), name="logs")


@app.get("/video_feed")
def video_feed():
    """MJPEG stream endpoint for real-time video feed."""
    def frame_generator():
        while True:
            if pipeline.latest_frame_jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + pipeline.latest_frame_jpeg + b"\r\n")
            time.sleep(0.033)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
def get_status():
    """Returns real-time status and telemetry metrics."""
    return pipeline.latest_metrics


@app.get("/api/videos")
def list_videos():
    """Lists available video files in the videos/ folder."""
    files = sorted(glob.glob("videos/*.mp4"))
    return {"videos": [f.replace("\\", "/") for f in files]}


@app.get("/api/visitors")
def get_visitors():
    """Returns list of registered unique visitors with their latest image thumbnail."""
    visitors = database.get_all_visitors(pipeline.db_path)
    # Attach latest image for each visitor
    with database.get_connection(pipeline.db_path) as conn:
        for v in visitors:
            row = conn.execute(
                "SELECT image_path FROM events WHERE face_id = ? AND image_path != '' ORDER BY id DESC LIMIT 1",
                (v["face_id"],)
            ).fetchone()
            v["latest_image"] = row[0] if row else ""
            v["is_inside"] = v["face_id"] in pipeline.inside_faces
    return {"visitors": visitors}


@app.get("/api/events")
def get_events(limit: int = 50, face_id: Optional[str] = None):
    """Returns recent entry/exit events."""
    if face_id:
        events = database.get_events_by_face(pipeline.db_path, face_id)
    else:
        events = database.get_recent_events(pipeline.db_path, limit=limit)
    return {"events": events}


@app.get("/api/stats")
def get_stats():
    """Returns aggregated stats from DB."""
    return database.get_statistics(pipeline.db_path)


@app.get("/api/logs")
def get_logs(lines: int = 100):
    """Returns the latest lines from logs/events.log."""
    log_path = os.path.join(pipeline.logs_dir, "events.log")
    if not os.path.exists(log_path):
        return {"logs": []}
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            return {"logs": [l.strip() for l in all_lines[-lines:]]}
    except Exception as e:
        return {"logs": [f"Error reading log file: {e}"]}


@app.post("/api/control/start")
def control_start(payload: dict = Body(...)):
    """Starts video processing with specified source."""
    source = payload.get("source")
    pipeline.start_pipeline(source)
    return {"status": "started", "source": pipeline.config.get("video_source")}


@app.post("/api/control/stop")
def control_stop():
    """Stops video processing."""
    pipeline.stop_pipeline()
    return {"status": "stopped"}


@app.post("/api/control/pause")
def control_pause():
    """Toggles pause/resume."""
    pipeline.toggle_pause()
    return {"status": "paused" if pipeline.paused else "resumed"}


@app.get("/api/config")
def get_config():
    """Returns current system configuration parameters."""
    return pipeline.config


@app.post("/api/config")
def update_config(payload: dict = Body(...)):
    """Updates configuration parameters dynamically."""
    pipeline.config.update(payload)
    pipeline._save_config()
    if pipeline.engine:
        if "yolo_confidence" in payload:
            pipeline.engine.yolo_conf = float(payload["yolo_confidence"])
        if "min_face_size" in payload:
            pipeline.engine.min_face_size = int(payload["min_face_size"])
    if pipeline.tracker:
        if "recognition_threshold" in payload:
            pipeline.tracker.threshold = float(payload["recognition_threshold"])
        if "tracker_max_age" in payload:
            pipeline.tracker.max_age = int(payload["tracker_max_age"])
        if "tracker_min_hits" in payload:
            pipeline.tracker.min_hits = int(payload["tracker_min_hits"])
    return {"status": "updated", "config": pipeline.config}


@app.get("/api/export/visitors.csv")
def export_visitors_csv():
    """Exports visitors table as CSV file."""
    visitors = database.get_all_visitors(pipeline.db_path)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["id", "face_id", "first_seen", "last_seen", "visit_count"])
    writer.writeheader()
    for v in visitors:
        writer.writerow(v)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=visitors.csv"}
    )


@app.get("/api/export/events.csv")
def export_events_csv():
    """Exports events table as CSV file."""
    with database.get_connection(pipeline.db_path) as conn:
        rows = conn.execute("SELECT * FROM events ORDER BY id ASC").fetchall()
        events = [dict(r) for r in rows]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["id", "face_id", "event_type", "timestamp", "image_path"])
    writer.writeheader()
    for e in events:
        writer.writerow(e)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=events.csv"}
    )


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the main Web Dashboard UI."""
    index_file = os.path.join("static", "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Dashboard loading...</h1>")


def start_web_server(port: int = 8000, config_path: str = "config.json"):
    """Starts the Uvicorn web server."""
    global pipeline
    pipeline = PipelineManager(config_path=config_path)
    # Automatically start pipeline on default video
    pipeline.start_pipeline()
    print(f"\n🌐 Web Dashboard running at: http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    start_web_server()
