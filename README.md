# Intelligent Face Tracker with Auto-Registration and Visitor Counting

A real-time AI pipeline that detects, tracks, and recognizes faces from a video stream or RTSP camera, automatically registers new visitors, and maintains an accurate unique visitor count — all backed by structured logging and a SQLite database.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          main.py (Orchestrator)                     │
│                                                                     │
│  Video Source ──► Frame Loop                                        │
│  (file/RTSP/cam)      │                                             │
│                        ▼  every N frames (detection_skip_frames)    │
│               ┌─────────────────┐                                   │
│               │  face_engine.py │                                   │
│               │  YOLOv8 Detect  │──► bboxes                        │
│               │  InsightFace    │──► 512-d embeddings               │
│               └────────┬────────┘                                   │
│                        │ detections [{bbox, embedding}]             │
│                        ▼                                            │
│               ┌─────────────────┐                                   │
│               │   tracker.py    │                                   │
│               │  IoU match      │──► confirmed tracks               │
│               │  Embedding re-ID│──► exited face_ids                │
│               │  Gallery (EMA)  │                                   │
│               └────────┬────────┘                                   │
│                        │                                            │
│           ┌────────────┴────────────┐                               │
│           ▼                         ▼                               │
│   ┌───────────────┐        ┌────────────────┐                       │
│   │  logger.py    │        │  database.py   │                       │
│   │  events.log   │        │  visitors table│                       │
│   │  face images  │        │  events  table │                       │
│   │  (entry/exit) │        │  (SQLite)      │                       │
│   └───────────────┘        └────────────────┘                       │
│                                                                     │
│   OpenCV Display Window (optional) ── Unique Visitor Counter HUD   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Detail |
|---|---|
| Face Detection | YOLOv8-face — real-time, GPU-accelerated |
| Face Recognition | InsightFace `buffalo_l` — 512-d ArcFace embeddings |
| Tracking | IoU-based bbox continuity + embedding re-ID across frames |
| Auto-Registration | New face → UUID assigned, stored in DB on first confirmed detection |
| Re-identification | Cosine similarity against persistent gallery (EMA-updated) |
| Entry/Exit Logging | One log entry per event, cropped JPEG saved locally |
| Unique Visitor Count | Derived from `visitors` table — re-entries never double-count |
| Detection Skip | Configurable `detection_skip_frames` to reduce CPU/GPU load |
| RTSP Support | Switch source via `--source rtsp://...` at runtime |
| SQLite Database | Zero-config, file-based, portable |
| Multi-video Gallery | Shared gallery across all videos in `run_all.py` |

---

## Project Structure

```
.
├── main.py              # Pipeline entry point (single video or RTSP)
├── run_all.py           # Batch processor — all videos with shared gallery
├── face_engine.py       # YOLO detection + InsightFace embeddings
├── tracker.py           # Multi-face tracker with re-ID
├── logger.py            # Event logging (file + images + DB)
├── database.py          # SQLite schema + CRUD helpers
├── config.json          # All tunable parameters
├── requirements.txt     # Python dependencies
├── yolov8n-face.pt      # Face-specific YOLO model (required)
├── data/
│   └── visitors.db      # SQLite database (auto-created)
└── logs/
    ├── events.log        # Mandatory system event log
    ├── entries/
    │   └── YYYY-MM-DD/  # Cropped face images on entry
    └── exits/
        └── YYYY-MM-DD/  # Cropped face images on exit
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional — system falls back to CPU automatically)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> If you don't have a CUDA GPU, replace `onnxruntime-gpu` with `onnxruntime` in `requirements.txt`.

### 3. Download YOLOv8-face model

```bash
# Download yolov8n-face.pt from HuggingFace:
curl -L -o yolov8n-face.pt "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"
```

Place the `.pt` file in the project root. The filename must match `yolo_model` in `config.json`.

### 4. InsightFace model

InsightFace downloads `buffalo_l` automatically on first run to `~/.insightface/models/` (~300 MB).

### 5. Place your video files

Copy video files to the `videos/` folder.

### 6. Run

```bash
# Process a single video file
python main.py --source videos/sample.mp4

# Process all videos in videos/ folder (shared gallery — recommended)
python run_all.py

# Live RTSP stream
python main.py --source "rtsp://user:pass@192.168.1.100:554/stream"

# Webcam
python main.py --source 0
```

Press `q` in the display window to stop.

---

## Sample `config.json`

```json
{
  "video_source": "videos/video_sample1.mp4",
  "detection_skip_frames": 1,
  "recognition_threshold": 0.50,
  "min_face_size": 40,
  "tracker_max_age": 45,
  "tracker_min_hits": 3,
  "embedding_model": "buffalo_l",
  "yolo_model": "yolov8n-face.pt",
  "yolo_confidence": 0.5,
  "database_path": "data/visitors.db",
  "logs_dir": "logs",
  "display_window": true,
  "rtsp_url": "rtsp://username:password@camera_ip:554/stream"
}
```

| Parameter | Description |
|---|---|
| `detection_skip_frames` | Run YOLO+InsightFace every N+1 frames (0 = every frame) |
| `recognition_threshold` | Cosine similarity cutoff for re-ID (higher = stricter) |
| `tracker_max_age` | Frames before an unmatched track is marked as exited |
| `tracker_min_hits` | Consecutive detections before a track is confirmed |
| `min_face_size` | Minimum face bounding box dimension in pixels |

---

## How Unique Visitor Counting Works

- Every new face that passes `tracker_min_hits` confirmation gets a UUID (`face_id`).
- The face embedding is stored in an in-memory gallery (EMA-updated).
- On subsequent appearances, cosine similarity is computed against the gallery.
- If similarity > `recognition_threshold`, the same `face_id` is reused — no new registration.
- The `visitors` table uses `UNIQUE` constraint on `face_id` — duplicates are impossible at DB level.
- `run_all.py` shares the gallery across all videos so the same person in different videos is counted once.

**Note:** Unique visitor count accuracy depends on video quality, face angle, lighting, and threshold tuning. No ground-truth evaluation has been performed on these specific videos.

---

## CPU Fallback

The system lists `CUDAExecutionProvider` first for InsightFace. If CUDA is unavailable, it automatically falls back to `CPUExecutionProvider`. A warning is printed but the system continues normally. No configuration change is needed.

For CPU-only machines, replace `onnxruntime-gpu` with `onnxruntime` in `requirements.txt` to avoid the warning.

---

## events.log Format

```
2026-10-03 16:11:39 | INFO  | SYSTEM | Pipeline started | source=videos/sample.mp4
2026-10-03 16:11:40 | INFO  | NEW_FACE_REGISTERED | face_id=a3f2c1b0 | image=logs/entries/...
2026-10-03 16:11:41 | INFO  | FACE_RECOGNITION | face_id=a3f2c1b0
2026-10-03 16:11:41 | INFO  | FACE_ENTRY | face_id=a3f2c1b0 | image=logs/entries/...
2026-10-03 16:11:42 | DEBUG | FACE_TRACKING | face_id=a3f2c1b0 | bbox=[x1,y1,x2,y2]
2026-10-03 16:11:55 | INFO  | FACE_EXIT | face_id=a3f2c1b0 | image=logs/exits/...
2026-10-03 16:12:00 | INFO  | SYSTEM | Pipeline ended | unique_visitors=7 | frames_processed=1800
```

`FACE_TRACKING` events are written to the log file at DEBUG level but suppressed from the console.

---

## Database Schema

### `visitors` table
| id | face_id | first_seen | last_seen | visit_count |
|---|---|---|---|---|
| 1 | a3f2c1b0 | 2026-10-03T16:11:40 | 2026-10-03T16:11:55 | 2 |

### `events` table
| id | face_id | event_type | timestamp | image_path |
|---|---|---|---|---|
| 1 | a3f2c1b0 | entry | 2026-10-03T16:11:40 | logs/entries/... |
| 2 | a3f2c1b0 | exit | 2026-10-03T16:11:55 | logs/exits/... |

---

## AI Planning Document

### Problem Decomposition

1. **Detect** faces in each frame reliably and fast → YOLOv8-face
2. **Identify** each face uniquely across frames → InsightFace ArcFace embeddings
3. **Track** faces continuously without re-running full recognition every frame → IoU bbox tracking
4. **Re-identify** faces that temporarily leave and re-enter → cosine similarity against gallery
5. **Log** every entry/exit with image evidence → structured filesystem + SQLite
6. **Count** unique visitors without double-counting → gallery size = unique count

### Design Decisions

- **YOLOv8-face over Haar/HOG**: Superior accuracy and speed; face-specific training.
- **InsightFace `buffalo_l` over `face_recognition` lib**: Production-grade ArcFace model, 512-d embeddings, more accurate at low resolution and partial occlusion.
- **IoU-first, embedding-second matching**: IoU is O(1) per pair and handles fast movement; embedding matching handles re-entries after occlusion.
- **EMA gallery update**: Prevents embedding drift from a single bad frame while adapting to lighting changes.
- **`confirmed` flag on Track**: Ensures `is_new=True` fires exactly once per face, regardless of `min_hits` value.
- **Shared gallery in `run_all.py`**: Single `FaceTracker` instance persists across all videos — same person in different videos gets one `face_id`.
- **SQLite**: Zero-config, ACID-compliant, sufficient for single-node deployment.
- **Detection skip with empty list**: On skipped frames, `[]` is passed to tracker so tracks age naturally and exits are detected correctly.

### Compute Load Estimate

| Component | CPU Load | GPU Load | Notes |
|---|---|---|---|
| YOLOv8n-face (640px) | ~5% | ~15–25% | ~30ms/frame on RTX 3060 |
| InsightFace buffalo_l | ~10% | ~20–30% | ~15ms/face on GPU |
| IoU Tracker | <1% | 0% | Pure NumPy |
| OpenCV decode/display | ~5–10% | 0% | |
| SQLite writes | <1% | 0% | |
| **Total (GPU mode)** | **~20%** | **~40–55%** | At 30 FPS, skip=1 |
| **Total (CPU only)** | **~70–90%** | **0%** | ~3–8 FPS depending on resolution |

---

## Assumptions

1. Input video contains frontal or near-frontal faces (±45° yaw). Extreme profile faces may not generate valid embeddings.
2. `yolov8n-face.pt` must be present in the project root (see Setup step 3).
3. InsightFace `buffalo_l` model is downloaded automatically on first run (~300 MB).
4. `run_all.py` processes all `.mp4` files in `videos/` with a shared gallery. For independent processing, use `main.py` per video.
5. GPU (CUDA) is optional; the system falls back to CPU automatically.
6. Face re-identification across separate `main.py` runs (different sessions) is not persistent — the gallery is in-memory. Use `run_all.py` for cross-video re-ID.
7. The `detection_skip_frames` parameter trades latency for throughput. On skipped frames, tracks age naturally and exits are still detected.

---

## Demo Video

🎥 [Watch the live demo on YouTube](https://youtu.be/MMscJkJS1Wo)

---

*This project is a part of a hackathon run by https://katomaran.com*
