"""
tracker.py — Multi-face tracker: IoU continuity + ArcFace re-ID + masked-face support.

Track lifecycle:
  TEMPORARY  hits < min_hits     → no box, no count, no identity
  CONFIRMED  hits >= min_hits    → box drawn, identity assigned, counted once
  LOST       active=False        → kept up to max_missed_frames for reconnection
  DEAD       missed > window     → purged

Identity resolution order (highest → lowest):
  Stage 1 — IoU match → existing active track
  Stage 2 — IoU or embedding match → recently lost track
  Stage 3 — Gallery embedding match → known returning person
  Stage 4 — New UUID at confirmation — absolute last resort

Counting rule:
  counted_face_ids is the ONLY source of unique_count.
  A face_id enters it exactly once: when confirmed=True and face_id is valid.
  A new track_id NEVER means a new visitor.
"""
import uuid
import numpy as np
from dataclasses import dataclass
from typing import Optional

IOU_MATCH_THRESHOLD      = 0.20
IOU_RECONNECT_THRESHOLD  = 0.05
EMBEDDING_REFRESH_FRAMES = 40
MAX_AGE_DEFAULT          = 30
MAX_MISSED_DEFAULT       = 150

# At confirmation, use a lower threshold to catch same person with head turned
CONFIRMATION_THRESHOLD_OFFSET = 0.05


@dataclass
class Track:
    track_id:               int
    bbox:                   list
    hits:                   int                  = 1
    age:                    int                  = 0
    active:                 bool                 = True
    confirmed:              bool                 = False
    face_id:                Optional[str]        = None
    embedding:              Optional[np.ndarray] = None
    frames_since_embedding: int                  = 0
    missed_frames:          int                  = 0
    last_frame:             int                  = 0
    embed_cooldown:         int                  = 0  # frames to skip before retrying embed


class FaceTracker:
    def __init__(self,
                 max_age: int = MAX_AGE_DEFAULT,
                 min_hits: int = 5,
                 recognition_threshold: float = 0.35,
                 max_missed_frames: int = MAX_MISSED_DEFAULT,
                 **kwargs):
        self.max_age           = max_age
        self.min_hits          = min_hits
        self.threshold         = recognition_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks: list[Track] = []
        self._next_track_id    = 0

        # face_id → normalized 512-d ArcFace embedding (EMA updated)
        self.gallery: dict[str, np.ndarray] = {}

        # Single authoritative visitor counter — only source of unique_count
        self.counted_face_ids: set[str] = set()

        # Diagnostic counters
        self.stat_new_ids     = 0
        self.stat_reconnected = 0
        self.stat_rejected    = 0

    # ── geometry ──────────────────────────────────────────────────────────────
    def _iou(self, a: list, b: list) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _valid(self, emb: Optional[np.ndarray]) -> bool:
        return emb is not None and not np.all(emb == 0)

    # ── gallery ───────────────────────────────────────────────────────────────
    def _gallery_match(self, emb: np.ndarray,
                       exclude: set[str] | None = None,
                       threshold: float | None = None) -> tuple[Optional[str], float]:
        cutoff = threshold if threshold is not None else self.threshold
        best_id, best_sim = None, cutoff
        for fid, g in self.gallery.items():
            if exclude and fid in exclude:
                continue
            s = self._cosine(emb, g)
            if s > best_sim:
                best_sim, best_id = s, fid
        return best_id, best_sim

    def _gallery_update(self, face_id: str, emb: np.ndarray):
        """EMA update — prevents drift from a single bad frame."""
        if face_id in self.gallery:
            u = 0.9 * self.gallery[face_id] + 0.1 * emb
            self.gallery[face_id] = u / (np.linalg.norm(u) + 1e-6)
        else:
            self.gallery[face_id] = emb.copy()

    # ── main update ───────────────────────────────────────────────────────────
    def update(self, detections: list[dict],
               frame_idx: int = 0,
               **kwargs) -> tuple[list[dict], list[str]]:
        """
        detections : [{bbox, embedding, conf}]  — embedding may be None
        Returns    : (confirmed_tracks, exited_face_ids)
        """
        # age all tracks
        for t in self.tracks:
            if t.active:
                t.age += 1
                t.frames_since_embedding += 1
                if t.embed_cooldown > 0:
                    t.embed_cooldown -= 1
            else:
                t.missed_frames += 1

        matched_tids: set[int] = set()
        matched_dets: set[int] = set()

        active = [t for t in self.tracks if t.active]
        lost   = [t for t in self.tracks
                  if not t.active and t.missed_frames <= self.max_missed_frames]

        # ── Stage 1: IoU match → active tracks ───────────────────────────────
        if active and detections:
            mat = np.array([[self._iou(t.bbox, d["bbox"])
                             for d in detections] for t in active])
            while mat.size:
                i, j = np.unravel_index(np.argmax(mat), mat.shape)
                if mat[i, j] < IOU_MATCH_THRESHOLD:
                    break
                t, d = active[i], detections[j]
                t.bbox, t.age, t.hits, t.last_frame = d["bbox"], 0, t.hits+1, frame_idx
                emb = d.get("embedding")
                if self._valid(emb):
                    t.embedding = emb
                    t.frames_since_embedding = 0
                    if t.face_id:
                        self._gallery_update(t.face_id, emb)
                # If emb is None, retain t.embedding (last valid) — do NOT clear it
                matched_tids.add(t.track_id)
                matched_dets.add(j)
                mat[i, :] = -1
                mat[:, j] = -1

        # ── Stage 2: reconnect unmatched detections → lost tracks ─────────────
        for j, d in enumerate(detections):
            if j in matched_dets:
                continue

            # 2a — spatial reconnect
            best_iou, best_t = 0.0, None
            for t in lost:
                v = self._iou(t.bbox, d["bbox"])
                if v > best_iou:
                    best_iou, best_t = v, t
            if best_t and best_iou >= IOU_RECONNECT_THRESHOLD:
                best_t.active = True
                best_t.bbox, best_t.age, best_t.missed_frames = d["bbox"], 0, 0
                best_t.hits += 1
                best_t.last_frame = frame_idx
                emb = d.get("embedding")
                if self._valid(emb):
                    best_t.embedding = emb
                    best_t.frames_since_embedding = 0
                    if best_t.face_id:
                        self._gallery_update(best_t.face_id, emb)
                # If emb is None, retain best_t.embedding (last valid)
                matched_tids.add(best_t.track_id)
                matched_dets.add(j)
                self.stat_reconnected += 1
                print(f"[RECONNECTED] frame={frame_idx} track_id={best_t.track_id}"
                      f" face_id={best_t.face_id} reason=iou iou={best_iou:.3f}")
                continue

            # 2b — embedding reconnect to lost tracks
            emb = d.get("embedding")
            if self._valid(emb):
                best_sim, best_t = 0.0, None
                for t in lost:
                    ref = t.embedding if self._valid(t.embedding) \
                          else self.gallery.get(t.face_id) if t.face_id else None
                    if ref is None:
                        continue
                    s = self._cosine(emb, ref)
                    if s > best_sim:
                        best_sim, best_t = s, t
                if best_t and best_sim >= self.threshold:
                    best_t.active = True
                    best_t.bbox, best_t.age, best_t.missed_frames = d["bbox"], 0, 0
                    best_t.hits += 1
                    best_t.last_frame = frame_idx
                    best_t.embedding = emb
                    best_t.frames_since_embedding = 0
                    if best_t.face_id:
                        self._gallery_update(best_t.face_id, emb)
                    matched_tids.add(best_t.track_id)
                    matched_dets.add(j)
                    self.stat_reconnected += 1
                    print(f"[RECONNECTED] frame={frame_idx} track_id={best_t.track_id}"
                          f" face_id={best_t.face_id} reason=embedding sim={best_sim:.4f}")
                    continue

        # ── Stage 3: gallery match for remaining unmatched detections ──────────
        # Only exclude face_ids matched THIS frame via IoU — not all confirmed tracks
        # This allows a drifted/re-entering person to reclaim their ID
        matched_fids: set[str] = {
            t.face_id for t in self.tracks
            if t.active and t.track_id in matched_tids and t.face_id
        }

        for j, d in enumerate(detections):
            if j in matched_dets:
                continue
            emb = d.get("embedding")

            if not self._valid(emb):
                nt = Track(track_id=self._next_track_id, bbox=d["bbox"],
                           last_frame=frame_idx)
                self._next_track_id += 1
                self.tracks.append(nt)
                matched_dets.add(j)
                continue

            face_id, sim = self._gallery_match(emb, exclude=matched_fids)

            if face_id:
                # check if a drifted active track already carries this face_id
                drifted = next((t for t in self.tracks
                                if t.active and t.face_id == face_id
                                and t.track_id not in matched_tids), None)
                if drifted:
                    drifted.bbox, drifted.age = d["bbox"], 0
                    drifted.hits += 1
                    drifted.last_frame = frame_idx
                    drifted.embedding = emb
                    drifted.frames_since_embedding = 0
                    self._gallery_update(face_id, emb)
                    matched_tids.add(drifted.track_id)
                    matched_dets.add(j)
                    matched_fids.add(face_id)
                    self.stat_reconnected += 1
                    print(f"[RECONNECTED] frame={frame_idx} track_id={drifted.track_id}"
                          f" face_id={face_id} reason=gallery_drift sim={sim:.4f}")
                else:
                    nt = Track(track_id=self._next_track_id, bbox=d["bbox"],
                               face_id=face_id, embedding=emb, last_frame=frame_idx)
                    self._next_track_id += 1
                    self.tracks.append(nt)
                    matched_tids.add(nt.track_id)
                    matched_dets.add(j)
                    matched_fids.add(face_id)
            else:
                nt = Track(track_id=self._next_track_id, bbox=d["bbox"],
                           embedding=emb, last_frame=frame_idx)
                self._next_track_id += 1
                self.tracks.append(nt)
                matched_dets.add(j)

        # ── mark stale active tracks as lost ──────────────────────────────────
        exited_ids: list[str] = []
        for t in self.tracks:
            if t.active and t.age > self.max_age:
                t.active = False
                t.missed_frames = 0
                if t.face_id:
                    exited_ids.append(t.face_id)
                print(f"[TRACK_LOST] frame={frame_idx} track_id={t.track_id}"
                      f" face_id={t.face_id} confirmed={t.confirmed}")

        self.tracks = [t for t in self.tracks
                       if t.active or t.missed_frames <= self.max_missed_frames]

        # ── Stage 4: confirmation gate + identity assignment ──────────────────
        confirmed_out: list[dict] = []

        # face_ids assigned THIS frame — to prevent two tracks claiming same ID
        assigned_this_frame: set[str] = {
            t.face_id for t in self.tracks
            if t.active and t.track_id in matched_tids and t.face_id
        }

        # sort by hits descending so most stable track confirms first
        active_tracks = sorted(
            [t for t in self.tracks if t.active and t.hits >= self.min_hits],
            key=lambda t: t.hits, reverse=True
        )

        for t in active_tracks:

            just_confirmed = not t.confirmed
            if just_confirmed:
                t.confirmed = True

                if t.face_id is None:
                    if self._valid(t.embedding):
                        # Use lower threshold at confirmation + only exclude
                        # face_ids assigned this frame (not all confirmed tracks)
                        conf_threshold = max(0.25, self.threshold - CONFIRMATION_THRESHOLD_OFFSET)
                        gid, gsim = self._gallery_match(
                            t.embedding,
                            exclude=assigned_this_frame,
                            threshold=conf_threshold
                        )
                        if gid:
                            t.face_id = gid
                            assigned_this_frame.add(gid)
                            print(f"[RECOGNITION] frame={frame_idx} track_id={t.track_id}"
                                  f" face_id={t.face_id} sim={gsim:.4f}")
                        else:
                            t.face_id = str(uuid.uuid4())[:8]
                            self.stat_new_ids += 1
                            assigned_this_frame.add(t.face_id)
                            print(f"[NEW_ID] frame={frame_idx} track_id={t.track_id}"
                                  f" face_id={t.face_id}")
                    else:
                        # no embedding yet — defer identity assignment
                        t.confirmed = False
                        continue

                if self._valid(t.embedding):
                    self._gallery_update(t.face_id, t.embedding)

            # ── single authoritative count ─────────────────────────────────
            if t.face_id and t.face_id not in self.counted_face_ids:
                self.counted_face_ids.add(t.face_id)
                print(f"[COUNT] frame={frame_idx} track_id={t.track_id}"
                      f" face_id={t.face_id} unique_visitors={len(self.counted_face_ids)}")

            confirmed_out.append({
                "track_id":      t.track_id,
                "face_id":       t.face_id,
                "bbox":          t.bbox,
                "is_new":        just_confirmed,
                "is_recognized": (not just_confirmed and t.face_id in self.gallery
                                  and t.frames_since_embedding == 0),
            })

        return confirmed_out, exited_ids

    # ── helpers ───────────────────────────────────────────────────────────────
    def needs_embedding(self, track_id: int) -> bool:
        t = next((tr for tr in self.tracks if tr.track_id == track_id), None)
        if t is None:
            return True
        if t.embed_cooldown > 0:
            return False  # throttled — InsightFace returned None recently
        if not t.confirmed or not self._valid(t.embedding):
            return True
        return t.frames_since_embedding >= EMBEDDING_REFRESH_FRAMES

    def set_embed_cooldown(self, track_id: int, frames: int = 8):
        """Called when InsightFace returned None — throttle retries."""
        t = next((tr for tr in self.tracks if tr.track_id == track_id), None)
        if t:
            t.embed_cooldown = frames

    @property
    def unique_count(self) -> int:
        return len(self.counted_face_ids)
