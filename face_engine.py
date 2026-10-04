"""
face_engine.py - Face detection (YOLOv8) + ArcFace embedding (InsightFace).
"""
import os
import warnings
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import onnxruntime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from insightface.app import FaceAnalysis

if not torch.cuda.is_available():
    torch.set_num_threads(min(4, torch.get_num_threads()))

NMS_IOU_THRESHOLD = 0.45
MIN_ASPECT_RATIO  = 0.45
MAX_ASPECT_RATIO  = 2.20


class FaceEngine:
    def __init__(self, yolo_model: str, embedding_model: str,
                 yolo_conf: float = 0.30, min_face_size: int = 30,
                 verbose_diagnostics: bool = False):
        self.yolo_conf     = yolo_conf
        self.min_face_size = min_face_size
        self.verbose       = verbose_diagnostics

        if not os.path.exists(yolo_model):
            print(f"[FaceEngine] {yolo_model} not found, falling back to yolov8n.pt")
            yolo_model = "yolov8n.pt"

        self.detector   = YOLO(yolo_model)
        self._face_model = "face" in yolo_model.lower()

        available_providers = onnxruntime.get_available_providers()
        is_cuda = "CUDAExecutionProvider" in available_providers and torch.cuda.is_available()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if is_cuda \
                    else ["CPUExecutionProvider"]
        ctx_id = 0 if is_cuda else -1

        self.recognizer = FaceAnalysis(
            name=embedding_model,
            allowed_modules=["recognition", "detection"],
            providers=providers,
        )
        self.recognizer.prepare(ctx_id=ctx_id, det_size=(640, 640))

        # Direct access to recognition model for fallback embedding extraction
        self._rec_model = self.recognizer.models.get("recognition", None)

        self.rejection_stats = {
            "too_small": 0, "bad_aspect": 0,
            "edge_boundary_artifact": 0, "total_rejected": 0,
        }
        self._warmup()

    def _warmup(self):
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            with torch.inference_mode():
                _ = self.detector.predict(dummy, imgsz=640, conf=self.yolo_conf, verbose=False)
            _ = self.recognizer.get(np.zeros((112, 112, 3), dtype=np.uint8))
        except Exception:
            pass

    def _nms(self, faces: list) -> list:
        if len(faces) <= 1:
            return faces
        faces = sorted(faces, key=lambda x: x["conf"], reverse=True)
        kept = []
        for c in faces:
            if not any(self._iou(c["bbox"], k["bbox"]) > NMS_IOU_THRESHOLD for k in kept):
                kept.append(c)
        return kept

    def _iou(self, a: list, b: list) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        denom = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / denom if denom > 0 else 0.0

    def detect_faces(self, frame: np.ndarray) -> list:
        if frame is None or frame.size == 0:
            return []

        fh, fw = frame.shape[:2]

        with torch.inference_mode():
            results = self.detector.predict(
                frame, imgsz=1280, conf=self.yolo_conf, verbose=False
            )[0]

        raw_faces = []
        for box in results.boxes:
            if int(box.cls[0]) != 0:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if not self._face_model:
                face_h = int((y2 - y1) * 0.35)
                y2 = y1 + face_h
                margin_x = int((x2 - x1) * 0.15)
                x1 += margin_x; x2 -= margin_x

            cx1 = max(0, min(x1, fw - 1))
            cy1 = max(0, min(y1, fh - 1))
            cx2 = max(cx1 + 1, min(x2, fw))
            cy2 = max(cy1 + 1, min(y2, fh))
            w, h = cx2 - cx1, cy2 - cy1

            if w < self.min_face_size or h < self.min_face_size:
                self.rejection_stats["too_small"] += 1
                self.rejection_stats["total_rejected"] += 1
                if self.verbose:
                    print(f"[FACE_REJECTED] reason=too_small w={w} h={h}")
                continue

            aspect = w / (h + 1e-6)
            if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
                self.rejection_stats["bad_aspect"] += 1
                self.rejection_stats["total_rejected"] += 1
                if self.verbose:
                    print(f"[FACE_REJECTED] reason=bad_aspect aspect={aspect:.2f}")
                continue

            touches_edge = (cx1 <= 2 or cy1 <= 2 or cx2 >= fw - 2 or cy2 >= fh - 2)
            if touches_edge and (w < self.min_face_size + 10 or h < self.min_face_size + 10
                                 or conf < self.yolo_conf + 0.08):
                self.rejection_stats["edge_boundary_artifact"] += 1
                self.rejection_stats["total_rejected"] += 1
                if self.verbose:
                    print(f"[FACE_REJECTED] reason=edge_artifact w={w} h={h} conf={conf:.2f}")
                continue

            if self.verbose:
                print(f"[FACE_DETECTED] conf={conf:.2f} w={w} h={h}")
            raw_faces.append({"bbox": [cx1, cy1, cx2, cy2], "conf": conf})

        return self._nms(raw_faces)

    def get_embedding(self, frame: np.ndarray, bbox: list) -> np.ndarray | None:
        if frame is None or frame.size == 0:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]

        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        pad = max(30, int(max(face_w, face_h) * 0.40))

        crop = frame[
            max(0, y1 - pad): min(fh, y2 + pad),
            max(0, x1 - pad): min(fw, x2 + pad),
        ]
        if crop.size == 0:
            return None

        # ensure crop is large enough for InsightFace internal detector
        ch, cw = crop.shape[:2]
        if cw < 112 or ch < 112:
            s = 112.0 / min(cw, ch)
            crop = cv2.resize(crop, (int(cw * s), int(ch * s)),
                              interpolation=cv2.INTER_LINEAR)

        detected = self.recognizer.get(crop)
        if detected:
            if len(detected) > 1:
                crop_cx, crop_cy = crop.shape[1] / 2.0, crop.shape[0] / 2.0
                target_face = min(detected, key=lambda f: (
                    ((f.bbox[0]+f.bbox[2])/2 - crop_cx)**2 +
                    ((f.bbox[1]+f.bbox[3])/2 - crop_cy)**2
                ))
            else:
                target_face = detected[0]
            emb = target_face.normed_embedding
            norm = np.linalg.norm(emb)
            return emb / (norm + 1e-6)

        # Fallback: InsightFace detector missed the face in the crop.
        # Resize the tight face crop (no padding) to 112x112 and run the
        # recognition model directly — no re-detection needed.
        if self._rec_model is None:
            if self.verbose:
                print(f"[EMB_FALLBACK] rec_model unavailable bbox={bbox}")
            return None
        try:
            tight = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
            if tight.size == 0:
                return None
            aligned = cv2.resize(tight, (112, 112), interpolation=cv2.INTER_LINEAR)
            emb = self._rec_model.get_feat(aligned[np.newaxis])[0]
            norm = np.linalg.norm(emb)
            if norm < 1e-6:
                return None
            if self.verbose:
                print(f"[EMB_FALLBACK] used direct rec model bbox={bbox}")
            return emb / norm
        except Exception as ex:
            if self.verbose:
                print(f"[EMB_FALLBACK_FAIL] {ex} bbox={bbox}")
            return None

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
