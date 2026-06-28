# 作成: 2026-06-28
# 目的: onnxruntime（MIT）だけでYOLO検出を行う再利用可能なモジュール。
#       ultralyticsを含めずに推論できるため、PointLock組み込みの基礎にもなる。

import cv2
import numpy as np
import onnxruntime as ort


class OnnxDetector:
    """YOLO(.onnx)をonnxruntimeで実行する検出器（多クラス対応・NMSあり）"""

    def __init__(self, onnx_path, conf=0.30, iou=0.45, imgsz=640):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def _preprocess(self, frame):
        """画像をレターボックスでimgsz正方形に整え、YOLO入力テンソルに変換する"""
        h, w = frame.shape[:2]
        scale = self.imgsz / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        padded[top:top + nh, left:left + nw] = resized
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        return blob, scale, top, left

    def detect(self, frame):
        """1フレームを推論し、検出結果のリスト[dict]を返す"""
        h, w = frame.shape[:2]
        blob, scale, top, left = self._preprocess(frame)
        out = self.session.run(None, {self.input_name: blob})[0]

        # 出力形状を判定（YOLO11: (1,4+nc,N) / NMSフリー: (1,N,6)）
        arr = out[0]
        if arr.shape[0] in (5, 6) or arr.shape[0] < arr.shape[1]:
            # (4+nc, N) 形式 → 転置して (N, 4+nc)
            preds = arr.T
            boxes_xywh = preds[:, :4]
            scores = preds[:, 4:]
            cls_ids = scores.argmax(axis=1)
            confs = scores.max(axis=1)
            # cx,cy,w,h → x1,y1,x2,y2（letterbox補正）
            cx = (boxes_xywh[:, 0] - left) / scale
            cy = (boxes_xywh[:, 1] - top) / scale
            bw = boxes_xywh[:, 2] / scale
            bh = boxes_xywh[:, 3] / scale
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
        else:
            # (N, 6) = x1,y1,x2,y2,conf,cls（NMSフリー・既にピクセル/letterbox座標）
            confs = arr[:, 4]
            cls_ids = arr[:, 5].astype(int)
            x1 = (arr[:, 0] - left) / scale
            y1 = (arr[:, 1] - top) / scale
            x2 = (arr[:, 2] - left) / scale
            y2 = (arr[:, 3] - top) / scale

        keep = confs > self.conf
        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
        confs, cls_ids = confs[keep], cls_ids[keep]
        if len(confs) == 0:
            return []

        # 元画像内にクリップ
        x1 = np.clip(x1, 0, w).astype(int)
        y1 = np.clip(y1, 0, h).astype(int)
        x2 = np.clip(x2, 0, w).astype(int)
        y2 = np.clip(y2, 0, h).astype(int)

        # NMSで重複除去
        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(boxes, confs.tolist(), self.conf, self.iou)
        if len(idxs) == 0:
            return []

        results = []
        for i in np.array(idxs).flatten():
            results.append({
                "x1": int(x1[i]), "y1": int(y1[i]), "x2": int(x2[i]), "y2": int(y2[i]),
                "conf": float(confs[i]), "cls": int(cls_ids[i]),
            })
        return results
