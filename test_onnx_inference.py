# 作成: 2026-06-28
# 目的: ultralyticsを一切使わず、onnxruntimeだけでL字マーク検出を行う
#       → PointLockへの組み込みはこのコードが基礎になる

import cv2
import numpy as np
import onnxruntime as ort
import time

ONNX_PATH = "runs/detect/runs/train/l_mark_v2/weights/best.onnx"
VIDEO_PATH = "samples/PointLock-mov.mp4"
OUTPUT_DIR = "samples/onnx_result"
CONF_THRESH = 0.3
IOU_THRESH = 0.45
INPUT_SIZE = 640

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# セッション初期化（ultralyticsは不要）
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print(f"onnxruntime バージョン: {ort.__version__}")
print(f"モデル入力: {input_name} {session.get_inputs()[0].shape}")

def preprocess(frame):
    """画像をYOLO入力形式に変換"""
    h, w = frame.shape[:2]
    # レターボックスリサイズ
    scale = INPUT_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_top = (INPUT_SIZE - new_h) // 2
    pad_left = (INPUT_SIZE - new_w) // 2
    padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    # HWC→CHW、0-1正規化
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, pad_top, pad_left

def postprocess(output, scale, pad_top, pad_left, orig_h, orig_w):
    """YOLO出力(1,5,8400)をバウンディングボックスに変換"""
    preds = output[0][0].T  # (8400, 5): cx cy w h conf
    mask = preds[:, 4] > CONF_THRESH
    preds = preds[mask]
    if len(preds) == 0:
        return []

    # 座標をパディング・スケール考慮で元画像座標に戻す
    cx = (preds[:, 0] - pad_left) / scale
    cy = (preds[:, 1] - pad_top) / scale
    w  = preds[:, 2] / scale
    h  = preds[:, 3] / scale
    confs = preds[:, 4]

    x1 = np.clip(cx - w/2, 0, orig_w).astype(int)
    y1 = np.clip(cy - h/2, 0, orig_h).astype(int)
    x2 = np.clip(cx + w/2, 0, orig_w).astype(int)
    y2 = np.clip(cy + h/2, 0, orig_h).astype(int)

    # NMS（Non-Maximum Suppression）
    boxes = np.stack([x1, y1, x2, y2], axis=1).tolist()
    scores = confs.tolist()
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH)
    if len(indices) == 0:
        return []

    results = []
    for i in indices.flatten():
        results.append({
            "x1": x1[i], "y1": y1[i], "x2": x2[i], "y2": y2[i],
            "conf": float(confs[i]),
            "cx": int((x1[i] + x2[i]) / 2),
            "cy": int((y1[i] + y2[i]) / 2),
        })
    return results

# 動画から5フレーム抽出して推論
cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

times = []
for i, pos in enumerate([int(total * t / 5) for t in range(5)]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if not ret:
        continue

    blob, scale, pad_top, pad_left = preprocess(frame)

    t0 = time.perf_counter()
    output = session.run(None, {input_name: blob})
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

    detections = postprocess(output, scale, pad_top, pad_left, orig_h, orig_w)

    for d in detections:
        cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 255, 0), 2)
        cv2.circle(frame, (d["cx"], d["cy"]), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"l_mark {d['conf']:.2f}", (d["x1"], d["y1"]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"frame{i:02d}: 中心=({d['cx']},{d['cy']}) 信頼度={d['conf']:.3f} 推論時間={times[-1]:.1f}ms")

    if not detections:
        print(f"frame{i:02d}: 未検出")

    cv2.imwrite(f"{OUTPUT_DIR}/onnx_{i:02d}.jpg", frame)

cap.release()
print(f"\n平均推論時間: {np.mean(times):.1f}ms（ultralytics不使用）")
print(f"結果: {OUTPUT_DIR}/")
