# 作成: 2026-06-28
# 目的: YOLO11n vs YOLO26n をONNX+onnxruntimeで推論速度比較する

import cv2
import time
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

VIDEO_PATH = "samples/PointLock-mov.mp4"
N_FRAMES = 20
INPUT_SIZE = 640
CONF_THRESH = 0.3
IOU_THRESH = 0.45

# --- Step1: YOLO26nをONNXエクスポート ---
print("YOLO26n → ONNX エクスポート中...")
model26 = YOLO("runs/detect/runs/detect/l_mark_yolo26/weights/best.pt")
onnx26_path = model26.export(format="onnx", imgsz=640, opset=12, dynamic=False, verbose=False)
print(f"  → {onnx26_path}")

ONNX_MODELS = {
    "YOLO11n-ONNX": "runs/detect/runs/train/l_mark_v2/weights/best.onnx",
    "YOLO26n-ONNX": str(onnx26_path),
}

# --- Step2: フレーム準備 ---
cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frames = []
for pos in [int(total * t / N_FRAMES) for t in range(N_FRAMES)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
cap.release()

def preprocess(frame):
    h, w = frame.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_top = (INPUT_SIZE - new_h) // 2
    pad_left = (INPUT_SIZE - new_w) // 2
    padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, pad_top, pad_left

def postprocess_with_nms(output, scale, pad_top, pad_left):
    """YOLO11n用（NMSあり）"""
    preds = output[0][0].T
    mask = preds[:, 4] > CONF_THRESH
    preds = preds[mask]
    if len(preds) == 0:
        return []
    cx = (preds[:, 0] - pad_left) / scale
    cy = (preds[:, 1] - pad_top) / scale
    w  = preds[:, 2] / scale
    h  = preds[:, 3] / scale
    confs = preds[:, 4]
    x1 = np.clip(cx - w/2, 0, orig_w).astype(int)
    y1 = np.clip(cy - h/2, 0, orig_h).astype(int)
    x2 = np.clip(cx + w/2, 0, orig_w).astype(int)
    y2 = np.clip(cy + h/2, 0, orig_h).astype(int)
    boxes = np.stack([x1, y1, x2, y2], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes, confs.tolist(), CONF_THRESH, IOU_THRESH)
    if len(indices) == 0:
        return []
    return [{"conf": float(confs[i])} for i in indices.flatten()]

def postprocess_nms_free(output):
    """YOLO26n用（NMSフリー）: 出力形式 (1,300,6) = [x1,y1,x2,y2,conf,class_id]"""
    preds = output[0][0]  # (300, 6) ピクセル座標
    mask = preds[:, 4] > CONF_THRESH
    preds = preds[mask]
    return [{"conf": float(p[4])} for p in preds] if len(preds) > 0 else []

# --- Step3: 比較実行 ---
print(f"\n{'モデル':<16} {'推論時間avg':>12} {'推論時間min':>12} {'後処理':>10} {'信頼度avg':>10}")
print("─" * 65)

for name, onnx_path in ONNX_MODELS.items():
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    times_infer = []
    confs = []

    for frame in frames:
        blob, scale, pad_top, pad_left = preprocess(frame)

        t0 = time.perf_counter()
        output = session.run(None, {input_name: blob})
        t1 = time.perf_counter()
        times_infer.append((t1 - t0) * 1000)

        if "11" in name:
            dets = postprocess_with_nms(output, scale, pad_top, pad_left)
            postproc = "NMSあり"
        else:
            dets = postprocess_nms_free(output)
            postproc = "NMSフリー"

        if dets:
            confs.append(max(d["conf"] for d in dets))

    avg_time = np.mean(times_infer)
    min_time = np.min(times_infer)
    avg_conf = np.mean(confs) if confs else 0.0
    print(f"{name:<16} {avg_time:>10.1f}ms {min_time:>10.1f}ms {postproc:>10} {avg_conf:>10.3f}")

print(f"\nonnxruntime: {ort.__version__}  ultralytics使用: 推論フェーズのみエクスポート用")
