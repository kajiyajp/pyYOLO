# 作成: 2026-06-28
# 目的: 学習済みモデルをONNX形式にエクスポートする（ultralyticsを推論時に不要にする）

from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/train/l_mark_v2/weights/best.pt"

model = YOLO(MODEL_PATH)

# ONNXエクスポート（動的バッチサイズ、opset=12は広く互換性あり）
path = model.export(format="onnx", imgsz=640, opset=12, dynamic=False)

print(f"エクスポート完了: {path}")
