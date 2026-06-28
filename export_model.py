# 作成: 2026-06-28
# 目的: best.pt を ONNX に変換する（学習・変換タブから呼ばれる）
# 使い方: python export_model.py <best.pt> [出力onnxパス]

import sys
import shutil

def main():
    """best.ptをONNX形式にエクスポートする"""
    if len(sys.argv) < 2:
        print("使い方: python export_model.py <best.pt> [out.onnx]")
        sys.exit(1)
    pt_path = sys.argv[1]

    from ultralytics import YOLO
    model = YOLO(pt_path)
    onnx_path = model.export(format="onnx", imgsz=640, opset=12, dynamic=False)

    # 出力先が指定されていればコピーする
    if len(sys.argv) > 2:
        shutil.copy(onnx_path, sys.argv[2])
        onnx_path = sys.argv[2]

    print(f"ONNX={onnx_path}")
    print("EXPORT_DONE")


if __name__ == "__main__":
    main()
