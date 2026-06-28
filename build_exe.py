# 作成: 2026-06-28
# 目的: アノテーションツールをPyInstallerでexe化する社内配布用ビルドスクリプト。
#       torch/ultralytics等の重い依存は使わないため除外してサイズを削減する。

import PyInstaller.__main__

# 除外する重量級モジュール（アノテーションツールでは未使用）
EXCLUDES = [
    "torch", "torchvision", "ultralytics", "ultralytics_thop",
    "matplotlib", "scipy", "pandas", "polars", "onnxruntime", "onnx",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtCharts",
]

def build():
    """PyInstallerを呼び出してexeを生成する"""
    args = [
        "annotation_tool.py",
        "--name", "pyYOLO-Annotator",
        "--windowed",          # コンソール窓を出さない（GUIアプリ）
        "--onedir",            # フォルダ形式（onefileより起動が速い）
        "--noconfirm",         # 既存distを確認なしで上書き
        "--clean",             # ビルドキャッシュをクリア
    ]
    for mod in EXCLUDES:
        args += ["--exclude-module", mod]

    PyInstaller.__main__.run(args)
    print("\nビルド完了: dist/pyYOLO-Annotator/pyYOLO-Annotator.exe")
    print("配布時は dist/pyYOLO-Annotator フォルダごとzip化して渡す")


if __name__ == "__main__":
    build()
