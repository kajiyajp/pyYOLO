# 作成: 2026-06-28
# 目的: アノテーションツールをPyInstallerでexe化する社内配布用ビルドスクリプト。
#       torch/ultralytics等の重い依存は使わないため除外してサイズを削減する。
#       ビルド時に既存の captures/ と *.onnx を退避・復元し、撮影/注釈を消さない。

import os
import shutil
import tempfile
import PyInstaller.__main__

DIST_DIR = os.path.join("dist", "pyYOLO-Annotator")

# 除外する重量級モジュール（アノテーションツールでは未使用）
# onnxruntime は推論タブで使うため除外しない（onnxは推論には不要なので除外可）
EXCLUDES = [
    "torch", "torchvision", "ultralytics", "ultralytics_thop",
    "matplotlib", "scipy", "pandas", "polars",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtCharts",
]


def _backup_userdata():
    """既存distの captures/ と *.onnx を一時退避し、退避先パスを返す"""
    if not os.path.isdir(DIST_DIR):
        return None
    stash = tempfile.mkdtemp(prefix="pyyolo_stash_")
    cap = os.path.join(DIST_DIR, "captures")
    if os.path.isdir(cap):
        shutil.move(cap, os.path.join(stash, "captures"))
    for f in os.listdir(DIST_DIR):
        if f.endswith(".onnx"):
            shutil.move(os.path.join(DIST_DIR, f), os.path.join(stash, f))
    return stash


def _restore_userdata(stash):
    """退避した captures/ と *.onnx をビルド後のdistへ戻す"""
    if not stash:
        return
    for name in os.listdir(stash):
        src = os.path.join(stash, name)
        dst = os.path.join(DIST_DIR, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy(src, dst)
    shutil.rmtree(stash, ignore_errors=True)


def build():
    """PyInstallerを呼び出してexeを生成する（ユーザーデータは保全）"""
    stash = _backup_userdata()  # 撮影/注釈/モデルを退避

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

    # 学習・変換スクリプトをexe同階層にコピー（別プロセスpythonから実行するため）
    for s in ("train_model.py", "export_model.py"):
        if os.path.exists(s):
            shutil.copy(s, os.path.join(DIST_DIR, s))

    _restore_userdata(stash)  # 撮影/注釈/モデルを復元

    print("\nビルド完了: dist/pyYOLO-Annotator/pyYOLO-Annotator.exe")
    print("captures/ と *.onnx は保全しました")
    print("配布時は dist/pyYOLO-Annotator フォルダごとzip化して渡す")


if __name__ == "__main__":
    build()
