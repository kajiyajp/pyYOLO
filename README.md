# pyYOLO

> 作成：2026-06-28  
> 目的：YOLO導入調査・実験リポジトリ（開発者の個人環境）

---

## このリポの位置づけ

```
【社内製品】PointLock（CNCビジュアルサーボシステム）
    ↑
    └── YOLO導入の調査・実験をここで行い、成果をPointLockに反映する
```

- PointLockのコードは**含まない**
- 学習済みモデル（`*.pt` `*.onnx`）はgit管理対象外（`.gitignore`）
- 動画ファイル（`*.mp4`）もgit管理対象外

---

## pyYOLO-ano(仮)のフロー
```
左のナビゲーションタブ
 キャプチャ　　　動画キャプチャ
    ↓
 アノテーション　ターゲット選択とキーポイントがあれば、点を打つ　→ 保存
    ↓ 
 学習・変換　　　モデル作成
    ↓ 
  推論　　　　　BB確認


---

## PointLockにおけるYOLOの役割分担

```
ライブ映像（UVCカメラ）
    ↓
YOLO（onnxruntime + best.onnx）
    → ターゲット発見 → BB座標・クラスを出力 → ROIを自動設定
    ↓
既存CV（detect_white_corner等）
    → 精密座標算出（±0.1px）
    ↓
ΔX/ΔY → CNC軸フィードバック制御
```

**YOLOは「見つける」、CVは「測る」の役割分担。**

---

## 実施済み実験

| 実験 | 内容 | 結果 |
|------|------|------|
| フレーム抽出 | PointLock-mov.mp4から100枚抽出 | ✅ |
| アノテーション | Make SenseでL字マーク100枚 | ✅ |
| YOLO11n学習（v1/v2） | 転移学習・データ拡張 | mAP50: 0.995 |
| Tracking検証 | model.track()で動画追跡 | 全523フレームID:1一貫 |
| OBB検出 | 回転対応BB・角度検出 | mAP50: 0.995 |
| ONNX推論 | ultralytics不使用で推論 | 22.6ms/frame |
| YOLO26n学習 | 最新モデルで比較 | mAP50: 0.995 |
| YOLO26n ONNX | NMSフリー推論 | 19.5ms/frame |

---

## ファイル構成

```
pyYOLO/
├── README.md                  # このファイル
├── CLAUDE.md                  # Claude Code作業ルール
│
├── annotation_tool.py         # 社内ツール本体（4タブ・撮影〜推論）
├── onnx_detector.py           # onnxruntime推論（PointLock移植の土台）
├── train_model.py             # 学習（引数対応・ツールから呼ばれる）
├── export_model.py            # ONNX変換（引数対応・ツールから呼ばれる）
├── build_exe.py               # PyInstallerでexe化
│
├── docs/                      # 運用・計画ドキュメント
│   ├── ROADMAP.md             #   開発ロードマップ
│   ├── RESUME.md              #   セッション引継ぎメモ
│   ├── STARTUP.md             #   立ち上げの記録
│   ├── TODO.md                #   残タスク
│   └── Proposal_for_Introducing_YOLO.md  # PointLock提案書
│
├── scripts/                   # 実験・データ準備スクリプト
│   ├── extract_frames.py      #   動画からフレーム抽出
│   ├── prepare_dataset.py     #   train/val分割
│   ├── prepare_real_dataset.py#   実機撮影データの分割
│   ├── convert_to_obb.py      #   ラベルをOBB形式に変換
│   ├── check_annotation.py    #   アノテーション可視化
│   ├── export_onnx.py         #   ONNXエクスポート
│   ├── train_real.py / train_obb.py / train_yolo26.py  # 各種学習
│   ├── test_*.py              #   推論/Tracking/OBB/ONNXのテスト
│   └── benchmark*.py / compare_yolo11_yolo26.py  # 速度・精度比較
│
├── configs/                   # データセット設定
│   ├── dataset.yaml
│   ├── dataset_obb.yaml
│   └── dataset_real.yaml
│
├── dataset/                   # 初期データ（動画100フレーム＋ラベル）
└── samples/                   # 実験結果の画像
    ├── inference/ obb_result/ onnx_result/ compare/ real_result/
    └── track_*.jpg

※ scripts/ は原則リポルートから実行する（例: python scripts/train_real.py）
※ runs/ dist/ build/ *.pt *.onnx captures/ datasets/ は .gitignore 対象
```

---

## ライセンス方針

| 用途 | ツール | ライセンス |
|------|--------|-----------|
| 学習（社内のみ） | ultralytics | AGPL-3.0（配布しない） |
| 推論（製品同梱） | onnxruntime | MIT ✅ |
| GUI開発 | PySide6 | LGPL-3.0 ✅ |
| 画像処理 | opencv-python | Apache 2.0 ✅ |

---

## 環境

```
Python    : 3.14.5
ultralytics: 8.4.80
onnxruntime: 1.27.0
PySide6   : 6.11.1
opencv-python: 4.13.0.92
OS        : Windows 11
GPU       : AMD Radeon RX9070 XT（CPU推論で実験済み）
```
