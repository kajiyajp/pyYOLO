# TODO — pyYOLO

> 作成：2026-06-28  
> 目的：PointLockへのYOLO導入に向けた残タスク管理

---

## 優先度：高

- [ ] 実機で各ターゲットを動画撮影する
  - L字マーク（白・コンクリート等）
  - ケガキ十字（zinc処理SM490鋼板）
  - 白丸＋ケガキ十字（zinc処理SM490鋼板）
  - L字マーク（黒皮SS400鋼板）

- [ ] 各ターゲットの動画からフレーム抽出・アノテーション
  - 各クラス50〜100枚目標
  - Make Sense または Roboflow でアノテーション

- [ ] 多クラスモデル（4クラス）で学習・推論テスト
  - `dataset.yaml` の `nc` と `names` を更新
  - 前回の `best.pt` を起点に転移学習

---

## 優先度：中

- [ ] PointLockに `vision/yolo_detector.py` を実装
  - YOLO推論 → ROI自動設定 → method自動選択
  - 既存 `pipeline.py` へ統合

- [ ] PointLockに `assets/models/best.pt` を同梱
  - `requirements.txt` に `ultralytics>=8.4.0` を追加

- [ ] PointLock ROADMAPにフェーズ1.5を追記
  - `Proposal_for_Introducing_YOLO.md` を参照して提案

---

## 優先度：低（将来）

- [ ] YOLO26n学習済みモデルをONNXエクスポートしてonnxruntimeで推論テスト
  - NMSフリーのためpostprocess()の簡略化を確認
  - YOLO11n ONNXと推論時間を比較（test_onnx_inference.pyを流用）

- [ ] PointLockに学習モード（アノテーションUI）を追加
  - PySide6で矩形描画UI（`calib_wizard.py` を参考に）
  - datalogのPNG → アノテーション → YOLO形式保存

- [ ] datalog蓄積トリガーで自動再学習ループを構築
  - 「新しい画像がN枚溜まったら再学習」
  - `best.pt` を自動更新

- [ ] Roboflowでのアノテーション作業効率化
  - 10枚手動 → AI自動補完で残りを効率化

- [ ] Keypoint検出の検討（将来）
  - 黒皮SS400などCVが苦手な素材が増えた時点で検討
  - Roboflow（Keypoint対応）でアノテーション作業が必要

---

## 完了済み

- [x] 環境構築（ultralytics / opencv-python）
- [x] サンプル動画（PointLock-mov.mp4）からフレーム100枚抽出
- [x] Make Senseでl_markを100枚アノテーション
- [x] train/val分割（80/20）
- [x] YOLOv11nで学習実験（v1・v2）
- [x] アノテーション可視化で座標の正確さを確認
- [x] PointLock ROADMAPへの提案書作成（Proposal_for_Introducing_YOLO.md）
- [x] README.md・.gitignore 整備・origin mainへpush
- [x] Tracking検証（test_tracking.py）：全523フレームでID:1を一貫追跡・信頼度0.88〜0.89
