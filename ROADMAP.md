# ROADMAP — pyYOLO

> 作成：2026-06-28  
> 目的：PointLockへのYOLO導入に向けた、このリポでの開発ロードマップ

---

## Phase 0：調査・実験（完了）

- [x] 環境構築（ultralytics / opencv-python / onnxruntime / PySide6）
- [x] サンプル動画からフレーム抽出・アノテーション（Make Sense）
- [x] YOLO11nで学習（mAP50: 0.995）
- [x] Tracking検証（全523フレームID一貫）
- [x] OBB検出検証（角度検出・mAP50: 0.995）
- [x] ONNX推論（ultralytics不使用・22.6ms）
- [x] YOLO26n学習・比較（NMSフリー・19.5ms）
- [x] ライセンス方針確定（学習=AGPL社内、推論=onnxruntime MIT）

---

## Phase 1：社内アノテーションツール（実装完了・実機確認待ち）

> Make Senseのような作業を社内で完結させるGUIツール（annotation_tool.py）

- [x] UVCライブ映像 / 静止画ファイルの読み込み
- [x] フレームキャプチャ機能（Spaceキー）
- [x] マウスドラッグで矩形描画
- [x] クラス選択（l_mark / cross / circle_cross / l_mark_black）
- [x] YOLOフォーマット（.txt）で保存（Ctrl+S）
- [x] アノテーション一覧表示・削除
- [x] 保存済みラベルの読み込み・再編集
- [ ] **実機UVCカメラでの動作確認（次回ノートPCで実施）**
- [x] PyInstallerでexe化（build_exe.py・社内配布用・約246MB）

---

## Phase 2：ONNX推論の可視化（予定）

- [ ] 学習済み best.onnx をGUIから読み込み
- [ ] ライブ映像にリアルタイムでBB表示
- [ ] 信頼度・クラス名のオーバーレイ
- [ ] 検出結果をログ出力

---

## Phase 3：多クラス対応（予定）

- [ ] 実機で各ターゲットを動画撮影（4クラス）
- [ ] アノテーションツールで各クラス50〜100枚作成
- [ ] 多クラスモデルで学習・推論テスト
- [ ] dataset.yamlを4クラスに更新

---

## Phase 4：PointLock連携（予定）

> 詳細は Proposal_for_Introducing_YOLO.md 参照

- [ ] 段階①：best.onnx をPointLockに渡してROI自動設定
- [ ] 段階②：プロセス間通信でBB座標を連携
- [ ] 段階③：PointLock内に学習モードUIを組み込み（PointLockリポ側作業）

---

## Phase 5：製品化準備（将来）

- [ ] Ultralytics Enterpriseライセンス交渉（最大15台・買い切り希望）
- [ ] 閉じたネットワーク環境でのオフライン動作検証
- [ ] RTX搭載PCでのCUDA推論速度検証
- [ ] datalog蓄積トリガーの自動再学習ループ
