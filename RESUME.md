# RESUME — pyYOLO

> セッション引継ぎメモ。再開時はこのファイル → ROADMAP.md → コードベースの順で確認する。

---

## 最終更新：2026-06-28

---

## 直近でやったこと

- CLAUDE.md / README.md / ROADMAP.md / RESUME.md を整備
- `feature/annotation-tool` ブランチで社内アノテーションツールの開発を開始
- `annotation_tool.py` を作成（PySide6 + UVC + YOLO形式保存）

---

## 次にやること

1. `annotation_tool.py` の動作確認
   - UVCカメラ（ノートPC）でライブ映像が表示されるか
   - 矩形描画 → クラス選択 → 保存 が正しく動くか
   - 保存した .txt が YOLOフォーマットになっているか（`check_annotation.py` で検証）
2. 動作OKなら `feature/annotation-tool` を `main` にマージ
3. Phase 2（ONNX推論のリアルタイム可視化）に着手

---

## 現在のブランチ状態

- 作業ブランチ：`feature/annotation-tool`
- マージ前に動作確認が必要

---

## 未解決・懸念点

- Python 3.14 + PySide6 6.11.1 でのUVCカメラ動作は未検証
- カメラが複数ある場合のデバイス選択UIは未実装（暫定でindex 0）
- YOLO26nの信頼度（0.808）がYOLO11n（0.884）より低い
  → 学習データが均質な動画1本のため。実機の多様な画像で改善見込み

---

## 重要な前提（忘れないこと）

- YOLO = 「見つける」、既存CV = 「測る」の役割分担
- ultralyticsは社内学習専用、製品推論は onnxruntime（MIT）
- 対象クラス：l_mark / cross / circle_cross / l_mark_black
