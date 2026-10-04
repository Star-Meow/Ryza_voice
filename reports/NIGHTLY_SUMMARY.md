# 夜跑總結

## 模塊一：資料切分
- 判定：成功
- 驗收清單：11/11 通過
- 產出：
  - `tools/split_train_test.py`（分層抽樣工具，seed=42）
  - `data/finetune_train.list`（467 段，49.11 分）
  - `data/finetune_test.list`（24 段，2.36 分；Q 4／EXCL 2／ELL 3／DECL 15）
  - `data/finetune_eval10.list`（10 段，⊆ test；Q 2／EXCL 1／ELL 1／DECL 6）
  - `data/split_manifest.csv`（491 行，含 md5／split／bucket／source_rel）
  - `data/ref/ref.txt`（參考音訊文字；ref.wav 因版權排除版控，來源 utt_id 00042）
  - `reports/split_report.md`
- `data/ryza_train.list` 未修改（SHA256 前後一致）

## 模塊二：環境與模型
- 判定：成功
- 驗收清單：9/9 通過
- 產出：
  - `H:\git\GPT-SoVITS\.venv`（Python 3.10.11，201 套件）
  - v4 權重：s2Gv4.pth 769 MB／vocoder.pth 57.8 MB／s1v3.ckpt 155 MB／BERT 651 MB／HuBERT 189 MB
  - `reports/env_pip_freeze.txt`、`reports/01_env_model_prep.md`

## 整體判定
- DONE（兩模塊驗收清單全過）

## 起床後第一步
- **TF32 觀察項（不預先處理）**：`GPT_SoVITS/s2_train_v3_lora.py:44-45` 預設開啟 TF32。本專案 `tools/spk_model.py` 記錄的 CUDA 記憶體損壞屬**其他環境（RTX 3060）過往經驗，本機（RTX 4070 SUPER）未再發生**，故先以官方原版設定直接訓練；若出現 CUDA 損壞徵兆再補關閉（僅關 TF32 matmul 與 cudnn TF32，保留 cudnn 開啟）。

## 已知偏差（詳見 01_env_model_prep.md）
- pyopenjtalk-prebuilt 0.3.0 替代 pyopenjtalk>=0.4.1；jieba_fast 未裝（日文路徑不需要）；opencc 用預編譯 wheel；numpy 降至 1.26.4
