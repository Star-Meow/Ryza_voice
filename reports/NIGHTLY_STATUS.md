# 夜跑狀態

## 整體
- 狀態：DONE
- 最後更新：2026-10-04 04:35:00
- 分支：feat/tts-v4-data-prep @ 2add1fa
- 工作目錄：H:\git\Ryza_voice
- GPT-SoVITS 倉庫：H:\git\GPT-SoVITS @ 48b1a01

## 模塊一：資料切分
- 狀態：DONE（commit 2add1fa）
- 輸入：data/ryza_train.list（491 行，SHA256 9f56934b…，未變）
- 驗收清單：
  - [x] 輸入 491 行
  - [x] train ∩ test = ∅
  - [x] train ∪ test = 491
  - [x] len(train) = 467
  - [x] len(test) = 24
  - [x] len(eval10) = 10
  - [x] 路徑存在、四欄格式、文字非空且無 |
  - [x] 三清單無重複路徑
  - [x] eval10 ⊆ test
  - [x] ryza_train.list SHA256 未變
  - [x] 參考音訊來自 train（00042，4–8s 無削波，清單最早列）
- 產出檔案：
  - tools/split_train_test.py
  - data/finetune_train.list（467，49.11 分）
  - data/finetune_test.list（24，2.36 分）
  - data/finetune_eval10.list（10，0.92 分）
  - data/split_manifest.csv（491 行）
  - data/ref/ref.wav（排除版控，版權）＋ data/ref/ref.txt
  - reports/split_report.md
- 失敗原因：（無）

## 模塊二：環境與模型
- 狀態：DONE
- 驗收清單：
  - [x] .venv 建立，Python 3.10.11
  - [x] torch.cuda.is_available() = True（torch 2.5.1+cu121，RTX 4070 SUPER）
  - [x] requirements.txt 安裝成功（201 套件；4 項 Windows 偏差已記錄）
  - [x] pyopenjtalk 可用，日文 G2P 測試通過（pyopenjtalk-prebuilt 0.3.0）
  - [x] s2Gv4.pth 存在且大小合理（769 MB，torch.load 4 鍵）
  - [x] vocoder.pth 存在且大小合理（57.8 MB，torch.load 194 鍵）
  - [x] BERT（651 MB）/ HuBERT（189 MB）/ s1v3.ckpt（155 MB）存在
  - [x] finetune_train.list 467 行、路徑存在、格式無誤
  - [x] TF32 衝突已記錄（s2_train_v3_lora.py:44-45 預設開啟；本階段不 patch）
- 產出檔案：
  - reports/env_pip_freeze.txt（201 行）
  - reports/01_env_model_prep.md
- 失敗原因：（無）

## 已知偏差（詳見 reports/01_env_model_prep.md）
- pyopenjtalk-prebuilt 0.3.0 替代 pyopenjtalk>=0.4.1（Windows 無 MSVC）
- jieba_fast 未安裝（日文路徑不需要；cleaner.py 延遲 import 已驗證）
- opencc 改用預編譯 wheel 1.4.2
- numpy 降級至 1.26.4（pyopenjtalk-prebuilt 二進位要求）

## 起床後建議動作
1. 看 reports/split_report.md（模塊一門禁 11/11）與 reports/01_env_model_prep.md（模塊二門禁 9/9）。
2. **訓練前最高優先**：處理 TF32 衝突——s2_train_v3_lora.py:44-45 預設開啟 TF32，與本機 CUDA 損壞記錄衝突，需在訓練前補關閉（僅關 TF32，保留 cudnn 開啟）。
3. 確認清單檔名：夜跑規格用 finetune_*.list，docs/03-01_dataset.md 草稿用 ryza_*_finetune.list，待統一。
4. 1Aa 首跑時觀察 clean_text 是否大量輸出 UNK（version 與符號表對應）。
