# 夜跑狀態

## 整體
- 狀態：RUNNING
- 最後更新：2026-10-04 01:12:00
- 分支：feat/tts-v4-data-prep（自 feat/voice-screening 建立）
- 工作目錄：H:\git\Ryza_voice
- GPT-SoVITS 倉庫：H:\git\GPT-SoVITS @ 48b1a01

## 模塊一：資料切分
- 狀態：RUNNING
- 輸入：data/ryza_train.list（491 行，SHA256 9f56934b…，唯讀）
- 驗收清單：
  - [x] 輸入 491 行（491，已 preflight 確認）
  - [ ] train ∩ test = ∅
  - [ ] train ∪ test = 491
  - [ ] len(train) = 467
  - [ ] len(test) = 24
  - [ ] len(eval10) = 10
  - [ ] 路徑存在、四欄格式、文字非空且無 |
  - [ ] 三清單無重複路徑
  - [ ] eval10 ⊆ test
  - [ ] ryza_train.list SHA256 未變
  - [ ] 參考音訊來自 train
- 產出檔案：（待產出）
- 失敗原因：（無）

## 模塊二：環境與模型
- 狀態：PENDING
- 驗收清單：
  - [ ] .venv 建立，Python 3.10
  - [ ] torch.cuda.is_available() = True
  - [ ] requirements.txt 安裝成功
  - [ ] pyopenjtalk 可用，日文 G2P 測試通過
  - [ ] s2Gv4.pth 存在且大小合理
  - [ ] vocoder.pth 存在且大小合理
  - [ ] BERT / HuBERT / s1v3.ckpt 存在
  - [ ] finetune_train.list 467 行、路徑存在
  - [ ] TF32 衝突已記錄
- 產出檔案：（待產出）
- 失敗原因：（無）

## 起床後建議動作
- 模塊一完成後先看 reports/split_report.md 的門禁勾選表。
- 模塊二完成後看 reports/01_env_model_prep.md 的模型清單與 G2P 驗證結果。
