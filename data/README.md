# data/

TTS 訓練用的資料清單。各檔的產生流程見 [`docs/04_transcript_pipeline.md`](../docs/04_transcript_pipeline.md)（訓練清單）與 [`docs/05_dataset_split.md`](../docs/05_dataset_split.md)（切分）。

音訊本身（`wav/`）與參考音訊 `ref/ref.wav` 因版權**不進版控**，清單中的路徑需要在本機有對應的音檔。

## 檔案

| 檔案 | 行數 | 格式 | 路徑形式 | 產生工具 |
|---|---|---|---|---|
| `ryza_train.list` | 491 | `音檔路徑\|ryza\|ja\|文字` | 相對路徑（`wav/NNNNN.wav`） | `tools/transcribe.py` |
| `finetune_train.list` | 467 | 同上 | **絕對路徑** | `tools/split_train_test.py` |
| `finetune_test.list` | 24 | 同上 | 絕對路徑 | `tools/split_train_test.py` |
| `finetune_eval10.list` | 10 | 同上（test 的子集） | 絕對路徑 | `tools/split_train_test.py` |
| `split_manifest.csv` | 491 ＋ 表頭 | 11 欄：`utt_id,audio_path,speaker,duration,text,split,bucket,md5,source_rel,class,eval10` | `audio_path` 絕對、`source_rel` 相對 | `tools/split_train_test.py` |
| `ref/ref.txt` | 1 | 參考音訊的台詞 | — | `tools/split_train_test.py` |
| `ref/ref.wav` | — | 參考音訊（**不進版控**） | — | `tools/split_train_test.py`，之後被覆蓋 |

## 使用注意

- **文字**是專有名詞校正後的版本（[`reports/asr_report.md`](../reports/asr_report.md) 附錄 A）。Whisper 原始輸出保存在 git 歷史 commit `672e4e5` 的 `tools/asr_screening.json`。
- **`finetune_*.list` 用絕對路徑**，好讓位於其他目錄的 GPT-SoVITS 直接讀取。換機器時請以 `split_manifest.csv` 的 `source_rel` 重建，或重跑切分。
- **`finetune_test.list` 不得進入任何訓練步驟**。
- **`class` 欄只是句末標點的文字推斷**，不是聲學情緒，也不進入訓練。它只決定了哪些句子被留作測試集，見 [`docs/05_dataset_split.md`](../docs/05_dataset_split.md) §2。
- **`ref.txt` 與 `ref.wav` 不同步**：`ref.txt` 是 `00042` 的台詞，`ref.wav` 已被覆蓋為 `wav/03536.wav`。推論時的 prompt text 要用 03536 的原文，見 [`docs/05_dataset_split.md`](../docs/05_dataset_split.md) §3。
