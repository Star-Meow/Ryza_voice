# 專案概述

> 本文說明專案的目標、範圍、模塊架構與評估標準，是 `docs/` 的起點。
> 安裝與執行方式見 [`README.md`](../README.md)。

---

## 1. 目標

以《Atelier Ryza 2》的遊戲語音，建立**ライザ（Ryza）日文聲線 TTS**，並驗證「用遊戲語音微調 TTS」是否可行。

為此需要先解決兩個問題：

1. 遊戲主語音庫沒有角色名稱，必須先**找出哪些語音是ライザ**。
2. 遊戲語音沒有文字稿，必須先**以 ASR 建出訓練用的逐字稿**。

接著才能以 GPT-SoVITS 為基礎模型進行微調。

## 2. 範圍與限制

| 項目 | 內容 |
|---|---|
| 資料來源 | `D:\SteamLibrary\steamapps\common\Atelier Ryza 2`，**全程唯讀** |
| 版權 | 音聲內容版權屬 Koei Tecmo／Gust，僅供個人用途。解出的音訊與模型權重**不進版控** |
| 語言 | 日文（ASR 固定 `language=ja`） |
| 硬體 | RTX 4070 SUPER 12 GB（決定了微調路線，見 [`06_model_selection.md`](06_model_selection.md)） |
| 基礎模型 | GPT-SoVITS；GPT（S1）與 SoVITS（S2）兩個階段都要微調 |

## 3. 模塊架構

```
遊戲音聲庫（KTSR）
  │  02 語音解包
  ▼
wav/ 9,569 段（無角色名稱）
  │  03 說話者篩選
  ▼
ryza_main/ 分級、top500
  │  04 轉譯資料集
  ▼
data/ryza_train.list 491 段（音檔路徑|ryza|ja|文字）
  │  05 資料切分
  ▼
train 467／test 24（含 eval10 10）
  │  06 選型 → 07 微調、推論、評估
  ▼
微調後模型、生成樣本、評估報告
```

| 模塊 | 做什麼 | 流程文件 | 結果 |
|---|---|---|---|
| 語音解包 | 解出 KTSR 音聲庫並建立索引 | [`02_unpack.md`](02_unpack.md) | `wav/`、`voice_index.csv` |
| 說話者篩選 | 用說話者嵌入，在沒有名稱的主庫中找出ライザ | [`03_speaker_id.md`](03_speaker_id.md) | `ryza_main/`、`ryza_main_index.csv` |
| 轉譯資料集 | ASR、過濾、去重、專有名詞校正，產出訓練清單 | [`04_transcript_pipeline.md`](04_transcript_pipeline.md) | [`asr_report.md`](../reports/asr_report.md) |
| 資料切分 | 分層抽樣，切出 train／test／eval10 | [`05_dataset_split.md`](05_dataset_split.md) | [`split_report.md`](../reports/split_report.md) |
| TTS 微調 | 選型，以及 v4 LoRA 的預處理、訓練、推論 | [`06_model_selection.md`](06_model_selection.md)、[`07_finetune.md`](07_finetune.md) | [`finetune_report.md`](../reports/finetune_report.md) |
| 評估 | 聲線相似度、可懂度、主觀試聽 | [`07_finetune.md`](07_finetune.md) §7 | [`finetune_report.md`](../reports/finetune_report.md)、[`defect_triage.md`](../reports/defect_triage.md) |

遇到問題時查 [`08_troubleshooting.md`](08_troubleshooting.md)，工具清單見 [`tools/README.md`](../tools/README.md)。

## 4. 評估標準

| 對象 | 標準 | 量測方式 | 結果 |
|---|---|---|---|
| 訓練清單 | 有效段數 ≥ 450，全數經人工確認為ライザ | 過濾、去重後的段數 | [`asr_report.md`](../reports/asr_report.md) §2 |
| 訓練清單 | 格式為 `音檔路徑\|ryza\|ja\|文字`，以專有名詞校正後的文字為準 | — | [`data/README.md`](../data/README.md) |
| ASR 品質 | 隨機抽 50 段人工校對，記錄 Whisper 原始輸出的 CER | reference＝人工校對文字，hypothesis＝Whisper 原始輸出 | [`asr_report.md`](../reports/asr_report.md) §3 |
| ASR 過濾 | 時長 < 2 秒或 > 15 秒、非說話聲、空輸出、重複、幻覺，皆須過濾並記錄原因 | — | [`asr_report.md`](../reports/asr_report.md) §2 |
| 測試集 | 訓練前保留約 5%（24 段），不得用於訓練 | train ∩ test = ∅ | [`split_report.md`](../reports/split_report.md) |
| 生成樣本 | (a) 測試集 10 句，以原台詞生成；(b) 全新台詞 10 句，涵蓋平敘、疑問、興奮、悲傷 | — | [`samples/`](../samples/README.md) |
| 聲線相似度 | 生成樣本對 A_high 中心點的平均 cosine ≥ 測試集真實音訊平均值的 **90%** | ECAPA-TDNN（`tools/spk_model.py`） | [`finetune_report.md`](../reports/finetune_report.md) §5.2 |
| 可懂度 | 生成音訊經 Whisper 轉寫，與輸入文字的平均 CER **≤ 10%** | Whisper large-v3 | [`finetune_report.md`](../reports/finetune_report.md) §5.3 |
| 主觀品質 | 聲線、發音、語氣自然度，各列出明顯缺陷 | 人工試聽 | [`finetune_report.md`](../reports/finetune_report.md) §6-3 |
| 訓練穩定性 | 訓練可穩定完成；遇到 GPU 不穩須記錄處理方式 | — | [`08_troubleshooting.md`](08_troubleshooting.md) §4 |

需求出處：[issue #1](https://github.com/Star-Meow/Ryza_voice/issues/1)；整理前的需求原文見
[`docs/01_issue.md` @ 262adda](https://github.com/Star-Meow/Ryza_voice/blob/262addab90ce98bc8229741ce2353c0d6fe8a8a8/docs/01_issue.md)。

## 5. 實際執行與原需求的差異

| 原需求 | 實際執行 | 原因與細節 |
|---|---|---|
| 轉譯範圍為 `A_high`（92）＋ `B_likely`（386） | 改走 `top500/`，依 `svm_dec` 排序取前 500 | 避免樣本數偏頗；B_likely 實際為 383。見 [`04_transcript_pipeline.md`](04_transcript_pipeline.md) §1.5 |
| 使用 Whisper large-v3 | 以 faster-whisper 載入同一份 large-v3 權重 | 權重相同，推論較快、顯存較省。見 [`asr_report.md`](../reports/asr_report.md) §1 |
| 記錄每段的排除原因 | 統計有保留，但逐段原因被 `--prune-raw` 從 `asr_screening.json` 移除 | 見 [`asr_report.md`](../reports/asr_report.md) §5 |
| 生成樣本與原音並列 | repo 內只提供生成檔與原音的對照表 | 原音因版權不進版控。見 [`samples/README.md`](../samples/README.md) |
