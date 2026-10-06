# Atelier Ryza 2 萊莎聲線 TTS 專案

以《Atelier Ryza 2》遊戲音聲建立**ライザ（Ryza）日文聲線 TTS**，分兩個模塊：

- **Part 1** — 語音解包、說話者篩選與 ASR 轉譯，產出 `data/ryza_train.list`
  （491 段日文語音，約 51.5 分鐘，48 kHz mono，格式 `音檔路徑|ryza|ja|文字`）。**已完成。**
- **Part 2** — 以此為基礎使用 GPT-SoVITS 微調萊莎聲線 TTS（RTX 4070 SUPER 12 GB）。**進行中。**

資料來源 `D:\SteamLibrary\steamapps\common\Atelier Ryza 2` **全程唯讀**，未做任何變更。
音聲內容版權屬 Koei Tecmo／Gust，僅供個人用途。

## 目錄結構

| 路徑 | 內容 |
|---|---|
| `wav/` | **VOICE 主語音庫**：9,569 個 WAV（48kHz／16bit／mono），13.76 小時、4.5 GB。版權，不進版控 |
| `voice_index.csv` | 主庫索引：cue 編號 ↔ sound_id ↔ 時長 ↔ stbin 偏移 |
| `battle_voice/` | **戰鬥角色語音庫**（014／015）646 個 WAV，檔名即遊戲內 cue 名稱。版權，不進版控 |
| `battle_voice_index.csv` | 戰鬥庫索引 |
| `ryza_battle/` | 戰鬥庫中節選的 **48 個ライザ戰鬥語音**。版權，不進版控 |
| `ryza_main/` | 說話者篩選的分級結果與 `top500/`。版權，不進版控 |
| `ryza_main_index.csv` | 主庫全 9,569 檔依說話者相似度排序的完整索引 |
| `ryza_main_review.csv` | `B_review/` 40 檔的優先待審清單 |
| `data/` | 訓練清單：`ryza_train.list`（491）、`finetune_{train,test,eval10}.list`、`split_manifest.csv` |
| `docs/` | 各階段詳細文件與選型分析（見「文件地圖」） |
| `configs/` | LoRA 訓練設定檔：`s1_ryza.yaml`、`s2_lora_ryza.json` 等 5 檔 |
| `samples/` | Part 2 生成樣本（16 epoch）：(a) eval10 測試集 10 句＋(b) 全新台詞 10 句，共 20 個 WAV。見 [`samples/README.md`](samples/README.md) |
| `reports/` | 環境準備、資料切分、ASR／CER 與 Gate 報告；`reports/logs/` 為訓練執行紀錄 |
| `tools/` | 解包、嵌入、ASR、切分等全部工具（見「工具入口」） |
| `requirements.txt` | Python 相依套件與版本 |
| `.gitignore` | 排除音訊、模型權重與中間產物的規則 |

## 專案流程總覽

| 模塊 | 內容 | 方法 | 關鍵結果 |
|---|---|---|---|
| **Part 1 ①** | 語音解包 | vgmstream r2117 批次解出 KTSR 音聲庫 | 9,569 + 646 個 WAV |
| **Part 1 ②** | 說話者篩選 | ECAPA-TDNN 嵌入比對 + LinearSVC 決策邊界 | `A_high` 92 檔全數人工確認為ライザ |
| **Part 1 ③** | ASR 轉譯 | faster-whisper large-v3 + 過濾、去重、專有名詞校正 | **491 段／51.5 分鐘，CER 1.0%** |
| **Part 2** | GPT-SoVITS 微調萊莎聲線 TTS | v4 LoRA 路線 | 進行中；範圍、選型與資料切分見 [`docs/06_dataset.md`](docs/06_dataset.md) |

Part 1 驗收要件已全數滿足：有效轉譯段數 491（門檻 450），且全數經人工確認為ライザ。

## 需求與文件對照（issue [#1](https://github.com/Star-Meow/Ryza_voice/issues/1)）

> 對應 PR：[#6](https://github.com/Star-Meow/Ryza_voice/pull/6)（已合併）。
> 逐條判定（PASS／PARTIAL）與證據的獨立稽核見 [`issue_compliance.md`](reports/issue_compliance.md)。

### Part 1：ASR 轉譯

- ASR 模型與參數（Whisper large-v3） : [`asr_report.md`](reports/asr_report.md) §1
- 轉譯工具（不寫死絕對路徑） : [`transcribe.py`](tools/transcribe.py)
- 過濾規則與排除統計 : [`screen_asr.py`](tools/screen_asr.py)、[`asr_report.md`](reports/asr_report.md) §2
- 訓練清單（491 段，`音檔路徑|ryza|ja|文字`） : [`ryza_train.list`](data/ryza_train.list)
- ASR 報告（模型／統計／時長／50 段 CER） : [`asr_report.md`](reports/asr_report.md)
- 50 段抽查 CER 資料 : [`cer_sample.csv`](reports/cer_sample.csv)
- 專有名詞校正規則與差異 : [`glossary.py`](tools/glossary.py)、[`correction_diff.md`](reports/correction_diff.md)
- 相依套件與版本 : [`requirements.txt`](requirements.txt)

### Part 2：GPT-SoVITS finetune

- 基礎模型版本與 commit hash : [`finetune_report.md`](reports/finetune_report.md) §10、[PR #6](https://github.com/Star-Meow/Ryza_voice/pull/6)
- 測試集切分（467／24／10） : [`split_report.md`](reports/split_report.md)、[`finetune_train.list`](data/finetune_train.list)、[`finetune_test.list`](data/finetune_test.list)
- 訓練設定檔 : [`s1_ryza.yaml`](configs/s1_ryza.yaml)、[`s2_lora_ryza.json`](configs/s2_lora_ryza.json)
- 訓練 log : [`stage_d_train.md`](reports/logs/stage_d_train.md)
- 重現步驟（資料準備→推論） : [`07_finetune.md`](docs/07_finetune.md)
- 生成樣本 (a)：測試集 10 句 : [`samples/`](samples/)（`eval10_*.wav`）
- 生成樣本 (b)：全新台詞 10 句 : [`samples/`](samples/)（`new_*.wav`）
- 樣本說明與原音對照表 : [`samples/README.md`](samples/README.md)
- 聲線相似度（0.8152／91.5%） : [`finetune_report.md`](reports/finetune_report.md) §5.2
- 可懂度 CER（2.99%）逐句文字稿 : [`cer_generated.csv`](reports/cer_generated.csv)
- 主觀試聽評語 : [`finetune_report.md`](reports/finetune_report.md) §6-3
- 缺陷排查（P-1～P-3） : [`defect_triage.md`](reports/defect_triage.md)
- GPU 不穩處置記錄 : [`09_pitfalls.md`](docs/09_pitfalls.md) PK-013

## Part 1 ①：語音解包

遊戲音聲為光榮特庫摩自有的 **KTSR「Sound Ring」** 封裝，共兩種語音庫：

| 語音庫 | 檔數 | cue 名稱 |
|---|---|---|
| `wav/`（VOICE 主庫，95.7% 台詞） | 9,569 | **全空（0/9569）**，僅存遞增的 sound_id |
| `battle_voice/`（戰鬥庫） | 646 | 已解密，可直接看出角色歸屬 |

- 主庫無名稱欄位，**無法從音聲庫本身分辨角色**，故 Part 1 ② 改以說話者嵌入篩選（見下）
- 音訊本體為 KOVS 容器 + Ogg 串流雙層混淆；戰鬥庫 cue 名稱另經 LCG XOR 加密
- 容器結構、加密演算法、索引欄位對應、重新解出指令：[`docs/03_unpack.md`](docs/03_unpack.md)

## Part 1 ②：說話者篩選（聲紋比對）

以 speechbrain ECAPA-TDNN（192 維）對主庫全 9,569 檔計算說話者嵌入，
以使用者確認的 3 個ライザ對話語音為參考在主庫內部互比，再依 LinearSVC 決策值分級：

| 分級 | 檔數 | 判斷 |
|---|---|---|
| `A_high/` | 92 | 已全部人工確認為ライザ |
| `B_likely/` | 383 | `svm_dec > −0.433`，可能正確 |
| `C_possible/` | 728 | `svm_dec > −0.433`，不確定 |
| `D_flagged/` | 165 | `svm_dec < −0.433`，強烈疑似錯誤 |
| `B_review/` | 40 | 剩餘檔中決策值最低者，優先待審 |
| `wrong/` | 22 | 已確認錯誤 |
| `top500/` | 497 | 依 `svm_dec` 排序，**STT 階段的處理範圍** |

嵌入 pipeline、驗證數據、以及**試過但行不通的方法**（戰鬥庫 centroid、kNN）：
[`docs/04_speaker_id.md`](docs/04_speaker_id.md)

## Part 1 ③：ASR 轉譯與訓練清單

### 3.1 ASR 引擎與逐字稿品質

faster-whisper **large-v3**、`language=ja`、Silero VAD 開啟、beam size 5、時長限制 2–15 秒。

```
500（top500）
  − 3（top500 內人工確認非ライザ）
= 497（ASR 輸入）
  − 6（跨檔文本重複）
= 491（有效段數）
```

- Issue 要求的五類過濾規則（時長／非說話聲／空輸出／重複／幻覺）**本批次 0 觸發**
- **CER 抽查 50 段：平均 1.0%，44/50 完全正確**；6 段錯誤中 5 段為遊戲專有名詞
- 專有名詞校正：6 條規則套用後共修正 14 筆，校正後與人工校對 **47/50 完全一致**

### 3.2 訓練清單與三處一致性

`data/ryza_train.list` 共 491 行，格式為 `音檔路徑|ryza|ja|文字`，有效總時長 **51.5 分鐘**（平均 6.3 秒）。

整個階段的核心約束是**三處編號集合必須完全相等**——
`tools/pos_label.py` 的 `SVM_DEC[]`、`tools/asr_screening.json` 的 PASS 筆數、
`data/ryza_train.list` 的行數。由 `tools/validate_pos_label.py` 從來源**獨立推導期望值**交叉檢查
（長度／唯一／範圍／音檔存在／順序／三處一致），不讀取任何產生器的內部狀態。

七步資料流、完整執行指令、歷史執行紀錄：[`docs/05_transcript_pipeline.md`](docs/05_transcript_pipeline.md)

## 工具入口

| 模塊 | 主要工具 | 產出 |
|---|---|---|
| ① 解包 | `tools/vgmstream/`、`parse_ktsr.py`、`parse_bare_ktsr.py`、`kovs_to_ogg.py` | `voice_index.csv`、`battle_voice_index.csv`、`wav/` |
| ② 篩選 | `spk_model.py`、`encode_all.py`、`final_index.py`、`refine_labels.py` | `embeddings.npy`、`ryza_main_index.csv`、`ryza_main/` |
| ③ 轉譯 | `build_pos_label.py`、`screen_asr.py`、`transcribe.py`、`validate_pos_label.py` | `asr_screening.json`、`data/ryza_train.list` |

完整目錄（每支腳本的用途、輸入、輸出）：[`docs/08_tools.md`](docs/08_tools.md)

## 文件地圖

**`docs/`** — 檔名序號即閱讀順序，依「需求 → 流程 → 參考」排列

| 文件 | 定位 |
|---|---|
| [`01_issue.md`](docs/01_issue.md) | 需求：原始 issue 與驗收要件 |
| [`02_issue_part2.md`](docs/02_issue_part2.md) | 需求：Part 2 選型論證 |
| [`03_unpack.md`](docs/03_unpack.md) | Part 1 ① 語音解包 |
| [`04_speaker_id.md`](docs/04_speaker_id.md) | Part 1 ② 說話者篩選 |
| [`05_transcript_pipeline.md`](docs/05_transcript_pipeline.md) | Part 1 ③ ASR 轉譯與訓練清單 |
| [`06_dataset.md`](docs/06_dataset.md) | Part 2 範圍、模型選型與資料切分 |
| [`07_finetune.md`](docs/07_finetune.md) | Part 2 從資料準備到推論的重現步驟 |
| [`08_tools.md`](docs/08_tools.md) | 參考：`tools/` 全部工具目錄 |
| [`09_pitfalls.md`](docs/09_pitfalls.md) | 參考：坑記錄（append-only，PK-001～PK-015） |

**`reports/`** — 各階段報告，依性質分五類

- 環境與模型：[`01_env_model_prep.md`](reports/01_env_model_prep.md)
- 資料切分：[`split_report.md`](reports/split_report.md)
- ASR 與品質：[`asr_report.md`](reports/asr_report.md)、[`audio_quality.md`](reports/audio_quality.md)、[`correction_diff.md`](reports/correction_diff.md)
- Gate 報告 G1–G4：[`G1_t0_t1_report.md`](reports/G1_t0_t1_report.md)、[`G2_stage_a_report.md`](reports/G2_stage_a_report.md)、[`G3_stage_b_report.md`](reports/G3_stage_b_report.md)、[`G4_stage_c_report.md`](reports/G4_stage_c_report.md)
- 訓練與評估：[`finetune_report.md`](reports/finetune_report.md)、[`defect_triage.md`](reports/defect_triage.md)、[`issue_compliance.md`](reports/issue_compliance.md)、[`split_report.md`](reports/split_report.md)、[`cer_generated.csv`](reports/cer_generated.csv)、[`logs/stage_d_train.md`](reports/logs/stage_d_train.md)、[`pr_description.md`](reports/pr_description.md)
- 歷史紀錄：[`issue_reply_part1.md`](reports/issue_reply_part1.md)；
  [`asr_filter_report.md`](reports/asr_filter_report.md) 已被 `asr_report.md` 取代

## 授權

- 解出的音聲內容版權屬 Koei Tecmo／Gust，僅供個人用途。
- vgmstream 為其作者所有（ISC-style license），見 `tools/vgmstream/COPYING`。