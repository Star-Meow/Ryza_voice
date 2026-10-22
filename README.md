# Atelier Ryza 2 萊莎聲線 TTS 專案

以《Atelier Ryza 2》遊戲音聲建立**ライザ（Ryza）日文聲線 TTS**，分兩個模塊：

- **Part 1** — 語音解包、說話者篩選與 ASR 轉譯，產出 `data/ryza_train.list`
  （491 段日文語音，約 51.5 分鐘，48 kHz mono，格式 `音檔路徑|ryza|ja|文字`）。**已完成。**
- **Part 2** — 以此為基礎使用 GPT-SoVITS v4 LoRA 微調萊莎聲線 TTS（RTX 4070 SUPER 12 GB），
  聲線相似度達真實基線的 **91.5%**、可懂度 CER **2.99%**。**已完成。**

需求來源：[issue #1](https://github.com/Star-Meow/Ryza_voice/issues/1)。

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
| `LICENSE` | 程式碼授權（MIT），不涵蓋遊戲音訊與生成樣本 |
| `.gitignore` | 排除音訊、模型權重與中間產物的規則 |

## 專案流程總覽

| 模塊 | 內容 | 方法 | 關鍵結果 |
|---|---|---|---|
| **Part 1 ①** | 語音解包 | vgmstream r2117 批次解出 KTSR 音聲庫 | 9,569 + 646 個 WAV |
| **Part 1 ②** | 說話者篩選 | ECAPA-TDNN 嵌入比對 + LinearSVC 決策邊界 | `A_high` 92 檔全數人工確認為ライザ |
| **Part 1 ③** | ASR 轉譯 | faster-whisper large-v3 + 過濾、去重、專有名詞校正 | **491 段／51.5 分鐘，CER 1.0%** |
| **Part 2** | GPT-SoVITS 微調萊莎聲線 TTS | v4 LoRA（S1 ＋ S2 LoRA 各 16 epoch） | **相似度 91.5%（門檻 90%）、CER 2.99%（上限 10%）** |

Part 1 有效轉譯段數 491（目標 ≥ 450），全數經人工確認為ライザ。

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

- 五類必要過濾規則（時長／非說話聲／空輸出／重複／幻覺）**本批次 0 觸發**
- **CER 抽查 50 段：平均 1.0%，44/50 完全正確**；6 段錯誤中 5 段為遊戲專有名詞
  - CER 的 reference（ground truth）是**這 50 段經人工校對後的文字**，hypothesis 是 **Whisper 原始輸出**（seed 42 抽樣）
- 專有名詞校正：6 條規則套用後共修正 14 筆，校正後與人工校對 **47/50 完全一致**

### 3.2 訓練清單與三處一致性

`data/ryza_train.list` 共 491 行，格式為 `音檔路徑|ryza|ja|文字`，有效總時長 **51.5 分鐘**（平均 6.3 秒）。

整個階段的核心約束是**三處編號集合必須完全相等**——
`tools/pos_label.py` 的 `SVM_DEC[]`、`tools/asr_screening.json` 的 PASS 筆數、
`data/ryza_train.list` 的行數。由 `tools/validate_pos_label.py` 從來源**獨立推導期望值**交叉檢查
（長度／唯一／範圍／音檔存在／順序／三處一致），不讀取任何產生器的內部狀態。

七步資料流、完整執行指令、歷史執行紀錄：[`docs/05_transcript_pipeline.md`](docs/05_transcript_pipeline.md)

## Part 2：GPT-SoVITS v4 LoRA 微調

- **選型**：v4 LoRA。12 GB 顯存下有官方 LoRA 微調路徑，且原生 48 kHz 與語料一致；
  v2ProPlus 列為備選。完整論證見 [`docs/02_model_selection.md`](docs/02_model_selection.md)
- **切分**：491 段依句型與時長分層，切成 train 467／test 24／eval10 10（[`docs/06_dataset.md`](docs/06_dataset.md)）
- **訓練**：基礎模型 GPT-SoVITS `main` @ `48b1a01`。S2 LoRA（rank 32、batch 4、`grad_ckpt=true`）與
  S1（batch 8）各 16 epoch，合計約 10 分鐘。12 GB 下不開 `grad_ckpt` 會顯存假死（[`docs/09_pitfalls.md`](docs/09_pitfalls.md) PK-013）

| 評估項目 | 結果 | 門檻 |
|---|---|---|
| 聲線相似度（ECAPA-TDNN 對 A_high 中心點） | 0.8152，真實基線的 **91.5%** | ≥ 90% |
| 可懂度（Whisper large-v3 CER） | **2.99%** | ≤ 10% |
| 主觀試聽 | 音色可辨識為萊莎；有語尾多音、情緒偏開心、電子音三項缺陷 | — |

生成樣本 20 句（測試集 10 句＋全新台詞 10 句）在 [`samples/`](samples/README.md)；
完整評估見 [`reports/finetune_report.md`](reports/finetune_report.md)，缺陷分析見 [`reports/defect_triage.md`](reports/defect_triage.md)。

## 安裝與執行

本 repo **不含遊戲音訊與模型權重**，需自備《Atelier Ryza 2》遊戲檔，並另行取得 GPT-SoVITS。

| 環境 | 用途 | 安裝 |
|---|---|---|
| 本 repo 的 `.venv` | Part 1 工具（解包、篩選、ASR、切分、評估） | `pip install -r requirements.txt` |
| GPT-SoVITS 的 `.venv` | Part 2 預處理、訓練、推論 | clone [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 並 checkout `48b1a01`；另需 ffmpeg |

從解包到推論的完整重現步驟，依 [`docs/07_finetune.md`](docs/07_finetune.md) §0 的工作流順序執行。

## 工具入口

| 模塊 | 主要工具 | 產出 |
|---|---|---|
| ① 解包 | `tools/vgmstream/`、`parse_ktsr.py`、`parse_bare_ktsr.py`、`kovs_to_ogg.py` | `voice_index.csv`、`battle_voice_index.csv`、`wav/` |
| ② 篩選 | `spk_model.py`、`encode_all.py`、`final_index.py`、`refine_labels.py` | `embeddings.npy`、`ryza_main_index.csv`、`ryza_main/` |
| ③ 轉譯 | `build_pos_label.py`、`screen_asr.py`、`transcribe.py`、`validate_pos_label.py` | `asr_screening.json`、`data/ryza_train.list` |

完整目錄（每支腳本的用途、輸入、輸出）：[`docs/08_tools.md`](docs/08_tools.md)

## 文件地圖

**`docs/`** — 檔名序號即閱讀順序，依「設計 → 流程 → 參考」排列

| 文件 | 定位 |
|---|---|
| [`02_model_selection.md`](docs/02_model_selection.md) | Part 2 模型選型論證 |
| [`03_unpack.md`](docs/03_unpack.md) | Part 1 ① 語音解包 |
| [`04_speaker_id.md`](docs/04_speaker_id.md) | Part 1 ② 說話者篩選 |
| [`05_transcript_pipeline.md`](docs/05_transcript_pipeline.md) | Part 1 ③ ASR 轉譯與訓練清單 |
| [`06_dataset.md`](docs/06_dataset.md) | Part 2 範圍與資料切分 |
| [`07_finetune.md`](docs/07_finetune.md) | Part 2 從資料準備到推論的重現步驟 |
| [`08_tools.md`](docs/08_tools.md) | 參考：`tools/` 全部工具目錄 |
| [`09_pitfalls.md`](docs/09_pitfalls.md) | 參考：坑記錄（append-only，PK-001～PK-015） |

**`reports/`** — 各階段的結果與紀錄

- 結果：
  - ASR：[`asr_report.md`](reports/asr_report.md)、[`correction_diff.md`](reports/correction_diff.md)
  - 切分：[`split_report.md`](reports/split_report.md)
  - 訓練與評估：[`finetune_report.md`](reports/finetune_report.md)、[`logs/stage_d_train.md`](reports/logs/stage_d_train.md)、[`cer_generated.csv`](reports/cer_generated.csv)
  - 缺陷分析：[`defect_triage.md`](reports/defect_triage.md)
- 開發過程紀錄：[`01_env_model_prep.md`](reports/01_env_model_prep.md)、[`G1_t0_t1_report.md`](reports/G1_t0_t1_report.md)、[`G2_stage_a_report.md`](reports/G2_stage_a_report.md)、[`G3_stage_b_report.md`](reports/G3_stage_b_report.md)、[`G4_stage_c_report.md`](reports/G4_stage_c_report.md)、[`smoke_test_stage_a.md`](reports/smoke_test_stage_a.md)、[`smoke_test_stage_c.md`](reports/smoke_test_stage_c.md)、[`audio_quality.md`](reports/audio_quality.md)
- 已取代：[`asr_filter_report.md`](reports/asr_filter_report.md)，由 `asr_report.md` 取代

**協作文件（預留）** — 有其他人參與時再建立：`CONTRIBUTING.md`（協作慣例）、
`.github/pull_request_template.md`、`.github/ISSUE_TEMPLATE/`、`AGENTS.md`（AI 協作者守則）。

## 授權

- 本 repo 的程式碼與文件採 MIT 授權，見 [`LICENSE`](LICENSE)。
- 解出的音聲內容版權屬 Koei Tecmo／Gust，**不在 MIT 授權範圍內**，僅供個人用途，不進版控。
  `samples/` 的生成樣本衍生自該音聲，僅供評估。
- vgmstream 為其作者所有（ISC-style license），見 `tools/vgmstream/COPYING`。