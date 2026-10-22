# tools/

本目錄所有腳本與資料檔的目錄，依模塊分組。各模塊的流程與執行順序見 [`docs/`](../docs/01_overview.md)。

**執行環境**：
- 本 repo 的 `.venv`：語音解包、說話者篩選、轉譯資料集、資料切分、評估。
- GPT-SoVITS 的 `.venv`：預處理、訓練、推論，從 `<GPT>` 目錄執行，見 [`07_finetune.md`](../docs/07_finetune.md) §1。

> `export_ryza.py`、`final_index.py`、`score_ryza.py`、`refine_labels.py` 沒有 argparse，路徑寫死在模組層級，必須在 repo 根目錄以無參數執行。
> `build_smoke_exp.py` 的 `<GPT>`、`<RYZA>` 路徑也寫死在檔內。

---

## 1. 語音解包

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`parse_ktsr.py`](parse_ktsr.py) | 解析 KTSC 容器，逐 cue 取出 metadata 與 KOVS 區塊偏移 | `VOICE.ktsl2asbin` | `voice_index.csv` |
| [`parse_bare_ktsr.py`](parse_bare_ktsr.py) | 解析裸 KTSR（戰鬥庫），以 LCG XOR 解密 cue 名稱 | `014_*.ktsl2asbin`、`015_*.ktsl2asbin` | `battle_voice_index.csv` |
| [`kovs_to_ogg.py`](kovs_to_ogg.py) | 把單一 KOVS 區塊去混淆為標準 Ogg（`--cue N` 可直接用索引） | `.ktsl2stbin` ＋ 偏移 | `.ogg` |
| [`ktsr_ref.c`](ktsr_ref.c) | vgmstream `ktsr.c` 原始碼，格式對照參考（**不可執行**） | — | — |
| `vgmstream/` | vgmstream **r2117** 預編譯 Windows 版（`vgmstream-cli.exe` ＋ 11 個 DLL） | 遊戲 `.ktsl2asbin` | WAV |
| [`voice_cues.csv`](voice_cues.csv) | **產物**：`parse_ktsr.py` 的解析輸出 | — | — |

vgmstream 的授權見 [`vgmstream/COPYING`](vgmstream/COPYING)，完整用法見 [`vgmstream/USAGE.md`](vgmstream/USAGE.md)。
批次解出指令見 [`02_unpack.md`](../docs/02_unpack.md) §6。

## 2. 說話者篩選

### 2.1 嵌入計算

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`spk_model.py`](spk_model.py) | ECAPA-TDNN 嵌入（只用 GPU、關閉 TF32） | wav | 192 維向量 |
| [`spk_model/`](spk_model/) | speechbrain 模型設定（`hyperparams.yaml`、`config.json`、`label_encoder.txt`）；`*.ckpt` 權重不進版控 | — | — |
| [`encode_all.py`](encode_all.py) | **單進程完整編碼**，9,569 檔約 7–8 分鐘 | `wav/` | `tools/embeddings.npy` |
| [`encode_chunk.py`](encode_chunk.py) | 編碼單一段 `[start, end)` | wav 區間 | `tools/chunks/NNNN.npy` |
| [`run_encode.sh`](run_encode.sh) | 分段 GPU 編碼驅動：每 500 檔一個進程、崩潰重試 3 次、**不退回 CPU** | — | 呼叫 `encode_chunk.py` |
| [`merge_chunks.py`](merge_chunks.py) | 合併分段結果 | `tools/chunks/*.npy` | `tools/embeddings.npy`（9570 × 192） |
| [`spk_feats.py`](spk_feats.py) | MFCC 特徵擷取（純 numpy／scipy），用於說話者相似度比對 | wav | 特徵向量 |

> `encode_all.py` 與 `run_encode.sh` 二選一：前者在穩定環境單進程跑完；後者以分段與重試吸收隨機的 CUDA 不穩。
> GPU 上**多執行緒並發 CUDA 呼叫會造成 `illegal memory access`**，兩者都循序處理。

### 2.2 評分、驗證與分級

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`score_ryza.py`](score_ryza.py) | 以正負樣本為主庫 9,569 檔評分 | `embeddings.npy` ＋ 戰鬥庫樣本 | 分數 |
| [`xdom_test.py`](xdom_test.py) | **跨條件角色辨識測試**：戰鬥庫 leave-one-out 分類 ＋ 3 個種子歸類 | `embeddings.npy` | 驗證結果（stdout） |
| [`analyze_ryza.py`](analyze_ryza.py) | 分數分析：檔名對應、連續區塊結構、門檻建議 | 分數 ＋ `voice_index.csv` | 分析（stdout） |
| [`cluster_ryza.py`](cluster_ryza.py) | KMeans（K=20/40/60）與 Agglomerative 聚類，找出種子所在的群 | `embeddings.npy` | 分群標籤 |
| [`refine_ryza.py`](refine_ryza.py) | 從 3 個種子出發迭代精煉，反覆以候選集 centroid 重排並檢查收斂 | `embeddings.npy` | 重排結果 |
| [`final_index.py`](final_index.py) | **產出最終索引**：錨點 cosine 平均得 `consensus_score` | `embeddings.npy` | `ryza_main_index.csv` |
| [`export_ryza.py`](export_ryza.py) | 依置信度門檻分級，複製到 `ryza_main/` | `ryza_main_index.csv` | `A_high/`、`B_likely/`、`C_possible/` |
| [`refine_labels.py`](refine_labels.py) | **依 SVM 邊界重整分級**：以 A_high（92 正）對 20 負訓練 LinearSVC(C=1) | `ryza_main_index.csv` ＋ `neg_labels.py` | 重整後分級、`svm_dec`、`margin_to_neg` |
| [`neg_labels.py`](neg_labels.py) | **資料檔**：`WRONG[]`（人工確認非ライザ）＋ `DUPLICATE[]`（文本重複排除） | — | — |
| [`ryza_scores.npy`](ryza_scores.npy)、[`ryza_min3.npy`](ryza_min3.npy) | **資料檔**：第一輪 `consensus_score` 與錨點最小值分數陣列 | — | — |
| [`svm_dec.npy`](svm_dec.npy)、[`margin.npy`](margin.npy) | **資料檔**：LinearSVC 決策值與到負樣本的 margin | — | — |
| [`battle_centroids.npy`](battle_centroids.npy)、[`battle_char_names.txt`](battle_char_names.txt) | **資料檔**：戰鬥庫各角色 centroid 與角色名（只用於驗證，不用於主庫判定） | — | — |

方法與驗證細節見 [`03_speaker_id.md`](../docs/03_speaker_id.md)。

## 3. 轉譯資料集

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`select_samples.py`](select_samples.py) | 依 `svm_dec` 排序產出 top500、跨層 `sound_id` 去重、**v 前綴正規化** | `ryza_main_index.csv` | `ryza_main/top500/` |
| [`build_pos_label.py`](build_pos_label.py) | 自 `ryza_main/top500/` **造冊並更新** `pos_label.py`（`--exclude-from`、`--prune-raw`） | `ryza_main/top500/` | `tools/pos_label.py` |
| [`screen_asr.py`](screen_asr.py) | faster-whisper large-v3 ASR 與過濾（時長、空輸出、幻覺、重複） | `pos_label.py` ＋ `wav/` | `tools/asr_screening.json` |
| [`glossary.py`](glossary.py) | **資料檔**：`PROPER_NOUNS`（正確形詞庫）＋ `CORRECTIONS`（6 條已確認誤聽） | — | — |
| [`apply_glossary.py`](apply_glossary.py) | 套用 `CORRECTIONS` 並**重算 `core`** | `asr_screening.json` | 更新後的 json |
| [`transcribe.py`](transcribe.py) | 產出 GPT-SoVITS 訓練清單（**純標準庫，不重跑 ASR**） | `pos_label.py` ＋ `asr_screening.json` | `data/ryza_train.list` |
| [`validate_pos_label.py`](validate_pos_label.py) | **三處一致性驗證**：只讀輸入，獨立推導期望值交叉檢查 | 全套產出檔 | 驗證結果 |
| [`make_cer_sample.py`](make_cer_sample.py) | CER 抽查：以固定 seed 隨機抽段產出清單 | `asr_screening.json` | `reports/cer_sample.csv` |
| [`pos_label.py`](pos_label.py) | **資料檔**：`SVM_DEC[]` 正樣本編號清單 | — | — |
| [`asr_screening.json`](asr_screening.json) | **產物**：491 筆 ASR 逐字稿與過濾紀錄 | — | — |

> `select_samples.py` docstring 中的 `--output ryza_train.list` 範例已過時；正式清單是 `data/ryza_train.list`，由 `transcribe.py` 產出。

資料流與執行順序見 [`04_transcript_pipeline.md`](../docs/04_transcript_pipeline.md)。

## 4. 資料切分

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`split_train_test.py`](split_train_test.py) | 自 `data/ryza_train.list` **分層**抽樣 train／test／eval10（seed 42），寫檔前做 11 項驗證 | `data/ryza_train.list` | `data/finetune_{train,test,eval10}.list`、`data/split_manifest.csv`、`data/ref/` |

設計與指令見 [`05_dataset_split.md`](../docs/05_dataset_split.md)。

## 5. TTS 微調與推論

在 `<GPT>` 目錄、以 GPT-SoVITS 的 venv 執行，見 [`07_finetune.md`](../docs/07_finetune.md)。

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`t0_convert_safetensors.py`](t0_convert_safetensors.py) | 官方預訓練 BERT／HuBERT 的 `.bin` → `model.safetensors`（不刪不改 `.bin`） | `pytorch_model.bin` | `model.safetensors` |
| [`run_preprocess.py`](run_preprocess.py) | 預處理驅動：以 venv Python 直接執行官方三步腳本（不經 webui），stdout／stderr 分檔 | `data/finetune_{train,eval10}.list` | `logs/ryza_v4_lora/` 預處理產出 |
| [`run_lora_train.py`](run_lora_train.py) | 訓練驅動：S2 LoRA 與 S1，內建 watchdog、GPU 記憶體取樣、推論權重目錄 preflight | `configs/*` | 權重、`reports/logs/raw/` |
| [`vram_runner.py`](vram_runner.py) | 以 `runpy` 載入官方訓練腳本，結束時印出 `torch.cuda.max_memory_allocated()`（不修改官方檔） | 官方腳本 ＋ 參數 | 顯存峰值 |
| [`stage_d_infer.py`](stage_d_infer.py) | 正式推論：產出 (a) eval10 10 句與 (b) 全新台詞 10 句 | 訓練後權重 | `logs/ryza_v4_lora/infer_e16/*.wav` |

## 6. 評估

在本 repo 的 venv 執行。

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`stage_d_eval.py`](stage_d_eval.py) | 聲線相似度（ECAPA-TDNN 對 A_high 中心點）與可懂度（Whisper CER） | 推論 summary ＋ 生成音檔 | 評估 json |
| [`cer_report.py`](cer_report.py) | CER 計算（**純標準庫 Levenshtein**）；ASR 抽查與生成樣本共用 | `reports/cer_sample.csv`、`reports/cer_generated.csv` | CER 報告 |
| [`audio_quality.py`](audio_quality.py) | 比對官方 wiki 的「低音質訓練集」判定條件（純 CPU 頻譜量測） | 訓練清單 | `reports/audio_quality.json` |

> `transcribe.py` 與 `cer_report.py` **刻意只用標準庫**，日常重產清單與計算 CER 時不需要載入 torch 或 faster-whisper。

## 7. 測試與驗證

在正式流程之外，用來確認環境或產出正確的工具。冒煙測試的步驟與通過條件見 [`07_finetune.md`](../docs/07_finetune.md) §5.2。

| 工具 | 用途 |
|---|---|
| [`t0_validate.py`](t0_validate.py) | T0 轉檔的獨立驗證（key、dtype、shape）；`--negative` 模式**故意製造損壞檔以證明驗證能辨識失敗** |
| [`t1_unk_analysis.py`](t1_unk_analysis.py) | UNK 來源分析：確認 UNK 來自 `#` 韻律邊界（唯讀，只輸出到 stdout） |
| [`stage_a_validate.py`](stage_a_validate.py) | 預處理檢查 PA-1～PA-13，對 train 與 eval10 各跑一次（唯讀） |
| [`run_lora_train_selftest.py`](run_lora_train_selftest.py) | 訓練驅動的無害自檢：以輸出一個字元的小程式替代訓練，不載入模型 |
| [`build_smoke_exp.py`](build_smoke_exp.py) | 建立冒煙 exp：從正式預處理產出取前 20 筆，不重跑預處理 |
| [`infer_smoke.py`](infer_smoke.py) | 推論冒煙：`--mode base` 用底模、`--mode smoke` 用冒煙權重，各生成 3 句 |
| [`stage_c_collect.py`](stage_c_collect.py) | 彙整冒煙測試數據：loss 趨勢、每 step 秒數、resume、S1 checkpoint 載入 |
| [`s1_lr_check.py`](s1_lr_check.py) | 模擬 S1 學習率排程，確認實際 LR 恆為 0.002（見 [PK-012](../docs/08_troubleshooting.md)） |

## 8. 生成產物（不進版控）

以下檔案由上述工具產生，體積較大或可重新生成，**刻意不進版控**：

| 產物 | 內容 | 產生者 |
|---|---|---|
| `tools/embeddings.npy` | 9,569 檔的 192 維說話者嵌入（9570 × 192） | `encode_all.py`／`merge_chunks.py` |
| `tools/chunks/*.npy` | 分段編碼中間檔 | `encode_chunk.py` |
| `tools/spk_model/*.ckpt` | ECAPA-TDNN 權重（83 MB，可重新下載） | speechbrain |
| `wav/`、`battle_voice/`、`ryza_battle/`、`ryza_main/` | **音訊本身**（版權屬 Koei Tecmo／Gust） | 解包與分級流程 |
| `reports/logs/raw/` | 訓練與檢查的原始 log、summary json | `run_lora_train.py`、`stage_a_validate.py` 等 |

## 9. 附帶資源

| 檔案 | 內容 |
|---|---|
| [`requirements.txt`](../requirements.txt) | 本 repo venv 的 Python 相依套件與版本 |
| [`reports/env_pip_freeze.txt`](../reports/env_pip_freeze.txt) | GPT-SoVITS venv 的完整 pip freeze |
