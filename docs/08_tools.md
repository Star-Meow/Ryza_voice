# 工具總覽

> `tools/` 內所有工具的完整目錄，依執行階段分組。
> **本文件中所有 markdown 連結皆指向已納入版控的檔案**；未納入版控者一律以純文字列於 §8，不做成連結。
>
> 專案總覽見 [`README.md`](../README.md)；各階段的完整記錄見 [`unpack.md`](03_unpack.md)、
> [`speaker_id.md`](04_speaker_id.md)、[`transcript_pipeline.md`](05_transcript_pipeline.md)。

---

## 1. Part 1 ①：語音解包

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`parse_ktsr.py`](../tools/parse_ktsr.py) | 解析 KTSC 容器，逐 cue 取出 metadata 與 KOVS 區塊偏移 | `VOICE.ktsl2asbin` | `voice_index.csv` |
| [`parse_bare_ktsr.py`](../tools/parse_bare_ktsr.py) | 解析裸 KTSR（戰鬥庫），LCG XOR 解密 cue 名稱 | `014_*.ktsl2asbin`、`015_*.ktsl2asbin` | `battle_voice_index.csv` |
| [`kovs_to_ogg.py`](../tools/kovs_to_ogg.py) | 單一 KOVS 區塊去混淆為標準 Ogg（`--cue N` 可直接用索引） | `.ktsl2stbin` + 偏移 | `.ogg` |
| [`ktsr_ref.c`](../tools/ktsr_ref.c) | vgmstream `ktsr.c` 原始碼，格式對照參考（**非可執行**） | — | — |
| `tools/vgmstream/` | vgmstream **r2117** 預編譯 Windows 版（`vgmstream-cli.exe` + 11 個 DLL） | 遊戲 `.ktsl2asbin` | WAV |

vgmstream 的授權見 [`tools/vgmstream/COPYING`](../tools/vgmstream/COPYING)，完整用法見
[`tools/vgmstream/USAGE.md`](../tools/vgmstream/USAGE.md)。批次解出指令見 [`unpack.md`](03_unpack.md) §6。

## 2. Part 1 ②：說話者篩選

### 2.1 嵌入計算

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`spk_model.py`](../tools/spk_model.py) | ECAPA-TDNN 嵌入（GPU-only、TF32 關閉） | wav | 192 維向量 |
| [`encode_all.py`](../tools/encode_all.py) | **單進程完整編碼**，9,569 檔約 7–8 分鐘 | `wav/` | `tools/embeddings.npy` |
| [`encode_chunk.py`](../tools/encode_chunk.py) | 編碼單一段 `[start, end)` | wav 區間 | `tools/chunks/NNNN.npy` |
| [`run_encode.sh`](../tools/run_encode.sh) | 分段 GPU 編碼驅動：每 500 檔一進程、崩潰重試 3 次、**不退回 CPU** | — | 呼叫 `encode_chunk.py` |
| [`merge_chunks.py`](../tools/merge_chunks.py) | 合併分段結果 | `tools/chunks/*.npy` | `tools/embeddings.npy`（9570 × 192） |
| [`spk_feats.py`](../tools/spk_feats.py) | MFCC 特徵擷取（純 numpy/scipy），說話者相似度比對 | wav | 特徵向量 |

> `encode_all.py` 與 `run_encode.sh` 二選一：前者供穩定環境單進程跑完，後者以分段與重試
> 吸收隨機 CUDA 不穩。GPU 上**多執行緒並發 CUDA 呼叫會造成 `illegal memory access`**，兩者皆循序處理。

### 2.2 評分、驗證與分級

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`score_ryza.py`](../tools/score_ryza.py) | 以正負樣本為主庫 9,569 檔評分 | `embeddings.npy` + 戰鬥庫樣本 | 分數 |
| [`xdom_test.py`](../tools/xdom_test.py) | **跨條件角色辨識測試**：戰鬥庫 leave-one-out 分類 + 3 個種子歸類 | `embeddings.npy` | 驗證報告（stdout） |
| [`analyze_ryza.py`](../tools/analyze_ryza.py) | 分數分析：檔名對應、連續區塊結構、門檻建議 | 分數 + `voice_index.csv` | 分析（stdout） |
| [`cluster_ryza.py`](../tools/cluster_ryza.py) | KMeans（K=20/40/60）與 Agglomerative 聚類，找種子所在群 | `embeddings.npy` | 分群標籤 |
| [`refine_ryza.py`](../tools/refine_ryza.py) | 從 3 個種子出發迭代精煉，反覆以候選集 centroid 重排並檢查收斂 | `embeddings.npy` | 重排結果 |
| [`final_index.py`](../tools/final_index.py) | **產出最終索引**：錨點 cos 平均得 `consensus_score` | `embeddings.npy` | `ryza_main_index.csv` |
| [`export_ryza.py`](../tools/export_ryza.py) | 依置信度門檻分級複製到 `ryza_main/` | `ryza_main_index.csv` | `A_high/`、`B_likely/`、`C_possible/` |
| [`refine_labels.py`](../tools/refine_labels.py) | **依 SVM 邊界重整分級**：A_high(92正) vs 20負 訓練 LinearSVC(C=1) | `ryza_main_index.csv` + `neg_labels.py` | 重整後分級、`svm_dec`、`margin_to_neg` |
| [`neg_labels.py`](../tools/neg_labels.py) | **資料檔**：`WRONG[]`（人工確認非ライザ）+ `DUPLICATE[]`（文本重複排除） | — | — |
| [`ryza_scores.npy`](../tools/ryza_scores.npy) | **資料檔**：第一輪 `consensus_score` 分數陣列 | — | — |
| [`svm_dec.npy`](../tools/svm_dec.npy) | **資料檔**：LinearSVC 決策值陣列 | — | — |

方法與驗證細節見 [`speaker_id.md`](04_speaker_id.md)。

## 3. Part 1 ③：ASR 轉譯與訓練清單

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`select_samples.py`](../tools/select_samples.py) | 依 `svm_dec` 排序產出 top500、跨層 `sound_id` 去重、**v 前綴正規化** | `ryza_main_index.csv` | 候選清單 / `ryza_main/top500/` |
| [`build_pos_label.py`](../tools/build_pos_label.py) | 自 `ryza_main/top500/` **造冊並更新** `pos_label.py`（`--exclude-from`／`--prune-raw`） | `ryza_main/top500/` | `tools/pos_label.py` |
| [`screen_asr.py`](../tools/screen_asr.py) | faster-whisper large-v3 ASR 與過濾（時長／空輸出／幻覺／重複） | `pos_label.py` + `wav/` | `tools/asr_screening.json` |
| [`glossary.py`](../tools/glossary.py) | **資料檔**：`PROPER_NOUNS`（正確形詞庫）+ `CORRECTIONS`（6 條已確認誤聽） | — | — |
| [`apply_glossary.py`](../tools/apply_glossary.py) | 套用 `CORRECTIONS` 並**重算 `core`** | `asr_screening.json` | 更新後的 json |
| [`transcribe.py`](../tools/transcribe.py) | 產 GPT-SoVITS 訓練清單（**純標準庫，不重跑 ASR**） | `pos_label.py` + `asr_screening.json` | `data/ryza_train.list` |
| [`validate_pos_label.py`](../tools/validate_pos_label.py) | **三處一致性驗證**：只讀輸入、獨立推導期望值交叉檢查 | 全套產出檔 | 驗證結果 |
| [`make_cer_sample.py`](../tools/make_cer_sample.py) | CER 抽查：固定 seed 隨機抽段產清單 | `asr_screening.json` | `reports/cer_sample.csv` |
| [`cer_report.py`](../tools/cer_report.py) | 對人工校對結果算 CER（**純標準庫 Levenshtein**） | `reports/cer_sample.csv` | CER 報告 |
| [`pos_label.py`](../tools/pos_label.py) | **資料檔**：`SVM_DEC[]` 正樣本編號清單 | — | — |
| [`asr_screening.json`](../tools/asr_screening.json) | **產物**：491 筆 ASR 逐字稿與過濾紀錄 | — | — |
| [`voice_cues.csv`](../tools/voice_cues.csv) | **產物**：`parse_ktsr.py` 的解析輸出 | — | — |

> `transcribe.py` 與 `cer_report.py` **刻意只用標準庫**，不引入三方相依，
> 確保日常重產清單與算 CER 不需要載入 torch／faster-whisper 環境。

資料流與執行順序見 [`transcript_pipeline.md`](05_transcript_pipeline.md)。

## 4. 資料切分與品質量測

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`split_train_test.py`](../tools/split_train_test.py) | 自 `data/ryza_train.list` **分層**抽樣 train/test/eval10（seed 42），內建五項驗證 | `data/ryza_train.list` | `data/finetune_{train,test,eval10}.list`、`data/split_manifest.csv` |
| [`audio_quality.py`](../tools/audio_quality.py) | 比對 GPT-SoVITS 官方 wiki 的「低音質訓練集」判定條件（純 CPU 頻譜量測） | 訓練清單 | `reports/audio_quality.{md,json}` |

## 5. 模型權重轉檔與 UNK 分析

| 工具 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| [`t0_convert_safetensors.py`](../tools/t0_convert_safetensors.py) | 官方預訓練 BERT／HuBERT 的 `.bin` → `model.safetensors`（不刪不改 `.bin`） | `pytorch_model.bin` | `model.safetensors` |
| [`t0_validate.py`](../tools/t0_validate.py) | T0 獨立驗證 V1–V4；另提供 `--negative` 模式**故意製造損壞檔以證明驗證能辨識失敗** | `.bin` + `.safetensors` | 驗證結果 |
| [`t1_unk_analysis.py`](../tools/t1_unk_analysis.py) | UNK 來源分析（唯讀，只輸出統計到 stdout，不寫入 `logs\`） | 預處理產出 | 統計（stdout） |

驗證報告見 [`reports/G1_t0_t1_report.md`](../reports/G1_t0_t1_report.md)。

## 6. 生成產物（不納入版控）

以下為執行上述工具所產生、体積較大或可重新生成的檔案，**刻意不進版控**：

| 產物 | 內容 | 產生者 |
|---|---|---|
| `tools/embeddings.npy` | 9,569 檔的 192 維說話者嵌入（9570 × 192） | `encode_all.py` / `merge_chunks.py` |
| `tools/chunks/*.npy` | 分段編碼中間檔 | `encode_chunk.py` |
| `tools/spk_model/*.ckpt` | ECAPA-TDNN 權重（83 MB，可重新下載） | speechbrain |
| `wav/`、`battle_voice/`、`ryza_battle/`、`ryza_main/` | **音訊本身**（版權屬 Koei Tecmo／Gust） | 解包與分級流程 |

## 7. 附帶資源

| 檔案 | 內容 |
|---|---|
| [`requirements.txt`](../requirements.txt) | Python 相依套件與版本 |
| [`reports/env_pip_freeze.txt`](../reports/env_pip_freeze.txt) | 環境完整 pip freeze（201 個套件） |

## 8. 尚未納入版控的工具

以下 5 支屬於後續的 LoRA 微調階段，**目前僅存在於本機工作樹，尚未納入版控**，
因此在本文件中以純文字列出而非連結。其他人 clone 後不會取得這些檔案。

| 檔名 | 用途 |
|---|---|
| `tools/run_preprocess.py` | 階段 A 預處理驅動：以 venv Python 直接執行官方三步腳本（不經 webui） |
| `tools/stage_a_validate.py` | 階段 A 驗證：PA-1～PA-13 對 train 與 eval10 各跑一次（唯讀） |
| `tools/run_lora_train.py` | 階段 B/C 啟動腳本：S2 LoRA 與 S1 訓練，內建 8 小時 watchdog 與 GPU 記憶體取樣 |
| `tools/run_lora_train_selftest.py` | 啟動腳本的無害驗證：以輸出一個字元的小程式替代訓練，不載入任何模型 |
| `tools/build_smoke_exp.py` | 冒煙 exp 建置：從正式產出過濾前 20 條，不重跑預處理 |

另有 `configs/`（4 份訓練設定檔）與 `docs/07_finetune.md`、`docs/09_pitfalls.md` 兩份文件同屬該階段，
亦尚未納入版控。