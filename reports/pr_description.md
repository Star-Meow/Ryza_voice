# PR 描述草案（Part 2：GPT-SoVITS v4 LoRA 微調）

> 用途：issue `docs/01_issue.md:34` 要求「須在 PR 中記錄使用的版本與 commit hash」。
> 本檔為可直接貼入 PR 描述的草案，內容已與 repo 內各報告一致。
> **尚未 push、尚未開 PR**——對外動作待授權。

---

## 選型：為何是 GPT-SoVITS v4 LoRA

完整分析見 [`../docs/02_issue_part2.md`](../docs/02_issue_part2.md)（§一 顯存與微調路徑、§五 選型結論、§六 決策鏈摘要）。

| 面向 | 內容 |
|---|---|
| **硬約束** | RTX 4070 SUPER **12 GB**。v4 LoRA 有官方 **8 GB** 訓練門檻；v2Pro 無官方 GB 數字，社群實測指向 ~12 GB，處於臨界 |
| **取樣率** | v4 原生 **48 kHz**，與語料（9,569 段遊戲語音皆 48 kHz）一致 |
| **相似度基準差距** | v4 官方 SIM 0.735 vs v2ProPlus 0.737，僅憑 benchmark 無法形成明顯差距 |
| **電音風險** | v3 有非整數倍上採樣的電音問題；v4 已修復 |
| **入口限制** | LoRA 是 v3/v4 在 12 GB 下的**唯一**可行微調路徑，非缺點而是必要選擇 |
| **備選** | v2ProPlus。若 v4 LoRA 實測無法滿足驗收條件再評估（**本次未啟用**，因已達標） |

---

## 使用的版本與 commit hash（issue 字面要求）

### 基礎模型

| 項目 | 值 |
|---|---|
| 框架 | **GPT-SoVITS**，main 分支，**無分支、無修改**（`git status` 全程乾淨，未改任何官方檔） |
| **commit hash** | **`48b1a0169a28582a8984402f82cf438d3bfa6aca`**（短碼 `48b1a01`，commit 訊息「Fix Fun-ASR-Nano Transformers requirement (#2824)」） |
| 目標版本 | **v4**（`model.version = "v4"`，config 硬性要求） |
| SoVITS 底模 | `GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth`（769,025,545 B） |
| Vocoder | `GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth`（57,781,109 B，**v4 路徑寫死於 `TTS.py:657`**） |
| GPT／AR 底模 | `GPT_SoVITS/pretrained_models/s1v3.ckpt`（155,284,856 B，v4 沿用 v3 權重，`config.py:25`） |
| 權重來源 | `huggingface.co/XXXXRT/GPT-SoVITS-Pretrained` |
| BERT／HuBERT | `chinese-roberta-wwm-ext-large`、`chinese-hubert-base`，經 **T0 轉檔為 safetensors**（原因見下） |

### 必要的模型轉檔（T0）

transformers 4.57.6 因 **CVE-2025-32434** 拒絕以 `torch.load` 載入 `.bin`，而本環境 torch 2.5.1+cu121 未達 2.6 門檻、升級會破壞 cu121 綁定。故將兩個模型轉為 `model.safetensors`：

| 模型 | `pytorch_model.bin` SHA256 | `model.safetensors` SHA256 |
|---|---|---|
| chinese-roberta-wwm-ext-large | `e53a693acc59ace251d143d068096ae0d7b79e4b1b503fa84c9dcf576448c1d8` | `ddb2d1e2aec45a149bb1b19213c0478cee464f9cc75531e70e673b5224ef4f05` |
| chinese-hubert-base | `24164f129c66499d1346e2aa55f183250c223161ec2770c0da3d3b08cf432d3c` | `25adc31d1889ff5d3da262189433bdc755f0ddb9f194342583dbd17e447daefe` |

驗證見 `reports/G1_t0_t1_report.md`。轉檔**未刪除也未修改**原始 `.bin`。

### 額外模型資產

| 項目 | 值 |
|---|---|
| fast_langdetect 語言偵測模型 | `GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin`，131,266,198 B，SHA256 `7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e` |

---

## 詳細內容與本專案 commit hash

> issue 的 commit hash 指的是**基礎模型**的（上方）。以下為**本專案交付內容**的 commit 對照，
> 用途是把 PR 的內容釘定到確切版本。

| commit | 內容 |
|---|---|
| `bb0a840` | 階段 B／C：訓練配置、啟動腳本、冒煙測試成果 |
| `0b7c743` | 階段 A 前置：T0 safetensors 轉檔與 T1 UNK 分析工具，G1 回報 |
| `49ca4b5` | **階段 D 主體**：正式訓練與品質評估（config 採 16 epoch、grad_ckpt=true、評估工具、報告） |
| `8c0ef23` | jieba_fast 永久不採用的裁示記錄 |
| `298bf4a` | 生成樣本的 Whisper 文字稿 `reports/cer_generated.csv` ＋報告附錄 A |
| `4f118e8` | 聲線相似度基線穩健性重算（9 種組合） |
| `33ae53b` | SD-8 證據補充 |
| `f9dc37f` | 記錄基礎模型與產出 commit hash |
| `ea8a5bc` | `docs/07_finetune.md` 補完完整重現步驟（177 → 590 行） |
| `2c4db07` | 整併主觀試聽評語 ＋ 量化交叉檢視 |
| `e53aab9`、`09e5810` | 主觀缺陷排查報告 `defect_triage.md`（P-1/P-2/P-3） |
| `9c0c6ab`、`b9aab67` | issue 合規稽核 `issue_compliance.md` ＋ 文件修正 |

**交付內容清單**（全部已 commit）：

| 類別 | 檔案 |
|---|---|
| 訓練設定檔 | `configs/s1_ryza.yaml`、`configs/s2_lora_ryza.json`、`configs/s1_ryza_smoke.yaml`、`configs/s2_lora_ryza_smoke*.json` |
| 訓練／推論／評估工具 | `tools/run_lora_train.py`、`tools/vram_runner.py`、`tools/stage_d_infer.py`、`tools/stage_d_eval.py`、`tools/stage_c_collect.py`、`tools/s1_lr_check.py`、`tools/run_preprocess.py`、`tools/build_smoke_exp.py`、`tools/stage_a_validate.py` |
| 重現文件 | `docs/07_finetune.md`（9 章，資料準備→推論）、`docs/09_pitfalls.md`（PK-001～015） |
| 報告 | `reports/finetune_report.md`、`reports/defect_triage.md`、`reports/issue_compliance.md`、`reports/cer_generated.csv`、`reports/logs/stage_d_train.md`、`reports/G2_stage_a_report.md`、`reports/G4_stage_c_report.md`、`reports/smoke_test_stage_c.md` |
| **不提交** | 模型權重、`reports/logs/raw/`（原始 log 與 metrics）、TensorBoard events、快取模型 |

---

## 驗收結果

| issue 驗收要件 | 結果 |
|---|---|
| 訓練前保留 5%（24 段）測試集且不用於訓練 | ✅ 467／24／10，11/11 門檻通過 |
| SoVITS 與 GPT 兩階段皆 finetune，記錄參數 | ✅ 各 16 epoch，exit 0 |
| `docs/07_finetune.md` 完整重現步驟 | ✅ 590 行 |
| 提交訓練設定檔與 log（不含權重） | ✅ |
| (a) 測試集 10 句生成 | ✅ |
| (b) 全新台詞 10 句，涵蓋四類語氣 | ✅ 語料庫 0 命中 |
| 聲線相似度 ≥ 測試集真實音訊平均 × 90% | ✅ **91.5%**（0.8152／門檻 0.8017） |
| 可懂度 CER ≤ 10% | ✅ **2.99%** |
| `finetune_report.md` 附主觀試聽評語 | ✅ 三項 ＋ 逐檔 |
| 訓練穩定完成，GPU 不穩須記錄 | ✅ 1 次顯存事件已處置 |

**訓練參數**：S2 batch 4／epochs 16／**grad_ckpt true**／lora_rank 32；S1 batch 8／epochs 16。合計約 **9.9 分鐘**（S2 8.7 分 ＋ S1 1.1 分）。

---

## 需要 reviewers 注意的三件事

1. **`grad_ckpt=true` 是必要設定，非可選**。`false` 時整卡峰值達 11958/12282 MiB（97.3%），
   進程在 epoch 2 假死且**不拋 OOM**。詳見 `docs/09_pitfalls.md` PK-013。
2. **epochs 由定案的 8 改為 16**。8 epoch 的成品聲線相似度僅達真實基線的 **86.9%**（門檻 90%），
   續訓至 16 後為 **91.5%**。S1 loss 至第 8 epoch 仍在單調下降、top-3 準確率僅 0.628 且無平台期。
   改回 8 只需調兩個 config 的 `epochs`，但相似度會未達標。
3. **兩個 gate 指標對語氣／情緒盲視**。成因是切分階段的 `class` 欄位以**句末標點**推導，
   決定了誰被留作測試集；主觀試聽發現 6/20 句情緒偏開心，而 cosine 與 CER 都無法反應。
   詳見 `reports/defect_triage.md` §P-2、`docs/09_pitfalls.md` PK-015。

---

## 未解決事項

- **PR 尚未建立**：本分支 6 個 commit（`9c0c6ab` 之後）尚未 push。issue 要求的
  「在 PR 中記錄」待 push ＋ 開 PR 後完成。
- **`v00596.wav` 資料品質**：`svm_dec = −0.4597`（源自專案自訂為「強烈疑似錯誤」的 `D_flagged` 層），
  因 `select_samples.py` 的 `v` 前綴自動採納規則進入 491 段訓練集。待裁示是否剔除。
- **人工查核的逐筆紀錄未入庫**：執行面成立（PM 2026-10-05 確認），但 repo 內無
  「音檔 → 判定 → 試聽者 → 日期」對照表，影響日後審計。
- **主觀缺陷 P-1/P-2/P-3** 尚在排查，量化 gate 已達標故不阻擋。
