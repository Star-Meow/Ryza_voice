# ASR 轉譯與訓練清單建構（Part 1 ③）

> 承接 [`speaker_id.md`](04_speaker_id.md) 的 top500 候選，將音訊轉成 GPT-SoVITS 可用的逐字稿訓練清單。
> 本階段的核心是**三處編號集合的一致性**——這是全文最需要注意的地方。
>
> ASR 引擎參數、過濾統計與 CER 抽查的完整報告見 [`reports/asr_report.md`](../reports/asr_report.md)；
> 專有名詞校正的逐段前後對照見 [`reports/correction_diff.md`](../reports/correction_diff.md)。

---

## 1. 三個核心檔案

整個階段的資料鏈是：

```
ryza_main/top500/  →  tools/pos_label.py  →  tools/asr_screening.json  →  data/ryza_train.list
```

| 檔案 | 角色 |
|---|---|
| [`tools/pos_label.py`](../tools/pos_label.py) | **正樣本編號清單** `SVM_DEC[]`，定義 STT 階段的處理範圍 |
| [`tools/asr_screening.json`](../tools/asr_screening.json) | **ASR 逐字稿與過濾紀錄**，每筆含 `status`／`reason`／分數／`text`／`core` |
| [`data/ryza_train.list`](../data/ryza_train.list) | **最終訓練清單**，491 行 |
| [`tools/neg_labels.py`](../tools/neg_labels.py) | `WRONG[]`（人工確認非ライザ）與 `DUPLICATE[]`（ASR 文本重複排除） |
| `wav/` | **原聲檔唯一來源**，清單與編號最終都指向此處 |

訓練清單格式為 `音檔路徑|speaker|lang|文字`，例如：

```
wav/00028.wav|ryza|ja|そっか、わかった。
```

**維護原則**：`asr_screening.json` 是唯一的事實來源，其餘兩檔皆由它推導。任何時候三者必須相等（見 §8）。

## 1.5 為何走 `top500/` 而非 issue 字面寫的 `A_high/ + B_likely/`

`docs/01_issue.md` 第 8 行的字面範圍是「`ryza_main/A_high/`（92）＋ `ryza_main/B_likely/`（386）＝約 478 段」。實際執行走的是 `ryza_main/top500/`。**這是刻意的決策，不是流程意外**（PM 2026-10-05 說明）：

> 早期排序時，因為**擔心樣本數偏頗**，所以將 A（人工確認的樣本，即 `A_high`）以外的直接進行 `svm_dec` 排序並取出前 500，進行去躁以及正負樣本確認。

| 面向 | 內容 |
|---|---|
| **動因** | 單靠 `A_high`（92 筆，已全數人工確認）樣本數偏少，不足以支撐後續 finetune；`B_likely` 又有 383 筆尚未全數人工確認 |
| **做法** | `A_high` 全數保留 ＋ 其餘層級合併後依 `svm_dec` 排序取前 500（`tools/select_samples.py`），再逐階段做去重與正負樣本確認 |
| **結果** | `A_high` 92/92 全數納入；另納入 `B_likely` 378 ＋ `C_possible` 20 → top500 497 → 去重後 **491** |
| **人工查核** | **早期視聽與人工檢查樣本皆由人工執行**（PM 2026-10-05 確認）。`tools/neg_labels.py` 的 `WRONG[]` 為人工確認非ライザ的結果，`DUPLICATE[]` 為去重結果 |

**複現時的注意事項**：

1. **不要照 issue 字面範圍重建。** 若改用 `A_high + B_likely`（383，非 issue 所寫的 386），得到的是另一份資料集，與 `data/ryza_train.list` 不一致，下游預處理、訓練、評估數據全部會變。正確入口是 `--source-dir ryza_main/top500`。
2. **issue 的 386 是過期數字**，`ryza_main/B_likely/` 實際為 **383**（`docs/04_speaker_id.md:145`、`README.md:64` 一致）。
3. **`v` 前綴自動採納規則未排除 `D_flagged` 層**：`tools/select_samples.py:8` 讓各層中檔名帶 `v` 前綴者全部採用、不參與排序，因此 `v00596.wav`（`svm_dec = −0.4597`，該層自訂為「強烈疑似錯誤」）也進入了最終 491 段。**PM 已於 2026-10-05 裁示不剔除此檔**（保留可維持 491 段，階段 A～D 數據全部有效）。複核記錄見 [`../reports/issue_compliance.md`](../reports/issue_compliance.md) §1.4。**日後複現時不應「順手修正」這條規則**——改了會得到不同的 491 段。
4. **人工查核的逐筆紀錄未留在 repo 內**。`WRONG[]` 保留了人工判定的**結果**，但沒有逐檔的「音檔編號 → 判定 → 試聽者 → 日期」對照表。若日後需稽核，需另行補建。

---

## 2. 編號正規化：v 前綴剝除

`ryza_main/top500/` 的檔名格式為 `NNNN_原檔名.wav`，其中一部分原檔名帶 `v` 前綴：

```
0093_v00086.wav  →  取底線後段「v00086」  →  剝除 v  →  00086
```

`v00086.wav` 與 `00086.wav` 視為**同一份音訊**（此邏輯定義於 [`tools/select_samples.py`](../tools/select_samples.py) 的
`index_key` / `sound_id` 處理）。

**清單與 `pos_label.py` 一律使用剝除後的純數字編號**，指向 `wav/` 主庫。
此規則同時套用於 [`tools/build_pos_label.py`](../tools/build_pos_label.py) 的造冊、
`screen_asr.py` 的音檔解析，以及 `validate_pos_label.py` 的期望值推導。

top500 中此類 v 前綴共 **6 個**：`v00086`、`v00105`、`v00142`、`v00339`、`v00596`、`v09508`。

## 3. ASR 引擎與參數

由 [`tools/screen_asr.py`](../tools/screen_asr.py) 執行。完整報告見 [`reports/asr_report.md`](../reports/asr_report.md) §1。

| 項目 | 設定 |
|---|---|
| 模型 | faster-whisper **large-v3**（CTranslate2 後端） |
| 裝置 / 精度 | CUDA（RTX 4070S 12G）/ **float16** |
| beam size | **5** |
| temperature | **梯度 fallback（0.0 → 1.0，faster-whisper 預設）** |
| VAD | 開啟（Silero VAD，經 onnxruntime） |
| 語言 | `ja`（固定，不偵測） |
| 時長門檻 | 2.0s ~ 15.0s（免模型，讀 WAV header） |
| initial_prompt | 無 |

**為何不固定 temperature=0.0**：實測固定時，`wav/04493.wav` 會陷入 `……` 無限重複幻覺
（compression_ratio 20.34，被 `HIGH_COMPRESSION` 規則捕捉）。改回預設梯度 fallback 後該段恢復為正常文字。
代價是重跑結果可能有少數段差異，故採預設。

> 所有路徑一律由參數傳入，**不寫死本機絕對路徑**。

## 4. 過濾規則

### 4.1 Issue 要求的規則

| 規則 | 門檻 | 觸發數 |
|---|---|---|
| `DURATION_SHORT` | < 2.0 s | 0 |
| `DURATION_LONG` | > 15.0 s | 0 |
| `EMPTY` | VAD 後無文字 | 0 |
| `HALLUCINATION` | 命中已知幻覺句 | 0 |
| `REPETITION` | 同字元連續 ≥ 4 | 0 |
| `NON_SPEECH` | `core` 為感嘆詞 | 0（標 REVIEW 交人工） |

**這六條規則本批次全數未觸發**——497 段音訊在 Issue 要求的條件下品質乾淨。

### 4.2 額外規則（非 Issue 要求）

| 規則 | 門檻 | 觸發數 | 說明 |
|---|---|---|---|
| `HIGH_COMPRESSION` | compression_ratio > 3.0 | 0（REVIEW） | Whisper 幻覺指標；實測最高僅 1.54 |
| `DUPLICATE_TEXT` | `core` 跨檔 ≥ 2 | **6（EXCLUDE）** | 跨檔同一台詞重複錄製，見 §5 |

> **`DUPLICATE_TEXT` 與 Issue 的「重複」是不同概念**：Issue 的「重複」指單一檔案內 ASR 輸出的
> 重複幻覺（即 `REPETITION`，觸發 0）；`DUPLICATE_TEXT` 指**不同音檔收錄了同一句台詞**。

## 5. 跨檔文本去重

比對 `asr_screening.json` 的 **`core` 欄**（逐字稿去除標點、只保留日文與英數字元）。

同一 `core` 出現 **≥ 2 次**時，**保留 `SVM_DEC` 順序中首次出現者**，其餘標 `EXCLUDE`
（`reason = DUPLICATE_TEXT`）。目前偵測到 **4 組共 10 筆**，排除其中 **6 筆**：

| 保留 | 排除 | 台詞 |
|---|---|---|
| `06332` | `04253`, `07018` | あれ?扉が開いてる。あんなしっかり閉まってたのに。 |
| `05676` | `07718`, `08413` | 扉、閉じちゃってるね。どうやって開けるんだろう。 |
| `01359` | `01991` | 私さ、そろそろ島に帰ろうかなって思うんだ。 |
| `01036` | `02998` | まだだよ。まだ諦めない |

> 這 10 個音檔的 **MD5 互不相同、時長也些微相差**，是同一句台詞的重複錄製，
> **非 Whisper 幻覺**。對 TTS 訓練而言保留一個版本即可。

被排除的 6 筆編號同步寫入 [`tools/neg_labels.py`](../tools/neg_labels.py) 的 `DUPLICATE[]`。

## 6. 專有名詞校正

CER 抽查發現的 6 段錯誤中，**5 段為專有名詞／固定詞組**（第 6 段為語頭語氣詞，不屬專有名詞）。
據此建立 [`tools/glossary.py`](../tools/glossary.py)，分兩層：

- **`PROPER_NOUNS`**：專有名詞**正確形**詞庫，完整保留、本身不參與字串取代，供人工校對與未來擴充時對照
- **`CORRECTIONS`**：**已確認**的誤聽修正（ASR 輸出 → 正確形）

| # | 規則（誤聽 → 正確） | 命中段數 | 依據 |
|---|---|---|---|
| 1 | `オート` → `王都` | 5 | CER id13；本庫 5 檔全數誤聽 |
| 2 | `ボース` → `ボオス` | 6 | 全庫掃描（抽樣外，6 檔全數誤聽） |
| 3 | `セイリ` → `セリ` | 1 | CER id38 |
| 4 | `ピー` → `フィー` | 1 | CER id50 |
| 5 | `諸子` → `書庫` | 1 | CER id32 |
| 6 | `タレジャーハンター` → `トレジャーハンター` | 1 | CER id16 |
| | **合計** | **15 次 / 14 檔** | |

`wav/02580.wav` 同時含兩條誤聽，故 15 次命中對應 14 個檔案——這就是「14 筆逐字稿修正」的由來。

**兩條刻意不收的規則**：
- 推測型誤聽（`リザ` → `ライザ` 等）本庫 **0 次觸發**，不收
- `ヴォルカ` → `ヴォルカー` 會把已正確的 `ヴォルカー` 改成 `ヴォルカーー`（子字串誤改），**不收**

套用工具 [`tools/apply_glossary.py`](../tools/apply_glossary.py)：修正 `text` 後**重算 `core`**
（與 `screen_asr.py` 的 `core_text` 同一字元過濾規則），避免改了 text 但 core 沒跟上、
導致後續跨檔去重比對失準。`segments` 逐段文字同步修正，`status` 不變。

**校正結果**：14 筆修正、`data/ryza_train.list` 重產為 491 行校正後文字；
校正後與 CER 50 段人工校對 **47/50 完全一致**（餘 3 段為標點／語氣詞差異，非專有名詞）。

## 7. 七步資料流

```
[步驟 1] 依 svm_dec 排序 → ryza_main/top500/（497 檔）
        │  實際檔名為 NNNN_原檔名.wav，已剔除 3 個人工確認負樣本
        ▼
[步驟 2] 人工篩查負樣本（top500 內 3 份）
        │       ├─► ryza_main/wrong/
        │       └─► neg_labels.py WRONG[]
        ▼
[步驟 3] 造冊 → pos_label.py SVM_DEC[]（497）
        │       ├── v 前綴剝除（6 個）
        │       └── 跨檔去重、依 top500 序號排序
        ▼
[步驟 4] ASR + 過濾 → tools/asr_screening.json
        │       ├── Whisper large-v3、language=ja、VAD 開啟
        │       ├── 時長 <2s / >15s
        │       ├── 非說話聲（喘氣、吶喊等感嘆詞）
        │       ├── ASR 空輸出（VAD 後無語音）
        │       └── 明顯幻覺（Whisper 常見輸出、單字重複）
        │       └─► 497 筆紀錄（單筆規則全 PASS）
        ▼
[步驟 5] 跨檔文本去重（門檻 ≥2，保留首次出現者）
        │       ├── 排除 6 筆 → neg_labels.py DUPLICATE[]
        │       └─► asr_screening.json 這 6 筆改標 EXCLUDE
        ▼
[步驟 6] 同步 pos_label.py SVM_DEC[]（497 → 491）
        │       └── asr_screening.json 精簡為 491 全 PASS
        ▼
[步驟 7] transcribe.py 產生清單（不重跑 ASR）
        │       ├── 讀 pos_label.py SVM_DEC[] + asr_screening.json 文字
        │       ├── 只取 PASS 且文字非空者，依 SVM_DEC 順序
        │       └─► data/ryza_train.list（491 行）
        ▼
[專有名詞校正] glossary.py CORRECTIONS（6 條）
        │       ├── apply_glossary.py 套用並重算 core
        │       └── transcribe.py 重產清單（491 行）
        ▼
reports/asr_report.md      模型參數、過濾統計、有效時長、CER 抽查
reports/correction_diff.md 專有名詞校正前後對照（14 段）
```

**數量關係**（每一步都必須對得上）：

```
500（top500 序號）
  − 3（top500 內 WRONG）
= 497（造冊 SVM_DEC）
  − 6（DUPLICATE_TEXT）
= 491（最終清單）
```

`WRONG[]` 共 23 筆，其中 **20 筆落在 top500 之外**，不影響本階段計算。

## 8. 三處一致性

步驟 6 完成後，三個檔案的編號集合必須**完全相等**：

```
pos_label.py SVM_DEC[]        = 491
asr_screening.json PASS       = 491
data/ryza_train.list           = 491 行

neg_labels.py WRONG            = 23  （全庫人工確認非ライザ；其中 3 筆在 top500 內）
neg_labels.py DUPLICATE        = 6   （ASR 文本重複）
```

驗證工具 [`tools/validate_pos_label.py`](../tools/validate_pos_label.py) **只讀輸入**（來源目錄與各產出檔），
**不讀任何產生器的內部狀態**，並從來源獨立推導期望值做交叉檢查：

| # | 檢查 | 條件 |
|---|---|---|
| 1 | **長度** | `pos_label.py` 的 `SVM_DEC` 與從來源目錄（top500，剝 v 前綴）推導的期望值一致 |
| 2 | **唯一** | 編號唯一、全為整數 |
| 3 | **範圍** | 編號落在主庫範圍 `1..9569` |
| 4 | **音檔存在** | 每個編號對應的 `wav/NNNNN.wav` 實際存在 |
| 5 | **順序** | 與來源目錄的序號順序一致（維持 `svm_dec` 高→低） |
| 6 | **三處一致** | 不含 `WRONG` 與 `DUPLICATE` 編號；`asr_screening.json` 與訓練清單的編號集合相同 |

> `WRONG[]` 23 筆但 `ryza_main/wrong/` 僅 22 檔：編號 `01120` 在 `WRONG[]` 中，但未被複製進
> `ryza_main/wrong/`（`wav/01120.wav` 本身存在於主庫）。兩處數量差 1 屬正常。

## 9. 執行順序

**從頭重跑**（會重跑 ASR，耗時數小時級）：

```bash
# 步驟 3：從 top500 造冊（497）
python tools/build_pos_label.py \
  --source-dir ryza_main/top500 \
  --output tools/pos_label.py

# 步驟 4+5：ASR 重跑，門檻 ≥2 生效 → json 497 筆（6 筆 EXCLUDE）
.venv/Scripts/python.exe tools/screen_asr.py \
  --pos-label tools/pos_label.py \
  --audio-root wav \
  --raw-out tools/asr_screening.json

# 步驟 6：排除 6 筆，pos_label 與 json 同步為 491
python tools/build_pos_label.py \
  --source-dir ryza_main/top500 \
  --exclude-from tools/asr_screening.json \
  --prune-raw tools/asr_screening.json \
  --output tools/pos_label.py

# 步驟 7：產清單（純標準庫，不重跑 ASR）
python tools/transcribe.py \
  --pos-label tools/pos_label.py \
  --asr-json tools/asr_screening.json \
  --output data/ryza_train.list

# 驗證：三處一致
python tools/validate_pos_label.py \
  --source-dir ryza_main/top500 \
  --pos-label tools/pos_label.py \
  --neg-label tools/neg_labels.py \
  --asr-json tools/asr_screening.json \
  --train-list data/ryza_train.list \
  --audio-root wav
```

**僅重產清單**（不重跑 ASR，日常最常用）：

```bash
python tools/transcribe.py \
  --pos-label tools/pos_label.py \
  --asr-json tools/asr_screening.json \
  --output data/ryza_train.list

python tools/validate_pos_label.py \
  --source-dir ryza_main/top500 \
  --pos-label tools/pos_label.py \
  --neg-label tools/neg_labels.py \
  --asr-json tools/asr_screening.json \
  --train-list data/ryza_train.list \
  --audio-root wav
```

## 10. 歷史執行紀錄

以下為資料鏈改道當時已執行完畢的一次性動作，**僅供追溯，不是可重跑的流程**。

| 紀錄 | 說明 |
|---|---|
| STT 資料鏈改道 top500，去重後 491 段訓練清單 | commit `672e4e5` |
| 刪除根目錄舊清單 `git rm ryza_train.list` | 清單已改道至 `data/`，根目錄檔案已刪除。**再次執行會失敗** |
| 491 段完整清單的 **Whisper 原始輸出**（校正前） | git 歷史 commit `672e4e5` 的 `tools/asr_screening.json` |

> 舊版篩選報告 [`reports/asr_filter_report.md`](../reports/asr_filter_report.md)
> 已被 `reports/asr_report.md` 取代，保留作為歷史紀錄。