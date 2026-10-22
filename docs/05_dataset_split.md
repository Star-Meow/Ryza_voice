# 資料切分

> 把 [`04_transcript_pipeline.md`](04_transcript_pipeline.md) 產出的 `data/ryza_train.list`（491 段）切成訓練集與測試集。
> 本文是切分的唯一說明來源；切分結果見 [`reports/split_report.md`](../reports/split_report.md)，各資料檔格式見 [`data/README.md`](../data/README.md)。

---

## 1. 目標

| 集合 | 段數 | 用途 |
|---|---|---|
| train | **467** | 訓練的唯一輸入 |
| test | **24**（約 5%） | 測試集，不得進入任何訓練步驟 |
| eval10 | **10** | test 的子集，用於生成樣本 (a) 與聲線相似度基線 |

為避免純隨機抽樣漏掉少數語氣類別，採**分層隨機抽樣**。稀有層下限為 2 段，不足即硬性失敗；其餘按比例分配。
seed 固定為 42，與 ASR 的 CER 抽查使用同一慣例。

## 2. 分層抽樣設計

兩層分層，seed 固定 42：

**第一層＝句末標點分類**（`split_train_test.py:58-70`）

| 類別 | 判斷 | test 配額 | eval10 配額 |
|---|---|---|---|
| `Q` | 結尾 `?` `？` | 4 | 2 |
| `EXCL` | 結尾 `!` `！` | 2 | 1 |
| `ELL` | 結尾 `…` 或 `...` | 3 | 1 |
| `DECL` | 其餘 | 15 | 6 |
| | **合計** | **24** | **10** |

**第二層＝各類別內的時長三分位**：依 `(duration, utt_id)` 排序，桶索引 `= (pos*3)//len(items)`，`d0` 最短、`d1` 中、`d2` 最長。配額依桶大小以**最大餘數法**分配，再由 `random.Random(seed)` 依固定 `CLASS_ORDER = [DECL, EXCL, ELL, Q]` 抽樣。

manifest 的 `bucket` 欄即 `f'{class}-d{bucket_index}'`，例如 `DECL-d1` = 平敘類、中時長三分位；`Q-d0` = 疑問類、最短三分位。

> ⚠️ **`class` 欄位的三個性質，複現時務必先知道**（主觀試聽後補充）
>
> 1. **它只是文字推斷，不是聲學情緒。** 分類只看句末標點，而標點表達的是語法功能，不是聲優實際的語氣。
>    主觀試聽已實證兩者會不符：全新台詞 (b) 10 句的語氣標註用的正是同一套規則，
>    而試聽結果顯示**至少 8/10 句標註與實際聲音不符，興奮與悲傷兩類全錯**。
> 2. **它不進入任何訓練腳本。** `data/finetune_train.list` 仍為 4 欄（`路徑|ryza|ja|文字`），
>    `prepare_datasets/1-get-text.py:129` 只解這 4 欄。模型**沒有**被灌入任何情緒標籤，
>    不會「學會 DECL → 開心的聲音」。
> 3. **但它決定了誰被留作測試集**，因此**評測軸是「文字看起來像什麼」而非「聲音實際是什麼」**。
>    `class` 分布：eval10 為 DECL 6／Q 2／EXCL 1／ELL 1，test24 為 DECL 15／Q 4／ELL 3／EXCL 2。
>    這是量化指標（cosine、CER）對「情緒偏開心」缺陷**結構性盲視**的成因——
>    兩個 gate 指標都不衡量聲學情緒，卻是在一個聲學情緒未經驗證的集合上量的。
>
> 詳見 [`../reports/defect_triage.md`](../reports/defect_triage.md) §P-2 與
> [`../reports/finetune_report.md`](../reports/finetune_report.md) §6-4。
> **若要把語氣品質納入驗收，需要另建指標並以聲學情緒重新分層，不能沿用 cosine。**

## 3. 參考音訊

規則（`split_train_test.py:300-310`）：在 **train** 內依原始清單順序，取第一個 `4.0 ≤ duration ≤ 8.0` 且峰值 `< 32767` 者；退而求其次放寬為 `3.0–10.0`。

本次選中 `wav/00042.wav`（清單第 3 行、utt_id `00042`、7.12 s、峰值 31488/32767）。

> ⚠️ **`data/ref/ref.txt` 與 `ref.wav` 不同步**：`ref.txt` 的內容是 `00042` 的台詞，而磁碟上的 `ref.wav` 實際已被覆蓋為 `wav/03536.wav`（md5 `1f63d4fceb5bdebb1b556e37c8331276`）。**推論時的 prompt text 必須用 `wav/03536.wav` 的原文**：`よかった。また手伝えることがあったらいつでも言ってね。`（見 [`07_finetune.md`](07_finetune.md) §12.1）。`ref.wav` 本身被 `.gitignore` 忽略，`ref.txt` 則為追蹤檔。

## 4. 產出與驗證

| 檔案 | 內容 |
|---|---|
| `data/finetune_train.list` | **467** 行，四欄，**絕對路徑** |
| `data/finetune_test.list` | **24** 行 |
| `data/finetune_eval10.list` | **10** 行（`eval10` 是 test 的子集） |
| `data/split_manifest.csv` | 491 列 ＋ 表頭，11 欄：`utt_id,audio_path,speaker,duration,text,split,bucket,md5,source_rel,class,eval10` |
| `data/ref/ref.wav` | 參考音訊（來源見 §3） |
| `reports/split_report.md` | 切分報告 |

**寫檔前的 11 項門檻全部通過才會落盤**（`split_train_test.py:279-318`，任一失敗 `sys.exit(1)`）：train=467、test=24、eval10=10、**train ∩ test = ∅**、train ∪ test = 491、eval10 ⊆ test、輸入 SHA256 未變、參考音訊取自 train、逐列路徑存在且四欄齊全、speaker=ryza、lang=ja、文字非空且不含 `|`、無重複路徑。

時長統計：train 49.11 分、test 2.36 分、eval10 0.92 分、全集 51.47 分；MD5 重複組 **0**。

切分工具為 [`tools/split_train_test.py`](../tools/split_train_test.py)，路徑一律由參數傳入。

## 5. 執行指令

```bash
.venv/Scripts/python.exe tools/split_train_test.py \
  --input data/ryza_train.list \
  --train-out data/finetune_train.list \
  --test-out data/finetune_test.list \
  --eval-out data/finetune_eval10.list \
  --manifest data/split_manifest.csv \
  --ref-dir data/ref \
  --report reports/split_report.md \
  --seed 42
```

> `data/ryza_train.list` 存的是**相對路徑**，而 `--audio-root` 的預設值已是 repo 根目錄。
> **不要加 `--audio-root wav`**，否則會解析成 `<root>/wav/wav/xxxxx.wav`。
