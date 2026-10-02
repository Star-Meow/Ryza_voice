# ASR 語音轉文字報告（STT 階段）

> 產生日期：2026-10-02
> 資料鏈：`ryza_main/top500/` → `tools/pos_label.py` → `tools/asr_screening.json` → `data/ryza_train.list`

## 1. 模型與參數

| 項目 | 設定 |
|---|---|
| ASR 模型 | faster-whisper **large-v3**（CTranslate2 後端） |
| 推論裝置 / 精度 | CUDA（RTX 4070S 12G）/ **float16** |
| **beam size** | **5** |
| **temperature** | **梯度 fallback（0.0 → 1.0，faster-whisper 預設）** |
| VAD | 開啟（Silero VAD，經 onnxruntime） |
| 語言 | `ja`（固定，不偵測） |
| 時長門檻 | 2.0s ~ 15.0s（免模型，讀 WAV header） |
| initial_prompt | 無 |
| 音訊來源 | `wav/`（48kHz / 16bit / mono） |

> **temperature 未固定的原因**：實測固定 `temperature=0.0` 時，`wav/04493.wav` 會陷入 `……` 無限重複幻覺（compression_ratio 20.34，被 HIGH_COMPRESSION 規則捕捉）。改回預設梯度 fallback 後該段恢復為正常文字「せめて、何かきっかけでも作れれば…ん?」。代價是重跑結果可能有少數段差異。

## 2. 處理量與過濾統計

### 資料漏斗

```
500  top500（依 svm_dec 排序）
 -3  WRONG（人工確認非ライザ）          → neg_labels.py WRONG[]
 ────
497  ASR 輸入（pos_label.py SVM_DEC[]）
 -6  DUPLICATE_TEXT（跨檔文本重複）      → neg_labels.py DUPLICATE[]
 ────
491  有效段數（data/ryza_train.list）
```

### 過濾規則觸發統計

| 規則 | 門檻 | 觸發數 | 說明 |
|---|---|---|---|
| DURATION_SHORT | < 2.0s | 0 | 免模型，讀 WAV header |
| DURATION_LONG | > 15.0s | 0 | 同上 |
| EMPTY | VAD 後無文字 | 0 | ASR 空輸出 |
| HALLUCINATION | 命中已知幻覺句 | 0 | 英/日 Whisper 常見幻覺 |
| REPETITION | 同字元連續 ≥4 | 0 | 單字重複 |
| NON_SPEECH | core 為感嘆詞 | 0（REVIEW） | 喘氣、吶喊等短促感嘆 |
| HIGH_COMPRESSION | compression_ratio > 3.0 | 0（REVIEW） | 幻覺指標；實測最高僅 1.54 |
| **DUPLICATE_TEXT** | **core 跨檔 ≥2** | **6（EXCLUDE）** | **保留首次出現者** |
| MISSING / READ_ERROR / ASR_ERROR | — | 0 | 檔案不存在與解碼錯誤 |

**結論**：單檔規則（時長、空輸出、幻覺、重複、感嘆詞、壓縮比）**全數未觸發**，497 段音訊品質乾淨；唯一生效的是跨檔文本去重，排除 6 段同一台詞的重複錄製。

### 被排除的 6 段（DUPLICATE_TEXT）

| 音檔 | 與下列檔案文本重複 | 台詞 |
|---|---|---|
| `wav/04253.wav` | `wav/06332.wav` | あれ?扉が開いてる。あんなしっかり閉まってたのに。 |
| `wav/07018.wav` | `wav/06332.wav` | （同上） |
| `wav/02998.wav` | `wav/01036.wav` | まだだよ。まだ諦めない |
| `wav/07718.wav` | `wav/05676.wav` | 扉、閉じちゃってるね。どうやって開けるんだろう。 |
| `wav/08413.wav` | `wav/05676.wav` | （同上） |
| `wav/01991.wav` | `wav/01359.wav` | 私さ、そろそろ島に帰ろうかなって思うんだ。 |

> 這 10 個音檔 MD5 互不相同、時長些微相差，是**同一句台詞的重複錄製**（遊戲多個 cue 引用同一段語音），非 Whisper 幻覺。對 TTS 訓練而言保留一個版本即可。

### 有效總時長（491 段）

- **合計 51.5 分鐘**
- 平均 6.3s｜最短 2.4s｜最長 12.0s

## 3. 品質抽查：50 段人工校對 CER

### 抽查方法

- 自 491 段以固定 **seed=42** 隨機抽 **50 段**（佔 10.2%，可重現）
- 人工逐段播放音檔，將正確台詞填入 `reference` 欄；無異常的段落留空（視同 Whisper 輸出正確）
- **CER = Levenshtein(reference, whisper) / len(reference)**，此處記錄的是 **Whisper 原始輸出**的錯誤率
- 比對前兩邊都過 core 正規化（去標點、只保留日文/英數字元），避免標點差異干擾

### 結果

| 指標 | 數值 |
|---|---|
| **平均 CER** | **1.0%** |
| 字元錯誤總數 | 10 / 1114 字元 |
| 完全正確（CER = 0） | 44 / 50 段（88%） |
| 有錯誤 | 6 / 50 段（12%） |

### 6 段錯誤明細

| # | 音檔 | Whisper 輸出 | 正確（人工校對） | 編輯距離 | CER |
|---|---|---|---|---|---|
| 13 | `wav/01124.wav` | オート | 王都 | 3 | 10.3% |
| 16 | `wav/01552.wav` | タレジャー | トレジャー | 1 | 2.9% |
| 32 | `wav/09557.wav` | 諸子 | 書庫 | 2 | 11.1% |
| 38 | `wav/05954.wav` | セイリさん | セリさん | 1 | 5.6% |
| 40 | `wav/08204.wav` | な男子学生 | 男子学生 | 1 | 4.2% |
| 50 | `wav/03173.wav` | ピー | フィー | 2 | 18.2% |

### 錯誤類型分析

6 段錯誤**全數為專有名詞／固定詞組的聽寫偏差**，無文法或語意錯誤：

- **同音異字**：オート→王都（おうと）、諸子→書庫（しょこ）
- **清濁音／長音／母音**：タレジャー→トレジャー、セイリ→セリ、ピー→フィー
- **語頭語氣詞**：多了「な」

> **解讀**：Whisper large-v3 對**一般句子的辨識接近完美**（44/50 完全正確），1.0% 的錯誤幾乎全由遊戲專有名詞造成。這類錯誤可透過建立遊戲詞表（王都 / トレジャー / 書庫 / セリ / フィー 等）在轉錄後統一校正，預期校正後 CER 降至 ~0%。
>
> 抽樣僅 50 段就發現 6 個錯誤詞，491 段全文可能還有同類未發現的專有名詞偏差，建議詞表建立後對全文掃描一遍。

## 4. 後續工作

- [ ] 建立遊戲專有名詞詞表，對 491 段逐字稿套用校正
- [ ] 校正後重算 CER，於本報告補「校正後 CER」對照欄（預期 ~0%）
- [ ] 人工試聽確認 491 段皆為ライザ（目前僅 A_high 92 段已人工確認）

## 附錄：相關檔案

| 檔案 | 內容 |
|---|---|
| `tools/asr_screening.json` | 491 筆 ASR 結果（全 PASS；6 筆 EXCLUDE 記錄已精簡，明細見 `asr_filter_report.md`） |
| `data/ryza_train.list` | 491 行 GPT-SoVITS 訓練清單 |
| `reports/cer_sample.csv` | 50 段抽查清單（含人工校對結果） |
| `reports/cer_report.md` | CER 逐段計算明細 |
| `asr_filter_report.md` | 預篩報告（含 6 段 EXCLUDE 明細） |
| `tools/screen_asr.py` | ASR 執行與過濾規則 |
