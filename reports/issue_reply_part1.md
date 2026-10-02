## Part 1：ASR 轉譯進度回報

### 背景

原開發環境為 RTX 3060 6G，於 STT 階段因顯存不足無法穩定執行 Whisper large-v3，因此先完成候選音檔排序，再將 STT 階段遷移至 RTX 4070S 12G 環境開發。套件庫依原環境 `.venv` 與 `requirements.txt` 重新構築。

### 目前進度

**模型與參數**

- 使用 faster-whisper large-v3（CTranslate2 後端），CUDA / float16。
- beam size 5，temperature 採梯度 fallback（0.0 → 1.0）。
- VAD 開啟（Silero VAD），language=ja 固定不偵測，時長門檻 2.0s ~ 15.0s。
- temperature 未固定的理由：開發期間觀察到固定 `temperature=0.0` 時 `wav/04493.wav` 輸出異常重複，改回預設梯度 fallback 後恢復正常，故未固定 temperature。

**處理量與過濾統計**

- 資料漏斗：500（top500）− 3（WRONG）＝ 497（ASR 輸入）− 6（DUPLICATE_TEXT）＝ 491（有效段數）。
- Issue 要求的過濾規則（時長 <2s / >15s、非說話聲、ASR 空輸出、重複、幻覺）全數未觸發，觸發數皆為 0。
- 額外過濾：`DUPLICATE_TEXT`（跨檔同一台詞重複錄製）排除 6 段，已於報告標示為「額外過濾（非 Issue 要求）」。
- 有效總時長：491 段合計 51.5 分鐘，平均 6.3s。

> **關於 WRONG[] 的兩個數字**：`neg_labels.py` 的 `WRONG[]` 為**全庫**人工確認非ライザ的累計清單，共 **23 筆**；其中落在 top500 範圍內的只有 **3 筆**，本次漏斗扣除的是這 3 筆。其餘 20 筆在 top500 之外，不參與本階段計算。草稿「500 − 3」中的 3 即 top500 內的剔除數，與 23 筆為不同口徑，非矛盾。

**品質抽查：50 段人工校對 CER**

- 固定 `seed=42` 隨機抽 50 段，人工逐段檢視。留空 = 判定無異常；修正 6 段 = 有誤。
- CER 計算基準為 Whisper 原始輸出，平均 CER 1.0%（10 / 1114 字元），完全正確 44 / 50 段。
- 6 段錯誤全數為聽寫偏差，無文法或語意錯誤；其中 5 段為專有名詞／固定詞組（同音異字、清濁音／長音／母音），1 段（id40 `wav/08204.wav`）為語頭語氣詞多一字「な」，非專有名詞。

**專有名詞校正**

建立詞庫 `tools/glossary.py`，並以 `tools/apply_glossary.py` 套用確認型修正 `CORRECTIONS`，共 6 條規則：

| # | 規則（誤聽 → 正確） | 命中段數 | 依據 |
|---|---|---|---|
| 1 | `オート` → `王都` | 5 | CER id13；本庫 5 檔全數誤聽 |
| 2 | `ボース` → `ボオス` | 6 | 全庫掃描；6 檔全數誤聽（抽樣外） |
| 3 | `セイリ` → `セリ` | 1 | CER id38 |
| 4 | `ピー` → `フィー` | 1 | CER id50 |
| 5 | `諸子` → `書庫` | 1 | CER id32 |
| 6 | `タレジャーハンター` → `トレジャーハンター` | 1 | CER id16 |
|  | **合計** | **15 次 / 14 檔** | |

與 CER 抽查的對應關係：

| CER 錯誤段 | 對應規則 | 是否收入詞庫 |
|---|---|---|
| id13 `wav/01124.wav` | オート → 王都 | ✅ |
| id16 `wav/01552.wav` | タレジャーハンター → トレジャーハンター | ✅ |
| id32 `wav/09557.wav` | 諸子 → 書庫 | ✅ |
| id38 `wav/05954.wav` | セイリ → セリ | ✅ |
| id50 `wav/03173.wav` | ピー → フィー | ✅ |
| id40 `wav/08204.wav` | （語頭語氣詞「な」，非專有名詞） | ❌ 不收 |

亦即：CER 的 6 段錯誤有 5 段被詞庫覆蓋；詞庫的第 6 條 `ボース→ボオス` 來自全庫掃描，不在 50 段抽樣內。

6 條規則合計命中 15 次，落在 14 個不同音檔（`wav/02580.wav` 同時含 `ボース` 與 `オート` 兩條誤聽），此即「**14 筆逐字稿修正**」的由來。`data/ryza_train.list` 逐字稿欄以校正後文字為準。

校正前後對照留存於 `reports/correction_diff.md`（14 段完整前後文）與 `reports/correction_diff.csv`（機器可讀），供直接查驗；完整 491 段的校正前輸出另可由 git 歷史（commit `672e4e5`）還原。

**交付物**

*Issue 要求*

- `data/ryza_train.list`：491 行，格式 `音檔路徑|ryza|ja|文字`（校正後）。
- `reports/asr_report.md`：模型參數、過濾統計、有效時長、CER 抽查、錯誤類型分析。
- `tools/transcribe.py`、`tools/screen_asr.py`：路徑一律以參數傳入，未寫死絕對路徑。
- `requirements.txt`：本階段所需套件與版本。

*額外新增（非 Issue 要求，供資料鏈維護與品質查驗）*

- `tools/glossary.py`：專有名詞詞庫（`PROPER_NOUNS` 正確形對照、`CORRECTIONS` 確認型修正）。
- `tools/apply_glossary.py`：套用 `CORRECTIONS` 並重算 `core`。
- `tools/build_pos_label.py`：造冊工具，產生與更新 `pos_label.py`。
- `tools/validate_pos_label.py`：三處編號一致性驗證（長度／唯一／範圍／音檔存在／順序）。
- `reports/correction_diff.md`、`reports/correction_diff.csv`：校正前後對照表。

*資料檔（由造冊工具產生，非腳本）*

- `tools/pos_label.py`：`SVM_DEC[]` 491 筆正樣本編號。
- `tools/neg_labels.py`：`WRONG[]` 23 筆（全庫，top500 內 3 筆）＋ `DUPLICATE[]` 6 筆。

**驗收對照**

| Issue 要求 | 狀態 |
|---|---|
| 使用 Whisper large-v3 或等效模型，language=ja，PR 說明理由 | 完成 |
| `tools/transcribe.py` 路徑參數化，不寫死絕對路徑 | 完成 |
| 過濾時長 <2s 或 >15s、非說話聲、ASR 空輸出、重複、幻覺 | 完成 |
| 產出 `data/ryza_train.list`，格式 `音檔路徑\|ryza\|ja\|文字` | 完成 |
| 有效轉譯段數 ≥ 450 | 完成（491 段） |
| 全數經人工確認為萊莎 | 完成 |
| `reports/asr_report.md` 含模型參數、過濾統計、有效時長、CER | 完成 |
| 專有名詞已人工校正，最終清單以校正後文字為準 | 完成 |
| 新增 `requirements.txt` | 完成 |

### 實作決策

- **只收確認型修正**：`CORRECTIONS` 每條都有 CER 人工校對或全庫掃描依據；推測型誤聽（如 `リザ→ライザ`）本庫 0 次觸發，不收。`ヴォルカ→ヴォルカー` 因會把正確的 `ヴォルカー` 誤改為 `ヴォルカーー`（子字串誤改），也不收。
- **詞庫分兩層**：`PROPER_NOUNS` 為正確形對照，完整保留不收斂，供人工校對與後續擴充；實際參與字串取代的只有 `CORRECTIONS`。
