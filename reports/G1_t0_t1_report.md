# G1 回報：T0（BERT/HuBERT 轉 safetensors）＋ T1（UNK 來源分析）

日期：2026-10-04
回報依據：計畫 §3 Gate G1、§15 回報格式
執行環境：`<GPT>` = `H:\git\GPT-SoVITS`（HEAD `48b1a01`，單一 commit）、`<RYZA>` = `H:\git\Ryza_voice`（分支 `feat/tts-v4-data-prep`，HEAD `24153eb`）
Python：`<GPT>\.venv\Scripts\python.exe`（3.10.11 / torch 2.5.1+cu121 / transformers 4.57.6 / safetensors 0.8.0）

---

## 1. 結果總表

| ID | 結論 | 證據等級 | 證據 | 與計畫是否相符 |
|---|---|---|---|---|
| T0-1 | 兩個 `model.safetensors` 已新增，`.bin` 未變動 | 【實測】 | `<GPT>\GPT_SoVITS\pretrained_models\chinese-{roberta-wwm-ext-large,hubert-base}\model.safetensors`；`.bin` SHA256 轉檔前後相同（見 §2） | 相符 |
| T0-2 | key 名稱、數量、dtype、shape 完全保留 | 【實測】 | `tools/t0_validate.py` V1：roberta 397/397、hubert 211/211，無 only_in_bin / only_in_st / dtype / shape 差異 | 相符 |
| T0-3 | `from_pretrained` 乾淨載入，無 missing/unexpected/mismatched keys、無警告 | 【實測】 | V2：roberta→`BertForMaskedLM`（325,545,608 params）；hubert→`HubertModel`（94,371,712 params）；四類清單皆 `[]`，`warnings: []` | 相符 |
| T0-4 | 新載入與手動讀 `.bin` 載入同一架構，輸出最大絕對差 = 0.0（≤ 1e-6） | 【實測】 | V3：roberta logits `[1,10,21128]` diff 0.0；hubert last_hidden_state `[1,49,768]` diff 0.0 | 相符（優於門檻） |
| T0-5 | 驗證已先證明能辨識失敗（negative fixture） | 【實測】 | `--negative`：roberta perturb→V3 diff 1.226；rename→V1/V2/V3 失敗。hubert perturb→V3 diff 6.126；rename→V1/V2/V3 失敗。`detects_failure: true` | 相符 |
| T0-6 | 對版本控制無影響 | 【實測】 | `git status --short` 仍只有 `?? =0.4.1`；新檔被 `pretrained_models/.gitignore`（內容 `*`／`!.gitignore`）忽略，以 `git ls-files --others --ignored` 確認存在 | 相符 |
| T0-7 | 計畫前提（transformers 拒絕 `.bin`）成立，T0 確有必要 | 【實測】 | `AutoModelForMaskedLM.from_pretrained(bin 目錄)` 與 `HubertModel.from_pretrained(...)` 皆 `ValueError: ... CVE-2025-32434 ... require users to upgrade torch to at least v2.6 ... does not apply when loading files with safetensors` | 相符 |
| T0-8 | 安全載入模式一次成功，未使用非安全模式 | 【實測】 | `torch.load(..., weights_only=True)` 兩檔皆成功（roberta 397 key；hubert 211 key） | 相符 |
| T1-1 | UNK 唯一來源是 `#`，無其他符號 | 【實測】 | `tools/t1_unk_analysis.py`：訓練集 UNK 來源 `{'#': 1776}`、eval10 `{'#': 35}`；全部分類為 `prosody_insert`（韻律邊界插入，非文字中的符號） | 相符 |
| T1-2 | 停止條件不觸發 | 【實測】 | 非 `#` 的 UNK 總數 = 0（門檻 > 20），影響行數 = 0（門檻 > 10） | 相符 |
| T1-3 | 推論端使用同一個 `clean_text` 與日文 G2P，前提成立 | 【讀碼】 | `TextPreprocessor.py:15` `from text.cleaner import clean_text`；`:206-210` `clean_text_inf`→`clean_text(text, language, version)`；`TTS.py:1171` 傳入 `self.configs.version`；ja 走 `text.japanese.g2p`（`with_prosody` 預設 `True`，`japanese.py:267`） | 相符 |
| T1-4 | `#` 不在 symbols2（732 符號），兩端同時產生 UNK | 【實測】 | `'#' in symbols2.symbols == False`；`len(symbols2.symbols) == 732`；`'[' in symbols2.symbols == True`（`[`/`]` 不會變 UNK） | 相符 |
| T1-5 | 無空輸出、無異常短行 | 【實測】 | 訓練集音素數 min 15／median 58／max 120，空輸出 0，`<3` 音素 0；eval10 min 24／median 52／max 104，空輸出 0，`<3` 0 | 相符 |
| T1-6 | 版控紀錄查詢完成 | 【實測】 | `<GPT>` 全庫僅 1 commit（`48b1a01`），`text/japanese.py` 與 `text/symbols2.py` 皆於該 commit 隨初始匯入加入；`git log -S"with_prosody"`、`-S'"#"'` 查無其他 commit | 相符（見 §5 風險 2） |

---

## 2. T0 明細表

| 項目 | chinese-roberta-wwm-ext-large | chinese-hubert-base |
|---|---|---|
| `.bin` 路徑 | `pretrained_models\chinese-roberta-wwm-ext-large\pytorch_model.bin` | `pretrained_models\chinese-hubert-base\pytorch_model.bin` |
| `.bin` SHA256（轉檔前） | `e53a693acc59ace251d143d068096ae0d7b79e4b1b503fa84c9dcf576448c1d8` | `24164f129c66499d1346e2aa55f183250c223161ec2770c0da3d3b08cf432d3c` |
| `.bin` SHA256（轉檔後） | 同上（未變動） | 同上（未變動） |
| `.bin` 大小 | 651,225,145 B | 188,811,417 B |
| `.bin` key 數 | 397 | 211 |
| `.bin` dtype | fp16 ×396＋int64 ×1（`bert.embeddings.position_ids`） | fp16 ×211 |
| 共享儲存（需 clone） | 2 組：`bert.embeddings.word_embeddings.weight`↔`cls.predictions.decoder.weight`；`cls.predictions.bias`↔`cls.predictions.decoder.bias` | 0 組 |
| `model.safetensors` SHA256 | `ddb2d1e2aec45a149bb1b19213c0478cee464f9cc75531e70e673b5224ef4f05` | `25adc31d1889ff5d3da262189433bdc755f0ddb9f194342583dbd17e447daefe` |
| `model.safetensors` 大小 | 694,454,896 B | 188,767,040 B |
| metadata | `{"format": "pt"}` | `{"format": "pt"}` |
| key 保留 | 397/397，dtype/shape 全同 | 211/211，dtype/shape 全同 |
| `from_pretrained` 載入類別 | `BertForMaskedLM`（經 `AutoModelForMaskedLM`，比照 `1-get-text.py:62`） | `HubertModel`（`local_files_only=True`，比照 `feature_extractor/cnhubert.py:31`） |
| missing / unexpected / mismatched keys | `[]` / `[]` / `[]`（error_msgs `[]`，無警告） | `[]` / `[]` / `[]`（error_msgs `[]`，無警告） |
| 輸出最大絕對差（vs 手動讀 `.bin`） | 0.0（logits，output_abs_max 6.88） | 0.0（last_hidden_state，output_abs_max 4.54） |
| 複製到正式目錄後重驗 | 全 V1–V4 通過（與溫存目錄結果相同） | 全 V1–V4 通過 |

執行方式：`tools/t0_convert_safetensors.py`（實作）先在 `tools\t0_work\` 暫存目錄轉檔（只複製 config 類檔案），`tools/t0_validate.py`（獨立驗證，只看 input `.bin` 與 output `.safetensors`）在暫存目錄驗證＋negative fixture，通過後才複製進 pretrained 目錄，並於最終位置重跑完整驗證；溫存目錄與偵察腳本（`t0_work\`、`t0_recon_tmp.py`）已刪除。

Negative fixture 證明（驗證能抓到損壞才信任正向結果）：

| 模型 | 故意損壞方式 | 預期 | 實際 |
|---|---|---|---|
| roberta | 數值偏移（某 weight +1.0） | 失敗 | V3 diff 1.226 > 1e-6 → FAIL |
| roberta | 改名一個 key（`..._BROKEN`） | 失敗 | V1/V2/V3 皆 FAIL |
| hubert | 數值偏移 | 失敗 | V3 diff 6.126 > 1e-6 → FAIL |
| hubert | 改名一個 key | 失敗 | V1/V2/V3 皆 FAIL |

---

## 3. T1 統計明細

比照 `1-get-text.py:92` 的呼叫：`clean_text(text.replace("%", "-").replace("￥", ","), "ja", "v4")`；另呼叫 `text.japanese.g2p` 取替換前的原始音素以還原 UNK 來源（兩者比對一致行數：訓練集 467/467、eval10 10/10，0 不一致）。

### 訓練集 `data\finetune_train.list`（467 行）

| 項目 | 值 |
|---|---|
| 解析成功行數 | 467（格式異常或非 ja：0） |
| 音素數分布 | min 15／median 58／max 120 |
| 含 UNK 的行數 | 451 / 467（96.6%） |
| 每行 UNK 數 | min 0／median 4／max 11 |
| 每行 UNK 比例 | min 0.0000／median 0.0633／max 0.1224 |
| UNK 來源符號 | `#` ×1776（100%） |
| 來源類別 | `prosody_insert` ×1776（韻律邊界，`japanese.py:248` 插入，不在原文中）／`in_text` ×0 |
| 非 `#` 的 UNK | 0 |
| 空輸出（0 音素）行數 | 0 |
| 異常短行（<3 音素） | 0 |
| 含 `%` 或 `￥` 的行數 | 0 |

UNK 最多前 5 行：`06844.wav`(11/106)、`03166.wav`(10/104)、`08365.wav`(10/120)、`02580.wav`(9/107)、`02700.wav`(9/105)。

### eval10 `data\finetune_eval10.list`（10 行）

| 項目 | 值 |
|---|---|
| 解析成功行數 | 10 |
| 音素數分布 | min 24／median 52／max 104 |
| 含 UNK 的行數 | 10 / 10 |
| 每行 UNK 數 | min 1／median 3.5／max 7 |
| 每行 UNK 比例 | min 0.0256／median 0.0681／max 0.0889 |
| UNK 來源符號 | `#` ×35（100%，全為 prosody_insert） |
| 非 `#` 的 UNK | 0 |
| 空輸出／異常短行 | 0 / 0 |

### 推論端比對（`#` → UNK 前提）

- `GPT_SoVITS\TTS_infer_pack\TextPreprocessor.py:15`：`from text.cleaner import clean_text`（與 `1-get-text.py:20` 同一函式）。
- `TextPreprocessor.py:206-210`：`clean_text_inf` → `phones, word2ph, norm_text = clean_text(text, language, version)`；`phones = cleaned_text_to_sequence(phones, version)`。
- `GPT_SoVITS\TTS_infer_pack\TTS.py:1171`：`self.text_preprocessor.preprocess(text, text_lang, text_split_method, self.configs.version)` — version 由 `TTS_Config` 傳入（`:331-333` 讀取、`:399-401` `update_version`），推論時帶 v4 即與訓練同路徑。
- ja 語系在 `clean_text` 中走 `text.japanese`，`g2p()` 預設 `with_prosody=True`（`japanese.py:267`），`#` 由 `pyopenjtalk_g2p_prosody` 的 accent phrase border 插入（`japanese.py:248`）。
- `symbols2.symbols` 共 732 個符號，不含 `#`（`symbols2.py:783-788` 建構），故推論端 `#` 同樣映射為 UNK — 兩端行為一致，**「不過濾 `#`」的前提成立**。

---

## 4. 與計畫不符或需 PM 留意的事項

1. **HuBERT 無 weight_norm 參數**：計畫 §6 提示「HuBERT 的 weight_norm 參數命名需特別留意」。實測 `chinese-hubert-base` 的 `.bin` 為標準 transformers 命名（211 key，無 `weight_g`/`weight_v`），`from_pretrained` 無任何 missing/unexpected keys。⚠️ 推斷不成立，但對結果無影響。
2. **RoBERTa 實際載入類別是 `BertForMaskedLM`**：`config.json` 指定 BERT 架構，`AutoModelForMaskedLM` 解析為 `BertForMaskedLM`（非 `RobertaForMaskedLM`）。已驗證乾淨載入，不影響使用。
3. **roberta safetensors 含 `bert.embeddings.position_ids`（int64）**：`from_pretrained` 內部吸收，無警告；但手動 `load_state_dict(strict=False)` 會列為 unexpected（V3 的 `load_info_b_unexpected` 有記錄）。屬預期行為，不影響數值等價（diff 0.0）。
4. **safetensors 比 `.bin` 略大**：roberta 694 MB vs 651 MB；hubert 188.8 MB vs 188.8 MB（ 幾乎相同）。safetensors 每張量有固定 header／alignment，為正常格式差異。
5. **共享張量只 clone 第二份**：`cls.predictions.decoder.weight`、`cls.predictions.decoder.bias` 被 clone（數值不變），`word_embeddings.weight` 等保留原樣；`from_pretrained` 會自動重新 tie，數值等價。
6. **推論端缺 `%`／`￥` 前處理**：`1-get-text.py:92` 有 `.replace("%","-").replace("￥",",")`，`TextPreprocessor` 無。本資料集 train/eval10 含 `%` 或 `￥` 的行數皆為 0，**無實際影響**；列為觀察供 `docs/07_finetune.md` 易錯點使用。

---

## 5. 風險與意外發現

1. **ja_userdic 的 `user.dict` 未生成**：`japanese.py:61-71` 會在首次載入時以 `pyopenjtalk.mecab_dict_index` 從 `userdict.csv`（17 MB，為 `<GPT>` 追蹤檔）建立 `user.dict`，但該區塊被 `try/except ... pass` 包住，本次 G2P 跑完後 `ja_userdic\` 仍只有 `userdict.csv`，無 `user.dict`／`userdict.md5` — 推測 `mecab_dict_index` 失敗被吞掉，G2P 實際使用 pyopenjtalk 內建詞典。`git status` 確認 `<GPT>` 無新增檔案。**風險**：訓練預處理與推論若都在本環境跑會一致；但若某次成功建立 user dict，G2P 結果可能改變，是重現性觀察點，建議在 `docs/07_finetune.md` 記錄。
2. **`<GPT>` 為單一 squashed commit**：全庫僅 `48b1a01`（`Fix Fun-ASR-Nano Transformers requirement (#2824)`），`text/japanese.py`、`text/symbols2.py`、`text/cleaner.py` 皆於該 commit 隨初始匯入加入，`git log -S` 查無與 `#` 或 `with_prosody` 相關的其他提交，無法追溯歷史變更。
3. **`<GPT>` 工作樹狀態**：始終只有預先存在的 `?? =0.4.1`（計畫 §2.1 已列為「工作樹乾淨」的唯一例外）；兩個 `model.safetensors` 被 `pretrained_models/.gitignore` 忽略。
4. **TF32**：依 PM 裁定（其他環境問題、本機無視），全程未修改任何腳本，冒煙階段若出現 illegal instruction／CUBLAS 錯誤才停止回報。

---

## 6. 無法判定的項目

- 無。T0 的四項驗證與 T1 的五項回報內容皆已取得實測或讀碼證據。

---

## 7. 目前工作樹與產出

- `<RYZA>`（分支 `feat/tts-v4-data-prep`，HEAD `24153eb`，尚未建立 §5 的工作分支——依計畫 §8 P0-2 於階段 0 建立）：新增未追蹤檔案 `tools\t0_convert_safetensors.py`、`tools\t0_validate.py`、`tools\t1_unk_analysis.py`、`reports\G1_t0_t1_report.md`。既有追蹤檔皆未修改。
- `<GPT>`：僅新增兩個被忽略的 `model.safetensors`，`git status` 仍只有 `?? =0.4.1`。

**停在 G1，等待 PM 放行後才進入階段 0（P0-1～P0-9）與階段 A（預處理）。**
