# 事實層級遺失稽核

**目的**：確認整理前（`docs-baseline` = 262adda）的內容，是否都還出現在新版本的某份文件中。

**方法**：[`audit_facts.py`](audit_facts.py) 從基準版本 28 份 `.md` 抽出 6,021 個「事實片段」，包括行內程式碼、檔名、數字（小數、三位數以上、百分比）與 hash。接著在新版本的全部 `.md` 加上 `LICENSE` 中逐一搜尋（比對前會正規化，去除空白與標點）。逐筆清單見 [`audit_missing.csv`](audit_missing.csv)。

這個方法和 `check.py` 互補：
- `check.py` 只看章節是否到了指定的目標檔。
- 本稽核看整個新版本，也涵蓋處置為 `delete` 的章節。

## 結果

| 階段 | 找不到的片段 |
|---|---|
| 初次稽核 | 787 |
| 補回漏收的技術內容後 | **725** |

### 補回的內容（初次稽核發現的真實遺失）

| 原位置 | 內容 | 補到 |
|---|---|---|
| `reports/G4_stage_c_report.md` §3、`G1` 推論端比對、`G3` §4 | 上游程式碼關鍵行號，其中幾項新版本完全沒寫：resume 失敗時裸 `except` 靜默退回底模（`s2_train_v3_lora.py:190`）、S2 時長過濾 0.6～54 秒（`data_utils.py:588`）、`TTS.run` 是 generator、`save_every_epoch`／`if_save_every_weights` 由 webui 注入 | `docs/07_finetune.md` 附錄 A |
| `reports/01_env_model_prep.md`「日文 G2P 驗證」 | pyopenjtalk 與 `clean_text` 的預期輸出，可用來確認環境 | `docs/07_finetune.md` §1.3 |
| `reports/G1_t0_t1_report.md` §4 | 推論端缺 `%`／`￥` 前處理（`1-get-text.py:92`） | `docs/07_finetune.md` §8 #10 |
| 同上 | `BertForMaskedLM` 載入類別、`position_ids` unexpected key、HuBERT 無 weight_norm | `docs/07_finetune.md` §2.2 |
| `docs/09_pitfalls.md` §6 | S1 資料擴充的出處 `AR/data/dataset.py` | `docs/07_finetune.md` §5.4 |
| `reports/smoke_test_stage_c.md` §5.3 | S2 step 數由 `DistributedBucketSampler` 分桶決定（`data_utils.py:994-1016`） | `docs/07_finetune.md` §5.4 |
| `reports/G4_stage_c_report.md` §5 | S1 沒有逐步 loss（Lightning 未設 `on_step=True`） | `docs/07_finetune.md` §5.5 |

### 剩餘 725 筆的分類（刻意不保留）

| 類別 | 筆數 | 說明 | 原文可否回溯 |
|---|---|---|---|
| A. 舊檔名、行號引用 | 173 | 例如 `03_unpack.md`、`docs/04_speaker_id.md:145`、`README.md:64`。檔案已改名或刪除，引用改指新檔名或章節 | 改名對照見 `check.py` 的 `RENAMES` |
| B. 階段回報的當次明細 | 401 | G1～G4、冒煙測試、環境準備中的當次量測值、檔案清單、branch／HEAD 狀態、停止點、epoch 外推表、逐檢查項明細。結論已在 07、08；明細屬於執行紀錄 | 08 坑記錄的證據欄以固定版本連結指向原報告；其餘以 `git show 262adda:<路徑>` 查閱 |
| C. 協作產物 | 84 | issue_compliance 的合規判定、pr_description 的交付清單、issue_reply 的進度回報 | issue #1、PR #6；`issue_compliance` 未曾張貼到 GitHub，只能用 `git show 262adda:reports/issue_compliance.md` 查閱 |
| D. 保留檔的措辭與更正 | 67 | 去除 Part／階段／PM 字樣；更正錯誤（`・・・` 不屬 ELL、`--audio-root wav` 錯誤）、差異描述與附錄 A 對齊；刪除本機桌面路徑；跨多行的程式碼片段因單行修改而整段比對不到 | 已逐筆人工檢查，抽查的兩筆（`NNNN_原檔名`、`eval10_*.wav`）確認內容仍在 |

## 結論

- 初次稽核找到 7 處真實遺失，都是技術參考內容，已補回 `docs/07_finetune.md`。
- 剩餘 725 筆都屬於刻意不保留的類別，並全部可以回溯：
  - A 類可依改名對照找到新位置。
  - B、C 類可在固定版本 commit `262adda` 查閱原文。
  - D 類都是措辭調整或錯誤更正。

**需要留意的一點**：`issue_compliance.md`（合規稽核）從未張貼到 GitHub，刪除後只存在於 git 歷史。它的已知限制已併入 `asr_report` §5，v00596 的來源說明已併入 `04` §1.5，其餘是 PASS／PARTIAL 判定。
