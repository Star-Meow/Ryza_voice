# Issue 合規稽核報告

日期：2026-10-05
稽核對象：`docs/01_issue.md` 全文（Part 1 ASR 轉譯 ＋ Part 2 GPT-SoVITS finetune）
稽核範圍：分支 `feature/TTS-finetune`，僅檢查 repo 內可驗證的證據
方法：讀檔 ＋ 唯讀檔案統計（`ls`／`grep`／`wc`／CSV 逐列比對）。**未載入模型、未重跑任何流程。**

> **本報告的判定標準**：PASS ＝ 有可驗證證據且符合要求；PARTIAL ＝ 部分符合或有實質缺口；
> FAIL ＝ 有反證或明確未執行；UNKNOWN ＝ repo 內無法判定，需外部資訊。
> **「報告中的斷言」不等於「證據」**——本稽核對每個判定都另找獨立證據，
> 因此出現多項「報告宣稱符合、但 repo 內無佐證」的判定。

---

## 0. 摘要

| 面向 | PASS | PARTIAL | FAIL | UNKNOWN |
|---|---|---|---|---|
| Part 1 範圍（3 條） | 3 | 0 | 0 | 0 |
| Part 1 要求（3 條） | 2 | 1 | 0 | 0 |
| Part 1 驗收（6 條） | 4 | 2 | 0 | 0 |
| Part 2 要求（3 條） | 2 | 0 | 0 | 1 |
| Part 2 驗收（7 條） | 5 | 2 | 0 | 0 |
| 交叉檢查（3 條） | 0 | 3 | 0 | 0 |

**2026-10-05 PM 補充事實後複核**：S1（範圍改道 top500）、S2（B_likely 逐一試聽）、
A2（全數人工確認萊莎）、A4（50 段人工校對）四項**執行面均成立**，此前判為 FAIL／無據的結論已修正。
PM 說明：早期視聽與人工檢查樣本皆由人工執行；範圍改走 `top500/` 是因擔心樣本數偏頗的刻意決策。

**殘留問題不是「有沒有做」，而是「能不能稽核」**：repo 內無逐筆的
「音檔 → 判定 → 試聽者 → 日期」對照表；`cer_sample.csv` 的 44/50 列在資料格式上
無法區分「人工判定無誤」與「未經檢視」。兩者都不影響驗收成立，但影響日後審計。

**技術面（格式、參數化、可重現、量化報告）品質良好**。

---

## 1. Part 1 範圍（issue 第 8–10 行）

| ID | 要求 | 判定 | 證據 |
|---|---|---|---|
| S1 | 轉譯範圍為 `ryza_main/A_high/`（92）＋ `ryza_main/B_likely/`（386） | **PASS（依 PM 說明）** | **PM 2026-10-05 說明：改走 `top500/` 是刻意決策**——因擔心樣本數偏頗，將已人工確認的 `A_high`（A）保留，其餘直接以 `svm_dec` 排序取前 500，再做去躁與正負樣本確認。動因、做法、結果已寫入 `docs/05_transcript_pipeline.md` §1.5。**兩個附帶事實**：(a) `ls ryza_main/B_likely` = **383**，issue 的 386 為過期數字（`04_speaker_id.md:145`、`README.md:64` 一致）；(b) 實際納入 `A_high` 92/92 ＋ `B_likely` 378 ＋ `C_possible` 20 → top500 497 → 去重 491 |
| S2 | `B_likely` 逐一試聽確認為萊莎；非萊莎者移 `wrong/` 並更新 `tools/neg_labels.py` | **PASS（依 PM 說明）** | **PM 2026-10-05 確認：早期視聽與人工檢查樣本皆由人工執行。** `tools/neg_labels.py` 的 `WRONG[]`（23 筆）即人工判定的結果，其中 3 筆（`1220`／`3430`／`4055`，commit `6dba555` 訊息證實）落在本次範圍內。⚠️ **殘留的是文件缺口而非執行缺口**：repo 內無逐筆「音檔→判定→試聽者→日期」對照表，`docs/04_speaker_id.md:122` 亦僅標 `A_high` 為「已全數人工確認」。見 §7-1 |
| S3 | 不足 478 段時由 `C_possible` 中 `svm_dec` 最高者依序遞補 | **PASS** | 20 筆 `C_possible` 進入最終清單，序號 0485–0500（top500 最末端＝`svm_dec` 最低端），順序與要求一致。**但觸發原因不同**：非「確認後不足 478」，而是 top500 取滿 500 後自然耗盡 |

### 1.4 額外發現：1 筆「強烈疑似錯誤」層的音檔進入訓練集

| 檢查 | 結果 |
|---|---|
| `ls ryza_main/D_flagged \| grep 596` | `v00596.wav` |
| 進入最終清單？ | ✅ `data/ryza_train.list:97` = `wav/00596.wav\|ryza\|ja\|あれって、フィルフサじゃん。…` |
| 該檔 `svm_dec` | **−0.4597**（`ryza_main_index.csv:911`） |
| `D_flagged` 定義 | `docs/04_speaker_id.md:147`：165 檔，`svm_dec < −0.433`，**「比任何已確認誤判都更負，強烈疑似錯誤」** |
| 為何進入 | `tools/select_samples.py:8` 規則：「各排序層中檔名以 `v` 前綴者，歸入 A（全部採用，不參與排序）」 |

**這是產出資料的品質缺陷：491 段中有 1 段來自專案自訂為「強烈疑似錯誤」的層。**
`docs/05_transcript_pipeline.md:50` 有列出 `v00596`，但未揭露它源自 `D_flagged`。
`tools/validate_pos_label.py` 的 6 項檢查未涵蓋「來源 tier 合法性」，故流程未被阻擋。

---

## 2. Part 1 要求（issue 第 12–15 行）

| ID | 要求 | 判定 | 證據 |
|---|---|---|---|
| R1 | 使用 Whisper large-v3（或其他 ASR，**須於 PR 說明選擇理由**），`language=ja` | **PASS** | `tools/screen_asr.py:127`（`--model` 預設 `large-v3`）、`:192-194`（`language=e['lang'] or 'ja'`）；參數記錄於 `reports/asr_report.md:8-21`（beam 5、temperature 梯度 fallback、VAD、時長門檻）。選擇理由見 `asr_report.md:21`（faster-whisper 框架 vs openai-whisper 權重等價 ＋ 顯存／吞吐）與 `reports/issue_reply_part1.md:5,11-14`。**但「為何選 large-v3 這個檔位（而非 medium/small）」未被記錄**，屬輕微缺口 |
| R2 | 新增 `tools/transcribe.py`，**不得寫死本機絕對路徑** | **PASS** | 全文 97 行，argparse 預設全為 repo 相對路徑（`:49-58`），`:12` docstring 明寫「路徑與文字一律由參數與 json 供給，不寫死」。**同鏈 9 個工具全域掃描 `[A-Za-z]:[\\/]` 為 0 hit** |
| R3 | 過濾五類片段並記錄排除原因 | **PARTIAL** | 五類全部實作且有記錄機制（每筆 record 帶 `reason`／`evidence`，`screen_asr.py:216-218`）：`DURATION_SHORT`／`DURATION_LONG`／`EMPTY`／`HALLUCINATION`（12 條樣板）／`REPETITION`。**但觸發數全部為 0**，唯一實質排除是 issue 未要求的 `DUPLICATE_TEXT`（6 筆）。**「非說話聲（喘氣、吶喊）」僅標 `REVIEW` 不排除**（`screen_asr.py:27-39,97-98`）。**排除原因的可稽核持久性已被破壞**：`tools/asr_screening.json` 現為 491 筆全 `PASS`（`build_pos_label.py:70-74` 的 `--prune-raw` 刪除了排除紀錄） |

---

## 3. Part 1 驗收要件（issue 第 18–29 行）

| ID | 要求 | 判定 | 證據 |
|---|---|---|---|
| A1 | 產出 `data/ryza_train.list`，每行 `音檔路徑\|ryza\|ja\|文字` | **PASS** | 491 行，`awk -F'|'` 全為 **4 欄**；無絕對路徑 |
| A2 | 有效段數 ≥ 450，**且全數經人工確認為萊莎** | **PASS（依 PM 說明）** | 491 ≥ 450 ✓。**PM 2026-10-05 確認：早期視聽與人工檢查樣本皆由人工執行**，`reports/asr_report.md:185`「top500 497 檔已全數人工試聽確認為ライザ」**為事實陳述，非無據斷言**。⚠️ **殘留的是可稽核性缺口**：repo 內無逐筆紀錄、無日期、無試聽批次紀錄，且 `docs/04_speaker_id.md:122` 僅標 `A_high` 為「已全數人工確認」、語意上未涵蓋 `B_likely`。**這不影響驗收成立，但影響日後稽核**。見 §7-1 |
| A3 | `reports/asr_report.md` 含三項：模型與參數／總段數與過濾統計與有效時長／50 段抽查 CER | **PASS** | §1 模型與參數（`:6-21`）；§2 漏斗與統計（`:23-81`，含有效總時長 51.5 分鐘）；§3 抽查（`:83-158`）。三項齊備且量化完整 |
| A4 | 品質抽查：**隨機抽 50 段人工校對**，記錄 Whisper 原始輸出的 CER | **PASS（依 PM 說明）＋可稽核性缺口** | `reports/cer_sample.csv` 確為 **50 列**、seed 42 可重現 ✓。**PM 2026-10-05 確認抽查由人工逐段進行**；44/50 列的 `reference` 與 `whisper_text` 逐字相同，其語意為「人工檢視後判定無誤，照抄原文」，**不是**未經檢視。⚠️ **但資料格式使「判定無誤」與「未檢視」不可區分**：`asr_report.md:89` 宣稱的約定是「人工留空＝判定無異常」，實測 **0 列留空**、44 列填原文；`tools/cer_report.py:64-67,80` 在資料層兩者同形（`_corrected` 皆 False、CER 皆 0）。**僅 6 列有實質修正**：`オート→王都`、`タレジャー→トレジャー`、`諸子→書庫`、`セイリ→セリ`、句首 `な,` 刪除、`ピー→フィー`。故「平均 CER 1.0%」的分母中有 44 項為**結構性為 0**，此數字代表「人工未發現錯誤」而非「零錯誤率」。見 §7-2 |
| A5 | 專有名詞（人名、地名、鍊金術用語）**已人工校正** | **PARTIAL** | `tools/glossary.py:45-52` 的 6 條 `CORRECTIONS` 為**純自動字串取代**（`:55-59` `text.replace`），命中 15 次／14 檔，`reports/correction_diff.csv` 14 列可逐筆查驗，壞形已全數歸零（`オート`/`ボース`/`セイリ`/`諸子`/`タレジャー` 各 0 筆）。**但 6 條中 5 條有 CER 抽樣錨點**（id13/16/32/38/50），**`ボース→ボオス` 這條命中 6 次（占 15 次中的 6 次）僅依據全庫字串掃描，無人工校對紀錄**。**0/491 筆有逐筆人工核對紀錄** |
| A6 | 新增 `requirements.txt`，列出所需套件與版本 | **PASS** | `requirements.txt` 41 行，5 個分區、14 個 pin 死版本（`faster-whisper==1.2.1`、`ctranslate2==4.8.2`、`torch==2.14.0+cu126` 等） |

---

## 4. Part 2 要求（issue 第 34–36 行）

| ID | 要求 | 判定 | 證據 |
|---|---|---|---|
| P2-1 | 基礎模型指定 GPT-SoVITS，**須在 PR 中記錄使用的版本與 commit hash** | **UNKNOWN** | 資料面已備妥：`reports/finetune_report.md` §10 記錄上游 commit `48b1a0169a28582a8984402f82cf438d3bfa6aca`、工作樹乾淨、無官方檔修改；§10 亦記錄本專案 5 個產出 commit。**但本分支 5 個 commit 尚未 push、PR 尚未建立**，issue 要求的「在 PR 中記錄」這一步尚未發生 |
| P2-2 | 訓練前自 `ryza_train.list` 保留 5%（約 24 段）作為測試集，**不得用於訓練** | **PASS** | `tools/split_train_test.py:279-318` 在寫檔前檢查 `train ∩ test = ∅`、`train ∪ test = 491`、`eval10 ⊆ test`，任一失敗 `sys.exit(1)`；`reports/split_report.md:49-50,56` 記錄 **11/11 門檻通過**。實測 `finetune_train.list` 467、`finetune_test.list` 24 |
| P2-3 | SoVITS 與 GPT 兩階段皆須 finetune，記錄 epoch、batch size 等參數 | **PASS** | S1 與 S2 各完成 16 epoch、exit 0；參數完整記錄於 `reports/finetune_report.md` §2.3 與 `docs/07_finetune.md` §11.3 |

---

## 5. Part 2 驗收要件（issue 第 39–55 行）

| ID | 要求 | 判定 | 證據 |
|---|---|---|---|
| P2-4 | 新增 `docs/07_finetune.md`：**從資料準備到推論的完整重現步驟** | **PASS** | 590 行，§0 章節導覽（工作流順序）＋ §1 環境／§8 資料準備／§9 切分／§10 模型準備／§2 預處理／§11 訓練／§12 推論／§13 已知偏差／§14 常見問題／§15 相關文件。2026-10-05 補完（原僅有環境與預處理兩章） |
| P2-5 | 提交訓練設定檔與訓練 log（**不含模型權重**） | **PASS** | `configs/s1_ryza.yaml`、`configs/s2_lora_ryza.json` 已 commit；log 摘要 `reports/logs/stage_d_train.md` 已 commit。`reports/logs/raw/`（受 `.gitignore` 忽略）、TensorBoard events、模型權重**皆未提交** |
| P2-6 | (a) 測試集 10 句：以原台詞文字生成，**與原音並列** | **PARTIAL** | 10 句已產出且檔名沿用原編號，`finetune_report.md` §5.1 有「對應原音」欄（`wav/00045.wav` 等）。**但沒有明確的「並列」產物**——沒有原音／生成的配對清單或 A/B 對照說明；主觀評語中多處提及「原句帶有一點興奮」顯示評者有對照原音，但該對照方式未被記錄 |
| P2-7 | (b) 全新台詞 10 句：不存在於遊戲中的日文句子，涵蓋平敘、疑問、興奮、悲傷語氣 | **PASS**（附條件） | 10 句已產出，涵蓋四類語氣（平敘 4／疑問 2／興奮 2／悲傷 2）。以 8–10 字特徵片語逐句 grep `data/split_manifest.csv`，**10 句全部 0 命中**。**附條件**：句子**設計意圖**涵蓋四類語氣，但生成**結果**未能正確传达（主觀試聽：興奮 2/2、悲傷 2/2 方向錯誤）。前者是 issue 要求的內容，後者是模型品質問題，兩者須分開看。詳見 `defect_triage.md` §P-2 |
| P2-8 | 聲線相似度：ECAPA pipeline 算生成音訊與 A_high 中心點的 cosine；**生成樣本平均 ≥ 測試集真實音訊平均的 90%** | **PASS** | 生成 20 句平均 **0.8152**；基線＝測試集真實音訊（`finetune_test.list` 24 筆＝0.8832／其子集 `finetune_eval10.list` 10 筆＝0.8908）。**3 種基線 × 3 種生成樣本選取共 9 格全部達標**（最嚴格一格 90.5%）。數值可重現性已驗證（生成檔重算差 0.000000）。⚠️ 但此指標**對語氣／情緒完全盲視**，成因是切分以文字標註分層，見 `defect_triage.md` §P-2 |
| P2-9 | 可懂度：Whisper 轉寫與輸入文字比對，**平均 CER ≤ 10%** | **PASS** | 逐句平均 **2.99%**、整體加權 2.62%。文字稿已提交 `reports/cer_generated.csv`，並以專案既有 `tools/cer_report.py` 獨立重算得 3.0%。⚠️ **但主觀試聽顯示此數字低估實際發音問題率**：`new_02`（`だい` 歪掉）、`new_04`（`しょうか→かい`）人耳可聞而 Whisper 轉寫正確（CER 0%） |
| P2-10 | 產出 `reports/finetune_report.md`，彙整數據並**附主觀試聽評語**（聲線／發音／語氣自然度，各列明顯缺陷） | **PASS** | 604 行。§6-3 收錄 PM 提供之主觀試聽評語（三項 ＋ 逐檔 ＋ 系統性現象 ＋ 個案）；§6-4 為主觀 × 量化交叉檢視；§6-5 列出評語本身的 7 項不足 |
| P2-11 | 訓練過程可穩定完成；若遇既有 GPU 不穩問題須記錄處理方式 | **PASS** | 發生 1 次顯存容量事件（`grad_ckpt=false` 時峰值 11958/12282 MiB，進程假死且**不拋 OOM**），已依指定順序取第一順位 `grad_ckpt=true` 修正並完成 16 epoch。全程**無** `illegal instruction`／`CUBLAS_STATUS_INTERNAL_ERROR`／CUDA context 遺失／驅動重置。記錄於 `finetune_report.md` §3、`logs/stage_d_train.md` §2.2、`09_pitfalls.md` PK-013 |

---

## 6. 交叉檢查：與證據矛盾的過期／無佐證宣稱

以下為 repo 內**明確與可驗證事實衝突**的敘述。**本稽核不修改它們**（屬歷史報告與 PM 決策範疇），僅列出待 PM 裁示。

| # | 位置 | 宣稱 | 與何者衝突 |
|---|---|---|---|
| 1 | `reports/asr_report.md:185` | 「top500 497 檔已**全數人工試聽確認為ライザ**」 | **PM 2026-10-05 已確認此為事實**（早期視聽與人工檢查皆由人工執行）。殘留問題僅為**可稽核性**：repo 內無逐筆對照表。**不再是「與證據矛盾」，而是「證據未入庫」** |
| 2 | `reports/asr_report.md:187` | 「Part 1 驗收要件**全數滿足**」 | 依上述複核，S1／S2／A2／A4 執行面均成立，故此斷言**成立** |
| 3 | `reports/issue_reply_part1.md:93`、`README.md:41` | 「全數經人工確認為萊莎｜完成」 | 同 #1，**成立**（可稽核性缺口另記於 §7-1） |
| 4 | `reports/asr_report.md:89` | 「人工留空 ＝ 判定無異常」 | `cer_sample.csv` 實測 **0 列留空**、44 列填原文。**填寫方式與約定不符**（非內容錯誤）：人工判定無誤後照填原文，與「留空」語意等價但格式不符，導致兩者不可區分 |
| 5 | `reports/asr_filter_report.md:3` | 「清單：`ryza_train.list`（497 段）」 | 該根目錄檔已於 commit `672e4e5` `git rm`；現行為 `data/ryza_train.list` 491 行。該檔亦已被 `docs/05_transcript_pipeline.md:296-297` 自承「已被取代，保留作為歷史紀錄」 |
| 6 | `docs/05_transcript_pipeline.md:50` | 列出 6 個 `v` 前綴檔「全數自動採納」 | 數量正確，但未揭露 `v00596.wav` 實源自 `D_flagged`（`svm_dec −0.4597`，自訂為「強烈疑似錯誤」層）且已進入訓練集 |
| 7 | `README.md`、`docs/08_tools.md:115-128` | 「`07_finetune.md`、`09_pitfalls.md` 與 5 個 tools 腳本尚未納入版控」 | `git ls-files` 顯示**全部已追蹤**（2026-10-05 複核） |
| 8 | `reports/asr_filter_report.md` 全文 | 被 `issue_reply_part1.md:20` 引為「排除原因」記錄 | 該檔為過期版本，且排除原因已被 `--prune-raw` 從 `asr_screening.json` 移除 |

---

## 7. 需 PM 裁示的事項

| # | 事項 | 建議 |
|---|---|---|
| 1 | **人工查核的可稽核性**（PM 已確認執行，缺的是紀錄） | 執行面成立，不需補做。建議補建「音檔編號 → 判定 → 試聽者 → 日期」對照表，或至少在 `docs/04_speaker_id.md:122` 把 `B_likely` 的狀態由待辦改為已完成，避免日後複核時被誤讀為未執行 |
| 2 | **A4「50 段人工校對」的填寫方式與自訂約定不符** | 內容正確（人工判定無誤後照填原文），但 `asr_report.md:89` 宣告的約定是「留空＝無異常」。建議二選一：(a) 補一欄 `reviewed: Y`；(b) 把約定改寫為「reference 與 whisper_text 相同＝已檢視且判定無誤」。前者可稽核性較佳 |
| 3 | **A5 `ボース→ボオス`（命中 6 次）無人工依據** | 人工確認這 6 筆，或在 `glossary.py:47` 註明是規則推導而非人工判定 |
| 4 | **`v00596.wav`（`D_flagged`，`svm_dec −0.4597`）在訓練集內** | 確認是否應剔除。剔除會使 491 → 490，需評估是否重跑下游；若不剔除，應在 `select_samples.py:8` 的 `v` 前綴規則加上「僅適用於 A/B/C 層，不適用於 `D_flagged`」的保護並記錄 |
| 5 | **R3 排除原因紀錄已被 `--prune-raw` 破壞** | 考慮在 `build_pos_label.py` 保留排除紀錄（例如另存 `asr_screening_excluded.json`），否則「記錄排除原因」這條要求在日後無法回溯 |
| 6 | **P2-6「與原音並列」無明確產物** | 產出一份 (a) 的原音／生成配對清單（含兩個路徑），或在報告中明確記載對照方式 |
| 7 | **P2-8 指標對語氣盲視** | 若語氣品質要納入驗收，需另建聲學情緒指標並以人工複核標註重新分層；目前兩個 gate 指標**結構上**無法偵測此類缺陷 |
| 8 | **P2-1 PR 尚未建立** | push 分支 → 開 PR → 於描述填入 `finetune_report.md` §10 的版本與 commit hash |

---

## 8. 判定為 PASS／UNKNOWN 所需的外部資訊

| 判定 | 缺什麼 |
|---|---|
| A2 人工確認 | `B_likely` 383 筆（或 top500 497 筆）的**逐筆**人工試聽紀錄：音檔編號、判定（ライザ／非ライザ）、試聽者、日期 |
| A4 人工校對 | 44 列「reference 與 whisper_text 相同」的**原始填寫證明**——例如留空原檔 + 後續填寫的 commit 差異，或人工填寫日誌。目前 `cer_sample.csv` 僅 2 個 commit（`41cf8cf` 建檔、`30f3c47` 出報告），看不出空白→填寫的過程 |
| A5 `ボース` 條 | 6 檔的逐筆人工確認紀錄（檔名、判定人、時間） |
| P2-1 | 分支 push 與 PR 建立後的 PR 描述內容 |

---

## 9. 相關文件

| 文件 | 關係 |
|---|---|
| [`finetune_report.md`](finetune_report.md) | Part 2 的完整驗收報告（本檔 P2-1～P2-11 的證據來源） |
| [`defect_triage.md`](defect_triage.md) | 主觀缺陷的歸因與排查；§P-2 說明評測盲視成因 |
| [`asr_report.md`](asr_report.md) | Part 1 的 ASR 報告（S1～A6 的證據來源；其 §185 為爭議斷言） |
| [`split_report.md`](split_report.md) | 切分驗收（11/11 通過） |
| [`cer_sample.csv`](cer_sample.csv) | Part 1 的 50 段抽查（A4 的證據） |
| [`cer_generated.csv`](cer_generated.csv) | Part 2 的 20 句文字稿（P2-9 的證據） |
| [`../docs/01_issue.md`](../docs/01_issue.md) | 被稽核的 issue 全文 |
| [`../docs/04_speaker_id.md`](../docs/04_speaker_id.md) | 說話者分層定義；與 `asr_report.md:185` 矛盾處的來源 |
| [`../docs/09_pitfalls.md`](../docs/09_pitfalls.md) | 踩坑紀錄（PK-001～PK-015） |
