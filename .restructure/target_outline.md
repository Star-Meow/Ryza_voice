# 目標結構框架（定案版）

各文件應該有什麼、不應該有什麼，後續修改以本文件為準；`inventory.csv` 的去向也依本文件更新。
本目錄只存在於 `docs/repo-boundary` branch，合併前移除。

## 共用原則

1. **不沿用 issue 的框架**：「Part 1／Part 2」、「已完成／進行中」、「驗收要件」、「對應 issue 要求」只留在 issue 與 PR。repo 內一律以模塊命名。
2. **每個主題只有一個來源**：其他地方只放一行摘要加連結。
3. **有慣例位置的，依慣例**：工具目錄放 `tools/README.md`，資料說明放 `data/README.md`。
4. **不保留專案管理產物**：issue 原文與回覆、PR 描述、合規稽核、階段回報、冒煙測試的執行結果，都不進 repo。issue 與 PR 一律引用，不複製。
5. **一律刪除的字樣**：「計畫 §x」（計畫不在 repo）、「本 session」、「等待放行」、「停止點」、「尚未 commit」、「階段 A～D」（改用模塊名稱）、「PM 裁示」（改為中性敘述：「決定…，理由…」）。

| 類型 | 位置 | 放什麼 | 不放什麼 |
|---|---|---|---|
| 入口 | `README.md` | 是什麼、怎麼裝、怎麼跑、去哪看 | 方法細節、數據、決策理由、進度 |
| 概述 | `docs/01_overview.md` | 目標、範圍、模塊架構、評估標準 | 實作細節 |
| 設計 | `docs/` | 為什麼這樣做 | 執行步驟、結果 |
| 流程 | `docs/` | 怎麼做 | 當次執行紀錄、驗收口吻 |
| 疑難排解 | `docs/08_troubleshooting.md` | 現象、原因、處置 | 過程敘事 |
| 結果 | `reports/` | 設定、數據、分析、限制 | 重現步驟、停止點、branch 狀態 |
| 目錄說明 | `data/`、`tools/`、`samples/` 的 README | 目錄內有什麼、怎麼產生 | 評估結論 |

## 文件總表

```
README.md                         入口
LICENSE
data/README.md                    新增：資料檔說明（dataset card）
tools/README.md                   由 docs/08_tools.md 移來：工具目錄
samples/README.md                 生成樣本說明
docs/01_overview.md               新增：目標、範圍、模塊架構、評估標準
docs/02_unpack.md                 ← 03_unpack
docs/03_speaker_id.md             ← 04_speaker_id
docs/04_transcript_pipeline.md    ← 05_transcript_pipeline
docs/05_dataset_split.md          ← 06_dataset ＋ 07 §9
docs/06_model_selection.md        ← 02_model_selection ＋ reports/audio_quality
docs/07_finetune.md               TTS 微調、推論、評估（含冒煙測試的執行方式）
docs/08_troubleshooting.md        ← 09_pitfalls
reports/asr_report.md             ＋ correction_diff 併入附錄
reports/split_report.md
reports/finetune_report.md
reports/defect_triage.md
reports/logs/stage_d_train.md
```

**刪除**：
- `reports/01_env_model_prep`、`G1`～`G4`、`smoke_test_stage_a`、`smoke_test_stage_c`：技術內容先移到 07 或 08，再刪除。
- `reports/asr_filter_report`、`reports/audio_quality.md`（併入 06）、`reports/correction_diff.md`（併入 asr_report；CSV 保留）。

## 模塊

| 模塊 | 做什麼 | 流程文件 | 結果 |
|---|---|---|---|
| 語音解包 | KTSR 音聲庫 → WAV 與索引 | 02 | — |
| 說話者篩選 | 在沒有名稱的主庫中找出ライザ | 03 | `ryza_main/` 分級 |
| 轉譯資料集 | ASR → 過濾 → 去重 → 校正 → 訓練清單 | 04 | asr_report |
| 資料切分 | 切出 train／test／eval10 | 05 | split_report |
| TTS 微調 | 選型、GPT-SoVITS v4 LoRA 預處理、訓練、推論 | 06（選型）、07 | finetune_report、logs/ |
| 評估 | 聲線相似度、可懂度 CER、主觀試聽 | 07 | finetune_report、defect_triage |

---

## 各檔框架

### README.md
```
# Ryza Voice           一句話簡介；聲明不含遊戲音訊與權重
## 模塊                 表格：模塊｜做什麼｜主要工具｜文件
## 安裝                 兩個環境、ffmpeg、遊戲資料
## 使用                 各模塊入口指令；完整順序連到 07 §0
## 目錄結構
## 文件                 docs／reports 各一張表
## 樣本與結果           只放連結
## 授權
```

### docs/01_overview.md
```
# 專案概述
## 1. 目標
## 2. 範圍與限制          資料唯讀、版權、硬體、語言
## 3. 模塊架構            資料流（輸入 → 產出），連到 02～07
## 4. 評估標準            門檻表；需求出處：issue #1 ＋ 原文 permalink（blob/262adda/docs/01_issue.md）
## 5. 實際執行與原需求的差異   每項一行，連到細節
```

### docs/02～04（解包、篩選、轉譯）
- 標題去掉「Part 1 ①②③」，章節結構維持不變。
- 03：刪除「尚未納入版控」；TF32 改為連到 08。
- 04：過濾規則標題改為「必要／附加」；刪除 §10「歷史執行紀錄」；1.5 已在 Step 1 改寫。

### docs/05_dataset_split.md
```
# 資料切分
## 1. 目標            467／24／10
## 2. 分層抽樣設計    取自 07 §9.1，含 class 欄位的三個性質
## 3. 參考音訊        取自 07 §9.3：切分選中 00042，之後被覆蓋為 03536
## 4. 產出與驗證      取自 07 §9.2，含 11 項驗證
## 5. 執行指令        修正 --audio-root
```
刪除：「二、模型選型」（改為連到 06）。

### docs/06_model_selection.md
- 結構維持「背景、一～六」。
- 刪除「七、待驗證項」。
- 新增「七、實測後的修正」與「附錄 A：音訊品質實測」。

### docs/07_finetune.md
```
# TTS 微調與推論
## 0. 全流程總覽        步驟｜文件｜產出（刪除 issue 欄）
## 1. 環境需求          ＋Windows 依賴偏差（01_env）
## 2. 預處理            （現 §2＋§3 的完成條件）
## 4. 易錯點
## 6. 推論端隱性依賴
## 7. 顯存
## 10. 模型準備         ＋權重表（01_env）、T0 SHA256（G1）
## 11. 訓練             ＋冒煙測試（執行方式與通過條件）、設定欄位依據（G3）、batch 推導
## 12. 推論與評估
## 13. 已知偏差
## 14. 常見問題         只連到 08
## 15. 相關文件
```
- 章節編號維持現狀，避免大量錨點失效。
- 刪除 §3 的敘事（只保留完成條件）、§5、§8（改為連到 02～04）、§9（移到 05）。

### docs/08_troubleshooting.md
- 結構：欄位定義／追加規則／坑記錄（PK-001～PK-015）／GPU 不穩專區。
- 刪除 §5～§7。
- PK-015 移到最後；修正被 `\r` 截斷的路徑；PM 字樣改中性。

### tools/README.md
- 依模塊分組，另加「測試與驗證」一組（冒煙、selftest、validate）。
- 涵蓋全部 43 支腳本。

### data/README.md
- 每個檔案的格式、筆數、產生工具與流程文件連結。
- 註明哪些檔案不進版控。

### reports
- `asr_report`：檔頭加 CER 定義；刪除驗收宣告；`correction_diff` 併入附錄。
- `split_report`：參考音訊加註後續覆蓋；刪除「下一步／尚未訓練」。
- `finetune_report`：刪除回報頭、「與計畫是否相符」欄、§9 重現步驟（改為連到 07）、§11 停止點；PM 字樣改中性；修正 ボース 與路徑。
- `defect_triage`：「我」改為中性敘述；刪除 PM 授權字樣。
- `logs/stage_d_train`：標題去掉「階段 D」字樣；刪除「依計畫」。
- `samples/README`：刪除「供 reviewer」；評估改為連結；修正「1 epoch」。
