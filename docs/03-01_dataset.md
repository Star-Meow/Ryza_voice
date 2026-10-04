# Part 2：GPT-SoVITS 微調

## 一、定位與目標

Part 1 已完成語音解包、說話者篩選與 ASR 轉譯，產出 `data/ryza_train.list`（491 段日文語音，約 51.5 分鐘，48kHz mono，格式為 `音檔路徑|ryza|ja|文字`）。

Part 2 以此為基礎，使用 GPT-SoVITS 微調萊莎聲線 TTS，並依 Issue 要求完成訓練、生成與驗收。環境為 RTX 4070 SUPER 12GB。

本文涵蓋**從選型到資料切分**的內容。訓練與驗收的完整流程見 [`docs/finetune.md`](finetune.md)。

---

## 二、模型選型

### 2.1 版本篩選

GPT-SoVITS 主分支可選版本為 v1 / v2 / v4 / v2Pro / v2ProPlus。篩選依**版本沿革與官方定位**逐步收斂，而非以 benchmark 分數反向淘汰。

| 版本 | 淘汰理由 |
|---|---|
| v1 | 最早版本，架構與訓練資源均已過時 |
| v2 | 被 v2Pro 直接取代；同硬體成本、更好效果 |
| v3 | 不在主分支 UI 選項中，官方將 v4 定位為其平替；原生只輸出 24kHz |
| v2ProPlus | 非淘汰，列為備選 |

保留的主要候選為 **v2Pro**（VQ 路線）與 **v4**（CFM / diffusion 路線）。

### 2.2 v2Pro 與 v4 的關鍵差異

兩者在 12GB GPU 下的可行性有本質差異：

| 項目 | v2Pro | v4 |
|---|---|---|
| 微調方式 | 目前主分支未提供 LoRA 入口 | 全量微調或 LoRA |
| 官方訓練配置顯存門檻 | 無明確 GB 數字 | 14GB（全量）/ 12GB（梯度檢查點）/ 8GB（LoRA） |
| 12GB 可行性 | VRAM 餘裕較小 | LoRA 路徑有明確餘裕 |

目前主分支 WebUI 的訓練流程中，v1/v2/v2Pro/v2ProPlus 使用 `s2_train.py`，未提供 LoRA 訓練入口；v3/v4 則使用 `s2_train_v3_lora.py`。

### 2.3 選型結論：v4 LoRA

主要理由：

1. **顯存可行性**：官方對 v3 的 LoRA 訓練記載約 8GB；v4 與 v3 共用 `s2_train_v3_lora.py`，預期具有相近的低顯存特性。
2. **取樣率匹配**：v4 原生輸出 48kHz，與本專案 491 段資料一致；v2Pro 原生 32kHz，需額外上採樣。
3. **相似度無明顯差距**：官方 zero-shot SIM 中，v4（0.735）與 v2ProPlus（0.737）接近，僅依 benchmark 無法形成明顯差距。
4. **避免 v3 已知問題**：v4 修正 v3 非整數倍上採樣可能造成的 metallic artifacts；本資料集 51.5 分鐘落在 v3 官方警示的 <100h 區間。

**備選**：v2ProPlus。若 v4 LoRA 實測無法滿足驗收條件，再評估此路徑；屆時需先實測其實際 VRAM 使用量，並以相同資料切分與驗收流程比較。

完整選型論證見 [`docs/issue_part2.md`](issue_part2.md)。

---

## 三、資料切分

### 3.1 切分原則

Issue 要求訓練前保留 5%（約 24 段）作為測試集，不得用於訓練。為避免純隨機抽樣漏掉少數情緒類別，採用**分層隨機抽樣**，以句尾標點作為情緒代理：

| 分層 | 判定規則（依序檢查） | 測試集配額 |
|---|---|---|
| Q 疑問 | 結尾為 `?` 或 `？` | 4 |
| EXCL 興奮／感嘆 | 結尾為 `!` 或 `！` | 2 |
| ELL 猶豫／餘韻／悲傷 | 結尾為 `…`、`...` 或 `・・・` | 3 |
| DECL 平敘 | 其餘（含 `。`、語尾助詞結尾） | 15 |

**稀有層下限**為 2 段：避免樣本數過少無法判斷該語氣表現。其餘嚴格按比例分配。

**種子**固定為 42，與 Part 1 的 CER 抽查一致，全專案同一慣例。

### 3.2 切分後的產出

| 檔案 | 行數 | 用途 |
|---|---|---|
| `data/finetune_train.list` | 467 | WebUI 訓練唯一輸入 |
| `data/finetune_test.list` | 24 | 測試集，嚴禁進入任何訓練步驟 |
| `data/finetune_eval10.list` | 10 | 驗收 3(a) 生成對象，自測試集分層抽出 |

`data/ryza_train.list`（491 行）為 Part 1 交付物，保持不動。

`finetune_eval10.list` 的 10 句自 24 段再分層抽取（同 seed 42）：Q 2、EXCL 1、ELL 1、DECL 6。

### 3.3 切分驗證

切分工具為 [`tools/split_train_test.py`](../tools/split_train_test.py)，路徑由參數傳入，不寫死本機絕對路徑。內建五項驗證，全部通過才寫檔：

| 檢查 | 條件 |
|---|---|
| 無重疊 | train ∩ test == ∅ |
| 聯集完整 | len(train) + len(test) == 491 |
| 路徑有效 | 每個路徑 `os.path.exists` 為真 |
| 格式一致 | 4 欄、speaker==ryza、lang==ja、文字非空 |
| 無重複 | 三份清單內路徑皆唯一 |

任一失敗即 abort 並印出第一個違規行。

切分統計與驗證結果見 [`reports/split_report.md`](../reports/split_report.md)。

### 3.4 執行指令

```bash
.venv/Scripts/python.exe tools/split_train_test.py \
  --input data/ryza_train.list \
  --train-out data/finetune_train.list \
  --test-out data/finetune_test.list \
  --eval-out data/finetune_eval10.list \
  --report reports/split_report.md \
  --seed 42 \
  --audio-root wav
```
