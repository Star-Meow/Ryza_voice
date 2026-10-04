# 煙霧測試：ffmpeg 就位 ＋ 階段 A 端到端跑通

日期：2026-10-05
性質：冒煙測試（smoke test）——驗證環境依賴齊全、預處理 pipeline 可端到端跑通、產出符合預期。
範圍：`<GPT>` = `H:\git\GPT-SoVITS`，`<RYZA>` = `H:\git\Ryza_voice`

> 本文整併本次任務五個區域的完整記錄。Gate 回報（§15 格式）見 [G2_stage_a_report.md](G2_stage_a_report.md)，環境與流程細節見 [../docs/07_finetune.md](../docs/07_finetune.md)。

---

## 區域 1：環境修正（ffmpeg 缺失）

### 問題
首次跑階段 A 第 2 步時，`load_audio`（`<GPT>\tools\my_utils.py:16-37`）呼叫 ffmpeg CLI，本機無 `ffmpeg.exe`，467 個 wav 全部 `FileNotFoundError: [WinError 2]`，log 中 934 次 traceback（467×2）。

**關鍵陷阱**：`2-get-hubert-wav32k.py:124-125` 的裸 `except` 把每行錯誤吞掉只印 traceback，**結束碼仍是 0**。若只看 exit code 會誤判成功。這是本專案「驗收必須以產出檔案數量與內容為準」原則的直接原因。

### 處置（方案 A，PM 放行）
- 下載 gyan.dev 靜態版 ffmpeg 9.0.2（release-essentials）：`https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip`
- zip SHA256：`60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba`
- 放置 `<GPT>\.venv\Scripts\ffmpeg.exe`（105,423,872 B）與 `ffprobe.exe`（105,221,120 B），不改名、不做 .cmd shim
- 驅動程式 `run_preprocess.py` 把 `.venv\Scripts` 加進子行程 PATH（`env["PATH"]`）

### 本 session 實測（2026-10-05）
```bash
source /h/git/GPT-SoVITS/.venv/Scripts/activate
python -c "import shutil; print(shutil.which('ffmpeg')); print(shutil.which('ffprobe'))"
```
結果：
```
ffmpeg  : H:\git\GPT-SoVITS\.venv\Scripts\ffmpeg.EXE
ffprobe : H:\git\GPT-SoVITS\.venv\Scripts\ffprobe.EXE
ffmpeg version 9.0.2-essentials_build-www.gyan.dev
ffprobe version 9.0.2-essentials_build-www.gyan.dev
```
**判定：通過。** 兩路徑皆指向 `.venv\Scripts`，版本正常印出。

二進位完整性：`.venv\Scripts` 與 `<RYZA>\tools\_ffmpeg_tmp\ffmpeg-9.0.2-essentials_build\bin\` 兩兩 md5 完全相同（ffmpeg `acd08751bc4120a0c10c2da133ae1ed7`、ffprobe `1f7afec013dacbc8b88302a8fdbdd684`），均為 PE32+ x86-64 真二進位（非 shim）。

---

## 區域 2：階段 A 重跑

### 首次失敗的殘留清理
刪除失敗產出（空目錄 `4-cnhubert\`、`5-wav32k\` 與 0 byte 的 `6-name2semantic-0.tsv`）。**首次失敗完全未寫出任何檔案**——`load_audio` 在寫檔前就丟例外，故無「部分損壞」可能。重跑後 467 個 `.pt` 與 wav 的 mtime 全部晚於 ffmpeg 就位時間（00:51）。

### 各步驟結束碼與耗時

驅動：`tools/run_preprocess.py`（§4.3 環境變數明確設定、stdout/stderr 分檔、非 0 即停）。

| exp | 步驟 | 腳本 | 起訖 | 耗時 | 結束碼 | stderr |
|---|---|---|---|---|---|---|
| train | 1 | 1-get-text.py | 00:17:27→00:17:34 | 7.3 s | 0 | 僅 pkg_resources 棄用警告 |
| train | 2 | 2-get-hubert-wav32k.py | 00:54:11→00:54:57 | 46.1 s | 0 | 0 byte |
| train | 3 | 3-get-semantic.py | 00:54:57→00:55:05 | 8.1 s | 0 | 0 byte |
| eval10 | 1 | 1-get-text.py | 01:00:38→01:00:45 | 6.8 s | 0 | 僅 pkg_resources 棄用警告 |
| eval10 | 2 | 2-get-hubert-wav32k.py | 01:00:45→01:00:53 | 8.0 s | 0 | 0 byte |
| eval10 | 3 | 3-get-semantic.py | 01:00:53→01:01:01 | 7.5 s | 0 | 0 byte |

**判定：通過。** 六步驟全數 exit 0，stderr 無實質錯誤（pkg_resources 棄用警告無害）。原始 log：`<RYZA>\reports\logs\raw\{train,eval10}_step{1,2,3}.{out,err}`。

**注意**：exit code 0 不可單獨作為成功依據（見區域 1 的陷阱）。本表搭配區域 3 的檔案級驗收才成立。

---

## 區域 3：產出驗收（PA-1～PA-11）

驗證腳本 `tools/stage_a_validate.py`（唯讀，只寫 json）。本 session 以修正後版本重跑，結果與 [G2_stage_a_report.md](G2_stage_a_report.md) 一致。

| PA | 項目 | train（467） | eval10（10） |
|---|---|---|---|
| PA-1 | 2-name2text 行數／欄位 | ✅ 467 行、4 欄 | ✅ 10 行、4 欄 |
| PA-2 | 4-cnhubert `.pt` 數 | ✅ 467（0 空檔，min 185,550 B） | ✅ 10 |
| PA-3 | 5-wav32k 數＋取樣率 | ✅ 467、32000 Hz | ✅ 10、32000 Hz |
| PA-4 | 6-name2semantic 行數＋表頭 | ✅ 468 行、LF、0 CR | ✅ 11 行 |
| PA-5 | 五處鍵集合與 list 全等 | ✅ 無缺漏 | ✅ 無缺漏 |
| PA-6 | HuBERT 無 NaN／形狀 | ✅ 抽 20 檔全 `[1,768,N]` | ✅ 全檢 10 檔 |
| PA-7 | 5-wav32k 總大小 | 179.9 MB | 3.4 MB |
| PA-8 | 3-bert 存在且為空 | ✅（ja 不產 BERT） | ✅ |
| PA-9 | 非 # UNK | ✅ 總數 0、影響行 0 | ✅ 0、0 |
| PA-10 | 音素與 G2P 一致 | ✅ 0 mismatch | ✅ 0 mismatch |
| PA-11 | 版本推斷／載入 | v3（無害）、`<All keys matched successfully>` | 同 |

**判定：通過。** PA-1～PA-11 兩 exp 全數 pass。PA-12／PA-13（git 狀態）見區域 4。

### validator 的兩個 false-failure bug（已修）

| bug | 根因 | 影響 | 修正 |
|---|---|---|---|
| PA-10 5/5 mismatch | 逐行讀 list 時未去除行尾換行，`text_normalize`→g2p 把結尾 `\n` 多轉一個 `.` | 恆判 fail，**非資料問題**（手動以官方讀法重跑全量：train 467/467、eval10 10/10 一致） | 預載 list 時 `read().strip("\n").split("\n")` |
| PA-9 affected_lines=451 | 門檻把 `#`（韻律邊界）造成的 UNK 行也算進「影響行數」 | train 誤判 fail（451 > 10） | `affected` 只收集含**非 #** 來源的行；`unk_sources` 統計不變 |

兩者皆為驗證腳本 bug，**資料本身正確**（T1 與 PA-9 再確認：非 # UNK 本就是 0）。

### json 與腳本不同步（本 session 修正）

`stage_a_pa.json`（01:05 舊版）記錄 PA-9／PA-10 為 fail，但 `stage_a_validate.py` 之後已修正卻**未重跑**，導致 json 落後於 G2 報告。本 session 以修正版重跑，json 已刷新至與報告一致（PA-9／PA-10 皆 pass）。

---

## 區域 4：收尾（git 狀態與文件）

### `<GPT>`（無分支，commit `48b1a01`）
- 工作樹**完全乾淨**：`git status --short` 空、`git diff` 空
- `logs\ryza_v4_lora`（399 MB）、`logs\ryza_v4_lora_eval10`（7.6 MB）、`.venv\Scripts\{ffmpeg,ffprobe}.exe` 全被 `.gitignore` 忽略
- **`?? =0.4.1` 已不存在**：G1 報告記載該檔「始終存在」，PA-12 亦期望「仍只有 =0.4.1」，但現已消失。該檔從未入版控（`git log --all -- "=0.4.1"` 無結果），**無法還原、無法查證刪除時間**。PA-12 據實改記「工作樹全乾淨」——結果優於預期，但偏離計畫文字

### `<RYZA>`（`feat/tts-v4-lora-prep`，HEAD `0b7c743`）
既有追蹤檔全未修改。untracked 項目：

| 項目 | 處置 | 理由 |
|---|---|---|
| `tools/run_preprocess.py` | 保留 | 階段 A 三步驅動 |
| `tools/stage_a_validate.py` | 保留 | PA 驗證（已修正） |
| `reports/logs/` | 保留 | `raw\` 已被內部 `.gitignore` 忽略，只整理後的 .md 會提交 |
| `tools/_ffmpeg_tmp/` | **已不存在** | 見下方註 |
| `docs/07_finetune.md` | **本 session 新建** | 記錄 ffmpeg 版本／來源／位置＋易錯點（解除 `03-01_dataset.md:9`、`issue.md:39` 引用缺口） |
| `reports/smoke_test_stage_a.md` | **本 session 新建** | 本文件 |

**註：`tools/_ffmpeg_tmp/` 已被刪除。** 對話開始時 git status 仍列 `?? tools/_ffmpeg_tmp/`（內含 114 MB `ffmpeg.zip` 與解壓目錄），本 session 收尾時發現該目錄已不存在、無任何 .gitignore 規則涵蓋它，**刪除時間與執行者無法查證**。影響評估：

- ffmpeg 功能**不受影響**：`.venv\Scripts\{ffmpeg,ffprobe}.exe` 仍在，md5 早已驗證，venv 啟動態實測通過（區域 1）
- 代價是**失去可重現的 zip 實體**——日後若需在新環境重放，只能憑 `docs/07_finetune.md` 記錄的 gyan.dev URL 與 SHA256 重新下載（一次連網限定，需重新申請）

**未 commit**——待 PM 放行後再依 §12.5 處理。

### 產出目錄結構（供階段 B/C 使用）

```
<GPT>\logs\ryza_v4_lora\          # train 467
  2-name2text.txt      (467 行, CRLF)
  2-name2text-0.txt    (原檔保留, md5 相同)
  3-bert\              (空, ja 不產 BERT)
  4-cnhubert\          (467 個 .pt)
  5-wav32k\            (467 個 wav, 179.9 MB, 32 kHz)
  6-name2semantic.tsv  (468 行含表頭, LF)
  6-name2semantic-0.tsv (原檔保留, 內容等價)
<GPT>\logs\ryza_v4_lora_eval10\   # eval10 10，同結構
```

---

## 區域 5：不符事項／無法判定／風險

### 與計畫不符（含本次新增）
1. ffmpeg 原本不存在，經 PM 放行後安裝（方案 A）；首次失敗 exit code 恆 0 的陷阱（裸 `except`）
2. PA-9／PA-10 首次誤判為 validator bug，非資料問題（已修）
3. **（新）`stage_a_pa.json` 落後於修正後的腳本與 G2 報告**——本 session 重跑已消除差異
4. `6-name2semantic-0.tsv` 原檔無結尾換行（`wc -l` 466 非 467），合併已正規化
5. `2-name2text.txt` 為 CRLF（官方 Windows 行為，不影響 S1 讀取）
6. `2-name2text-0.txt`／`6-name2semantic-0.tsv` 合併後**未刪除**（官方 webui 會 `os.remove`），內容已驗證等價
7. **（新）`=0.4.1` 檔案已消失**，與 G1 記載及 PA-12 期望牴觸（見區域 4）
8. `train_summary.json` 曾被 `open(..., "w")` 覆寫而遺失 step1 記錄；step1 的 7.3 s 由補測取得
9. **（新）`tools/_ffmpeg_tmp/` 已被刪除**——對話開始時仍存在，收尾時已消失，刪除時間與執行者無法查證。zip 消失代表「重現下載」需重新連網；ffmpeg 功能不受影響（見區域 4 註）

### 無法判定
- `=0.4.1` 何時由誰刪除（從未入版控，無軌跡）
- `tools/_ffmpeg_tmp/` 何時由誰刪除（本 session 期間發生，無軌跡）
- 今早執行輪次是否為本指令的直接結果（時間軸完全吻合但無文字記錄）

### 風險與意外發現
1. **官方腳本裸 `except` 吞錯**（3 處）：階段 C 冒煙測試不可只看結束碼，必須以產出數量與內容為準
2. **ffmpeg 是隱性系統依賴**：`load_audio` 只走 ffmpeg CLI，無 Python fallback；已記入 `docs/07_finetune.md`
3. **`3-bert\` 在 ja 時確實為空**，但第 1 步仍載入 RoBERTa——T0 轉檔必要
4. **第 2 步的 NaN retry 路徑未觸發**：`nan_fails` 為空，全程維持 fp16

---

## 結論

**冒煙測試通過。** ffmpeg 環境依賴已就位並實測確認；階段 A 六步驟全部跑通；467 + 10 筆產出經 PA-1～PA-13 檔案級驗收全數通過；兩 repo 版控狀態符合預期（`<GPT>` 全乾淨、`<RYZA>` 僅新增未追蹤檔，無任何追蹤檔被修改）。

**停在 G2，等待 PM 放行後才進階段 B（訓練配置與啟動腳本）。**
