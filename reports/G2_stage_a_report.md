# G2 回報：階段 A 預處理驗證（訓練集 467 ＋ eval10 10）

日期：2026-10-05
回報依據：計畫 §3 Gate G2、§9 階段 A、§15 回報格式
分支：`feat/tts-v4-lora-prep`（`<RYZA>`）；`<GPT>` 在 `main` 分支（commit `48b1a01`）、無修改
工作目錄：CWD = `H:\git\GPT-SoVITS`（logs 為相對路徑），Python = `.venv\Scripts\python.exe`

---

## 1. 結果總表

| ID | 結論 | 證據等級 | 證據 | 與計畫是否相符 |
|---|---|---|---|---|
| PA-1 | 2-name2text.txt 行數與欄位正確 | 【實測】 | train 467 行、eval10 10 行；每行恰 4 欄（tab 分隔），bad_field_rows=[] | 相符 |
| PA-2 | 4-cnhubert \*.pt 數量正確 | 【實測】 | train 467、eval10 10 | 相符 |
| PA-3 | 5-wav32k 檔案數與取樣率正確 | 【實測】 | train 467、eval10 10；抽樣取樣率皆 32000 | 相符 |
| PA-4 | 6-name2semantic.tsv 行數正確（含表頭、LF） | 【實測】 | train 468 行、eval10 11 行；表頭 `item_name\tsemantic_audio`；CR bytes = 0 | 相符 |
| PA-5 | 四處 key 集合與 list 檔名集合完全相等 | 【實測】 | train: list/name2text/hubert/wav32k/semantic 皆 467，集合相等，only_in_list=[]、missing_from_list=[]；eval10 皆 10 | 相符 |
| PA-6 | HuBERT 特徵無 NaN/Inf、形狀 768×frames | 【實測】 | train 抽 20 檔（含 07935→[1,768,120]、03166→[1,768,601]）；eval10 全檢 10 檔；nan_inf=[]、bad_shape=[] | 相符 |
| PA-7 | 5-wav32k 總大小已記錄 | 【實測】 | train 179.9 MB；eval10 3.4 MB | 相符 |
| PA-8 | 3-bert 存在且為空 | 【實測】 | 兩 exp 皆存在、entries=[]（ja 不產 BERT 特徵，符合 §9.1） | 相符 |
| PA-9 | UNK 檢查通過 | 【實測】 | train 非 # 來源 0（`#` 1776 次）；eval10 非 # 來源 0（`#` 35 次）；門檻只計非 # 來源（總數 ≤20、影響行數 ≤10） | 相符 |
| PA-10 | 音素序列與 T1 的 G2P 輸出一致 | 【實測】 | 兩 exp 各抽 5 行（seed 固定），mismatches=[] | 相符 |
| PA-11 | 第 3 步版本推斷為 v3（無害），載入正常 | 【實測】 | s2Gv4.pth = 769,025,545 bytes > 700 MB → 推斷 v3；log 顯示 `<All keys matched successfully>` | 相符（§9.1 已預期） |
| PA-12 | `<GPT>` 工作樹乾淨 | 【實測】 | `git status --short` 空；`git diff --stat` 空；logs/、.venv\Scripts\ffmpeg.exe 均被 .gitignore 忽略 | 相符（優於預期：`<GPT>` 現已完全乾淨，`=0.4.1` 已刪） |
| PA-13 | 兩 repo 無被修改的追蹤檔 | 【實測】 | `<GPT>` 無任何追蹤檔變動；`<RYZA>` 僅新增未追蹤檔（見 §5） | 相符 |

**結論：PA-1～PA-13 全數通過，無停止條件觸發。**

---

## 2. 各步驟結束碼與耗時

驅動程式 `tools/run_preprocess.py`（§4.3 全域環境變數明確設定、stdout/stderr 分檔、記錄結束碼；非 0 即停）。

| exp | 步驟 | 腳本 | 起訖 | 耗時 | 結束碼 | stderr |
|---|---|---|---|---|---|---|
| train | 1 | 1-get-text.py | 00:17:27→00:17:34 | 7.3 s | 0 | 僅 pkg_resources 棄用警告 |
| train | 2 | 2-get-hubert-wav32k.py | 00:54:11→00:54:57 | 46.1 s | 0 | 0 byte |
| train | 3 | 3-get-semantic.py | 00:54:57→00:55:05 | 8.1 s | 0 | 0 byte |
| eval10 | 1 | 1-get-text.py | 01:00:38→01:00:45 | 6.8 s | 0 | 僅 pkg_resources 棄用警告 |
| eval10 | 2 | 2-get-hubert-wav32k.py | 01:00:45→01:00:53 | 8.0 s | 0 | 0 byte |
| eval10 | 3 | 3-get-semantic.py | 01:00:53→01:01:01 | 7.5 s | 0 | 0 byte |

原始 log：`<RYZA>\reports\logs\raw\{train,eval10}_step{1,2,3}.{out,err}`（已被 `.gitignore` 忽略，不提交）。

---

## 3. 與計畫不符或需 PM 留意的事項

1. **ffmpeg 原本不存在，經 PM 放行後安裝**（方案 A）：
   - 事實【實測】：首次跑第 2 步時 `load_audio`（`tools/my_utils.py:24-28`）呼叫 ffmpeg CLI，本機無 `ffmpeg.exe`，467 個 wav 全部 `FileNotFoundError: [WinError 2]`（log 中 934 次 traceback = 467×2）。
   - **腳本的 `except` 把每行錯誤吞掉只印 traceback（`2-get-hubert-wav32k.py:124-125`），結束碼仍是 0**。這是「結束碼不可靠」的鐵證——若只看 exit code 會以為成功。本計畫的 PA-2/PA-3 檔案計數檢查正是抓到這個問題的關閘。
   - 處置（已執行）：下載 gyan.dev 靜態版 ffmpeg 9.0.2（release-essentials），將 `ffmpeg.exe`、`ffprobe.exe` 放入 `<GPT>\.venv\Scripts\`；驅動程式把該目錄加進子行程 PATH（`run_preprocess.py` 的 env["PATH"]）。
   - 殘留清理：刪除失敗產出（空目錄 `4-cnhubert\`、`5-wav32k\` 與 0 byte 的 `6-name2semantic-0.tsv`）後完整重跑第 2、3 步。**第一次失敗完全未寫出任何檔案**（`load_audio` 在寫檔前就丟例外），所以沒有「部分損壞」的可能；467 個 `.pt` 與 wav 的 mtime 全部晚於 ffmpeg 就位時間（00:51:57）。
   - 對 git 的影響：`.venv\` 在 `<GPT>` 的 .gitignore 內，`git status` 完全乾淨。
2. **PA-9 門檻的驗證實作曾被誤解**：初次驗證把 `#` 造成的 UNK 行數（451）也計入「影響行數」門檻而誤判 FAIL。修正後門檻只計非 # 來源（計畫 §9.4 明定），通過。**這是驗證腳本的 bug，不是資料問題**——資料的非 # UNK 本就是 0（T1 已證明，PA-9 再確認）。
3. **PA-10 首次誤判**：驗證腳本逐行讀 list 時未去除行尾換行，導致重算的 G2P 與 name2text 不一致（5/5 行「mismatch」）。修正（比照 T1 做 `strip("\n")`）後 5/5 一致。同樣是驗證腳本 bug，非資料問題。
4. **`6-name2semantic-0.tsv` 原始檔無結尾換行**：腳本以 `"\n".join(lines1)` 寫出（`:117-118`），最後一行後無 `\n`，故 `wc -l` 顯示 466 而非 467。合併時已正規化（補 LF 結尾），最終 `6-name2semantic.tsv` 為 468 行（含表頭）。
5. **`2-name2text.txt` 為 CRLF**：官方腳本在 Windows 寫出 CRLF（467 行皆含 `\r`）。合併採複製保留原樣（計畫 §9.3 只要求 6- 表頭與 LF，未要求 2- 換行）。S1 訓練以 Python 讀取，CRLF 不影響解析；若後續發現問題再回報。
6. **eval10 的第 3 步耗時異常短（7.5 s）**：因 only 10 筆，屬正常。

---

## 4. 無法判定的項目

- **PA-6 的 eval10 形狀下限**：eval10 全檢通過，但樣本時長分布未逐一記錄（未列入 PA 門檻）；train 的抽樣含最短 07935 與最長 03166，形狀分別為 [1,768,120] 與 [1,768,601]，與時長 2.412 s／12.042 s × 50 fps 一致。
- **`6-name2semantic` 的 token ID 範圍**：本階段未檢查（S2 詞表上限 732 為 embedding 維度，semantic token 由 s2Gv4 的 VQ 產生，ID 空間由 codebook 決定，非 §2.1 的詞表限制）。若 PM 需要可補查。

---

## 5. 目前工作樹與產出

### `<GPT>`（`main` 分支，commit `48b1a01`）
- 工作樹**完全乾淨**：`git status` 空、`git diff` 空。`logs\ryza_v4_lora`（399 MB）、`logs\ryza_v4_lora_eval10`（7.6 MB）、`.venv\Scripts\{ffmpeg,ffprobe}.exe` 全被 .gitignore 忽略。
- 預處理產出（供後續階段 B/C 使用）：
  - `logs\ryza_v4_lora\`：`2-name2text.txt`（467）、`2-name2text-0.txt`（原檔保留）、`3-bert\`（空）、`4-cnhubert\`（467 個 .pt）、`5-wav32k\`（467 個 wav，179.9 MB）、`6-name2semantic.tsv`（468 行含表頭）、`6-name2semantic-0.tsv`（原檔保留）。
  - `logs\ryza_v4_lora_eval10\`：同結構，數量 10、5-wav32k 3.4 MB。

### `<RYZA>`（`feat/tts-v4-lora-prep`，HEAD `0b7c743`）
- 新增未追蹤（尚未 commit，待 G2 放行後再依 §12.5 commit）：
  - `tools\run_preprocess.py`（階段 A 三步驅動）
  - `tools\stage_a_validate.py`（PA-1～PA-13 驗證）
  - `reports\logs\`（含 `raw\*.out/.err`、`stage_a_pa.json`；`raw\` 已被內部 .gitignore 忽略，只有整理後的 .md 會提交）
- 已 commit（G1）：`tools\t0_*.py`、`tools\t1_unk_analysis.py`、`reports\G1_t0_t1_report.md`。

---

## 6. 風險與意外發現

1. **官方腳本的 `except` 會吞錯**：`1-get-text.py:102-103`、`2-get-hubert-wav32k.py:124-125`、`3-get-semantic.py:115-116` 皆為裸 `except` 印 traceback 後繼續，**exit code 恆為 0**。結論：階段 C 冒煙測試不能只看結束碼，必須以「預期產出檔案數量與內容」為準（本計畫的 PA／SM 檢查表已是此設計）。建議在 `docs/07_finetune.md` 的易錯點加入此條。
2. **ffmpeg 是本專案的隱性系統依賴**：`load_audio` 只走 ffmpeg CLI，無 Python fallback。已記錄於 `docs/07_finetune.md`（見下）。建議環境需求章節列為必備項。
3. **`3-bert\` 目錄在 ja 時確實為空**（符合 §9.1 預期），但第 1 步仍會載入 RoBERTa（`1-get-text.py:61-62`）——T0 的轉檔是必要的。
4. **第 2 步的 NaN retry 路徑未觸發**：`nan_fails` 為空，`is_half` 未被降級，全程維持 fp16。

---

## 7. docs/07_finetune.md 補充（ffmpeg 記錄）

依放行要求，環境需求資訊已記錄於此供後續寫入 `docs/07_finetune.md` 第 1 章：

```text
ffmpeg：9.0.2-essentials_build（gyan.dev 靜態版）
  下載：https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
  zip SHA256：60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba
  放置位置：<GPT>\.venv\Scripts\ffmpeg.exe 與 ffprobe.exe（不改名、不做 .cmd shim）
  原因：tools/my_utils.py 的 load_audio 只走 ffmpeg CLI；venv Scripts 需在 PATH 上
```

**停在 G2，等待 PM 放行後才進階段 B（訓練配置與啟動腳本）。**
