# .restructure — 文件邊界整理的比對工具

只存在於 `docs/repo-boundary` branch；合併前最後一個 commit 刪除本目錄（記錄保留在 git 歷史）。

| 檔案 | 用途 |
|---|---|
| `inventory.csv` | 基準版本（tag `docs-baseline` = `origin/main` 262adda）每個 `.md` 章節一列，記錄去向。可用 Excel 開啟（UTF-8 BOM） |
| `build_inventory.py` | 由規則產生 `inventory.csv`。調整去向時改 `step2_rules.py` 再重跑 |
| `check.py` | 唯讀比對 inventory 與工作樹，每一步結束執行 |
| `target_outline.md` | 各文件應有內容的框架（定案版），修改以此為準 |
| `step2_rules.py` | 依定案框架更新的去向規則，優先於 `build_inventory.py` 內的舊規則 |

## 欄位

- `action`：
  - `keep`：不動，內容 hash 必須不變。
  - `edit`：原處修改，內容必須改變；若 `dest` 有值，代表要改標題。
  - `move`／`merge`：移到 `dest`（`檔案#標題`）。`merge` 表示與既有內容合併。目標關鍵字命中不到一半時，`reason` 須寫「人工確認：」說明缺的是什麼。
  - `delete`：必須寫 `reason`；協作產物還要附上 GitHub 位置。
  - `add`：計畫新增的章節。
- `category`：專案說明／流程／設計／結果／參考／過程紀錄／協作產物／新增。
- `hash`：正規化後內文（去除空白與標點）的 SHA-1 前 10 碼。

## check.py 的狀態

| 狀態 | 意義 |
|---|---|
| OK | 已達成 |
| PENDING | 尚未執行 |
| PARTIAL | 做了一半，例如源還在、新舊標題並存、`keep` 的內容被改了、搬過去的關鍵字不到一半 |
| FAIL | 違反規則，例如章節消失卻沒有去向、`delete` 沒寫理由 |

另外三項檢查：
- `UNMAPPED`：目前的章節對不回 inventory，代表來路不明的新內容。
- `BROKEN`：相對連結的目標不存在。
- `STALE`：仍在引用已整檔移出的檔案（指向固定版本 permalink 的連結不算）。

有 FAIL、UNMAPPED 或 BROKEN 時，exit code 為 1。

**限制**：同一檔內出現同名標題時（例如 `defect_triage` 的三組「我發現了什麼」），只用標題比對，不比對內容 hash。

```
python -I .restructure/check.py        # 摘要與待處理清單
python -I .restructure/check.py -v     # 列出每一列
```
