# 說話者篩選：說話者嵌入比對（Part 1 ②）

> 主語音庫的 cue 名稱欄位全為空（0/9569），無法從音聲庫本身分辨角色。本階段改用**說話者嵌入**在主庫內部互比，找出與已知ライザ語音同一說話者的音軌。
>
> 前置階段見 [`unpack.md`](03_unpack.md)，後續的 ASR 與訓練清單建構見 [`transcript_pipeline.md`](05_transcript_pipeline.md)。

---

## 1. 動機

主庫 9,569 個 cue 沒有任何名稱資訊，只留下遞增的 `sound_id`（原因見 [`unpack.md`](03_unpack.md) §2）。
直接可行的做法有兩條：解 PAK 事件腳本比對 cue 引用（成本高、未做），或用**說話者嵌入**在主庫內部做無監督聚類（成本低、可立即執行）。

本階段採後者：以使用者確認的 3 個ライザ對話語音為參考，在 9,569 檔主庫中找出同一名說話者。

## 2. 嵌入 pipeline

**模型**：speechbrain ECAPA-TDNN（VoxCeleb 訓練，**192 維**），權重存於 `tools/spk_model/`
（`hyperparams.yaml`、`config.json`、`label_encoder.txt` 已納入版控；`*.ckpt` 權重 83 MB 可重新下載，未納入版控）。

**特徵流程**（實作見 [`tools/spk_model.py`](../tools/spk_model.py)）：

```
wav → Fbank(80mel) → sentence-norm → ECAPA_TDNN → global-norm → L2 正規化 → 192 維向量
```

**執行特性**：
- **GPU-only**（`DEVICE = 'cuda'`，禁止 CPU 回退），執行前須 `torch.cuda.is_available()` 為 True
- [`tools/encode_all.py`](../tools/encode_all.py) 單進程循序編碼：單檔約 **46 ms**，9,569 檔全程約 **7–8 分鐘**。
  循序是必要的——GPU 上多執行緒並發 CUDA 呼叫會造成 `illegal memory access`
- [`tools/run_encode.sh`](../tools/run_encode.sh) 為分段策略：每 **500 檔**一個獨立進程，
  以 `CUDA_LAUNCH_BLOCKING=1` 序列化 CUDA 核心，崩潰重試最多 3 次且**不退回 CPU**；
  再由 [`tools/merge_chunks.py`](../tools/merge_chunks.py) 合併
- 產出 `tools/embeddings.npy`，形狀 **9570 × 192**（index 0 保留不使用，實際檔號 = index + 1）

> **TF32 設定**：`spk_model.py` 沿用 `torch.backends.cuda.matmul.allow_tf32 = False`。
> 這是**其他環境（RTX 3060）曾發生 CUDA 記憶體損壞的過往經驗，本機（RTX 4070 SUPER）從未發生**。
> 完整記錄見 `docs/09_pitfalls.md` §4（該檔尚未納入版控）。本機無需理會、不 patch、不預防、不監控。

## 3. 校準來源與評分

**戰鬥庫校準樣本**（[`tools/score_ryza.py`](../tools/score_ryza.py)）：

- 正樣本：`battle_voice/` 中 **54 個 RYZA 戰鬥語音**（不同錄音條件）
- 負樣本：`battle_voice/` 中 KLAU／TAO／LENT／PAT／ANCIENT 等其他角色

**主庫錨點**（[`tools/final_index.py`](../tools/final_index.py) 的錨點策略）：

| 錨點 | 角色 | 說明 |
|---|---|---|
| `wav/00030.wav` | **主錨點** | 乾淨對話錄音 |
| `wav/00042.wav` | **主錨點** | 乾淨對話錄音（與 00030 互比 cos 0.769） |
| `wav/00025.wav` | 次要參考 | 背景噪音較多、嵌入空間中偏離，排名 #4500 |

`consensus_score` = 與 `00030`、`00042` 兩錨點的 **cosine 平均**。
戰鬥庫各角色 centroid 的分數一併列出，但**僅供參考**（已知跨條件不可靠，見 §4）。

## 4. 關鍵驗證

### 4.1 嵌入模型本身可靠

戰鬥庫內 **leave-one-out 角色分類 100% 正確**（RYZA／KLAU／TAO／LENT／PAT／ANCIENT 六角色，共 **131 檔**）。
驗證工具：[`tools/xdom_test.py`](../tools/xdom_test.py)。
這證明嵌入向量確實承載了說話者差異，不是雜訊。

### 4.2 跨條件比對不可靠

以戰鬥庫 centroid 分類 3 個對話種子，**全部被判為 TAO**。
原因是**領域偏移大於說話者差異**——戰鬥語音與對話語音的錄音條件差異，壓過了角色本身的特徵差異。

**因此 Part 1 ② 的篩選一律只在主庫內部互比**，不使用任何戰鬥庫 centroid 作為判定依據。

### 4.3 高分群是真實聚類

| 集合 | 內聚度（互相 cos） |
|---|---|
| Top-100 候選 | **0.843** |
| Top-500 候選 | **0.795** |
| 隨機基準 | 0.52 – 0.55 |

高分群的內聚度遠高於隨機基準，確認其為真實的說話者聚類。

## 5. 兩輪使用者回饋精煉

首輪依 `consensus_score` 分級匯出後，使用者逐批試聽回饋：
`A_high`（92 檔）**全部確認正確**；`B_likely` / `C_possible` 中累計確認 **20 個誤判**
（皆為同一相近角色，清單見 [`tools/neg_labels.py`](../tools/neg_labels.py) 的 `WRONG[]`）。

以 **A_high 的 92 個正樣本 vs 20 個負樣本**訓練 `LinearSVC(C=1)`（[`tools/refine_labels.py`](../tools/refine_labels.py)）：

| 指標 | 結果 |
|---|---|
| **LOO 正確率** | **99.1%**（正樣本 0 錯、負樣本 1 錯） |
| 正樣本決策值 | 92 個**全部 ≥ +0.163** |
| 負樣本決策值 | 20 個**全部 ≤ −0.433** |
| 負樣本互相 cos 中位數 | 0.805（負樣本本身構成連貫聚類） |

正負決策值之間有**明確間隙**，故以 `svm_dec > −0.433` 作為分界。
`ryza_main_index.csv` 隨之加入 `svm_dec` 與 `margin_to_neg` 兩個欄位。

## 6. 方法學排除記錄

本節記錄**試過但行不通**、或**已知限制**的路徑，供後續重現時不必重走。

### 6.1 戰鬥庫 centroid 無法用於主庫精煉

正負樣本在戰鬥空間的最佳匹配**皆為 TAO**（領域偏移），無法區分。
戰鬥庫 centroid 只在 §4.1 的 leave-one-out 情境下有效（同一條件內互比），不可跨條件移植。

### 6.2 kNN 無效，僅線性邊界可用

20 個負樣本的嵌入**位於正樣本聚類的內部**，而非外側。
因此以近鄰距離分類（kNN）無法把負樣本剔除——它們在特徵空間裡本來就與正樣本混在一起。
**只有線性決策邊界能分離**兩者，這是採 LinearSVC 而非 kNN 的直接原因。

### 6.3 分數平滑衰減，門檻為經驗值

`consensus_score` 沿遞減方向**平滑衰減、無明顯間隙**。
`0.72` / `0.76` / `0.80` 三個門檻是**經驗值**，不是從資料分布推導出的統計顯著分界。

實務建議：
- **`A_high` 優先試聽**（已全數人工確認）
- **`C_possible` 假陽性風險較高**

要提高精度，需要的是**遊戲腳本（PAK）或更多已確認樣本**來校準，而不是更換分類演算法。

### 6.4 曾嘗試的其他聚類路徑

- [`tools/cluster_ryza.py`](../tools/cluster_ryza.py)：KMeans（K = 20／40／60）與
  AgglomerativeClustering，用於找出 3 個種子所在的群
- [`tools/refine_ryza.py`](../tools/refine_ryza.py)：從 3 個種子出發，反覆用候選集 centroid 重排、
  檢查收斂的迭代精煉
- [`tools/analyze_ryza.py`](../tools/analyze_ryza.py)：修正檔名對應、檢視連續區塊結構、
  據以建議門檻

這幾支支撐了門檻的選擇，但最終判定改以 §5 的 SVM 決策值為準。

## 7. 產出

### 7.1 分級目錄（`ryza_main/`）

| 目錄 | 檔數 | 判斷依據 |
|---|---|---|
| `A_high/` | 92 | `consensus ≥ 0.80`，**已全部人工確認為ライザ** |
| `B_likely/` | 383 | `svm_dec > −0.433`，可能正確 |
| `C_possible/` | 728 | `svm_dec > −0.433`，不確定 |
| `D_flagged/` | 165 | `svm_dec < −0.433`，比任何已確認誤判都更負，**強烈疑似錯誤** |
| `B_review/` | 40 | 剩餘檔中 `svm_dec` 最低者，**優先待審** |
| `wrong/` | 22 | 已人工確認錯誤 |
| `top500/` | 497 | 依 `svm_dec` 排序的 top500（序號 0365／0442／0484 為人工確認的負樣本，已剔除） |

> `wrong/` 22 檔對應 `WRONG[]` 23 筆：編號 `01120` 在 `WRONG[]` 中，但未被複製進
> `ryza_main/wrong/`（`wav/01120.wav` 本身存在於主庫）。兩處數量差 1 屬正常。

### 7.2 索引檔

- **`ryza_main_index.csv`** — 主庫全 9,569 檔依 `consensus_score` 排序，含各錨點 cos、`min3`、
  戰鬥庫參考分、`svm_dec`、`margin_to_neg`
- **`ryza_main_review.csv`** — `B_review/` 那 40 檔的優先待審清單

> `ryza_main/` 與其中的 WAV 因版權與體積**不納入版控**；分級資訊以索引 CSV 為準。