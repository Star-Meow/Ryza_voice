# Atelier Ryza 2 語音解包專案

從 `D:\SteamLibrary\steamapps\common\Atelier Ryza 2`（**唯讀**，未做任何變更）
解出的遊戲音聲檔。遊戲資料為光榮特庫摩的 **KTSR「Sound Ring」** 封裝格式。

## 目錄結構

| 路徑 | 內容 |
|---|---|
| `wav/` | **VOICE 主語音庫**：9,569 個 WAV（48kHz / 16bit / mono），共 13.76 小時，4.5GB |
| `voice_index.csv` | 主庫索引：WAV 檔名 ↔ cue 編號 ↔ sound_id ↔ 時長 ↔ stbin 偏移 |
| `battle_voice/` | 戰鬥角色語音庫（014/015）全解出，**檔名即為遊戲內 cue 名稱** |
| `ryza_battle/` | 從戰鬥庫節選的 **48 個 ライザ(Ryza) 戰鬥語音** |
| `tools/` | 解包工具與解析器（說明見下） |
| `ryza_main/` | 主庫ライザ候選（分級 A/B/C，見下節） |
| `ryza_main/top500/` | 依 svm_dec 排序的 top500 候選音訊（實際 497 檔，剔除 3 個人工確認負樣本；STT 階段輸入，見下節） |
| `ryza_main/wrong/` | 已人工確認非ライザ的負樣本音檔 |
| `data/` | GPT-SoVITS 格式訓練清單（見「ASR 語音轉文字處理」） |
| `reports/` | ASR 報告（本次暫緩，尚未建立） |
| `ryza_main_index.csv` | 主庫全檔之說話者相似度排序索引 |

## 兩種語音庫的差別（重要）

- **戰鬥角色語音**（`014_battle_SP_character`、`015_battle_character`）：
  KTSR 子項目**內建加密的 cue 名稱**，已解密，可直接看出角色歸屬。
- **VOICE 主語音庫**（95.7% 的遊戲台詞）：
  經全數 9,569 個 cue 逐一掃描確認，**名稱欄位全為空**（0/9569），
  只留下遞增的數值 sound_id。來源名稱只存在遊戲腳本/事件資料（PAK 封裝）內，
  音聲庫本身不保存。因此主庫只能以編號順序輸出。

## ライザ (Ryza) 相關語音

`ryza_battle/` 中 48 個檔案，命名規則為 `SE_BCH_NNN_RYZA_*`（戰鬥語音）
與 `SE_BSP_NNN_RYZA_*`（必殺技語音），例如：

```
015_SE_BCH_001_RYZA01.wav
015_SE_BCH_003_RYZA_BLAZE_RUSH03.wav
014_SE_BSP_060_RYZA_WAND01_a.wav
014_SE_BSP_063_RYZA_CHARGE01.wav
```

同一批 cue 名稱的 `_a` / `_b` / `_c` 後綴為 vgmstream 對重複名稱的自動編號。

戰鬥庫的原始編碼為 Microsoft 4-bit ADPCM，解成 WAV 後取樣率保留原值
（47976–48019 Hz 之間浮動，非整數 48000），多數為單聲道、6 個為立體聲，
皆屬正常。全部 48 個檔案已驗證為有效 16bit PCM WAV。

> 主語音庫（13.76 小時台詞）因無名稱欄位，無法直接從音聲庫本身分辨哪幾軌是
> ライザ。若需要可進一步解包 PAK 封裝的事件腳本，比對語音 cue 引用。

## 主庫ライザ語音篩選（說話者嵌入比對）

主庫 9569 個 cue 全無名稱欄位（見上），無法從音聲庫本身分辨角色。
改用**說話者嵌入**比對：以使用者確認的 3 個ライザ對話語音
（`00025.wav`、`00030.wav`、`00042.wav`）為參考，在主庫中找同一名說話者。

### 方法

1. **嵌入模型**：speechbrain ECAPA-TDNN（VoxCeleb，192 維），權重存於
   `tools/spk_model/`。Pipeline：wav → Fbank(80mel) → sentence-norm
   → ECAPA-TDNN → global-norm → L2 正規化向量。
2. **GPU 執行**：嵌入階段於 RTX 3060（CUDA）完成；後續 ASR 階段改用
   **RTX 4070S 12G**。本機 TF32 matmul 會造成 CUDA 記憶體
   損壞（約 20–30 個檔後 illegal instruction / CUBLAS 錯誤），須
   `torch.backends.cuda.matmul.allow_tf32 = False`；並以
   `CUDA_LAUNCH_BLOCKING=1` + 分段進程（`tools/run_encode.sh`）吸收剩餘的
   隨機驅動不穩。9569 檔嵌入結果：`tools/embeddings.npy`。
3. **評分**：`consensus_score` = 與 `00030`、`00042` 兩個乾淨錨點的 cos 平均。
   （`00025` 背景噪音較多、在嵌入空間中偏離，排名 #4500，僅作次要參考。）

### 關鍵驗證

- 戰鬥庫內 leave-one-out 角色分類 **100% 正確**（RYZA/KLAU/TAO/LENT/PAT/ANCIENT
  6 角色、131 檔），嵌入模型本身可靠。
- **跨條件（戰鬥→對對話）比對不可靠**：3 個對話種子用戰鬥庫 centroid
  全被判為 TAO——領域偏移大於說話者差異。因此只用主庫內部互比。
- Top-100 候選彼此內聚度 0.843、Top-500 0.795（隨機基準 0.52–0.55），
  確認高分群是真實的說話者聚類，而非雜訊。

### 產出

| 路徑 | 內容 |
|---|---|
| `ryza_main_index.csv` | 主庫全 9569 檔依 consensus 分數排序，含各錨點 cos、min3、戰鬥庫參考分 |
| `ryza_main/A_high/` | 92 檔，consensus ≥ 0.80（**已全部人工確認為ライザ**） |
| `ryza_main/B_likely/` | 401 檔，0.76–0.80（部分已確認，見下） |
| `ryza_main/B_review/` | 40 檔，SVM 判別值最低、最可疑，優先待審 |
| `ryza_main/C_possible/` | 894 檔，0.72–0.76 |
| `ryza_main/wrong/` | 5 檔已確認錯誤（00108/01120/01819/01916/03033） |

### 使用者回饋精煉（兩輪）

`A_high` 全部確認正確。`B_likely`/`C_possible` 中累計確認 **20 個誤判**
（同一相近角色，清單見 `tools/neg_labels.py`）。以 A_high(92 正) vs 20 負
訓練 LinearSVC(C=1)：

- **LOO 正確率 99.1%**（正樣本 0 錯、負樣本 1 錯）。
- 決策值：92 個正樣本 **≥ +0.163**，20 個負樣本 **≤ −0.433**，間隙明確。
- 負樣本互相 cos 中位數 0.805，本身構成連貫聚類。
- `ryza_main_index.csv` 含 `svm_dec`、`margin_to_neg` 欄位。

依此邊界整理（`tools/refine_labels.py`）：

| 資料夾 | 檔數 | 判斷 |
|---|---|---|
| `A_high/` | 92 | 已人工確認為ライザ |
| `B_likely/` | 386 | svm_dec > −0.433，可能正確 |
| `C_possible/` | 729 | svm_dec > −0.433，不確定 |
| `D_flagged/` | 165 | svm_dec < −0.433（比任何確認誤判都更負），強烈疑似錯誤 |
| `B_review/` | 40 | 剩餘檔中 svm_dec 最低，優先待審（`ryza_main_review.csv`） |
| `wrong/` | 20 | 已確認錯誤 |

> 戰鬥庫角色 centroid 無法用於此精煉：正負樣本在戰鬥空間的最佳匹配
> 皆為 TAO（領域偏移）。負樣本嵌入位於正樣本聚類內部，kNN 無效；
> 僅線性邊界能分離。

> 分數平滑衰減、無明顯間隙，門檻為經驗值。`A_high` 建議優先試聽；
> `C_possible` 假陽性風險較高。如需更精確的邊界，需遊戲腳本（PAK）或
> 更多已確認樣本來校準。

## ASR 語音轉文字（STT）資料處理

接續說話者嵌入篩選，將 top500 候選音訊轉成可供 GPT-SoVITS 使用的
逐字稿訓練清單。整個階段的核心是**三處編號集合的一致性**：
`pos_label.py`、`asr_screening.json`、`data/ryza_train.list` 必須指向
同一批音訊。

### 檔案角色

| 路徑 | 角色 |
|---|---|
| `wav/` | **原聲檔唯一來源**：9,569 個主庫 WAV，ASR 與清單的路徑最終都指向此處 |
| `ryza_main/top500/` | 依 svm_dec 排序的 top500 音訊，檔名格式 `NNNN_原檔名.wav`（如 `0001_00028.wav`） |
| `ryza_main/wrong/` | 人工確認非ライザ的負樣本音檔 |
| `tools/neg_labels.py` | `WRONG[]` 負樣本編號；`DUPLICATE[]` ASR 文本重複排除編號 |
| `tools/pos_label.py` | `SVM_DEC[]` 正樣本編號清單（STT 階段的處理範圍） |
| `tools/asr_screening.json` | ASR 逐字稿與過濾紀錄（每筆含 status / reason / 分數） |
| `tools/build_pos_label.py` | 造冊與更新 `pos_label.py`（`--source-dir` / `--exclude-from` / `--prune-raw`） |
| `tools/screen_asr.py` | ASR 與過濾，產 `asr_screening.json` |
| `tools/transcribe.py` | 專責產 `data/ryza_train.list`（純標準庫，不重跑 ASR） |
| `tools/validate_pos_label.py` | 三處一致性驗證 |
| `tools/make_cer_sample.py` | CER 抽查：固定 seed 從 json 隨機抽段產清單 |
| `tools/cer_report.py` | CER 抽查：對人工校對結果算字元錯誤率（純標準庫 Levenshtein） |
| `reports/cer_sample.csv` | CER 抽樣清單（50 段，reference 欄待人工校對填入） |
| `data/ryza_train.list` | GPT-SoVITS 格式訓練清單：`路徑\|ryza\|ja\|文字` |
| `reports/asr_report.md` | 統計、參數、CER、專有名詞校正（**本次暫緩**，待後續） |

### 編號正規化（v 前綴）

`ryza_main/top500/` 的檔名為 `NNNN_原檔名.wav`，其中一部分原檔名帶 `v` 前綴：

```
0093_v00086.wav   →  取底線後段 v00086  →  剝 v  →  00086
```

`v00086.wav` 與 `00086.wav` 視為**同一份音訊**（`tools/select_samples.py` 的
`index_key`/`sound_id` 邏輯）。清單與 `pos_label.py` 一律使用剝除後的純數字
編號，指向 `wav/` 主庫。top500 中此類 v 前綴共 6 個：
`v00086`、`v00105`、`v00142`、`v00339`、`v00596`、`v09508`。

### 七步驟資料流

```
[步驟 1] 依 svm_dec 排序 → ryza_main/top500/（已執行）
        │
        ▼
[步驟 2] 人工篩查負樣本（top500 內 3 份）
        │       ├─► ryza_main/wrong/
        │       └─► neg_labels.py WRONG[]
        │
        ▼
[步驟 3] 造冊 → pos_label.py SVM_DEC[]（497）
        │                └── v 前綴剝除、跨行去重
        │
        ▼
[步驟 4] ASR + 雜訊過濾 → tools/asr_screening.json
        │       ├── Whisper large-v3, language=ja, VAD 開啟
        │       ├── 時長 <2s / >15s
        │       ├── 非說話聲（喘氣、吶喊等感嘆詞）
        │       ├── ASR 空輸出（VAD 後無語音）
        │       └── 明顯幻覺（Whisper 常見輸出、單字重複）
        │       └─► 497 筆紀錄（單筆規則全 PASS）
        │
        ▼
[步驟 5] 跨檔文本去重（門檻 >=2，保留首次出現者）
        │       ├── 排除 6 筆 → neg_labels.py DUPLICATE[]
        │       └─► asr_screening.json 這 6 筆改標 EXCLUDE
        │
        ▼
[步驟 6] 同步 pos_label.py SVM_DEC[]（497 → 491）
        │       └─► asr_screening.json 精簡為 491 全 PASS
        │
        ▼
[步驟 7] transcribe.py 產生清單（不重跑 ASR）
        │       ├── 讀 pos_label.py SVM_DEC[] + asr_screening.json 文字
        │       ├── 只取 PASS 且文字非空者，依 SVM_DEC 順序
        │       └─► data/ryza_train.list（491 行）
        │
        ▼
[reports/asr_report.md] ← 本次暫緩，待後續
```

### 文本去重

跨檔比對 `asr_screening.json` 的 `core` 欄位（逐字稿去除標點與非日文字元）。
同一 `core` 出現 >=2 次時，**保留 SVM_DEC 順序中首次出現者**，其餘標
`EXCLUDE`（`reason=DUPLICATE_TEXT`）。目前偵測到 4 組共 10 筆，排除其中 6 筆：

| 保留 | 排除 | 文本 |
|---|---|---|
| `06332` | `04253`, `07018` | あれ?扉が開いてる。あんなしっかり閉まってたのに。 |
| `05676` | `07718`, `08413` | 扉、閉じちゃってるね。どうやって開けるんだろう。 |
| `01359` | `01991` | 私さ、そろそろ島に帰ろうかなって思うんだ。 |
| `01036` | `02998` | まだだよ。まだ諦めない |

> 這 10 個音檔的 MD5 互不相同、時長也些微相差，是**同一句台詞的重複錄製**，
> 非 Whisper 幻覺；對 TTS 訓練而言保留一個版本即可。

### 三處一致性

步驟 6 完成後，三個檔案的編號集合必須完全相等：

```
pos_label.py SVM_DEC[]        = 491
asr_screening.json PASS       = 491
data/ryza_train.list          = 491
neg_labels.py WRONG           = 3   （top500 內人工剔除）
neg_labels.py DUPLICATE       = 6   （ASR 文本重複）
```

數量關係：`500（top500）− 3（WRONG）= 497（造冊）− 6（DUPLICATE）= 491（清單）`。

由 `tools/validate_pos_label.py` 從來源獨立推導期望值交叉檢查
（長度 / 唯一 / 範圍 / 音檔存在 / 順序 / 三處一致）。

### 執行順序

```bash
# 步驟 3：從 top500 造冊（497）
python tools/build_pos_label.py \
  --source-dir ryza_main/top500 \
  --output tools/pos_label.py

# 步驟 4+5：ASR 重跑，門檻 >=2 生效 → json 497（6 EXCLUDE）
.venv/Scripts/python.exe tools/screen_asr.py \
  --pos-label tools/pos_label.py \
  --audio-root wav \
  --raw-out tools/asr_screening.json

# 步驟 6：排除 6 筆，pos_label 與 json 同步為 491
python tools/build_pos_label.py \
  --source-dir ryza_main/top500 \
  --exclude-from tools/asr_screening.json \
  --prune-raw tools/asr_screening.json \
  --output tools/pos_label.py

# 步驟 7：產清單（純標準庫，不重跑 ASR）
python tools/transcribe.py \
  --pos-label tools/pos_label.py \
  --asr-json tools/asr_screening.json \
  --output data/ryza_train.list

# 驗證：三處一致
python tools/validate_pos_label.py \
  --source-dir ryza_main/top500 \
  --pos-label tools/pos_label.py \
  --neg-label tools/neg_labels.py \
  --asr-json tools/asr_screening.json \
  --train-list data/ryza_train.list \
  --audio-root wav

# 步驟 9：刪除根目錄舊清單（清單已改道至 data/）
git rm ryza_train.list
```

### STT 相關工具

| 檔案 | 用途 | 狀態 |
|---|---|---|
| `select_samples.py` | svm_dec 排序、`sound_id` 編號正規化 | 已完成 |
| `build_pos_label.py` | 步驟 3、6 造冊與更新 `pos_label.py` | 已完成（`--source-dir`、`--exclude-from`、`--prune-raw`） |
| `screen_asr.py` | 步驟 4、5 ASR 與過濾 | 已完成（DUPLICATE_TEXT 門檻 `>=2`、標 EXCLUDE） |
| `transcribe.py` | 步驟 7 專責產 `data/ryza_train.list` | 已完成（純標準庫） |
| `validate_pos_label.py` | 三處一致性驗證 | 已完成（`--source-dir` 與交叉檢查） |
| `neg_labels.py` | `WRONG[]` + `DUPLICATE[]` | 已完成（`DUPLICATE[]` 6 筆） |
| `asr_filter_report.md` | 現行篩選報告（根目錄） | `reports/asr_report.md` 暫緩，此檔維持現狀 |

### 目前進度

| 項目 | 狀態 |
|---|---|
| 步驟 1–4 | ✅ top500 497 檔、ASR 497 筆 |
| 步驟 5 去重 | ✅ 門檻 `>=2`，6 筆 EXCLUDE，`DUPLICATE[]` 已填 |
| 步驟 6–7 | ✅ 同步至 491、`transcribe.py` 產 `data/ryza_train.list`（491 行） |
| 步驟 8–9 | ✅ 三處驗證通過、`git rm ryza_train.list` |
| `data/` 目錄 | ✅ 已建立（`data/ryza_train.list`） |
| `reports/cer_sample.csv` | ✅ 50 段抽樣清單已產生（reference 待人工校對） |
| `reports/asr_report.md` | ⬜ 本次暫緩（CER 與專有名詞校正待後續） |

> 驗收條件：有效轉譯段數 ≥ 450（目標 491），且全數人工確認為ライザ。
> 目前 491 段已全數通過 ASR 過濾；**人工確認為ライザ**尚待試聽
> （491 段中僅 `A_high` 92 段已人工確認）。

## 音聲格式筆記

VOICE 主庫的音訊為 **KOVS 容器包裹的 Ogg Vorbis**，兩層混淆：

1. 每個 KOVS 區塊有 0x20 位元組標頭（`KOVS` magic + block_size + loop + 通道數），
   Ogg 串流由 +0x20 開始。
2. Ogg 串流的**前 0x100 位元組與自身偏移做 XOR**（`buf[i] ^= i`），之後為標準 Ogg。

相關工具 `tools/kovs_to_ogg.py` 可把單一 KOVS 區塊還原為標準 `.ogg`，
已驗證解碼後的 PCM 與 vgmstream 輸出的 WAV **位元級完全一致**。

戰鬥庫的 cue 名稱另外經 **LCG XOR 加密**（標準 rand：
`seed = 0x343FD*seed + 0x269EC3`，取 `(seed>>16)&0xFF` 逐位元組 XOR），
seed 為各庫檔案偏移 0x0c 的 `audio_id`（本作固定為 `0xAF4905A9`）。

## tools/ 工具說明

| 檔案 | 用途 |
|---|---|
| `vgmstream/` | vgmstream r2117（官方支援 KTSC/KTSR/KOVS），批次解出主庫與戰鬥庫 |
| `parse_ktsr.py` | 解析 VOICE.ktsl2asbin（KTSC 容器）→ 產出 `voice_cues.csv` |
| `parse_bare_ktsr.py` | 解析裸 KTSR 庫（戰鬥庫）→ 解密 cue 名稱 |
| `kovs_to_ogg.py` | 單一 KOVS 區塊去混淆 → 標準 Ogg（`--cue N` 用索引） |
| `ktsr_ref.c` | vgmstream 的 ktsr.c 原始碼（格式對照參考） |
| `spk_model.py` | ECAPA-TDNN 嵌入（GPU，TF32 關閉避免記憶體損壞） |
| `encode_chunk.py` / `run_encode.sh` | 分段 GPU 編碼（崩潰重試，不退 CPU） |
| `encode_all.py` | 單進程完整編碼（穩定環境用） |
| `final_index.py` / `export_ryza.py` | 評分、索引產出、分級子集匯出 |
| `embeddings.npy` | 9569 檔的 192 維說話者嵌入 |
| `select_samples.py` / `screen_asr.py` / `build_pos_label.py` / `validate_pos_label.py` | STT 階段：top500 排序、ASR 過濾、造冊、三處一致性驗證（見「ASR 語音轉文字（STT）資料處理」） |

### 重新解出（如需）

```bash
# 主庫（注意：須指向 .ktsl2asbin，vgmstream 會自動找同名的 .ktsl2stbin）
tools/vgmstream/vgmstream-cli.exe -s 1 -S 0 -o "wav/?05s.wav" \
  "D:\SteamLibrary\steamapps\common\Atelier Ryza 2\Data\x64\Sound\VOICE.ktsl2asbin"

# 戰鬥庫（?n 為 cue 名稱）
tools/vgmstream/vgmstream-cli.exe -s 1 -S 0 -o "battle_voice/015_?n.wav" \
  "D:\SteamLibrary\steamapps\common\Atelier Ryza 2\Data\x64\Sound\015_battle_character.ktsl2asbin"
```

## 索引欄位對應

`voice_index.csv` 中 `cue_index = N` 對應 `wav/(N+1):05d.wav`
（vgmstream 的 subsong 編號為 1 起算）。已驗證：cue 0 的 num_samples
（223043）與 vgmstream 第 1 軌、以及 `wav/00001.wav` 的幀數完全一致；
200 個抽樣 cue 的 stbin 偏移全數指向有效的 KOVS 區塊，大小總和佔 stbin 的 99.9%。

## 授權

- 解出的音聲內容版權屬 Koei Tecmo / Gust，僅供個人用途。
- vgmstream 為其作者所有（ISC-style license），見 `tools/vgmstream/COPYING`。
