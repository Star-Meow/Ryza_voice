# 訓練坑記錄（Pitfall Log）

> 持續累積文件。記錄 GPT-SoVITS v4 微調（Ryza 語音）從資料準備到正式訓練遭遇的坑、根因與處置，供後續階段 B（訓練配置）、C（冒煙測試）、正式訓練與新環境重現時查閱。

**issue 引用**：`docs/01_issue.md:55` —「訓練過程可穩定完成；若遇到既有的 GPU 不穩問題，須記錄處理方式。」

**相關文件**：[finetune.md](07_finetune.md)（環境與流程）、[../reports/G1_t0_t1_report.md](../reports/G1_t0_t1_report.md)、[../reports/G2_stage_a_report.md](../reports/G2_stage_a_report.md)、[../reports/smoke_test_stage_a.md](../reports/smoke_test_stage_a.md)

---

## 1. 欄位定義

每筆記錄固定欄位：

| 欄位 | 說明 |
|---|---|
| **編號** | `PK-NNN`，全庫唯一、連號，不因分類而重編 |
| **日期** | 首次發現日（YYYY-MM-DD）；事後追溯加註「追溯」 |
| **現象** | 可觀察的症狀，含錯誤訊息關鍵字 |
| **根因** | 確定的原因；未確定者寫「待查」 |
| **處置** | 已採取的動作；無動作者寫「觀察」 |
| **證據等級** | 【實測】親自執行並取得輸出／【讀碼】由原始碼行號推得／【推斷】間接推理／【缺失】僅由「東西不見了」判定，無軌跡 |
| **狀態** | 已解／觀察中／無法判定 |
| **證據** | 檔案與行號，或指令與輸出摘要 |

## 2. 追加規則

1. **新坑依編號往下連號**（不插隊、不改既有編號），寫在所屬分類節的**最末**。
2. **既有記錄原則上不修改**；僅狀態變化時在原記錄補一行「▶ YYYY-MM-DD 更新：…」，不覆寫原文。
3. 「GPU／CUDA」分類另見 §4 專區，該專區獨立維護（issue 特別要求）。
4. 無法判定根因者一律標【缺失】或【推斷】並寫明缺什麼資訊，不可補腦成已解。

---

## 3. 坑記錄

### 3.1 環境依賴

#### PK-001 ffmpeg 缺失導致 467 檔全 FileNotFoundError
- **日期**：2026-10-05（追溯，首次發生於階段 A 第 2 步首次執行）
- **現象**：`2-get-hubert-wav32k.py` 對 467 個 wav 全部 `FileNotFoundError: [WinError 2]`，log 中 934 次 traceback（467×2，`load_audio` 內部 except 重跑一次）。`4-cnhubert\`／`5-wav32k\` 為空目錄，`6-name2semantic-0.tsv` 為 0 byte。
- **根因**：`tools/my_utils.py:16-37` 的 `load_audio` 透過 `ffmpeg-python` 以 `cmd=["ffmpeg","-nostdin"]` 起 subprocess，**只認 PATH 上的 `ffmpeg`**，無 Python fallback；本機系統 PATH 無 ffmpeg。
- **處置**：依 PM 放行（方案 A），下載 gyan.dev 靜態版 `ffmpeg 9.0.2-essentials_build`，將 `ffmpeg.exe`／`ffprobe.exe` 放入 `<GPT>\.venv\Scripts\`（不改名、不做 .cmd shim），使 venv 啟動即生效；驅動程式 `run_preprocess.py` 另把該目錄加進子行程 PATH。刪除失敗產出後重跑第 2、3 步成功。
- **證據等級**：【實測】
- **狀態**：已解
- **證據**：`reports/smoke_test_stage_a.md` 區域 1；`docs/07_finetune.md` §1；`shutil.which('ffmpeg')` = `H:\git\GPT-SoVITS\.venv\Scripts\ffmpeg.EXE`，`ffmpeg -version` = `9.0.2-essentials_build-www.gyan.dev`
- **備註**：首次失敗完全未寫出任何檔案（`load_audio` 在寫檔前丟例外），故無「部分損壞」；與 PK-004（exit code 恆 0）複合，才使問題可被檔案計數檢查抓到。

#### PK-002 transformers 拒載 `.bin`（CVE-2025-32434）
- **日期**：2026-10-04（G1）
- **現象**：`AutoModelForMaskedLM.from_pretrained(bin 目錄)` 與 `HubertModel.from_pretrained(...)` 皆 `ValueError: ... CVE-2025-32434 ... require users to upgrade torch to at least v2.6 ... does not apply when loading files with safetensors`。
- **根因**：transformers 4.57.6 因該 CVE 預設禁止以 `torch.load` 載入 `.bin`；本環境 torch 2.5.1+cu121 未達 2.6 門檻，且升級 torch 會破壞 cu121 綁定。
- **處置**：執行 T0——將 `chinese-roberta-wwm-ext-large`／`chinese-hubert-base` 的 `pytorch_model.bin` 轉為 `model.safetensors`（`tools/t0_convert_safetensors.py`），以 `tools/t0_validate.py` 驗 key/dtype/shape 全保留、`from_pretrained` 乾淨載入、輸出最大絕對差 0.0，並附 negative fixture 證明驗證能辨識損壞。
- **證據等級**：【實測】
- **狀態**：已解
- **證據**：`reports/G1_t0_t1_report.md` T0-1～T0-8、§2 明細表；`tools/t0_validate.py`

#### PK-011 推論端缺 `jieba_fast`，且 `fast_langdetect` 想連網下載模型
- **日期**：2026-10-05（階段 C）
- **現象**：推論冒煙 E1 首次執行直接 `ModuleNotFoundError: No module named 'jieba_fast'`，整個 `TTS_infer_pack` 無法 import；繞過後又遇 `FileNotFoundError: fast-langdetect: Cache directory not found: H:\git\GPT-SoVITS\GPT_SoVITS\pretrained_models\fast_langdetect`。
- **根因**：兩個**模組層級**的依賴，與語言無關——
  1. `text/chinese.py:19-23` 與 `text/tone_sandhi.py:17` 在 import 期就 `import jieba_fast`，而 `TTS_infer_pack/TextPreprocessor.py:13` 又直接 `from text import chinese`，故**即使只做日文推論也會被擋下**。venv 內只有 `jieba` 0.42.1，沒有 `jieba_fast`。
  2. `text/LangSegmenter/langsegmenter.py:11` 把 `fast_langdetect` 的 `cache_dir` 指到 `GPT_SoVITS/pretrained_models/fast_langdetect/`，該目錄不存在。`fast_langdetect` 只在使用「套件預設 cache 目錄」時才自動建立；對自訂目錄會直接拋錯並轉而嘗試從 `dl.fbaipublicfiles.com` 下載 `lid.176.bin`——而本專案 §4.2 禁止連網。
- **處置**：**不安裝任何套件、不下載、不複製模型檔**，改在推論行程內做兩處設定：
  1. `sys.modules["jieba_fast"]` 別名到已安裝的 `jieba`（jieba_fast 本就是 jieba 的分支，API 相同），並一併掛上 `jieba_fast.posseg`。
  2. 把 `fast_langdetect.infer._default_detector` 改指向**套件自己內附**的 `resources/lid.176.ftz`（938 KB），經由 `LangDetectConfig(custom_model_path=...)`。
  實作見 `tools/infer_smoke.py` 的 `install_jieba_fast_alias()` 與 `point_fast_langdetect_at_bundled_model()`。
- **證據等級**：【實測】＋【讀碼】
- **狀態**：已解（繞過），但**根因仍在環境中**，新環境需重做上述設定
- **證據**：`reports/G4_stage_c_report.md` SM-INF-1；`tools/infer_smoke.py`
- **備註**：兩處設定都必須在 `from text.LangSegmenter import LangSegmenter` **之後**才生效——`langsegmenter.py:11` 會在 import 時把 `_default_detector` 重新指回壞路徑，順序寫反會靜默失效。
- **教訓**：計畫 §12.3 曾把「jieba_fast 未安裝」列為**僅影響中文**的已知偏差，實測證明它擋下整個推論管線，連日文都用不了。「lang 假設」要實際跑一次才能確認。

### 3.2 程式行為

#### PK-003 `#` 音素映射為 UNK
- **日期**：2026-10-04（G1，T1）
- **現象**：`2-name2text.txt` 中 451/467 行含 UNK，訓練集共 1776 個、eval10 共 35 個。
- **根因**：日文 G2P 的 `pyopenjtalk_g2p_prosody` 在 accent phrase 邊界插入 `#`（`text/japanese.py:248`），而 `symbols2.symbols`（732 符號）**不含 `#`**（`symbols2.py:783-788`），故 `#` 映射為 UNK。屬官方行為，非文字中的符號（`in_text` 來源為 0，全為 `prosody_insert`）。
- **處置**：不過濾。T1 證明推論端同一 `clean_text` 路徑（`TextPreprocessor.py:15`、`TTS.py:1171` 帶 v4）亦產生同樣 UNK，**訓練／推論兩端一致**，故無害。
- **證據等級**：【實測】＋【讀碼】
- **狀態**：已解（確認無害，不需處置）
- **證據**：`reports/G1_t0_t1_report.md` T1-1～T1-4、§3；`tools/t1_unk_analysis.py`
- **備註**：PA-9 門檻因此只計「非 # 來源」（見 PK-007）。

#### PK-004 官方腳本裸 `except` 吞錯，exit code 恆 0
- **日期**：2026-10-05（G2）
- **現象**：`1-get-text.py:102-103`、`2-get-hubert-wav32k.py:124-125`、`3-get-semantic.py:115-116` 皆為裸 `except` 印 traceback 後繼續；即使 467 檔全失敗，**結束碼仍是 0**。
- **根因**：官方預處理腳本的錯誤處理設計——逐項容錯、不中止，也不改結束碼。
- **處置**：不改官方腳本。確立原則：**所有驗收與冒煙測試不可只看 exit code，必須以「預期產出檔案數量與內容」為準**（本專案 PA／SM 檢查表即此設計）。`load_audio` 的 except 另會把真錯誤包裝成 `RuntimeError("音频加载失败")`，排查 ffmpeg 問題時先 `shutil.which('ffmpeg')`。
- **證據等級**：【讀碼】＋【實測】
- **狀態**：已解（對策確立）
- **證據**：`reports/G2_stage_a_report.md` §3.1、§6.1；`docs/07_finetune.md` §4.1

#### PK-005 `ja_userdic` 的 `user.dict` 從未生成
- **日期**：2026-10-04（G1）
- **現象**：`text/japanese.py:61-71` 應在首次載入時以 `pyopenjtalk.mecab_dict_index` 從 `userdict.csv` 建立 `user.dict`，但該區塊被 `try/except ... pass` 包住，G2P 跑完後 `ja_userdic\` 仍只有 `userdict.csv`（17 MB，`<GPT>` 追蹤檔）。
- **根因**：推測 `mecab_dict_index` 失敗被 `pass` 吞掉（同 PK-004 模式），G2P 實際使用 pyopenjtalk 內建詞典。未確認失敗具體原因。
- **處置**：觀察。目前訓練預處理與推論若都在本環境跑會一致；**風險是**若某次成功建立 user dict，G2P 結果可能改變，是重現性觀察點。
- **證據等級**：【實測】（觀察到缺失）＋【推斷】（根因）
- **狀態**：觀察中
- **證據**：`reports/G1_t0_t1_report.md` §5.1；`git status` 確認 `<GPT>` 無新增檔案

#### PK-010 推論權重目錄不存在 → `savee` 吞錯 → exit 0 卻沒有推論權重
- **日期**：2026-10-05（階段 C，04:57 首次重現）
- **現象**：S2 冒煙跑完 26 step、寫出 `logs_s2_v4_lora_32\G_233333333333.pth`、**結束碼 0**、無任何 Traceback，但 `SoVITS_weights_v4\` 下**沒有**推論權重。`train.log` 只有一行看起來像 INFO 的
  `saving ckpt ryza_v4_lora_smoke_e1:Traceback (most recent call last): ... FileNotFoundError: [Errno 2] No such file or directory: 'SoVITS_weights_v4/ryza_v4_lora_smoke_e1_s26_l32.pth'`。
- **根因**：三段式——
  1. `SoVITS_weights_v4`／`GPT_weights_v4` 這兩個推論權重目錄，在整個 repo 裡**只有 `webui.py:200-201` 會建立**；`config.py` 沒有任何 `os.makedirs`，訓練腳本也只 `os.makedirs(save_root)`（`s2_train_v3_lora.py:143`，那是 ckpt 目錄）。手動驅動繞過 webui，目錄就不存在。
  2. `process_ckpt.py:37` 的 `my_save2` 直接 `open(path, "wb")`，不做 `makedirs`。
  3. `process_ckpt.py:59-60` 的 `savee` 以裸 `except` 收尾並 `return traceback.format_exc()`；呼叫端 `s2_train_v3_lora.py:370-386` 只把回傳字串丟給 `logger.info`。**於是 traceback 被當成 INFO 訊息印出，訓練照常結束，exit code 維持 0。**
- **處置**：在自有的啟動腳本 `tools/run_lora_train.py` 加 preflight，於每輪訓練前建立 `SoVITS_weights_v4`／`GPT_weights_v4`（§4.1 明列允許寫入），並在 summary json 回報 `weight_dirs_created`。建立後同一份 config 重跑，`saving ckpt ...:Success.`，權重 75,550,062 bytes、檔頭 `04`。
- **證據等級**：【實測】＋【讀碼】
- **狀態**：已解
- **證據**：`logs/ryza_v4_lora_smoke/_archive_pre_stagec/train.log`（失敗現場）；`reports/G4_stage_c_report.md` SM-S2-5；`tools/run_lora_train.py` 的 `preflight()`
- **備註**：這是 PK-004「裸 `except` 吞錯、exit code 恆 0」的第二次實例，**同一個坑在訓練端與預處理端各中一次**，再次證明驗收必須看產出檔而不看結束碼。
- **教訓**：手動驅動官方腳本時，凡是把權重寫進「由 webui 建立」的目錄的路徑，都要自己補 `makedirs`；否則錯誤會被 INFO log 吞掉。

### 3.3 驗證腳本

#### PK-006 validator PA-10 讀檔未 strip 換行（false failure）
- **日期**：2026-10-05（G2）
- **現象**：`stage_a_validate.py` 的 PA-10 抽 5 行比對，train／eval10 皆 5/5「mismatch」。
- **根因**：validator 逐行讀 list 時保留行尾 `\n`，`text_normalize`→g2p 把結尾 `\n` 多轉一個 `.`，使重算音素比 name2text 多一個句點。
- **處置**：改為預載 list 時 `read().strip("\n").split("\n")`（與官方 `1-get-text.py:108` 讀法一致）。修正後 0 mismatch。**資料本身正確**，純驗證腳本 bug。
- **證據等級**：【實測】
- **狀態**：已解
- **證據**：`reports/smoke_test_stage_a.md` 區域 3；`tools/stage_a_validate.py:124-131`
- **備註**：同類陷阱——任何「重跑 G2P 並與儲存結果比對」的驗證都必須比照官方讀法 strip 換行。

#### PK-007 validator PA-9 門檻誤把 `#` UNK 計入（false failure）
- **日期**：2026-10-05（G2）
- **現象**：PA-9 train 判 `pass:false`（`affected_lines=451 > 10`）。
- **根因**：validator 門檻把所有含 UNK 的行算進「影響行數」，但計畫 §9.4 的停止條件是「**非 # 的 UNK** 總數 > 20 或影響行數 > 10」；`#` 是韻律邊界（PK-003），不應計入。
- **處置**：`affected` 只收集含**非 #** 來源 UNK 的行；`unk_symbols` 統計不變（仍記錄 `#` 供參）。修正後 train／eval10 皆 pass（非 # UNK = 0）。
- **證據等級**：【實測】
- **狀態**：已解
- **證據**：`reports/smoke_test_stage_a.md` 區域 3；`tools/stage_a_validate.py:121-156`

### 3.4 版控

#### PK-008 `<GPT>` 根目錄 `=0.4.1` 檔案消失
- **日期**：2026-10-05（G2 收尾發現）
- **現象**：G1 報告多次記載 `<GPT>` 工作樹「始終只有預先存在的 `?? =0.4.1`」，PA-12 亦期望「仍只有 =0.4.1」；G2 收尾時該檔**已不存在**，`git status --short` 全空。
- **根因**：待查。該檔從未入版控（`git log --all -- "=0.4.1"` 無結果），repo 僅單一 squashed commit `48b1a01`，reflog 只有 `clone` 一筆，**無任何軌跡可還原或查證刪除時間**。
- **處置**：PA-12 據實改記「工作樹全乾淨」。結果優於預期，但偏離計畫文字。
- **證據等級**：【缺失】
- **狀態**：無法判定
- **證據**：`reports/smoke_test_stage_a.md` 區域 4；`git log --all -- "=0.4.1"` 無結果
- **缺什麼**：刪除時間與執行者的任何記錄。

#### PK-009 `<RYZA>\tools\_ffmpeg_tmp\` 目錄被刪
- **日期**：2026-10-05（本 session 收尾發現）
- **現象**：對話開始時 git status 仍列 `?? tools/_ffmpeg_tmp/`（內含 114 MB `ffmpeg.zip` 與解壓目錄），收尾時已消失，且無任何 .gitignore 規則涵蓋。
- **根因**：待查。刪除發生在本 session 期間，無 git 軌跡可查。
- **處置**：觀察。ffmpeg 功能不受影響（`.venv\Scripts` 二進體仍在，md5 已驗證）；代價是**失去可重現的 zip 實體**，日後新環境重放需憑 `docs/07_finetune.md` 記錄的 gyan.dev URL 與 SHA256 重新下載（一次連網限定）。
- **證據等級**：【缺失】
- **狀態**：無法判定
- **證據**：`reports/smoke_test_stage_a.md` 區域 4 註；`git check-ignore tools/_ffmpeg_tmp` 無輸出
- **缺什麼**：刪除時間與執行者的任何記錄。
- **教訓**：下載型暫存若需保留作重現依據，應在文件建立時就明確標註保留策略，並避免放在 untracked 目錄被誤清。

### 3.5 GPU／CUDA
> 詳見 §4 GPU 不穩專區。

### 3.6 其他

#### PK-012 S1 的學習率排程器被上游硬寫成常數 0.002
- **日期**：2026-10-05（階段 C，SM-S1-5）
- **現象**：S1 冒煙只跑了 13 step，`warmup_steps=2000` 根本不可能走完，但進度棒與 TensorBoard 的 `lr_step` 顯示 **0.002**，而不是應有的約 7.5e-05。
- **根因**：`AR/modules/lr_schedulers.py` 的 `WarmupCosineLRSchedule` 被上游改寫過——
  - `set_lr()`（`:38-42`）直接 `g["lr"] = self.end_lr`，完全忽略傳入的 `lr`；
  - `step()`（`:58`）在算完 warmup/cosen 公式之後，又執行 `self.lr = lr = self.end_lr = 0.002  ###锁定用线性###不听话，直接锁定！`，把剛算出來的值覆蓋掉。
  故不論 `lr_init`／`lr`／`lr_end`／`warmup_steps`／`decay_steps` 怎麼設，**每一步實際套用到 optimizer 的學習率都是 0.002**。
- **處置**：不修改官方排程器（§4.2）。`tools/s1_lr_check.py` 以實測取代推論：分別模擬 13／472／1180／2000 step，確認四種長度的學習率軌跡都只有單一值 0.002，並與「若排程器正常」的理論值對照（2000 step 的理論值為 0.01）。**計畫 §11.3 SM-S1-5 的待驗證推論（「warmup_steps=2000 是否使正式訓練全程停留在 warmup」）答案是否定的，但原因不是 warmup 太長，而是排程器根本不會 warmup。**
- **證據等級**：【實測】＋【讀碼】
- **狀態**：已解（機制已查明，無需處置）
- **證據**：`reports/logs/raw/s1_lr_check.json`；`reports/G4_stage_c_report.md` SM-S1-5；`GPT_SoVITS/AR/modules/lr_schedulers.py:38-62`
- **教訓**：設定檔裡的 `optimizer.*` 五個欄位在 v4 官方實作中**全是無效設定**。日後若看到「學習率怎麼調都沒反應」，先確認排程器有沒有被鎖。

---

## 4. GPU 不穩專區

> 依 `docs/01_issue.md:55`：「訓練過程可穩定完成；若遇到既有的 GPU 不穩問題，須記錄處理方式。」本節專門記錄 GPU／CUDA 不穩的現象與處置，與 §3.5 分離維護。

**截至階段 C（2026-10-05）：本機（RTX 4070 SUPER 12 GB，12282 MiB，driver 591.86，torch 2.5.1+cu121）未發生任何 GPU 不穩。** 階段 A 全程 fp16（`is_half=1`）零 NaN retry、零 CUDA 錯誤；階段 C 五輪（S2 epochs=1、S2 resume epochs=2、S1 epochs=1、推論 E1、推論 E2）共 52 個 S2 step 與 13 個 S1 step，**未出現 `illegal instruction`、`CUBLAS_STATUS_INTERNAL_ERROR`、CUDA context 遺失、無故 OOM，loss 無 NaN／Inf**（S2 52 筆逐筆檢查、S1 epoch loss 3972.732 為有限值）。S2 整卡顯存峰值最高 11146 MiB（約 91%），仍在 12 GB 內，無 OOM。**本節目前無本機事件，以下唯一一筆是其他環境的過往異常，本機無需理會。**

### 冒煙實測顯存（階段 C，附 §11.2 SM-S2-7／§12.2 監測結果）

| 輪次 | 整卡峰值（nvidia-smi） | `max_memory_allocated` | `max_memory_reserved` |
|---|---|---|---|
| S2 epochs=1 | 10572 MiB | 8428.2 MiB | 8582.0 MiB |
| S2 resume epochs=2 | 11146 MiB | 8295.7 MiB | 9190.0 MiB |
| S1 epochs=1 | 2551 MiB（**取樣漏峰，見下**） | 3351.7 MiB | 4370.0 MiB |
| 推論 E1／E2 | — | 1898.4 MiB | 2264.0 MiB |

閒置基準實測 1734–1758 MiB（計畫 §11.1 寫的 1455 MiB 與本機不符，屬記錄誤差，非問題）。

⚠️ **nvidia-smi 每 5 秒取樣會漏掉短輪次的峰值**：S1 整輪僅 30.6 秒、實際訓練不到 2 秒，取樣點沒有落在顯存高點上，導致整卡讀值 2551 MiB **小於** torch 自己的 3351.7 MiB——這在物理上不可能，證明該讀數不可信。**短輪次一律以 `torch.cuda.max_memory_allocated()` 為準**（本次由 `tools/vram_runner.py` 以 `runpy` 載入官方腳本、於 `atexit` 印出，未修改任何官方檔案）。

### TF32 —— 其他環境過往異常，本機無需理會

> ⚠️ **本筆記錄的是其他環境（RTX 3060）的過往異常，非本機經驗。本機（RTX 4070 SUPER）從未發生，本地環境跑訓練時不需要理會 TF32，依官方預設設定即可。記錄僅為符合 issue「既有 GPU 不穩問題須記錄處理方式」之要求，並供未來若移地到其他環境時參考。**

- **來源記錄**：`tools/spk_model.py:14-19` — TF32 matmul **曾在其他環境（RTX 3060／當時 driver）造成 CUDA 記憶體損壞**（跑約 20–30 個檔後 `illegal instruction`／`CUBLAS_STATUS_INTERNAL_ERROR`），該環境關閉 TF32 後 3200 檔連續測試穩定。
- **本機狀態**：**未發生、不適用。** `spk_model.py` 關閉 TF32 僅沿用該其他環境的穩定設定，非本機已確認的問題；本機訓練腳本維持官方預設（`allow_tf32 = True`，`GPT_SoVITS/GPT_SoVITS/s2_train_v3_lora.py:44-45`、`s2_train.py:44-45`、`s2_train_v3.py:44-45`；推論端 `TTS_infer_pack/TTS.py:210` 為 `False`），**不 patch、不預防、不監控 TF32**。
- **處置（本機）**：無需處置。依官方原版設定訓練即可。
- **處置（若未來移地至其他環境，且發生 CUDA 損壞）**：於該環境的 `s2_train_v3_lora.py` TF32 設定之後補關閉（**保留 `cudnn.enabled` 開啟**，僅關 TF32 matmul 與 cudnn TF32；`spk_model.py` 連 `cudnn.enabled` 都關是推論專用做法，訓練不可照抄）。此為其他環境的應急方案，**本機不需 preemptive 套用**。
- **狀態**：本機不適用（過往其他環境經驗）
- **證據等級**：【推斷】（過往其他環境經驗，非本機實測）
- **證據**：`tools/spk_model.py:14-19`；`reports/01_env_model_prep.md`「TF32 記錄（不 patch）」；`reports/G1_t0_t1_report.md` §5.4

### 觸發本節記錄的條件

正式訓練（階段 D）或冒煙測試（階段 C）**在本機**出現以下任一徵兆時，才在此新增本機記錄（含發生 epoch／step、GPU 溫度與顯存占用、driver／torch 版本、當時 batch size）：

- `illegal instruction` 或 `CUBLAS_STATUS_INTERNAL_ERROR`
- CUDA context 遺失／`device-side assert`
- 無故 OOM（顯存需求與 batch size 不成比例）
- loss 突然 NaN 或爆炸（排除資料問題後）

**注意**：上列徵兆出現時，先依現場證據判定根因，**不要預設與 TF32 有關**——TF32 問題屬其他環境過往經驗，本機無此記錄。

---

## 5. 與前因／計畫不符處

1. **前因所列「transformers 拒載 .bin」**：非本次新發現，已於 G1（2026-10-04）以 T0 解決，本文件追填為 PK-002。
2. **前因所列「`#`→UNK」**：同為 G1 已確認無害，追填為 PK-003（狀態=已解，處置=不需處置）。
3. **前因未列的兩筆**：本文件補入 PK-005（`ja_userdic` user.dict 未生成，G1 風險項，觀察中）與 §4 的 TF32 記錄。後者依 PM 指示明確標註為**其他環境（RTX 3060）過往異常、本機無需理會**，狀態為「本機不適用」，不列為本機觀察項。

---

## 6. 階段 C 追加（2026-10-05）

計畫 §11／§12 與實際執行的落差，詳見 `reports/G4_stage_c_report.md` §4；本節只列與坑記錄直接相關者。

4. **計畫 §12.3 把「jieba_fast 未安裝」列為僅影響中文**——實測證明它擋下整個推論管線（見 PK-011）。已在本階段解決，但該文件的「已知偏差」章節需於階段 D 更正。
5. **計畫 §10.1 預期「S1 冒煙實際 batch 會被壓到 5」**——實測為 8。原因是 `AR/data/data_module.py:51` 的 `len(dataset)//4` 作用在**已自動擴充為 100** 的資料集上（100//4 = 25），不是 20 條（20//4 = 5）。兩階段的資料集都會先被擴充到至少 100（`data_utils.py:546-551`、`AR/data/dataset.py`），所以這個下限在正式訓練時同樣不會生效。
6. **計畫 §11.1 寫的顯存基準 1455 MiB** 與本機實測 1734–1758 MiB 不符，屬計畫側的記錄誤差，不影響任何判斷。
