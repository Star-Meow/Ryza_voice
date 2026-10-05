# 階段 C 回報：冒煙測試（S2 LoRA ＋ S1 ＋ 推論）

日期：2026-10-05
回報依據：計畫 §11 階段 C、§15 回報格式
分支：`feat/tts-v4-lora-prep`（`<RYZA>`）；`<GPT>` 在 `main` 分支（commit `48b1a01`）、工作樹乾淨
樣本：冒煙 exp `logs/ryza_v4_lora_smoke`（20 筆，§10.1）
工作目錄：CWD = `H:\git\GPT-SoVITS`，Python = `.venv\Scripts\python.exe`

> 冒煙明細（§12.1 格式：輪次對照表、loss 趨勢表、epoch 外推表、GPU 異常表）見 [smoke_test_stage_c.md](smoke_test_stage_c.md)。
> **正式訓練未啟動。**

---

## 1. 結果總表

| ID | 結論 | 證據等級 | 證據 | 與計畫是否相符 |
|---|---|---|---|---|
| SM-S2-1 | 結束碼 0；stderr 無 Traceback | 【實測】 | `s2_smoke_summary.json` `exit_code: 0`；`s2_smoke.err` 與 `s2_smoke.out` 的 `Traceback` 計數皆 0；stdout 只含 hparams、資料集統計與 26 筆 loss | 相符 |
| SM-S2-2 | 載入樣本數與預期相符，時長過濾未剔除任何一筆 | 【實測】 | stdout：`phoneme_data_len: 20`、`wav_data_len: 100`、`skipped_phone: 0 , skipped_dur: 0`、`total left: 100` | 相符（擴充 5× 後 100 筆，如 §10.1 預期） |
| SM-S2-3 | 有 loss 紀錄（log_interval=1）；無 NaN／Inf | 【實測】 | `train.log` 26 筆 `[loss, step, lr]`（step 0–25），首 `0.0216744449`、末 `0.0193039812`；52 筆（含輪次 2）逐筆檢查 NaN／Inf 皆 0 | 相符 |
| SM-S2-4 | `logs_s2_v4_lora_32\G_233333333333.pth` 存在 | 【實測】 | `logs/ryza_v4_lora_smoke/logs_s2_v4_lora_32/G_233333333333.pth`，1,845,612,582 bytes；log `Saving model and optimizer state at iteration 1` | 相符 |
| SM-S2-5 | `SoVITS_weights_v4` 下出現 LoRA 權重；檔頭 `04`；大小遠小於底模 | 【實測】 | `SoVITS_weights_v4/ryza_v4_lora_smoke_e1_s26_l32.pth` 75,550,062 bytes；前 2 bytes = `3034`（ASCII `04`）；`lora_rank=32`、`info="1epoch_26iteration"`、258 個 tensor、無 `enc_q`；對比底模 `s2Gv4.pth` 769,025,545 bytes（**佔 9.8%**）。log `saving ckpt ryza_v4_lora_smoke_e1:Success.` | 相符（**04:57 那次因權重目錄不存在而失敗，見 §4-1**） |
| SM-S2-6 | resume 從第 2 個 epoch 接續，並新增 `_e2_` 推論權重 | 【實測】 | stdout `start training from epoch 2`；`global_step` 由 26 續至 51（未歸零）；`saving ckpt ryza_v4_lora_smoke_e2:Success.`；新增 `ryza_v4_lora_smoke_e2_s52_l32.pth`（75,550,062 bytes、檔頭 `04`） | 相符 |
| SM-S2-7 | 顯存峰值（整張卡）、每 step 平均秒數已記錄 | 【實測】 | 整卡峰值 10572 MiB（`s2_smoke.gpu`）／11146 MiB（`s2_smoke_resume.gpu`）；`torch.cuda.max_memory_allocated()` 8428.2／8295.7 MiB；每 step 0.5425／0.5386 s（算法見 `smoke_test_stage_c.md` §1） | 相符（**額外補上 torch 層級顯存**，官方腳本本身無此輸出） |
| SM-S2-8 | 未出現 illegal instruction 或 CUBLAS 錯誤 | 【實測】 | 三份 log 全文檢索無 `illegal instruction`／`CUBLAS`；無 OOM；無 CUDA context 遺失 | 相符 |
| SM-S1-1 | 結束碼 0；無 Traceback；無 BERT 特徵時資料載入正常 | 【實測】 | `s1_smoke_summary.json` `exit_code: 0`；`s1_smoke.out`／`.err` 的 `Traceback` 計數皆 0；`3-bert` 為空目錄（0 檔）仍完成 13 step 訓練。補零邏輯在 `AR/data/dataset.py:223-237`（缺檔 → `bert_feature=None`）與 `:276-282`（collate 時 `zero_()` 補零） | 相符（**階段 B 列為「無法判定」者，本輪實測確認**） |
| SM-S1-2 | loss 無 NaN／Inf | 【實測】 | TensorBoard `total_loss_epoch` = 3972.732178（有限值）、`top_3_acc_epoch` = 0.49297；進度棒 `total_loss_step` = 1564.616 | 相符 |
| SM-S1-3 | `logs_s1_v4\ckpt` 有 checkpoint（只留最新）；`GPT_weights_v4\ryza_v4_lora_smoke-e*.ckpt` 存在、可 CPU 載入、頂層三鍵 | 【實測】 | `logs_s1_v4/ckpt/` 僅 `epoch=0-step=3.ckpt`（310,538,850 bytes，唯一檔）；`GPT_weights_v4/ryza_v4_lora_smoke-e1.ckpt`（155,312,957 bytes）以 `torch.load(map_location="cpu")` 載入成功，頂層鍵 = `["config", "info", "weight"]`、`info="GPT-e1"`、295 個 fp16 tensor | 相符 |
| SM-S1-4 | 顯存峰值、每 step 秒數、實際 batch size 已記錄 | 【實測】 | `max_memory_allocated` 3351.7 MiB（整卡讀值 2551 MiB 為 5 秒取樣漏峰，不可採信）；每 step 0.104 s（9.60 it/s）；**實際 batch 8**（13 step 中前 12 步為 8、最後 1 步為 4） | 相符（**實際 batch 與計畫 §10.1 預期的 5 不符，見 §4-3**） |
| SM-S1-5 | 學習率檢查完成；`warmup_steps=2000` 是否使正式訓練全程停留 warmup | 【實測】＋【讀碼】 | `reports/logs/raw/s1_lr_check.json`：13／472／1180／2000 step 四種長度的實際 LR **全為 0.002**（單一值）。`AR/modules/lr_schedulers.py:58` 硬寫 `self.lr = lr = self.end_lr = 0.002`，`:38-42` 的 `set_lr` 亦忽略傳入值 | **相符，但結論與推論相反**（見 §4-4） |
| SM-INF-1 | E1 無例外；BERT 與 HuBERT 能載入 | 【實測】 | `infer_base.log`：`Loading BERT weights from ...chinese-roberta-wwm-ext-large`、`Loading CNHuBERT weights from ...chinese-hubert-base` 均無例外；`infer_smoke.log` 同。**T0 safetensors 轉檔成果在推論端驗證通過** | 相符（**需先繞過兩個缺失依賴，見 §4-2**） |
| SM-INF-2 | 實際生效的權重路徑與預期相符（custom 陷阱檢查） | 【實測】 | E1：`[ACTUAL] t2s=s1v3.ckpt`、`vits=s2Gv4.pth`。E2：`[ACTUAL] t2s=GPT_weights_v4/ryza_v4_lora_smoke-e1.ckpt`、`vits=SoVITS_weights_v4/ryza_v4_lora_smoke_e2_s52_l32.pth`。兩輪 `custom 生效 = True`。陷阱機制確認於 `TTS.py:318` | 相符（**陷阱為真，必須傳 `{"custom": {...}}`**） |
| SM-INF-3 | 3 個目標文字都產生音訊；sr 48000；時長 0.5–20 s；無 NaN；非全靜音 | 【實測】 | 6 個 wav 全部 48000 Hz。E1：`base_01498` 1.640 s／RMS 0.109769、`base_00045` 4.480 s／RMS 0.110901、`base_00638` 7.200 s／RMS 0.121733。E2：`smoke_01498` 1.640 s／0.108040、`smoke_00045` 4.480 s／0.110746、`smoke_00638` 7.200 s／0.120973。全部 `nan=False` | 相符 |
| SM-INF-4 | `tts_infer.yaml` 前後 SHA256 已比對；被改動則還原並再比 | 【實測】 | 兩輪 `本輪是否改動 = True`（`save_configs()` 確實改寫）；已 `git checkout --` 單檔還原。最終 on-disk SHA256 = `b9481d69…5e9af5`（＝本階段起始基線），`git diff --quiet HEAD` = 一致。`infer_{base,smoke}_summary.json` 的 `tts_infer_yaml` 欄位 | 相符（**過程中曾因推論中途中斷而污染一次，已重跑兩輪取乾淨證據，見 §4-5**） |
| SM-INF-5 | `<GPT>` 工作樹仍只有 `=0.4.1` | 【實測】 | 全階段結束 `git -C <GPT> status --short` 為空。**實為「完全乾淨」，`=0.4.1` 已於階段 A 刪除（PK-008）** | 相符（優於預期：比計畫預期的更乾淨） |
| SC-1 | §11.5 epoch 外推表完成 | 【實測】 | `smoke_test_stage_c.md` §4：epochs 8／5／3 三列 ＋ S1 沿用官方 20 的對照列。8 epoch 合計約 **9.9 分鐘**，遠低於 8 小時，**不觸發自動降級** | 相符 |
| SC-2 | 正式 exp `ryza_v4_lora` 內無任何訓練產物（PE-3 前置） | 【實測】 | `logs/ryza_v4_lora/` 仍只有 `2-name2text*.txt`、`3-bert/`、`4-cnhubert/`、`5-wav32k/`、`6-name2semantic*.tsv`；無 `logs_s2_*`、無 `logs_s1_*` | 相符 |
| SC-3 | `<GPT>` 無任何官方追蹤檔被修改 | 【實測】 | `git status --short` 空、`git diff --stat` 空 | 相符 |
| SC-4 | 未執行任何正式訓練或預處理 | 【實測】 | 本階段僅新增：`<GPT>\SoVITS_weights_v4\`、`<GPT>\GPT_weights_v4\`（§4.1 允許寫入），以及 `logs/ryza_v4_lora_smoke/` 下的冒煙產物。正式 exp 無變動（見 SC-2） | 相符 |

**結論：SM-S2-1～8、SM-S1-1～5、SM-INF-1～5、SC-1～4 共 27 項全數通過。** 其中 5 項的實際行為與計畫預期不同（SM-S2-5、SM-S1-4、SM-S1-5、SM-INF-1、SM-INF-4），詳見 §4。

---

## 2. 產出檔案清單

### 2.1 `<RYZA>` 新增／修改（皆未 commit）

| 路徑 | 用途 |
|---|---|
| `tools/vram_runner.py` | 以 `runpy` 載入官方訓練腳本、`atexit` 印 `torch.cuda.max_memory_allocated()`／`max_memory_reserved()`。**官方腳本零修改** |
| `tools/infer_smoke.py` | 推論冒煙 E1／E2（SM-INF-1～5）；含 `jieba_fast` 別名與 `fast_langdetect` 本地模型兩處行程內設定 |
| `tools/stage_c_collect.py` | 彙整 S2 loss（`train.log`）與 S1 loss（TensorBoard events）、CPU 驗證 S1／S2 權重 → `reports/logs/raw/stage_c_metrics.json` |
| `tools/s1_lr_check.py` | SM-S1-5 學習率實測 → `reports/logs/raw/s1_lr_check.json` |
| `tools/run_lora_train.py`（改） | 新增 preflight 建立推論權重目錄、`--stall-timeout-min`（預設 15）、`--vram`；summary json 補 4 個欄位 |
| `configs/s2_lora_ryza_smoke_e2.json` | resume 輪用；**僅 `train.epochs` 1→2，其餘與 smoke config 逐字相同**（已 diff 確認） |
| `reports/smoke_test_stage_c.md` | 本階段冒煙明細（§12.1 格式） |
| `reports/G4_stage_c_report.md` | 本檔（§15 格式） |
| `docs/09_pitfalls.md`（改） | 新增 PK-010、PK-011、PK-012；§4 補階段 C 顯存實測與 TF32 監測結果；新增 §6 階段 C 與計畫落差 |

### 2.2 `<GPT>` 產出（全部落在 §4.1 允許清單內，且 `logs`、`SoVITS_weights*/`、`GPT_weights*/` 皆被 `.gitignore` 忽略）

| 路徑 | 內容 |
|---|---|
| `SoVITS_weights_v4/ryza_v4_lora_smoke_e1_s26_l32.pth` | 75,550,062 bytes，檔頭 `04` |
| `SoVITS_weights_v4/ryza_v4_lora_smoke_e2_s52_l32.pth` | 75,550,062 bytes，檔頭 `04` |
| `GPT_weights_v4/ryza_v4_lora_smoke-e1.ckpt` | 155,312,957 bytes，CPU 載入驗證通過 |
| `logs/ryza_v4_lora_smoke/logs_s2_v4_lora_32/G_233333333333.pth` | 1,845,612,582 bytes（epoch 2 覆寫後） |
| `logs/ryza_v4_lora_smoke/logs_s1_v4/ckpt/epoch=0-step=3.ckpt` | 310,538,850 bytes |
| `logs/ryza_v4_lora_smoke/infer/` | 6 個 wav（E1 3 個 ＋ E2 3 個） |
| `logs/ryza_v4_lora_smoke/_archive_pre_stagec/` | 04:57 那次失敗的 ckpt 與 `train.log`（**未刪除**，見 §4-1） |

### 2.3 `<RYZA>\reports\logs\raw\` 新增（`.gitignore` 忽略，不提交）

`s2_smoke.{out,err,gpu}`、`s2_smoke_summary.json`、`s2_smoke_resume.{out,err,gpu}`、`s2_smoke_resume_summary.json`、`s1_smoke.{out,err,gpu}`、`s1_smoke_summary.json`、`infer_base.log`、`infer_smoke.log`、`infer_{base,smoke}_summary.json`、`stage_c_metrics.json`、`s1_lr_check.json`。

---

## 3. 讀碼依據（本輪新增的關鍵行號）

| 事項 | 位置 | 說明 |
|---|---|---|
| 推論權重目錄只有 webui 會建立 | `webui.py:200-201` | `config.py` 全文無 `os.makedirs`；這是 SM-S2-5 失敗的根因 |
| `savee` 的裸 `except` | `process_ckpt.py:59-60` | `return traceback.format_exc()`，呼叫端只 `logger.info` → exit 0 |
| `my_save2` 不建目錄 | `process_ckpt.py:37` | `open(path, "wb")` 前無 `makedirs` |
| resume 分支 | `s2_train_v3_lora.py:177-189` | `utils.latest_checkpoint_path(save_root, "G_*.pth")`；`:190` 的裸 `except` 會靜默退回 pretrained |
| S2 樣本自動擴充 | `module/data_utils.py:546-551` | `min_num=100`，重複 `max(2, 100//leng)` 次 |
| S2 時長過濾 | `data_utils.py:588` | `if 54 > duration > 0.6` |
| S1 batch 下限 | `AR/data/data_module.py:51` | `max(min(batch_size, len(dataset)//4), 1)`，作用在**已擴充**的 dataset 上 |
| S1 BERT 缺檔補零 | `AR/data/dataset.py:223-237`、`:276-282` | 缺檔 → `bert_feature=None` → collate `zero_()` |
| S1 checkpoint 三鍵 | `s1_train.py:62-81` | `weight`（fp16）／`config`／`info` |
| `TTS.run` 是 generator | `TTS.py:997`（`@torch.no_grad()` ＋ 內含 `yield`） | 呼叫端須 `next()` 取 `(sr, audio)`，`api_v2.py:441` 即如此 |
| **custom 陷阱** | `TTS.py:318` | `self.configs = configs_.get("custom", configs_["v2"])`；頂層傳入的鍵被靜默忽略 |
| `tts_infer.yaml` 被改寫 | `TTS.py:384-385`、`:590`、`:597` | 每次 `init_vits_weights`／`init_t2s_weights` 都 `save_configs()` |
| `jieba_fast` 模組層級依賴 | `text/chinese.py:19-23`、`text/tone_sandhi.py:17`、`TTS_infer_pack/TextPreprocessor.py:13` | 日文推論也會被擋 |
| `fast_langdetect` 路徑 | `text/LangSegmenter/langsegmenter.py:11` | 自訂 `cache_dir` 不存在 → 轉而嘗試連網下載 |
| **S1 排程器硬寫常數** | `AR/modules/lr_schedulers.py:38-42`、`:58` | `self.lr = lr = self.end_lr = 0.002` |
| 全 repo 無顯存輸出 | `grep -rn max_memory_allocated`（排除 `.venv`）= 0 命中 | 故自建 `vram_runner.py` |

---

## 4. 與計畫不符或需 PM 決定的事項

### 4-1 SM-S2-5 首次失敗：推論權重目錄不存在（已自行修正，但根因值得記錄）

04:57（階段 C 開始前）的 S2 冒煙跑完 26 step、寫出 1.8 GB 的 resume checkpoint、**exit code 0**、無任何 Traceback，但 `SoVITS_weights_v4\` 下沒有推論權重。錯誤只以 INFO 級別出現在 `train.log`。根因與處置見 `docs/09_pitfalls.md` PK-010；已加 preflight 修正並重跑，SM-S2-5 現為通過。

**依 PM 指示，舊 ckpt 未刪除**，移至 `logs/ryza_v4_lora_smoke/_archive_pre_stagec/`（1.8 GB，該處不在 `G_*.pth` 的 glob 範圍內，不影響 resume 判定）。**清理清單見 §6。**

### 4-2 推論端缺兩個依賴，需 PM 裁示處理方式

計畫 §12.3 把「jieba_fast 未安裝」列為「僅影響中文」的已知偏差。**實測證明它擋下整個推論管線，連純日文推論都無法 import。** 另有一項計畫未預期的問題：`fast_langdetect` 因 `langsegmenter.py:11` 指定的 cache 目錄不存在，會轉而嘗試**從網路下載模型**，與 §4.2「不連網下載」衝突。

本次處置（**不安裝、不下載、不複製任何模型檔**）：在推論行程內把 `jieba_fast` 別名到已安裝的 `jieba`，並把 `fast_langdetect` 指向套件自己內附的 `resources/lid.176.ftz`。兩處都在 `tools/infer_smoke.py` 內，程式化、可重現。

**請 PM 裁示**：正式推論是否沿用同一設定（目前這是唯一可行解），或改為在 `requirements.txt` 補 `jieba_fast` 並預先放置 fast_langdetect 模型檔。兩者都不影響訓練。

### 4-3 S1 實際 batch 是 8，不是計畫預期的 5

§10.1 預期「20 條樣本時實際 batch 上限為 5」。實測為 **8**：`data_module.py:51` 的 `len(dataset)//4` 作用在**已自動擴充為 100** 的資料集上（100//4 = 25），`min(8, 25) = 8`。13 step/epoch 中前 12 步 batch 8、最後 1 步 batch 4。

**對正式訓練的影響**：467 筆同樣會先擴充（467 > 100，不擴充），`467//4 = 116 > 8`，故**正式訓練的實際 batch 就是設定值 8**。結論不變，但計畫 §10.1 的「batch 被壓低」機制在本專案**永遠不會生效**——§12.1 輪次表的「實際 batch」欄在正式訓練時可直接等同設定值。

### 4-4 SM-S1-5：排程器被上游鎖死，學習率恆為 0.002

計畫的待驗證推論是「`warmup_steps=2000` 是否使正式訓練全程停留在 warmup」。**答案是否定的，但原因不是 warmup 太長。** `AR/modules/lr_schedulers.py` 的 warmup／cosine 公式在 `:58` 被一行 `self.lr = lr = self.end_lr = 0.002` 覆蓋，`:38-42` 的 `set_lr` 也忽略傳入值。實測 13／472／1180／2000 step 的 LR 全為 0.002。

**後果**：`configs/s1_ryza.yaml` 的 `optimizer.{lr, lr_init, lr_end, warmup_steps, decay_steps}` **五個欄位全部無效**。因此「是否要調 warmup_steps」在 v4 官方實作下不成立。

**請 PM 裁示**：維持不動（唯一改法是修改官方排程器，違反 §4.2），或接受修改。本階段已依 §4.2 保持不動，僅記錄。

### 4-5 `tts_infer.yaml` 曾被推論中途中斷污染（已修正，證據已重取）

E1 首次執行的兩次嘗試分別因 `jieba_fast` 與 generator 解包錯誤而 crash，過程中 `save_configs()` 已改寫 `tts_infer.yaml` 但未走到還原步驟。導致後續 E1 的「前後 SHA256 一致」是**假通過**（起跑前就已髒）。已改為同時記錄 `git diff --quiet HEAD` 的結果並在起跑前警告，**兩輪重跑取乾淨證據**：兩輪皆 `本輪是否改動 = True` → 還原 → 最終 SHA256 = 起始基線且 `git diff --quiet HEAD` 一致。

**教訓（已寫入 PK-010 同類思路）**：驗收腳本若中途失敗，磁碟副作用會污染下一輪的證據；恢復步驟必須放在 `finally` 或由外部補救。

### 4-6 需 PM 決定：S1 的 epochs

§11.5 要求在沿用官方預設 20 時另列對照列（已列：合計約 11.5 分鐘）。8 與 20 的時間差僅約 1.7 分鐘，對 8 小時上限無影響。本報告與 `smoke_test_stage_c.md` 均**建議 8**（與 S2 對齊、降低 GPT 端過擬合風險），**最終由 PM 決定**。

### 4-7 記錄誤差（非問題）

計畫 §11.1 寫的顯存基準 1455 MiB，本機實測閒置 1734–1758 MiB。

---

## 5. 無法判定的項目

- **S1 逐步 loss 無法取得**：Lightning module 的 `self.log` 未設 `on_step=True`，TensorBoard 只有 4 個 epoch 層 tag；stdout 進度棒只保留最後一步的 `total_loss_step`／`lr_step`。§12.4 允許的兩種來源都拿不到逐步值。如實記錄最後可得的值，未推測中間值。
- **S1 正式訓練的每 step 秒數**：0.104 s 來自 Lightning 自報的 `9.60it/s`，**不含資料載入等待**。冒煙只有 100 筆（且是 20 筆重複），正式 467 筆時是否轉為 I/O bound（`num_workers=4` ＋ `persistent_workers=True`）需實際跑才知道。§4 的 epoch 表因此對 S1 較不確定。
- **E1／E2 的整輪耗時未記錄**：本階段未對推論輪計時（SM-INF-1～5 不含此項），故輪次表該欄留白，未以推估填補。
- **`docs/09_pitfalls.md`／`reports/smoke_test_stage_a.md` 的既有內容**：非本階段產出，本階段只做追加（PK-010～012、§4、§6），未逐條核對原有內容是否需更正。

---

## 6. 風險與意外發現

1. **S2 顯存餘裕比預期薄**：`docs/07_finetune.md` §5 寫「12 GB 有餘裕」，實測峰值 11146 MiB ≈ 整卡 **91%**，餘裕約 9%。官方 8 GB 門檻已超過。正式訓練若出現記憶體壓力，降 batch（S2 4→2）是第一手段，但每 step 秒數會上升，需重算 §4 的外推。
2. **nvidia-smi 的 5 秒取樣在短輪次會漏峰**：S1 的整卡讀值（2551 MiB）小於 torch 自報值（3351.7 MiB），物理上不可能。短輪次的顯存結論一律以 `torch.cuda.max_memory_allocated()` 為準。
3. **手動驅動官方腳本有兩處「webui 隱性前提」**：`SoVITS_weights_v4`／`GPT_weights_v4` 由 webui 建立（PK-010）、`fast_langdetect` 模型路徑由 langsegmenter 指定（PK-011）。兩者都會在離開 webui 後以難以察覺的方式失敗（前者被 INFO log 吞掉，後者轉為連網下載）。
4. **官方 `latest_checkpoint_path` 的排序用整個絕對路徑的數字**（`utils.py:114`），不是檔名。本輪只有一個 `G_*.pth` 故無影響；若日後同一目錄同時存在 `G_<step>.pth` 與 `G_233333333333.pth`，resume 可能挑錯。
5. **`=0.4.1` 已不存在**（PK-008），故 SM-INF-5／PE-10 的「工作樹僅 =0.4.1」實際驗證為「完全乾淨」。
6. **`<RYZA>` 工作樹在本階段執行期間被其他程序改動（非本階段產出）**：階段 C 執行期間（05:43–05:48）出現 `README.md` 大幅改寫（97 增／439 刪）、`reports/NIGHTLY_STATUS.md` 與 `reports/NIGHTLY_SUMMARY.md` 被刪除，以及新增 `docs/speaker_id.md`、`docs/tools.md`、`docs/transcript_pipeline.md`、`docs/unpack.md`。這些**不是本階段建立的檔案，本階段亦未修改 `README.md`**，推測為 PM 或另一個 session 在平行作業。**已原樣保留、未還原、未 commit**；本報告所有關於「未修改」的判定僅涵蓋本階段自身的產出與 `<GPT>`。若這些改動非預期，請 PM 自行確認來源。

### 冒煙產物清理清單（依 PE-11 提交，Agent 不自行刪除）

| 項目 | 路徑 | 大小 | 建議 |
|---|---|---|---|
| 舊失敗 ckpt | `logs/ryza_v4_lora_smoke/_archive_pre_stagec/G_233333333333.pth` | 1.8 GB | 可刪（已無用途，且是 1.8 GB） |
| 舊失敗 log | `logs/ryza_v4_lora_smoke/_archive_pre_stagec/train.log` | 6.6 KB | 建議保留（PK-010 的現場證據） |
| 冒煙 resume ckpt | `logs/ryza_v4_lora_smoke/logs_s2_v4_lora_32/G_233333333333.pth` | 1.8 GB | 可刪（正式訓練不會用此目錄） |
| S1 ckpt | `logs/ryza_v4_lora_smoke/logs_s1_v4/ckpt/epoch=0-step=3.ckpt` | 310 MB | 可刪 |
| 推論權重 | `SoVITS_weights_v4/ryza_v4_lora_smoke_e{1,2}_*.pth` | 各 75 MB | 可刪 |
| 推論權重 | `GPT_weights_v4/ryza_v4_lora_smoke-e1.ckpt` | 155 MB | 可刪 |
| 冒煙音訊 | `logs/ryza_v4_lora_smoke/infer/` | 2.5 MB | 建議保留（E1/E2 對照） |
| 冒煙 exp 資料 | `logs/ryza_v4_lora_smoke/{2-name2text*,6-name2semantic*,4-cnhubert,5-wav32k,3-bert}` | 約 8 MB | 建議保留（§13 PE-4 的重跑依據） |

合計可回收約 **2.3 GB**。**以上均未刪除，等 PM 指示。**

---

## 7. 停止點

**停在階段 C 完成，等 PM 驗收。**

本階段未啟動任何正式訓練；`<GPT>` 工作樹乾淨、無任何官方追蹤檔被修改；`<RYZA>` 新增的 `configs\`、`tools\`、`reports\`、`docs/09_pitfalls.md` 皆未 commit（§12.5 的 commit 屬階段 D，且前序指示為不 commit）。

**§14 停止條件全部未觸發**：無 OOM、無 loss NaN／Inf、無 illegal instruction／CUBLAS、無 15 分鐘無 log、推論 E1 未失敗。

待 PM 裁示：§4-2（推論依賴處理方式）、§4-4（S1 學習率）、§4-6（S1 epochs）、§6（冒煙產物清理）。