# 階段 B 回報：訓練配置與啟動腳本

日期：2026-10-05
回報依據：計畫 §10、§15 回報格式
分支：`feat/tts-v4-lora-prep`（`<RYZA>`）；`<GPT>` 無分支、無修改

---

## 1. 結果總表

| ID | 結論 | 證據等級 | 證據 | 與計畫是否相符 |
|---|---|---|---|---|
| SB-1 | S2 config 7 欄位齊全，`version="v4"` | 【讀碼＋實測】 | `configs/s2_lora_ryza.json`；selftest 17 項 S2 檢查全 PASS（model.version／頂層 version 皆 v4、lora_rank 32、grad_ckpt false、name、save_weight_dir、pretrained_s2G、pretrained_s2D、exp_dir、s2_ckpt_dir、gpu_numbers） | 相符 |
| SB-2 | S2 四項已定案參數已覆蓋 | 【實測】 | batch_size 4、log_interval 20、if_save_latest 1、save_every_epoch 1（selftest 逐項 PASS） | 相符 |
| SB-3 | S1 yaml 8 欄位齊全，路徑指向階段 A 產出 | 【讀碼＋實測】 | `configs/s1_ryza.yaml`；selftest 12 項 S1 檢查全 PASS；`train_semantic_path`／`train_phoneme_path`／`output_dir`／`pretrained_s1` 皆頂層 | 相符 |
| SB-4 | S1 覆蓋 batch_size=8、if_save_latest=1 | 【實測】 | selftest PASS（其餘維持官方 s1longer-v2.yaml 預設） | 相符 |
| SB-5 | 啟動腳本環境變數與 CWD 正確 | 【實測】 | selftest：PYTHONUTF8=1、PYTHONIOENCODING=utf-8、PYTHONPATH 兩路徑、version=v4、is_half=True、hz=25hz、_CUDA_VISIBLE_DEVICES=0、`.venv\Scripts` 在 PATH；子行程 `os.getcwd()` = `H:\git\GPT-SoVITS` | 相符 |
| SB-6 | stdout/stderr 分檔、結束碼記錄 | 【實測】 | selftest 以小程式（輸出一字元、exit 3）驗證：STDOUT_MARKER 只在 .out、STDERR_MARKER 只在 .err、結束碼 3 正確捕捉、兩檔不同路徑 | 相符（§10.4 全部要求） |
| SB-7 | 8 小時 watchdog 已內建且參數可設定 | 【讀碼】 | `run_lora_train.py` watchdog thread：`--watchdog-hours`（預設 8.0，0=關閉）；到點 `proc.terminate()`，不續跑、不自動刪除任何檔案。冒煙不驗證其觸發（依 §2.1 裁定） | 相符 |
| SB-8 | 顯存取樣已內建 | 【讀碼】 | `sample_gpu()`：每 5 秒 nvidia-smi 查詢整張卡 memory.used／total，寫入 `<tag>.gpu`；記錄峰值（`--no-gpu-sample` 可關閉） | 相符（§11.1） |
| SB-9 | 未執行任何訓練或預處理 | 【實測】 | selftest 只跑輸出一字元的小程式；無任何 logs_s2_*／logs_s1_*／G_*.pth 產出；`logs\ryza_v4_lora\` 仍只有階段 A 產出 | 相符（§4.2 硬性邊界） |
| SB-10 | 未修改任何官方追蹤檔 | 【實測】 | `<GPT>` `git status --short` 全空、`git diff --stat` 全空 | 相符 |

**結論：SB-1～SB-10 全數通過。**

---

## 2. 產出檔案清單

| 路徑 | 用途 |
|---|---|
| `configs\s2_lora_ryza.json` | S2 LoRA 訓練設定（基底官方 `s2.json` ＋ open1Ba 注入的 7 欄位 ＋ 4 項定案覆蓋） |
| `configs\s1_ryza.yaml` | S1 訓練設定（基底官方 `s1longer-v2.yaml` ＋ open1Bb 注入的 8 項欄位 ＋ 2 項定案覆蓋） |
| `tools\run_lora_train.py` | 啟動腳本：`--stage s2`／`--stage s1`、`--config`、`--tag`、`--watchdog-hours`、`--no-gpu-sample` |
| `tools\run_lora_train_selftest.py` | 無害驗證（47 項檢查，不跑訓練） |

**使用方式**：

```bash
cd <GPT>
# 正式訓練（8 小時 watchdog 預設開啟）
python <RYZA>/tools/run_lora_train.py --stage s2
python <RYZA>/tools/run_lora_train.py --stage s1
# 冒煙（階段 C 會用獨立 config 與 --tag smoke，watchdog 關閉）
python <RYZA>/tools/run_lora_train.py --stage s2 --config <RYZA>/configs/s2_lora_ryza_smoke.json --tag s2_smoke --watchdog-hours 0
```

---

## 3. 讀碼依據（欄位為何這樣填）

### S2（`s2_train_v3_lora.py` ＋ `webui.py:489-560` open1Ba）

| config 位置 | 讀取處 | 說明 |
|---|---|---|
| `train.batch_size` | `s2_train_v3_lora.py:101` | 定案 4 |
| `train.log_interval` | `:330` `global_step % hps.train.log_interval` | 定案 20 |
| `train.epochs` | `:229` `range(epoch_str, hps.train.epochs + 1)` | 8（§2.1，冒煙後可降） |
| `train.if_save_latest` | `:345` | 定案 1（固定檔名 G_233333333333.pth，每 epoch 覆寫） |
| `train.if_save_every_weights` | `:361` `== True`（須為 bool） | true |
| `train.save_every_epoch` | `:344` `epoch % hps.train.save_every_epoch` | **官方 `s2.json` 無此欄，open1Ba 才注入**；`s2_train_v3_lora.py:344` 直接讀取，缺欄會 AttributeError，故必補 1 |
| `train.lora_rank` | `:142,144` `hps.train.lora_rank` | 32 |
| `train.grad_ckpt` | `:319` `use_grad_ckpt=hps.train.grad_ckpt` | false（官方 `s2.json` 已有） |
| `train.pretrained_s2G` | `:196-205` 載入底模 | `GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth`（存在） |
| `train.pretrained_s2D` | **不被 `s2_train_v3_lora.py` 讀取**（只有 webui 寫入它） | 官方 v4 預設路徑 `.../s2Dv4.pth`；**該檔不存在**，webui 的 `check_pretrained_is_exist` 對 `s2Dv3`／`s2Dv4` 特別豁免（`webui.py:177`），LoRA 訓練只訓 G 不訓 D，故無害 |
| `train.gpu_numbers` | `:9` `hps.train.gpu_numbers.replace("-", ",")` | "0" |
| `model.version` | `:96-97` 選 V4 的 DataLoader／Collate | **"v4"（硬性）** |
| `data.exp_dir`、`s2_ckpt_dir` | `:78,81,142`；`:142` save_root = `{exp_dir}/logs_s2_{version}_lora_{lora_rank}` | 皆 `logs/ryza_v4_lora`（exp_root="logs" 相對路徑，**CWD 必須是 `<GPT>`**，`config.py:137`） |
| `save_weight_dir` | `:375` 推論權重輸出目錄 | `SoVITS_weights_v4`（`config.py:60-67`） |
| `name` | `:375,379` 推論權重檔名前綴 `{name}_e{epoch}_s{step}_l{lora_rank}` | `ryza_v4_lora` |

### S1（`s1_train.py` ＋ `webui.py:590-650` open1Bb）

| yaml 位置 | 讀取處 | 說明 |
|---|---|---|
| `train.batch_size` | `s1_train.py:101` 經 Trainer | 定案 8 |
| `train.epochs` | `:112` `max_epochs` | 20（官方預設；是否沿用由 PM 決定，§10.3） |
| `train.save_every_n_epoch` | `:105` `every_n_epochs` | 1（官方已有） |
| `train.if_save_latest` | `:40,52-58` my_model_ckpt：存新 ckpt 後刪舊的 | 定案 1 |
| `train.if_save_every_weights` | `:62-84` 每 epoch 存 `{half_weights_save_dir}/{exp_name}-e{epoch}.ckpt`（半精度） | true |
| `train.half_weights_save_dir` | `:77` | `GPT_weights_v4`（`config.py:68-75`） |
| `train.exp_name` | `:78` 檔名前綴 | `ryza_v4_lora` |
| `train.if_dpo` | `t2s_lightning_module.py:45` `config["train"].get("if_dpo", False)` | **`s1_train.py` 不讀，但 lightning module 讀**；open1Bb 會注入，故補 false |
| `pretrained_s1`（頂層） | `t2s_lightning_module.py` 經 config 載入 | `GPT_SoVITS/pretrained_models/s1v3.ckpt`（`config.py:25` v4 沿用 v3） |
| `train_semantic_path`（頂層） | `s1_train.py:134` | `logs/ryza_v4_lora/6-name2semantic.tsv` |
| `train_phoneme_path`（頂層） | `:135` | `logs/ryza_v4_lora/2-name2text.txt` |
| `output_dir`（頂層） | `:88-91`；`output_dir/ckpt` 為 checkpoint 目錄 | `logs/ryza_v4_lora/logs_s1_v4` |
| `optimizer.warmup_steps` | `t2s_lightning_module.py:142` | 2000（官方預設，SM-S1-5 學習率檢查會用到） |

---

## 4. 與計畫不符處

1. **`save_every_epoch` 官方 `s2.json` 原本沒有此欄**：`s2_train_v3_lora.py:344` 直接讀 `hps.train.save_every_epoch`，是 webui open1Ba 才注入的欄位（`webui.py:528`）。手動跑若不補會 AttributeError。已補 1。這在計畫 §10.2 的「需設定欄位」表中未列出，屬補充發現。
2. **`pretrained_s2D` 指向不存在的 `s2Dv4.pth`**：任務指示「指向對應 D 模型」。事實是官方 v4 發行包**只附 G 不附 D**（`gsv-v4-pretrained\` 只有 `s2Gv4.pth` 與 `vocoder.pth`），webui 對 v4 的預設值就是 `s2Dv4.pth` 並在 `check_pretrained_is_exist` 特別豁免（`webui.py:177`：`if "s2Dv3" not in i and "s2Dv4" not in i`）。`s2_train_v3_lora.py` 從頭到尾不讀 `pretrained_s2D`（grep 僅命中註解 `:47`）。**結論：填官方預設路徑、無害**，與計畫 §10.2「⚠️ 讀碼確認 v4 LoRA 腳本是否實際使用；未使用則記錄」的推斷一致——**確認未使用**。
3. **`if_save_every_weights` 官方 `s2.json` 無此欄**：同為 open1Ba 注入（`webui.py:527`），`:361` 以 `== True` 判斷，故須為 bool `true`。已補。
4. **`train.if_dpo` 的讀取點不在 `s1_train.py`**：計畫 §10.3 ⚠️ 要求確認。事實：`s1_train.py` 本身不讀，但 `GPT_SoVITS/AR/models/t2s_lightning_module.py:45` 讀 `config["train"].get("if_dpo", False)`——**腳本確實會讀**，open1Bb 也注入（`webui.py:622`）。已補 false（官方 webui 預設行為）。
5. **`docs/09_pitfalls.md` 與 `reports/smoke_test_stage_a.md` 已由前序對話建立**：階段 B 開始前才發現，其內容我未逐一核對。本次未修改它們。

---

## 5. 無法判定項目

- **`s1_train.py` 是否會在 `output_dir` 缺 BERT 特徵時自動補零**：`s1_train.py:134-135` 只讀 semantic／phoneme 兩個路徑，未直接讀 `3-bert`；BERT 特徵的補零邏輯在 `AR/data` 的 dataset 類別內（本階段未深讀）。SM-S1-1 要求「無 BERT 特徵時資料載入正常（缺檔自動補零）」，將在階段 C 實測確認。
- **S1 checkpoint「頂層含權重、設定與資訊三類鍵」的實際內容**：`my_model_ckpt` 存 `weight`／`config`／`info` 三鍵（`s1_train.py:68-80`），SM-S1-3 會在冒煙後以 CPU 載入實證。
- **`docs/09_pitfalls.md`（14.6 KB）與 `reports/smoke_test_stage_a.md` 的完整內容**：非本階段產出，未核對是否需補階段 B 內容。

---

## 6. 停止點

**停在階段 B 驗收前，等 PM 放行進階段 C（冒煙測試）。**

本階段未執行任何訓練或預處理，未修改任何官方追蹤檔（`<GPT>` `git status` 全空）。`<RYZA>` 新增 `configs\`、`tools\run_lora_train.py`、`tools\run_lora_train_selftest.py`，皆未 commit（依指示）。
