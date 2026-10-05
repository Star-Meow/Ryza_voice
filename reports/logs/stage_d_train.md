# 階段 D 訓練 log 歸檔

日期：2026-10-05
分支：`feature/TTS-finetune`
格式：依計畫 §12.4（指令與參數、起訖時間、結束碼、關鍵輸出摘要、完整 log 的關鍵段落、顯存峰值）

> 原始 log（`*.out`／`*.err`／`*.gpu`／`*_summary.json`）在 `reports/logs/raw/`，受 `.gitignore` 忽略，依 §12.4 不提交。
> 模型權重同樣不提交。

---

## 1. 執行環境

| 項目 | 值 |
|---|---|
| 工作目錄 | `H:\git\GPT-SoVITS`（`exp_dir` 為相對路徑，CWD 必須在此） |
| Python | `H:\git\GPT-SoVITS\.venv\Scripts\python.exe`（3.10.11、torch 2.5.1+cu121、numpy 1.26.4） |
| 驅動 | `tools/run_lora_train.py`（階段 B 產物，階段 D 沿用） |
| 環境變數 | `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`、`PYTHONPATH=<GPT>;<GPT>\GPT_SoVITS`、`version=v4`、`is_half=True`、`hz=25hz`、`_CUDA_VISIBLE_DEVICES=0`、`.venv\Scripts` 在 PATH |
| GPU | RTX 4070 SUPER，12282 MiB，driver 591.86 |
| 顯存量測 | `--vram` 經 `tools/vram_runner.py` 於 `atexit` 印 `torch.cuda.max_memory_allocated()`（官方腳本本身無此輸出，未修改官方檔）；另每 5 秒以 nvidia-smi 取整卡用量 |
| watchdog | 8 小時上限 ＋ 停滯 watchdog |

---

## 2. 各輪指令與結果

### 2.1 第 1 輪 — S1，8 epoch

```bash
python <RYZA>/tools/run_lora_train.py --stage s1 \
  --config <RYZA>/configs/s1_ryza.yaml --tag s1_train --vram
```

| 項目 | 值 |
|---|---|
| 起訖 | 2026-10-05 07:16:33 → 07:18:11 |
| 耗時 | 98.5 s |
| 結束碼 | **0** |
| 整卡峰值 | 10218 MiB（5 秒取樣，**短輪次會漏峰，不採信**） |
| `max_memory_allocated` | 4642.3 MiB |
| `max_memory_reserved` | 8294.0 MiB |
| 產出 | `GPT_weights_v4/ryza_v4_lora-e1..e8.ckpt`（各 155,312,957 B）、`logs/ryza_v4_lora/logs_s1_v4/ckpt/epoch=7-step=112.ckpt`（931,419,556 B，`if_save_latest=1` 只留最新） |

關鍵輸出：

```text
dataset.__len__(): 467
Epoch 7/7  ---------------- 59/59
total_loss_epoch: 4056.58 → 2822.74   top_3_acc_epoch: 0.4629 → 0.6280
```

### 2.2 第 2 輪 — S2，grad_ckpt=**false**，8 epoch（**崩潰**）

```bash
python <RYZA>/tools/run_lora_train.py --stage s2 \
  --config <RYZA>/configs/s2_lora_ryza.json --tag s2_train --vram
```

| 項目 | 值 |
|---|---|
| 起訖 | 2026-10-05 07:18:36 → 07:38:29 |
| 耗時 | 1193.5 s（強制終止） |
| 結束碼 | **4294967295**（−1，被 `Stop-Process -Force` 終止） |
| 整卡峰值 | **11958 MiB / 12282 MiB = 97.3%** |
| 產出 | epoch 1 正常：`SoVITS_weights_v4/ryza_v4_lora_e1_s119_l32.pth`（75,550,062 B）、`G_233333333333.pth`（1,845,612,582 B） |

崩潰徵兆（完整判斷過程見 `reports/finetune_report.md` §3）：

```text
# epoch 1 正常結束
INFO:ryza_v4_lora:saving ckpt ryza_v4_lora_e1:Success.
INFO:ryza_v4_lora:====> Epoch: 1
# epoch 2 進行到 46/119 後停滯，最後一次 loss 記錄停在 07:20:34（step 160）
2026-10-05 07:20:34,724	ryza_v4_lora	INFO	[0.03128746524453163, 160, 9.99750015625e-05]
# tqdm 於 13:42 時的狀態（正常應為 0.67 s/step）
 41%|███▉      | 49/119 [13:42<2:15:34, 116.20s/it]
```

同時觀測到：GPU utilization 持續 100%、顯存讀值平穩在 11905–11958 MiB、行程 CPU 時間持續上升（約 200%）、`.out`／`train.log`／`events.out.tfevents` 連續 15 分鐘零增長。**全程未拋出 CUDA OOM 例外**——是 caching allocator 在接近上限時反覆 `cudaFree`／重試所致的假死，非死鎖。

### 2.3 第 3 輪 — S2，grad_ckpt=**true**，8 epoch

```bash
python <RYZA>/tools/run_lora_train.py --stage s2 \
  --config <RYZA>/configs/s2_lora_ryza.json --tag s2_train_gckpt --vram --stall-timeout-min 5
```

| 項目 | 值 |
|---|---|
| 起訖 | 2026-10-05 07:38:59 → 07:47:02 |
| 耗時 | 482.9 s |
| 結束碼 | **0** |
| 整卡峰值 | **4603 MiB（降 61.5%）** |
| `max_memory_allocated` | 3507.5 MiB |
| 產出 | `ryza_v4_lora_e1..e8_s*_l32.pth` 全數 `Success.`、`G_233333333333.pth` |

```text
phoneme_data_len: 467
wav_data_len: 467
skipped_phone:  0 , skipped_dur:  0
total left:  467
INFO:ryza_v4_lora:saving ckpt ryza_v4_lora_e8:Success.
INFO:ryza_v4_lora:====> Epoch: 8
```

### 2.4 第 4 輪 — S1 續訓 8 → 16

```bash
python <RYZA>/tools/run_lora_train.py --stage s1 \
  --config <RYZA>/configs/s1_ryza_e16.yaml --tag s1_train_e16 --vram
```

| 項目 | 值 |
|---|---|
| 起訖 | 2026-10-05 07:53:28 → 07:54:53 |
| 耗時 | 85.8 s |
| 結束碼 | **0** |
| `max_memory_allocated` | 4705.3 MiB |

由 `logs_s1_v4/ckpt/epoch=7-step=112.ckpt` resume。

### 2.5 第 5 輪 — S2 續訓 8 → 16

```bash
python <RYZA>/tools/run_lora_train.py --stage s2 \
  --config <RYZA>/configs/s2_lora_ryza_e16.json --tag s2_train_e16 --vram --stall-timeout-min 6
```

| 項目 | 值 |
|---|---|
| 起訖 | 2026-10-05 07:55:10 → 08:02:35 |
| 耗時 | 445.0 s |
| 結束碼 | **0** |
| 整卡峰值 | 5093 MiB |
| `max_memory_allocated` | 3504.3 MiB |

```text
INFO:ryza_v4_lora:start training from epoch 9      ← 確認為 resume，非重頭
INFO:ryza_v4_lora:[0.02138558216392994, 1040, 9.978771236724554e-05]   ← global_step 由 952 續到 1904
INFO:ryza_v4_lora:saving ckpt ryza_v4_lora_e16:Success.
```

> `configs/s1_ryza_e16.yaml` 與 `configs/s2_lora_ryza_e16.json` 為當時的探測設定檔，驗收後已刪除；16 epoch 的設定已寫回 `configs/s1_ryza.yaml` 與 `configs/s2_lora_ryza.json`。

---

## 3. Loss 摘錄

### 3.1 S1（16 epoch，59 step/epoch）

來源：`<GPT>\logs\ryza_v4_lora\logs_s1_v4\logs_s1_v4\version_*`（TensorBoard events）

| epoch | total_loss_epoch | top_3_acc_epoch | lr_epoch |
|---|---|---|---|
| 1 | 4056.58 | 0.4629 | 0.002 |
| 4 | 3358.75 | 0.5432 | 0.002 |
| 8 | 2822.74 | 0.6280 | 0.002 |
| 12 | 2345.79 | 0.7110 | 0.002 |
| 16 | **1876.81** | **0.7948** | 0.002 |

**−53.7%，至第 16 epoch 尚未收斂。** `lr_epoch` 恆為 0.002，與階段 C SM-S1-5 發現的排程器鎖死一致。

### 3.2 S2（16 epoch，119 step/epoch，log_interval=20）

來源：`<GPT>\logs\ryza_v4_lora\train.log`，格式 `[loss, global_step, learning_rate]`

```text
2026-10-05 07:39:10,355	ryza_v4_lora	INFO	[0.0003603532..., 0, 9.99875e-05]
...
2026-10-05 08:02:26,...	ryza_v4_lora	INFO	[0.0..., 1900, 9.96...e-05]
```

共 96 筆，**NaN／Inf = 0 筆**。

⚠️ 每 epoch 僅 6 個取樣點，且每點是單一 step、隨機 0.64 秒片段的損失，**不是累計平均**。此粒度不足以判斷收斂，詳見 `reports/finetune_report.md` §2.4。

---

## 4. 顯存峰值彙總

| 輪 | 階段 | 整卡峰值（nvidia-smi） | `max_memory_allocated` | `max_memory_reserved` |
|---|---|---|---|---|
| 1 | S1 8ep | 10218 MiB（取樣漏峰） | 4642.3 MiB | 8294.0 MiB |
| 2 | S2 8ep grad_ckpt=false | **11958 MiB** | — | — |
| 3 | S2 8ep grad_ckpt=true | 4603 MiB | 3507.5 MiB | 3928.0 MiB |
| 4 | S1 →16 | 6465 MiB（取樣漏峰） | 4705.3 MiB | 5818.0 MiB |
| 5 | S2 →16 | 5093 MiB | 3504.3 MiB | 4420.0 MiB |
| — | 推論（20 句） | — | 見 `stage_d_infer_summary_e16.json` | — |

---

## 5. 產出權重

| 檔案 | 大小 | 說明 |
|---|---|---|
| `GPT_weights_v4/ryza_v4_lora-e16.ckpt` | 155,312,893 B | **交付用** S1 推論權重（fp16，頂層 `weight`/`config`/`info`） |
| `SoVITS_weights_v4/ryza_v4_lora_e16_s1904_l32.pth` | 75,550,062 B | **交付用** S2 LoRA 權重（檔頭 `04`、`lora_rank=32`） |
| `logs/ryza_v4_lora/logs_s2_v4_lora_32/G_233333333333.pth` | 1,845,612,582 B | resume 用 checkpoint |
| `logs/ryza_v4_lora/_archive_pre_gradckpt/` | 1.9 GB | grad_ckpt=false 那輪的 epoch 1 產物（**保留未刪**） |