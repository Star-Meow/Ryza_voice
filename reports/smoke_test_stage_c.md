# 煙霧測試：階段 C（S2 LoRA ＋ S1 ＋ 推論）

日期：2026-10-05
性質：冒煙測試（smoke test）——驗證 S2 LoRA 訓練、S1 訓練、resume、推論管線可端到端跑通，並量測顯存與速度以定正式訓練參數。
範圍：`<GPT>` = `H:\git\GPT-SoVITS`，`<RYZA>` = `H:\git\Ryza_voice`
回報依據：計畫 §11 階段 C、§12.1 報告格式

> Gate 回報（§15 格式）見 [G4_stage_c_report.md](G4_stage_c_report.md)。原始 log 與 metrics 於 `reports/logs/raw/`（受 `.gitignore` 忽略，不提交）。
> **正式訓練未啟動。** 本階段所有訓練都在獨立冒煙 exp `logs/ryza_v4_lora_smoke`（20 筆）進行，正式 exp `logs/ryza_v4_lora` 全程無任何訓練產物。

---

## 1. 測試輪次對照表

| 輪次 | 階段 | 設定 | batch（設定／實際） | epoch | 樣本數 | 結果 | 顯存峰值（整卡／torch allocated） | 每 step 秒數 | 整輪耗時 | 備註 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S2 LoRA | `s2_lora_ryza_smoke.json` | 4／4 | 1 | 20 → **100**（自動擴充 ×5） | ✅ 通過 | 10572 MiB／**8428.2 MiB** | **0.5425** | 27.7 s | 26 step（step 0–25），SM-S2-1～5,7,8 |
| 2 | S2 resume | `s2_lora_ryza_smoke_e2.json` | 4／4 | 2 | 20 → 100 | ✅ 通過 | 11146 MiB／**8295.7 MiB** | **0.5386** | 27.0 s | 由 epoch 2 接續，step 26–51，SM-S2-6 |
| 3 | S1 | `s1_ryza_smoke.yaml` | 8／**8**（末 step 為 4） | 1 | 20 → **100** | ✅ 通過 | 2551 MiB（取樣漏峰）／**3351.7 MiB** | **0.104** | 30.6 s | 13 step，SM-S1-1～4 |
| 4 | 推論 E1 | 底模 `s1v3.ckpt` ＋ `s2Gv4.pth` | — | — | 3 句 | ✅ 通過 | —／**1898.4 MiB** | — | 未記錄 | SM-INF-1～5 |
| 5 | 推論 E2 | 冒煙權重 `ryza_v4_lora_smoke-e1.ckpt` ＋ `ryza_v4_lora_smoke_e2_s52_l32.pth` | — | — | 3 句 | ✅ 通過 | —／**1898.4 MiB** | — | 未記錄 | SM-INF-1～5 |

**樣本數說明**：S2 與 S1 的資料集都會在樣本少於 100 時自動複製擴充（`data_utils.py:546-551` 的 `min_num=100`；S1 為 `AR/data/dataset.py` 的同構寫法）。20 筆 → 100 筆，故兩者的「每 epoch step 數」都不代表正式訓練。

**每 step 秒數算法**：
- S2 ＝（最後一筆 loss 時間 − 第一筆 loss 時間）÷（step 數 − 1）。輪次 1：(05:44:13.275 − 05:43:59.713) ÷ 25 = **0.5425 s**；輪次 2：(05:45:20.462 − 05:45:06.998) ÷ 25 = **0.5386 s**。
- S1 ＝ Lightning 進度棒自報 `13/13 … 9.60it/s` → **0.104 s/step**。此值**不含**資料載入等待與 checkpoint 寫入，故為樂觀估計（見 §5 觀察）。

**指令**

```bash
cd <GPT>
# 輪次 1／2／3（S2 resume 的 config 僅 train.epochs 由 1 改 2，其餘逐字相同）
python <RYZA>/tools/run_lora_train.py --stage s2 --config <RYZA>/configs/s2_lora_ryza_smoke.json    --tag s2_smoke        --watchdog-hours 0 --vram
python <RYZA>/tools/run_lora_train.py --stage s2 --config <RYZA>/configs/s2_lora_ryza_smoke_e2.json --tag s2_smoke_resume --watchdog-hours 0 --vram
python <RYZA>/tools/run_lora_train.py --stage s1 --config <RYZA>/configs/s1_ryza_smoke.yaml         --tag s1_smoke        --watchdog-hours 0 --vram
# 輪次 4／5
python <RYZA>/tools/infer_smoke.py --mode base
python <RYZA>/tools/infer_smoke.py --mode smoke
```

---

## 2. loss 趨勢表

### S2（`log_interval=1`，log 鍵為 `[loss, global_step, learning_rate]`）

| 輪次 | 首 step | 中 step | 末 step | NaN／Inf |
|---|---|---|---|---|
| 1（epoch 1） | step 0：`0.0216744449` | step 12：`0.0660839975` | step 25：`0.0193039812` | 無 |
| 2（epoch 2，resume） | step 26：`0.0081972610` | step 38：`—` | step 51：`0.0284611732` | 無 |

共 52 筆紀錄，逐筆檢查 `nan`／`inf` 皆 0 筆。學習率在輪次 1 全程 `9.99875e-05`、輪次 2 為 `9.9962504687e-05`（`lr_decay=0.999875` 逐 step 衰減），與 `s2_train_v3_lora.py:319-333` 一致。

> 單 epoch 內 loss 在 0.0002～0.40 之間大幅波動屬正常——LoRA 冒煙只有 26 個 step、且 `segment_size=20480`（0.64 s）的隨機片段，統計量不足，**本階段不評估音質、不據此判斷收斂**。

### S1（Lightning 只在 epoch 層記錄，逐步值僅見於進度棒）

| 輪次 | 首 step | 中 step | 末 step | NaN／Inf |
|---|---|---|---|---|
| 1（epoch 1，13 step） | 進度棒未保存逐步值 | — | `total_loss_step` = 1564.616 | 無 |

epoch 層：`total_loss_epoch` = **3972.732**、`top_3_acc_epoch` = 0.49297、`lr_epoch` = 0.00392。

> `self.log` 未設 `on_step=True`，TensorBoard 只有 `total_loss_epoch`／`lr_epoch`／`top_3_acc_epoch`／`epoch` 四個 tag（`reports/logs/raw/stage_c_metrics.json` 的 `s1.tags`）。計畫 §12.4 允許的兩種取值來源（tensorboard 事件、stdout）都拿不到逐步值，如實記錄。S1 的 loss 量級（3,972）與 S2 不可比——前者是語意 token 對數損失、後者是 mel／duration／kl 合成損失。

---

## 3. 最終採用參數

| 項目 | S2（LoRA） | S1 |
|---|---|---|
| batch size | **4** | **8**（`data_module.py:51` 的 `//4` 下限在本設定下不生效） |
| epochs | **8**（§11.5 外推約 8.7 分鐘，遠低於 8 小時，不需自動降級） | **8**（見下方對照列；若沿用官方預設 20 亦僅約 2.8 分鐘） |
| log_interval | **20**（正式） | 不適用（Lightning epoch 層） |
| if_save_latest | 1（固定檔名 `G_233333333333.pth`，每 epoch 覆寫） | 1（存新 ckpt 後刪舊） |
| if_save_every_weights | true | true |
| save_every_epoch／save_every_n_epoch | 1 | 1 |
| lora_rank | 32 | 不適用 |
| 預估總時間 | 見 §4 epoch 表 | 見 §4 epoch 表 |
| 依據 | §4 外推 ＋ 顯存餘裕（峰值 8428 MiB，12 GB 內無 OOM） | §4 外推；實際 LR 恆為 0.002，見 §5 |

**S1 epochs 的選擇**：§11.5 要求在沿用官方預設 20 時另列一列供 PM 對照。兩者時間差約 1.7 分鐘（67.5 s vs 168.7 s），對 8 小時上限完全不構成壓力。**建議 8 epoch**，理由是與 S2 對齊、且 GPT 端在 467 筆資料上 20 epoch 有過擬合風險；此為建議，最終由 PM 決定。

---

## 4. §11.5 epoch 外推表

計算基礎：S2 每 epoch 117 step（467 ÷ 4 進位）× 0.5425 s/step ＋ 每 epoch checkpoint 約 2.0 s（實測 1.576／2.429 s）；S1 每 epoch 59 step（467 ÷ 8 進位）× 0.104 s/step ＋ 每 epoch checkpoint 約 2.3 s（實測 465 MB 寫入）。

| epochs | S2 預估 | S1 預估 | 合計 | 是否在 8 小時內 |
|---|---|---|---|---|
| **8** | 936 step → 507.8 + 16.0 = **523.8 s ≈ 8.7 分** | 472 step → 49.1 + 18.4 = **67.5 s ≈ 1.1 分** | **591.3 s ≈ 9.9 分** | ✅ 是（僅用掉 8 小時上限的 2.1%） |
| 5 | 585 step → 317.4 + 10.0 = **327.4 s ≈ 5.5 分** | 295 step → 30.7 + 11.5 = **42.2 s ≈ 0.7 分** | **369.6 s ≈ 6.2 分** | ✅ 是 |
| 3 | 351 step → 190.4 + 6.0 = **196.4 s ≈ 3.3 分** | 177 step → 18.4 + 6.9 = **25.3 s ≈ 0.4 分** | **221.7 s ≈ 3.7 分** | ✅ 是 |
| （對照）S1 = 20 | 523.8 s ≈ 8.7 分 | 1180 step → 122.7 + 46.0 = **168.7 s ≈ 2.8 分** | **692.5 s ≈ 11.5 分** | ✅ 是 |

**結論：8 epoch 合計約 9.9 分鐘，遠低於 8 小時，依 §11.5 規則不觸發自動降級（8 → 5 → 3）。建議維持 epochs=8。**

⚠️ **外推的不確定性（依 §10.1「只能用每 step 秒數外推」精神如實列出）**
1. **S2 每 step 秒數來自 20 個不重複 wav**（擴充 5 次後全在 page cache）。正式 467 個不重複檔（399 MB）每 step 的磁碟 I/O 會略高，實際時間只會比上表**長**。
2. **`segment_size=20480`（0.64 s）是固定的**，故單步計算量與來源音檔長度無關——這是外推能成立的主要理由。
3. **S1 的 0.104 s/step 是 Lightning 自報值，不含資料載入等待**。S1 用 `num_workers=4` ＋ `persistent_workers=True`，正式 467 筆時是否轉為 I/O bound 需實際跑才知道。這是本表最弱的一項估計。
4. **一次性啟動時間未列入表內**：S1 實測 interpreter ＋ import ＋ Trainer 建構約 6.5 s，資料集建構與 Windows worker spawn 在本輪合計不超過約 21 s（由 05:45:44.5 的 Trainer 初始化完成到 05:46:06.8 的 checkpoint 落盤，含 13 step 訓練）。此為一次性成本，正式訓練只付一次。
5. **8 小時 watchdog 是 S2 與 S1 合計上限**；本外推顯示約 10 分鐘，watchdog 不會觸發。

---

## 5. 觀察

### 5.1 顯存

| 輪次 | 整卡峰值（nvidia-smi，每 5 s） | `torch.cuda.max_memory_allocated()` | `max_memory_reserved()` |
|---|---|---|---|
| S2 epochs=1 | 10572 MiB | 8428.2 MiB | 8582.0 MiB |
| S2 resume epochs=2 | 11146 MiB | 8295.7 MiB | 9190.0 MiB |
| S1 epochs=1 | 2551 MiB ⚠️ | 3351.7 MiB | 4370.0 MiB |
| 推論 E1／E2 | — | 1898.4 MiB | 2264.0 MiB |

- **S2 峰值 11146 MiB ≈ 整卡 91%**，12 GB 內無 OOM。官方門檻 8 GB 已超過，但仍在安全範圍；`docs/07_finetune.md` §5 寫的「12 GB 有餘裕」偏樂觀——餘裕只有約 9%。**若正式訓練需降 batch（S2 4→2），顯存會明顯下降，但每 step 秒數會上升。**
- ⚠️ **nvidia-smi 的 5 秒取樣會漏掉短輪次峰值**：S1 整卡讀值 2551 MiB **小於** torch 自報的 3351.7 MiB，物理上不可能。這是取樣間隔大於峰值持續時間造成，非測量錯誤。**短輪次以 `torch.cuda.max_memory_allocated()` 為準。**
- 計畫 §11.1 寫的閒置基準 1455 MiB 與本機實測 1734–1758 MiB 不符（記錄誤差，不影響判斷）。

### 5.2 速度

S2 每 step 0.54 s（兩輪一致，差異 0.7%），S1 每 step 0.104 s。S2 是瓶頸，約為 S1 的 5.2 倍——這與兩者模型規模一致（S2 為 77.6M 的 CFM/DiT 全模型 LoRA，S1 為 77.6M 的 AR decoder 全參數）。

### 5.3 資料載入行為

- **S2**：20 筆 → 自動擴充 100 筆（`data_utils.py:546-551`，`min_num=100`，重複次數 `max(2, 100//20)=5`）。`skipped_phone: 0, skipped_dur: 0`——**時長過濾 0.6～54 秒未剔除任何一筆**，符合 SM-S2-2 的預期。最終 26 step/epoch（`DistributedBucketSampler` 依長度分桶並補齊，`data_utils.py:994-1016`），**不是** 100÷4=25。
- **S1**：同樣 20 → 100 筆。**實際 batch 是 8 而非計畫 §10.1 預期的 5**，因為 `data_module.py:51` 的 `len(dataset)//4` 作用在擴充後的 100 上（100//4=25），`min(8, 25)=8`。13 step/epoch 中前 12 步為 batch 8、最後 1 步為 batch 4（`bucket_sampler.py` 的 `num_samples=ceil(100/1)=100`，DataLoader 以 8 分批）。

### 5.4 resume 結果（SM-S2-6）

通過，且為**真正的接續而非重頭**：

- stdout 出現 `start training from epoch 2`（`s2_train_v3_lora.py:183-188` 的 resume 分支）。
- `global_step` 由 26 續到 51，**未歸零**。
- `Saving model and optimizer state at iteration 2` → 新增推論權重 `ryza_v4_lora_smoke_e2_s52_l32.pth`（檔頭 `04`、`lora_rank=32`、`info="2epoch_52iteration"`），log 為 `saving ckpt ryza_v4_lora_smoke_e2:Success.`。

### 5.5 SM-S1-5 學習率結論

計畫的待驗證推論是「`warmup_steps=2000` 是否使正式訓練全程停留在 warmup」。**實測答案是否定的，但原因與推論不同。**

`AR/modules/lr_schedulers.py` 的 `WarmupCosineLRSchedule` 被上游改寫成**常數學習率**：
- `set_lr()`（`:38-42`）直接 `g["lr"] = self.end_lr`，忽略傳入的 `lr`；
- `step()`（`:58`）在算完 warmup/cosine 公式後，又執行 `self.lr = lr = self.end_lr = 0.002  ###锁定用线性###不听话，直接锁定！`，覆蓋剛算出的值。

`tools/s1_lr_check.py` 直接跑該排程器實測，四種長度全部得到單一值：

| 情境 | step 數 | warmup 進度 | **實際最後 LR** | 若排程器正常的理論值 |
|---|---|---|---|---|
| 冒煙 | 13 | 0.65% | **0.002** | 6.994e-05 |
| 正式 8 epoch | 472 | 23.6% | **0.002** | 0.002363 |
| 正式 20 epoch | 1180 | 59.0% | **0.002** | 0.005899 |
| warmup 邊界 | 2000 | 100% | **0.002** | 0.01 |

**結論：S1 每一步的實際學習率都是 0.002，與總 step 數無關；排程器根本不會 warmup。** 冒煙觀測值（進度棒 `lr_step: 0.002`、TensorBoard `lr_epoch: 0.00392`）與此一致。

**對 PM 的意義**：`configs/s1_ryza.yaml` 的 `optimizer.{lr, lr_init, lr_end, warmup_steps, decay_steps}` **五個欄位全部無效**，調它們不會改變任何行為。因此「是否要調 warmup_steps」這個問題在 v4 官方實作下不成立。唯一能改 LR 的方式是修改官方排程器——**本專案不做（§4.2），此項僅記錄，是否處置由 PM 決定。**

---

## 6. GPU 異常表

**無。**

| 時間／輪次 | 症狀 | 處理方式 | 結果 |
|---|---|---|---|
| — | 無。階段 C 五輪共 52 個 S2 step、13 個 S1 step、6 次音訊合成，未出現 OOM、`illegal instruction`、`CUBLAS_STATUS_INTERNAL_ERROR`、CUDA context 遺失、驅動重置；loss 無 NaN／Inf | 不適用 | 不適用 |

（依 §12.2，TF32 監測結果＝**未出現**。本機維持官方預設設定，未 patch。）

---

## 7. 結論

**冒煙測試通過。** S2 LoRA（epochs=1）與 resume（epochs=2）、S1（epochs=1）、推論 E1（底模）與 E2（冒煙權重）五輪全部跑通，SM-S2-1～8、SM-S1-1～5、SM-INF-1～5 共 23 項判定全數通過，驗收一律以產出檔案數量與內容為準。

**建議的正式參數**：S2 batch 4／epochs 8／lora_rank 32；S1 batch 8／epochs 8。預估 S2＋S1 合計約 **9.9 分鐘**（§4），遠低於 8 小時上限，不需降級。

**是否建議放行正式訓練**：**可以放行**，但有兩件事請 PM 先裁示（詳見 [G4_stage_c_report.md](G4_stage_c_report.md) §4）——
1. 計畫 §12.3 把「jieba_fast 未安裝」列為僅影響中文，實測證明它擋下整個推論管線；本階段以行程內設定繞過（未安裝任何套件），正式推論需沿用同一設定。
2. S1 的學習率被上游硬寫成常數 0.002，設定檔的 `optimizer.*` 五欄位無效。

---

## 8. 實際訓練（待訓練後補）

> 以下為空白版型，**訓練後依實際數據填寫，不得填入推測值**。

### 8.1 測試輪次對照表（空白）

| 輪次 | batch | epoch | 樣本數 | 結果 | 顯存峰值 | 每 step 秒數 | 估計完整訓練時間 | 備註 |
|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | |
| 2 | | | | | | | | |

### 8.2 loss 趨勢表（空白）

| 輪次 | 首 step | 中 step | 末 step | log_interval | NaN／Inf |
|---|---|---|---|---|---|
| S2 | | | | | |
| S1 | | | | | |

### 8.3 GPU 異常表（空白）

| 時間／輪次 | 症狀（OOM、illegal instruction、CUBLAS 錯誤、驅動重置等） | 處理方式 | 結果 |
|---|---|---|---|
| | | | |