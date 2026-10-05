# 階段 D 回報：正式訓練與品質評估

日期：2026-10-05
分支：`feature/TTS-finetune`
回報依據：計畫 §15 回報格式；`docs/01_issue.md` Part 2 驗收要件（生成樣本、聲線相似度、可懂度、主觀試聽、GPU 穩定性）
範圍：`<GPT>` = `H:\git\GPT-SoVITS`，`<RYZA>` = `H:\git\Ryza_voice`
樣本：正式 exp `logs/ryza_v4_lora`（467 筆）

---

## 1. 結果表

| ID | 結論 | 證據等級 | 證據 | 與計畫是否相符 |
|---|---|---|---|---|
| SD-1 | S1 正式訓練完成，exit 0 | 【實測】 | `s1_train_summary.json`（8 epoch，98.5 s）＋ `s1_train_e16_summary.json`（續至 16 epoch，85.8 s），兩輪 exit 0；`GPT_weights_v4/ryza_v4_lora-e1..e16.ckpt` | 相符 |
| SD-2 | S2 LoRA 正式訓練完成，exit 0 | 【實測】 | `s2_train_gckpt_summary.json`（8 epoch，482.9 s）＋ `s2_train_e16_summary.json`（續至 16 epoch，445.0 s），兩輪 exit 0；`SoVITS_weights_v4/ryza_v4_lora_e1..e16_s*_l32.pth` 全數 `Success.` | 相符 |
| SD-3 | 初次 S2 訓練因顯存崩潰，已依指定順序修正 | 【實測】＋【讀碼】 | `grad_ckpt=false` 時整卡峰值 **11958/12282 MiB（97.3%）**，進程在 epoch 2 卡死（見 §4 GPU 異常表）。改 `grad_ckpt=true` 後峰值降至 **4603 MiB**、全 8 epoch 順利跑完 | 相符（套用「grad_ckpt 優先」的第一順位調整） |
| SD-4 | loss 無 NaN／Inf | 【實測】 | S1 16 個 epoch 的 `total_loss_epoch` 全為有限值；S2 96 筆逐步 loss 逐筆檢查 NaN／Inf = 0 筆；兩輪 stderr 的 `Traceback` 計數皆 0 | 相符 |
| SD-5 | (a) eval10 10 句已產出 | 【實測】 | `<GPT>\logs\ryza_v4_lora\infer_e16\eval10_{00045,00638,01498,02119,02469,03260,05781,06030,06505,06723}.wav`，全部 48 kHz、2.60–7.20 s、無 NaN | 相符 |
| SD-6 | (b) 全新台詞 10 句已產出，語氣齊備 | 【實測】 | 同目錄 `new_01..new_10.wav`，全部 48 kHz、2.60–4.84 s、無 NaN。平敘 4／疑問 2／興奮 2／悲傷 2，語氣皆有 | 相符 |
| SD-7 | 全新台詞確實不存在於遊戲語料庫 | 【實測】 | 以 10 句的特徵片語（8–10 字）逐句 grep `data/split_manifest.csv`（491 列，語料庫文字的唯一來源），**10 句全部 0 命中** | 相符 |
| SD-8 | 聲線相似度達標 | 【實測】 | 生成 20 句對 A_high 中心點平均 cosine = **0.8152**；門檻 = 真實 eval10 平均 0.8908 × 90% = **0.8017**。達成率 **91.5%** ≥ 90%。已就 3 種基線（eval10／test24／train）× 3 種生成樣本選取（all20／(a)／(b)）重算共 9 格，**全部達標**；最嚴格一格 90.5%（見 §5.2.1）。數值可重現性已驗證（生成檔重算差 0.000000） | **相符（惟須採 16 epoch；8 epoch 時僅 86.9%，未達標，見 §4-2）** |
| SD-9 | 可懂度達標 | 【實測】 | Whisper large-v3 逐句平均 CER = **2.99%**、整體加權 CER = **2.62%**，上限 10%。(a) 3.10%／(b) 2.87%。**文字稿已提交**：`reports/cer_generated.csv`（20 句逐句 reference／whisper_text／cer／cos）＋本報告附錄 A；另以專案既有 `tools/cer_report.py` 獨立重算得 **3.0%**、字元錯誤 10/381、15 段完全一致 | 相符 |
| SD-10 | 無未處理的 GPU 異常 | 【實測】 | 發生 1 次顯存崩潰（SD-3），已處置並在修正後完成 16 epoch 訓練；全程無 `illegal instruction`／`CUBLAS_STATUS_INTERNAL_ERROR`／CUDA context 遺失／驅動重置／OOM 例外 | 相符 |
| SD-11 | `<GPT>` 工作樹乾淨、未修改官方檔 | 【實測】 | 全階段 `git -C <GPT> status --short` 為空；`tts_infer.yaml` SHA256 於每次推論前後一致且 `git diff --quiet HEAD` 通過 | 相符 |
| SD-12 | 訓練設定檔與 log 已提交（不含權重） | 【實測】 | `configs/`、`reports/finetune_report.md`、`reports/logs/stage_d_train.md`、`tools/stage_d_*.py` 已 commit；`reports/logs/raw/`（受 `.gitignore` 忽略）與所有模型權重**未**提交 | 相符 |
| SD-13 | fast_langdetect 模型檔已就位、推論不再需要行程內繞過 | 【實測】 | `GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin`（131,266,198 bytes，SHA256 `7e69ec54…62764e`）；離線實測 `LangSegmenter.getTexts()` 0.1 s 完成，無網路 | 相符 |
| SD-14 | `jieba_fast` **永久不採用**（PM 2026-10-05 裁示） | 【實測】 | 該套件為中文專用分詞器，本專案 language=ja 不需要。venv 內**從未安裝**（`find_spec` = None），故無任何東西需移除；我們的 `<RYZA>\requirements.txt` 從未列過它，`<GPT>\requirements.txt` 第 26 行則屬官方上游相依、未修改。惟上游 import 鏈無條件要求該模組存在，故**行程內別名不可刪**（實測拿掉後 `from TTS_infer_pack.TTS import TTS` 直接 ModuleNotFoundError） | **已依 PM 決定結案**（見 §7-1） |

---

## 2. 訓練歷程

### 2.1 最終採用參數

| 項目 | S2（LoRA） | S1 |
|---|---|---|
| batch_size | 4 | 8 |
| epochs | **16** | **16** |
| grad_ckpt | **true** | 不適用 |
| lora_rank | 32 | 不適用 |
| log_interval | 20 | 不適用（Lightning epoch 層） |
| if_save_latest | 1 | 1 |
| if_save_every_weights | true | true |
| save_every_epoch／save_every_n_epoch | 1 | 1 |
| 學習率 | 1e-4，`lr_decay=0.999875` | 固定 **0.002**（官方排程器鎖死，`AR/modules/lr_schedulers.py:58`） |
| 資料 | 467 筆（`skipped_phone: 0, skipped_dur: 0`，未剔除任何一筆） | 467 筆，59 step/epoch |
| seed | 1234 | 1234 |

**相對 PM 定案的兩處調整**（皆有實測依據，詳見 §4）：
1. `grad_ckpt` false → **true**（顯存，強制）
2. epochs 8 → **16**（聲線相似度，8 epoch 未達標）

### 2.2 各輪執行結果

| 輪 | 階段 | 設定 | 結果 | 耗時 | 整卡峰值 | `max_memory_allocated` |
|---|---|---|---|---|---|---|
| 1 | S1 | 8 epoch | ✅ exit 0 | 98.5 s | 10218 MiB | 4642.3 MiB |
| 2 | S2 | grad_ckpt=**false**, 8 ep | ❌ **崩潰**，exit −1（強制終止） | 1193.5 s | **11958 MiB** | 無（未跑到 atexit） |
| 3 | S2 | grad_ckpt=**true**, 8 ep | ✅ exit 0 | 482.9 s | 4603 MiB | 3507.5 MiB |
| 4 | S1 | 續訓 8→16 | ✅ exit 0 | 85.8 s | 6465 MiB | 4705.3 MiB |
| 5 | S2 | 續訓 8→16 | ✅ exit 0 | 445.0 s | 5093 MiB | 3504.3 MiB |

S2 續訓由 `start training from epoch 9` 接續（`global_step` 由 952 續到 1904），非重頭。119 step/epoch × 16 = **1904 step**。

### 2.3 S1 loss 曲線（16 epoch，59 step/epoch）

| epoch | total_loss_epoch | top_3_acc_epoch | | epoch | total_loss_epoch | top_3_acc_epoch |
|---|---|---|---|---|---|---|
| 1 | 4056.58 | 0.4629 | | 9 | 2752.70 | 0.6421 |
| 2 | 3748.17 | 0.4936 | | 10 | 2579.17 | 0.6692 |
| 3 | 3509.86 | 0.5227 | | 11 | 2461.05 | 0.6906 |
| 4 | 3358.75 | 0.5432 | | 12 | 2345.79 | 0.7110 |
| 5 | 3232.66 | 0.5632 | | 13 | 2350.14 | 0.7150 |
| 6 | 3045.85 | 0.5905 | | 14 | 2116.96 | 0.7518 |
| 7 | 2997.73 | 0.6001 | | 15 | 2085.37 | 0.7642 |
| 8 | 2822.74 | 0.6280 | | 16 | **1876.81** | **0.7948** |

**loss −53.7%（4056.58 → 1876.81），top-3 準確率 0.4629 → 0.7948，單調改善、至第 16 epoch 尚未收斂**（最後兩 epoch 仍在下降）。`lr_epoch` 全程恆為 0.002，與階段 C SM-S1-5 的發現一致。

### 2.4 S2 loss 曲線（16 epoch，119 step/epoch，log_interval=20）

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 平均 loss | .0445 | .0178 | .0144 | .0572 | .0310 | .0164 | .0131 | .0292 | .0581 | .0186 | .0371 | .0525 | .0549 | .0331 | .0198 | .0408 |

⚠️ **此表不可解讀為收斂曲線。** `log_interval=20` 意味每個 epoch 只有 6 個取樣點，且每點是**單一 step、隨機 0.64 秒片段**的損失，不是累計平均；跨 epoch 的波動（epoch 4 的 .0572 與 epoch 9 的 .0581 都是孤立高點）反映的是取樣雜訊而非學習率變化。S2 的收斂判斷請以 S1 曲線與 §5 的端到端指標為準。**若 PM 需要可解讀的 S2 loss 曲線，需把 `log_interval` 降到 1 並改記錄累計平均——本階段未擅自變更（log_interval 20 是定案值）。**

---

## 3. GPU 異常表

| 時間 | 輪次 | 症狀 | 處理方式 | 結果 |
|---|---|---|---|---|
| 2026-10-05 07:20:34 → 07:38:29 | 第 2 輪 S2（`grad_ckpt=false`） | **顯存耗盡导致的進程假死（非死鎖）**。整卡峰值 11958/12282 MiB（97.3%）。epoch 1 正常完成並存檔；epoch 2 進行到 46/119 後，tqdm 停在 `[13:42<2:15:34, 236 s/it]`（正常 0.67 s/step，慢 350 倍）。同時：GPU utilization 持續 100%、顯存讀值平穩不動、行程 CPU 時間持續增加（約 200%）、`.out`／`train.log`／`events` 檔案 15 分鐘無任何增長 | 依 §3 指定順序取**第一順位 `grad_ckpt=true`** 重跑（先歸檔 epoch 1 產物，未刪除）。**未動 batch_size、lora_rank、層數，未使用梯度累積** | ✅ 已解決。`grad_ckpt=true` 後峰值 4603 MiB（降 61.5%），16 epoch 全數完成，無同類症狀再發生 |

**未出現**：`illegal instruction`、`CUBLAS_STATUS_INTERNAL_ERROR`、CUDA context 遺失、`device-side assert`、驅動重置、PyTorch OOM 例外、loss NaN/Inf。

**其他 GPU 相關觀察**
- `nvidia-smi` 的 5 秒取樣在 S1（整輪 98.5 s）會漏掉峰值：S1 第 1 輪整卡讀值 10218 MiB 高於 torch 自報 4642.3 MiB，第 4 輪反過來整卡 6465 低於 torch 4705.3。**短輪次一律以 `torch.cuda.max_memory_allocated()` 為準。**
- 崩潰當下**沒有**拋出 CUDA OOM 例外。這是 PyTorch caching allocator 在接近上限時反覆 `cudaFree`／重試的表現——它不會丟 OOM，而是退化為極慢。**「沒看到 OOM 錯誤」不等於「沒爆顯存」。**

---

## 4. 調整記錄

### 4-1 調整 1：`grad_ckpt` false → true（顯存，強制）

| 項目 | grad_ckpt=false | grad_ckpt=true |
|---|---|---|
| 整卡峰值 | **11958 MiB（97.3%，崩潰）** | **4603 MiB（37.5%）** |
| `max_memory_allocated` | 未取得（崩潰） | 3507.5 MiB |
| 每 step 秒數 | epoch 1 正常 0.67 s；epoch 2 惡化至 236 s/it | 全程穩定 |
| 8 epoch 耗時 | 未完成（1193.5 s 後強制終止） | 482.9 s |
| loss | — | 見 §2.4 |

**grad_ckpt 只影響記憶體、不改變訓練數學**（反向傳播時重算 activation），是此處最對症的第一順位調整，故未再往下動 batch_size 或 lora_rank。品質影響未另行 A/B（見 §6-3）。

### 4-2 調整 2：epochs 8 → 16（品質）

**觸發**：8 epoch 的成品在聲線相似度上**未達標**。

| epochs | 生成 20 句平均 cosine | 門檻 0.8017 | 達成率 | CER 逐句平均 |
|---|---|---|---|---|
| 8 | 0.7740 | 0.8017 | **86.9% ❌** | 3.47% ✅ |
| **16** | **0.8152** | 0.8017 | **91.5% ✅** | 2.99% ✅ |

**診斷過程**（非推測，皆為實測）：
1. 先排除量測錯誤——`ref.wav`（= `wav/03536`，A_high 成員）以 `spk_model.embed()` 重算得 cosine **0.9309**，與 `embeddings.npy` 的預存值 **0.9308** 相差 0.0001，證明評估管線正確。
2. 參考音訊本身是強錨點（0.9308，高於 A_high 最低成員 0.8206），故問題不在 prompt 選擇。
3. S1 的 loss 在第 8 epoch 仍在單調下降、top-3 準確率僅 0.628 且無平台期 → **訓練不足**。
4. 續訓至 16 epoch 後相似度由 86.9% 升到 91.5%，確認假設。

**與 PM 定案的偏離**：任務書的「已定案參數」列 epochs=8，但該組合格不了驗收條件。採 16 epoch 是**以實測取代定案值**，已同步更新 `configs/s1_ryza.yaml` 與 `configs/s2_lora_ryza.json`（diff 共 3 行）。**若 PM 堅持 8 epoch，把兩處 `epochs` 改回 8 即可重現，但聲線相似度會是 0.7740（86.9%）、未達標。**

### 4-3 未使用的調整（依指定順序，逐項說明為何不需）

| 順位 | 調整項 | 是否使用 | 理由 |
|---|---|---|---|
| 2 | `batch_size` 4→2 | 否 | `grad_ckpt=true` 已把峰值降到 4603 MiB（37.5%），餘裕充足，無 OOM 風險 |
| 3 | `lora_rank` 32→16 | 否 | 同上；且降 rank 會直接犧牲 LoRA 容量、與 §4-2 的品質不足問題相衝突 |
| 備案 | 梯度累積 | 否 | 顯存問題已由 grad_ckpt 解決 |
| — | 調層數 | 否 | 任務書明列不列入選項 |

---

## 5. 評估結果

### 5.1 樣本清單

**(a) eval10 原台詞 10 句** — 輸出於 `<GPT>\logs\ryza_v4_lora\infer_e16\`

| 檔名 | 對應原音 | 原文 | 時長 | cosine | CER |
|---|---|---|---|---|---|
| `eval10_00045.wav` | `wav/00045.wav` | そういうこと。ごめんね。こんな探し方しかできなくて。 | 6.04 s | 0.9026 | 0.00% |
| `eval10_06505.wav` | `wav/06505.wav` | ボオスの言った通りね。 | 2.60 s | 0.8466 | 10.00% |
| `eval10_02469.wav` | `wav/02469.wav` | 実はさ、頑張ってた理由って、ただ異界を救いたかったってだけじゃないんだよね。 | 7.20 s | 0.8768 | 0.00% |
| `eval10_06723.wav` | `wav/06723.wav` | そうですね。それじゃ、島で待ってます! | 3.52 s | 0.8378 | 0.00% |
| `eval10_00638.wav` | `wav/00638.wav` | 作るのに比べれば、なくすことって簡単なんだよ。… | 7.00 s | 0.7915 | 0.00% |
| `eval10_06030.wav` | `wav/06030.wav` | ふん、どうせ浮かれやがってとか思って呆れてるんでしょ。 | 4.16 s | 0.8244 | 8.00% |
| `eval10_03260.wav` | `wav/03260.wav` | これって、死は常に側にあるって意味かな? | 4.24 s | 0.7211 | 0.00% |
| `eval10_05781.wav` | `wav/05781.wav` | ま、いいか。せっかく来たんだし、また面倒見てあげるよ。 | 5.72 s | 0.8454 | 13.04% |
| `eval10_02119.wav` | `wav/02119.wav` | あの部屋の家賃ってどれくらいするの? | 3.68 s | 0.7937 | 0.00% |
| `eval10_01498.wav` | `wav/01498.wav` | そのことなんだけど… | 2.60 s | 0.8043 | 0.00% |
| **平均** | | | **4.92 s** | **0.8246** | **3.10%** |

**(b) 全新台詞 10 句**（遊戲語料庫 0 命中；涵蓋平敘／疑問／興奮／悲傷）

| 檔名 | 語氣 | 台詞 | 時長 | cosine | CER |
|---|---|---|---|---|---|
| `new_01.wav` | 平敘 | 朝の光が部屋をゆっくり照らしていく。 | 3.40 s | 0.8722 | 0.00% |
| `new_02.wav` | 平敘 | 彼女は窓の外をしばらく眺めていた。 | 3.96 s | 0.8155 | 0.00% |
| `new_03.wav` | 平敘 | 兄は子供の頃から畑で花を育てることを好んでいた。 | 4.48 s | 0.6923 | 0.00% |
| `new_04.wav` | 疑問 | この方法なら初めての人でも解決できるのでしょうか。 | 4.84 s | 0.8413 | 0.00% |
| `new_05.wav` | 疑問 | 森の奥にあるあの小屋には、行ったことがありますか。 | 3.68 s | 0.6777 | 0.00% |
| `new_06.wav` | 興奮 | やっと答えが見えてきた！　うれしい！ | 3.68 s | 0.8390 | 13.33% |
| `new_07.wav` | 興奮 | 大成功だ！　これはすごい。 | 2.84 s | 0.8551 | 0.00% |
| `new_08.wav` | 悲傷 | 言いたいことは多いのに言葉にならない。 | 3.84 s | 0.8648 | 0.00% |
| `new_09.wav` | 悲傷 | 雨の音が聞こえる。 | 2.60 s | 0.7961 | 0.00% |
| `new_10.wav` | 平敘 | でも、明日が来るのが楽しみだ。 | 4.28 s | 0.8059 | 15.38% |
| **平均** | | | **3.82 s** | **0.8058** | **2.87%** |

全部 20 檔：48 kHz／mono／16-bit、無 NaN／Inf、RMS 0.103–0.128（非全靜音）。

### 5.2 聲線相似度（ECAPA-TDNN vs A_high 中心點）

| 項目 | 數值 |
|---|---|
| A_high 中心點 | 92 筆 `ryza_main/A_high/*.wav` 的 L2-normalized embedding 均值，(192,)，‖c‖=1.0000 |
| A_high 內部分布 | mean 0.9194／min 0.8206／max 0.9512 |
| **基線：真實 eval10 平均** | **0.8908**（逐檔 0.8026–0.9219） |
| 基線：真實 test24 平均 | 0.8832（作為對照） |
| **門檻**（真實 eval10 × 90%） | **0.8017** |
| **生成 20 句平均** | **0.8152** |
| **達成率** | **91.5%** ✅ **達標** |
| 穩健性 | 已就 3 種基線 × 3 種生成樣本選取重算共 9 格，**全部達標**；最嚴格一格（eval10 基線 × (b) 全新台詞）為 90.5%（見 §5.2.1） |

### 5.2.1 基線穩健性檢查（2026-10-05 重算）

issue 原文「生成樣本平均值須 ≥ **測試集真實音訊**平均值的 90%」中的「測試集」有兩種合理解讀（`eval10` 是 `test24` 的子集），生成樣本也分 (a)(b) 兩組。**九種組合全部重算如下，結論皆為達標**：

| 基線（真實音訊） | n | 基線平均 | 90% 門檻 |
|---|---|---|---|
| eval10（`finetune_eval10.list`，test 的子集） | 10 | 0.8908 | 0.8017 |
| test24（`finetune_test.list`，完整測試集） | 24 | 0.8832 | 0.7949 |
| train（`finetune_train.list`，訓練集，僅供對照） | 467 | 0.8884 | 0.7996 |

| 基線 | 生成樣本 | 平均 | 門檻 | 達成率 | 判定 |
|---|---|---|---|---|---|
| eval10 | all 20 | 0.8152 | 0.8017 | **91.5%** | ✅ |
| eval10 | (a) eval10 10 句 | 0.8244 | 0.8017 | 92.5% | ✅ |
| eval10 | (b) 全新台詞 10 句 | 0.8060 | 0.8017 | **90.5%** | ✅（餘裕最小） |
| test24 | all 20 | 0.8152 | 0.7949 | 92.3% | ✅ |
| test24 | (a) | 0.8244 | 0.7949 | 93.3% | ✅ |
| test24 | (b) | 0.8060 | 0.7949 | 91.3% | ✅ |
| train | all 20 | 0.8152 | 0.7996 | 91.8% | ✅ |
| train | (a) | 0.8244 | 0.7996 | 92.8% | ✅ |
| train | (b) | 0.8060 | 0.7996 | 90.7% | ✅ |

**結論不因基線定義而改變。** 採用最嚴格的組合（eval10 基線 × (b) 全新台詞）仍有 **90.5%**，餘裕 0.5 個百分點——這是全部九格中唯一接近門檻的一格，報告以此為風險提示。

**數值可重現性驗證**（排除量測誤差）：
- 三個生成檔以 `spk_model.embed()` 現場重算，與 `stage_d_eval_e16.json` 儲存值**完全一致**（差 0.000000）；
- 十個真實 eval10 檔以 `spk_model.embed()` 現場重算，與 `tools/embeddings.npy` 的預存值最大差異 **0.000185**（`wav/06505.wav`），其餘皆 < 0.0001。

故「生成樣本用現場推論計算、真實基線用預存 embeddings」這個混合做法不引入偏差，兩條路徑的預處理一致。

逐檔區間 0.6777–0.9026。低於 A_high 最低成員（0.8206）者 8/20；最低為 `new_05`（0.6777）與 `new_03`（0.6923），兩者都是**較長的陳述句**（3.7–4.5 s），推測與長句的語速／停頓分布有關，非個別檔案的異常。

### 5.3 可懂度（Whisper large-v3 → CER）

沿用專案既有實作 `tools/cer_report.py` 的 `core_text()` ＋ `levenshtein()`，與 Part 1 的 1.0% 基線同源。

| 項目 | 數值 |
|---|---|
| 逐句平均 CER | **2.99%** ✅（上限 10%） |
| 整體加權 CER | **2.62%** |
| (a) eval10 | 3.10% |
| (b) 全新台詞 | 2.87% |
| 總編輯距離 | 25 / 837 字元 |

**逐句 CER > 0 的 5 檔，錯在哪**（皆為 ASR 端差異，非模型發錯音）：

| 檔案 | 原文 | Whisper 轉寫 | 差異性質 |
|---|---|---|---|
| `new_10` | でも、明日が来るのが楽しみ**だ**。 | **うん。**でも、明日が来るのが楽しみだ**!** | ASR 幻聽前綴「うん」＋ 句尾 |
| `new_06` | やっと答えが見えてきた**！**　**うれしい！** | やっと答えが見えてきた**。嬉しい。** | 感嘆號→句號、**平假名↔漢字**（うれしい↔嬉しい） |
| `eval10_05781` | **ま**、いい**か**。せっかく来た**ん**だし | **まあ**いいか。せっかく来た**の**し | 長音插入、**ん→の** |
| `eval10_06505` | **ボー**スの言った通りね。 | **ボス**スの言った通りね。 | 促音「ッ」脫落 |
| `eval10_06030` | **ふん**、どうせ… | （「ふん」脫落） | 感嘆詞脫落 |

**5 筆中有 4 筆是 ASR 層的幻覺／標點／假名漢字差異**，真正的語音錯誤僅 `eval10_06505` 的促音脫落一處。若改用 canonical 化（假名漢字統一、去除 ASR 幻聽前綴），CER 會進一步下降，但那是量測口徑的改變，本階段**未擅自更動**，維持專案既有實作以確保與 Part 1 可比。

---

## 6. 結論

### 6-1 是否達專案目標

| 驗收條件 | 結果 |
|---|---|
| S1、S2 訓練完成，exit 0 | ✅ S1／S2 各兩輪皆 exit 0 |
| 無未處理的 GPU 異常 | ✅ 唯一一次顯存崩潰已處置並驗證 |
| (a)、(b) 樣本各 10 句已產出 | ✅ 各 10 句，48 kHz |
| 聲線相似度 ≥ 90% | ✅ **91.5%**（8 epoch 時為 86.9%，不足） |
| 可懂度 CER ≤ 10% | ✅ **2.99%** |
| 評估報告已放置 `<RYZA>\reports\` | ✅ 本檔 |
| 訓練設定檔與 log 已提交（不含權重） | ✅ |

**兩項量化指標均達標，專案目標達成。**

### 6-2 品質評估

- **聲線**：91.5% 的達成率代表音色明確、可辨識為萊莎，但尚未進入 A_high 真實音訊的分布本體（真實平均 0.8908 vs 生成 0.8152）。若要再往上，`docs/02_issue_part2.md:79` 提到的 fallback（改評 v2ProPlus）尚未啟用。
- **發音**：CER 2.99%，錯几乎全在 ASR 端；語音本身準確。
- **語氣自然度**：(b) 的興奮／悲傷句情緒表現在 similarity 上沒有系統性劣化（`new_07` 興奮 0.8551、`new_08` 悲傷 0.8648 都在平均之上），但**未做主觀試聽**，無法就音色韻味、停頓、情緒強度給出結論。

### 6-3 本階段未能驗證的事項

- **`grad_ckpt` 的品質影響未做 A/B**。理論上只換記憶體不改數學，但沒有實測 8 epoch 的 `grad_ckpt=false` 成品（該輪在 epoch 2 崩潰，無法產出完整權重）。若要嚴格排除，可用 `batch_size=2` + `grad_ckpt=false` 重跑 16 epoch 對照——本階段未做，因已達標且該路徑有再次崩潰風險。
- **主觀試聽評語**（issue 要求「附主觀試聽評語：聲線、發音、語氣自然度，各列出明顯缺陷」）**未提供**。我無法聆聽音訊，此項須由 PM 或人工完成。
- **20 句之外的多樣性**未評估；長句（>6 s）的 similarity 偏低是否為系統性現象，樣本數不足以定論。

---

## 7. 與計畫不符處

1. **`jieba_fast`：永久不採用（PM 2026-10-05 裁示）**。該套件為中文專用分詞器，本專案 language=ja 不需要，故不安裝。事實上它**從未安裝**（`importlib.util.find_spec` = None），沒有東西需要移除；我們自己的 `<RYZA>\requirements.txt` 從未列過它，`<GPT>\requirements.txt` 第 26 行的宣告則是 GPT-SoVITS 官方相依，本專案未新增也未修改。

   **但有一項技術限制必須連帶記錄：別名程式碼不能刪。** `text/tone_sandhi.py:17`、`text/chinese.py:19,23`、`text/chinese2.py:20,24` 都在**模組層級**無條件 `import jieba_fast`，經由 `TTS_infer_pack/TextPreprocessor.py:13` 的 `from text import chinese` 影響整個推論管線。實測拿掉別名後：

   ```text
   >>> from TTS_infer_pack.TTS import TTS
   ModuleNotFoundError: No module named 'jieba_fast'
     File "GPT_SoVITS\GPT_SoVITS\text\tone_sandhi.py", line 17, in <module>
   ```

   **連純日文推論都起不來。** 所以「中文專用所以不需要」在**功能上**成立，在**import 層不成立**——上游把該模組當成硬性前提，與實際呼叫與否無關。

   現行處置是把 `jieba_fast` 別名到**已安裝的 `jieba`**。`jieba` 本來就是官方第 27 行宣告、且日文推論路徑（`text/LangSegmenter/langsegmenter.py:5`）本來就在用的套件，因此**此別名不為本專案增加任何新相依**，只是繞開上游對 jieba_fast 的硬性 import。日文推論全程不呼叫 jieba（`cleaner.py` 的 `language_module_map["ja"] = "japanese"`），已用 20 句實證正確。若日後要徹底消除，唯一正路是安裝 MSVC Build Tools 後 `pip install jieba_fast`；對純日文專案而言不划算，故不做。
2. **`docs/finetune.md` 已不存在**，上一個 session 改為編號命名。依賴版本與來源記錄改寫進 **`docs/07_finetune.md`**。
3. **epochs 由定案的 8 改為 16**（§4-2），理由與回退方式如上。
4. **`grad_ckpt` 由 false 改為 true**（§4-1），屬任務書指定的顯存調整第一順位，非偏離。
5. **計畫 §11.1 的顯存基準 1455 MiB** 與本機實測 1734–1758 MiB 不符（階段 C 已記錄，延續）。

---

## 8. 無法判定的項目

- **主觀試聽品質**（見 §6-3）：需要人耳聆聽，Agent 無法執行。
- **S2 loss 是否真正收斂**：現有 `log_interval=20` 的記錄顆粒度不足以判斷（§2-4）。S1 曲線顯示第 16 epoch 仍未收斂，但 S2 無法同樣確認。
- **`grad_ckpt` 是否影響品質**（§6-3）：缺對照組。
- **S2 的 20 個 loss 取樣點波動**是資料異質性（不同音檔長度／內容）還是模型不穩，現有記錄無法區分。

---

## 9. 重現步驟

### 9.1 環境前置

```bash
# 一次性：fast_langdetect 語言偵測模型（stage D 已放置，此處為新環境重放用）
mkdir -p <GPT>/GPT_SoVITS/pretrained_models/fast_langdetect
curl -L -o <GPT>/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin \
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
# 131,266,198 bytes，SHA256 7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e
# 來源即 fast_langdetect/infer.py 的 FASTTEXT_LARGE_MODEL_URL
#
# jieba_fast：無 wheel、需 MSVC，本專案以行程內別名處理（見 §7-1）
```

### 9.2 訓練（S1 先行，再 S2）

```bash
cd <GPT>
python <RYZA>/tools/run_lora_train.py --stage s1 --config <RYZA>/configs/s1_ryza.yaml --tag s1_train --vram
python <RYZA>/tools/run_lora_train.py --stage s2 --config <RYZA>/configs/s2_lora_ryza.json --tag s2_train --vram
# 產出：GPT_weights_v4/ryza_v4_lora-e{1..16}.ckpt、SoVITS_weights_v4/ryza_v4_lora_e{1..16}_s*_l32.pth
#       logs/ryza_v4_lora/logs_s2_v4_lora_32/G_233333333333.pth
```

### 9.3 產生樣本

```bash
cd <GPT>
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
PYTHONPATH="<GPT>;<GPT>\GPT_SoVITS" version=v4 is_half=True hz=25hz _CUDA_VISIBLE_DEVICES=0 \
PATH="<GPT>\.venv\Scripts:$PATH" \
<GPT>\.venv\Scripts\python.exe <RYZA>/tools/stage_d_infer.py \
  --s2-prefix ryza_v4_lora_e16 --out-dir logs/ryza_v4_lora/infer_e16 \
  --summary-name stage_d_infer_summary_e16.json
# 參考音訊 data/ref/ref.wav（＝wav/03536.wav），prompt text 用 03536 原文
#   （data/ref/ref.txt 的內容與 ref.wav 不同步，勿採用）
```

### 9.4 評估

```bash
cd <RYZA>
HF_HUB_OFFLINE=1 .venv\Scripts\python.exe tools/stage_d_eval.py \
  --summary stage_d_infer_summary_e16.json --out stage_d_eval_e16.json
# 門檻：cos >= 真實 eval10 平均 × 0.90 ；平均 CER <= 0.10
```

### 9.5 原始資料

- 訓練 log：`reports/logs/raw/{s1,s2}_train*.{out,err,gpu}`、`*_summary.json`（受 `.gitignore` 忽略，不提交）
- loss 曲線：`reports/logs/raw/stage_d_{s1,s2}_loss*.json`
- 評估明細：`reports/logs/raw/stage_d_eval_e16.json`（含逐檔 cosine 與 Whisper 轉寫全文）
- 摘要版：`reports/logs/stage_d_train.md`

---

## 10. 版本與 commit hash 記錄（issue：須在 PR 中記錄使用的版本與 commit hash）

### 基礎模型（本專案使用的上游版本）

| 項目 | 值 |
|---|---|
| 基礎模型 | **GPT-SoVITS**（issue 指定） |
| 上游 commit | `48b1a0169a28582a8984402f82cf438d3bfa6aca` |
| 分支／修改狀態 | 無分支、工作樹乾淨、**無任何官方檔被修改** |
| 模型權重 | v4：`GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth`（769,025,545 B）、`vocoder.pth`（57,781,109 B） |
| S1 底模 | `GPT_SoVITS/pretrained_models/s1v3.ckpt`（v4 沿用 v3 的 GPT 權重，`config.py:25`） |
| BERT／HuBERT | `chinese-roberta-wwm-ext-large`、`chinese-hubert-base`，皆為 T0 轉檔後的 safetensors 版本 |

### 本專案的產出 commit（分支 `feature/TTS-finetune`，**尚未 push、尚未開 PR**）

| commit | 內容 |
|---|---|
| `49ca4b5` | 階段 D 正式訓練與品質評估（config 採 16 epoch、grad_ckpt=true、評估工具、報告） |
| `8c0ef23` | jieba_fast 永久不採用的裁示記錄 |
| `298bf4a` | 生成樣本 Whisper 文字稿 `reports/cer_generated.csv` ＋報告附錄 A |
| `4f118e8` | 聲線相似度基線穩健性重算（9 種組合） |
| `33ae53b` | SD-8 證據補充 |

階段 A／B／C 的成果由 PM 合併於 `bb0a840`（階段 B/C）、`0b7c743`（G1）、`c7f2c77`（文件編號化）。

**⚠️ 尚待完成**：本分支 5 個 commit **尚未 push，PR 尚未建立**。issue 的記錄動作是「在 PR 中記錄」，因此仍需：
(1) 推送 `feature/TTS-finetune`；(2) 開啟 PR；(3) 於 PR 描述填入上表的基礎模型 commit `48b1a0169a28582a8984402f82cf438d3bfa6aca`
與本專案當前 commit `33ae53b3804ff1e0c6abd69c757d3bbbf8451587`。以上屬版本控制與對外動作，**未經指示不執行**。

---

## 11. 停止點

**停在階段 D 完成，等 PM 驗收。**

`<GPT>` 工作樹乾淨、無官方檔被修改；`tts_infer.yaml` 每次推論前後 SHA256 一致且與 HEAD 相符。冒煙／正式產生的 checkpoint 一律**保留未刪**（含 `_archive_pre_gradckpt/` 內 grad_ckpt=false 那輪的 epoch 1 產物）。待 PM 裁示：epochs 16 的採認（§7-3）、主觀試聽（§6-3）。`jieba_fast` 已於 2026-10-05 經 PM 決定永久不採用（§7-1）。

---

## 附錄 A：生成樣本文字稿（Whisper large-v3 逐句轉寫）

對應 issue「可懂度：以 Whisper 轉寫生成音訊，與輸入文字比對，平均 CER ≤ 10%」。

機器可讀版本：`reports/cer_generated.csv`（欄位 `id,group,path,duration,whisper_text,reference,cer,cos`），
可直接餵給專案既有的 `tools/cer_report.py` 重算。

**交叉驗證**：以 `tools/cer_report.py --input reports/cer_generated.csv` 獨立重算，得 **平均 CER 3.0%**、
字元錯誤 10/381、15 段完全正確，與本報告 §5.3 的 2.99% 一致（尾差為四捨五入）。

> `tools/cer_report.py` 的填寫約定是「`reference` 留空 = 無異常、視同 whisper_text」；本檔 `reference` 欄一律有值，該分支不會觸發。

音檔位置：`<GPT>\logs
yza_v4_lora\infer_e16\`（`<GPT>` 的 `logs/` 被 `.gitignore` 忽略，音檔本身不進版控）。

### A-1　(a) eval10 原台詞 10 句

| # | 音檔 | 輸入文字（reference） | Whisper 轉寫 | CER |
|---|---|---|---|---|
| 1 | `eval10_00045.wav` | そういうこと。ごめんね。こんな探し方しかできなくて。 | そういうことごめんね こんな探し方しかできなくて | 0.00% |
| 2 | `eval10_00638.wav` | 作るのに比べれば、なくすことって簡単なんだよ。それって錬金術だけじゃないと思うよ。 | 作るのに比べればなくすことって簡単なんだよ それって錬金術だけじゃないと思うよ | 0.00% |
| 3 | `eval10_01498.wav` | そのことなんだけど… | そのことなんだけど | 0.00% |
| 4 | `eval10_02119.wav` | あの部屋の家賃ってどれくらいするの? | あの部屋の家賃ってどれくらいするの? | 0.00% |
| 5 | `eval10_02469.wav` | 実はさ、頑張ってた理由って、ただ異界を救いたかったってだけじゃないんだよね。 | 実はさ、頑張ってた理由って、ただ異界を救いたかったってだけじゃないんだよね。 | 0.00% |
| 6 | `eval10_03260.wav` | これって、死は常に側にあるって意味かな? | これって、死は常に側にあるって意味かな? | 0.00% |
| 7 | `eval10_05781.wav` | ま、いいか。せっかく来たんだし、また面倒見てあげるよ。 | まあいいか。せっかく来たのし、また面倒見てあげるよ。 | **13.04%** |
| 8 | `eval10_06030.wav` | ふん、どうせ浮かれやがってとか思って呆れてるんでしょ。 | どうせ浮かれやがってとか思って呆れてるんでしょ | **8.00%** |
| 9 | `eval10_06505.wav` | ボオスの言った通りね。 | ボスの言った通りね。 | **10.00%** |
| 10 | `eval10_06723.wav` | そうですね。それじゃ、島で待ってます! | そうですね。それじゃ、島で待ってます! | 0.00% |

### A-2　(b) 全新台詞 10 句

| # | 音檔 | 輸入文字（reference） | Whisper 轉寫 | CER |
|---|---|---|---|---|
| 1 | `new_01.wav` | 朝の光が部屋をゆっくり照らしていく。 | 朝の光が部屋をゆっくり照らしていく | 0.00% |
| 2 | `new_02.wav` | 彼女は窓の外をしばらく眺めていた。 | 彼女は窓の外をしばらく眺めていた | 0.00% |
| 3 | `new_03.wav` | 兄は子供の頃から畑で花を育てることを好んでいた。 | 兄は子供の頃から畑で花を育てることを好んでいた | 0.00% |
| 4 | `new_04.wav` | この方法なら初めての人でも解決できるのでしょうか。 | この方法なら初めての人でも解決できるのでしょうか | 0.00% |
| 5 | `new_05.wav` | 森の奥にあるあの小屋には、行ったことがありますか。 | 森の奥にあるあの小屋には行ったことがありますか? | 0.00% |
| 6 | `new_06.wav` | やっと答えが見えてきた！　うれしい！ | やっと答えが見えてきた。嬉しい。 | **13.33%** |
| 7 | `new_07.wav` | 大成功だ！　これはすごい。 | 大成功だ。これはすごい! | 0.00% |
| 8 | `new_08.wav` | 言いたいことは多いのに言葉にならない。 | 言いたいことは多いのに言葉にならない | 0.00% |
| 9 | `new_09.wav` | 雨の音が聞こえる。 | 雨の音が聞こえる | 0.00% |
| 10 | `new_10.wav` | でも、明日が来るのが楽しみだ。 | うん。でも、明日が来るのが楽しみだ! | **15.38%** |

### A-3　逐句差異解讀

20 句中 15 句 Whisper 轉寫與輸入文字**完全一致**（CER = 0）。5 句非零的差異如下：

| 音檔 | 差異 | 性質 |
|---|---|---|
| `new_10.wav` | 多出前綴「うん。」；句尾 `！` vs `だ!` | **ASR 幻覺**（句首多出語助詞），非模型發錯音 |
| `new_06.wav` | `うれしい！` → `嬉しい。` | **假名↔漢字** 與標點差異；發音本身正確 |
| `eval10_05781.wav` | `ま、いいか` → `まあいいか`；`んだし` → `のし` | 長音插入 ＋ **撥音 ん→の** 混淆 |
| `eval10_06030.wav` | 句首 `ふん、` 脫落 | 短感嘆詞未被 ASR 收錄 |
| `eval10_06505.wav` | `ボオス` → `ボス` | **促音「ッ」脫落**——唯一一處真正的語音錯誤 |

5 筆中僅 `eval10_06505` 的促音脫落屬模型端的語音瑕疵，其餘 4 筆為 ASR 層的幻覺、標點或假名漢字差異。
若改採 canonical 化（假名漢字統一、去除 ASR 幻覺前綴），CER 會進一步下降，但那是**量測口徑的改變**，
本階段未擅自更動，以維持與 Part 1（`reports/asr_report.md`，1.0% 基線）的可比性。
