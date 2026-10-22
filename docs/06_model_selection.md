# 基礎模型選型：GPT-SoVITS 版本與微調路線

> 說明為什麼選擇 GPT-SoVITS **v4 LoRA**，以及 v2ProPlus 為何列為備選。
> 選型的約束見 [`01_overview.md`](01_overview.md) §2；選定後的微調流程見 [`07_finetune.md`](07_finetune.md)。

## 背景：版本篩選準則

GPT-SoVITS 主分支可選版本為 v1 / v2 / v4 / v2Pro / v2ProPlus。篩選依**版本沿革與官方定位**逐步收斂，而非以 benchmark 分數反向淘汰。

| 版本 | 淘汰理由 |
|---|---|
| **v1** | 最早版本，架構與訓練資源均已過時，官方與社群主力已不在 v1 |
| **v2** | 被 v2Pro 直接取代。官方 v2Pro release note 明說 v2Pro 維持 v2 的硬體成本與速度，但 zero-shot 相似度提升到 v3/v4 同級——同成本、更好效果，v2 無選擇理由 |
| **v3** | 不在主分支 UI（`webui.py:1498` 選項為 v1/v2/v4/v2Pro/v2ProPlus），官方明說「作者認為 v4 是 v3 的平替」；且 v3 原生只輸出 24kHz、有 <100h 電音警告 |
| **v2ProPlus** | 非淘汰，為 v2Pro 的增強版，列為備選 |

經此篩選，主要候選為 **v2Pro**（VQ 路線）與 **v4**（CFM / diffusion 路線），v2ProPlus 列備選。

---

## 一、顯存與微調路徑（核心約束）

RTX 4070 SUPER 12GB 為硬體上限，兩候選的可行路徑有本質差異：

| 項目 | v2Pro | v4 |
|---|---|---|
| **微調方式** | 目前主分支未提供 LoRA 入口 | **全量微調 或 LoRA** |
| **官方訓練配置中的顯存門檻** | 無（僅定性） | **14GB（全量）/ 12GB（梯度檢查點）/ 8GB（LoRA）** |
| **社群實測** | ~12GB（第三方整理，未標明訓練模式） | 8GB LoRA 為官方校準值 |
| **12GB 可行性** | **臨界甚至不足** | LoRA 路徑 **有 ~4GB 餘裕** |

**訓練腳本分流**：目前主分支 WebUI 的訓練流程中，v1/v2/v2Pro/v2ProPlus 使用 `s2_train.py`，未提供 LoRA 訓練入口；v3/v4 則使用 `s2_train_v3_lora.py`。LoRA 選項僅在 v3/v4 時可見，v2Pro 無法使用。

**WebUI batch_size 校準**：`webui.py:119` 公式為 `default_batch_size = minmem // 2 if version not in v3v4set else minmem // 8`。12GB 卡（minmem ≈ 12）對應 v2Pro batch = 6，v3/v4 batch = 1。此校準反映官方對兩者訓練記憶體特性的保守估計。

**12GB 下的實際含義**：
- v2Pro 全量微調：第三方實測 GPT-SoVITS 在 FP16 下微調普遍佔用 10.8–11.5GB，batch_size=4、音頻長度 60 秒時即佔用 11GB 以上。12GB 卡扣除 CUDA context 與 PyTorch 開銷後餘裕極少。
- v4 LoRA：官方訓練配置標示 8GB，12GB 下有明確餘裕。

---

## 二、版本規格與官方基準

| 版本 | 取樣率 | 架構 | SoVITS 參數量 | 官方 SIM（zero-shot） |
|---|---|---|---|---|
| v2Pro | 32kHz | VQ + Speaker Verification | 133M+77M | 0.709 |
| v2ProPlus | 32kHz | VQ + 增強 Speaker Verification | 152M+77M | 0.737 |
| v4 | **48kHz** | CFM + HiFiGAN | 330M+77M | 0.735 |

官方 wiki 記載：v2Pro「比v2显存占用稍高一点点，超v4的性能，**v2的硬件成本和速度**」；v4「**修復了v3非整數倍上採樣可能導致的電音問題，原生輸出48k音頻防悶**」。

官方 zero-shot SIM 數據顯示兩者數值接近（v2ProPlus 0.737 vs. v4 0.735），因此僅依此 benchmark 無法形成明顯的相似度差距。此數據支持「v4 不會在相似度上明顯犧牲」，但非淘汰 v1/v2 的依據。

---

## 三、v2Pro 的具體風險

1. **目前主分支未提供 LoRA 訓練入口**：v2Pro 只能走 `s2_train.py` 全量微調路徑，無法透過 LoRA 壓縮顯存。
2. **12GB 臨界**：社群實測 11GB+ 與 12GB 卡實際可用顯存高度重疊，OOM 風險顯著。
3. **取樣率不匹配**：v2Pro 原生 32kHz，本專案資料為 48kHz，需額外上採樣。

---

## 四、v3 路線不作為候選

官方已將 v4 定位為 v3 的替代方案，且 v4 修正 v3 的非整數倍上採樣可能造成的 metallic artifacts。本資料集僅 51.5 分鐘，遠低於 v3 電音警告的 <100h 門檻。在 12GB 硬體限制下，沒有必要額外選擇 v3。

---

## 五、選型結論

**優先路徑：v4 LoRA**

1. **顯存可行性有官方訓練配置保障**：8GB LoRA 門檻下，12GB 有餘裕；v2Pro 無官方 GB 數字，社群實測指向 ~12GB，處於臨界。
2. **取樣率匹配**：v4 原生 48kHz，與 491 段資料一致。
3. **相似度差距不明顯**：v4 SIM 0.735 與 v2ProPlus 0.737 接近，僅依 benchmark 無法形成明顯差距。
4. **消除 v3 電音風險**：v4 修復了 v3 非整數倍上採樣的電音問題。
5. **LoRA 是 v3/v4 唯一微調入口**：這不是缺點，而是 12GB 卡下的必要路徑。

**備選：v2ProPlus**

若 v4 LoRA 實測無法達到評估標準（[`01_overview.md`](01_overview.md) §4），再評估 v2ProPlus；屆時需先實測其實際 VRAM 使用量，並以相同資料切分與評估流程比較相似度、CER 及主觀音質。

---

## 六、決策鏈摘要

```
主分支版本
    │
    ├─ v1 → 淘汰：舊版本
    ├─ v2 → 淘汰：v2Pro 已取代其定位
    ├─ v3 → 淘汰：v4 已作為替代方案
    │
    ├─ v2Pro ────────┐
    │                 │
    ├─ v2ProPlus ────┤ → VQ 路線
    │                 │
    └─ v4 ───────────┘ → CFM / diffusion 路線
             │
             ↓
        RTX 4070 SUPER 12GB
             │
       ┌─────┴─────┐
       ↓           ↓
   v2Pro 全量    v4 LoRA
       │           │
   VRAM 臨界      官方低顯存路徑
       │           │
       └─────┬─────┘
             ↓
         優先 v4 LoRA
             │
             ↓
      實測 VRAM + 評估
```

核心不是「v4 benchmark 比 v2Pro 好」，而是：**在 12GB GPU 的實際約束下，v4 提供官方明確的 LoRA 低顯存訓練路徑，而 v2Pro 沒有；同時 v4 的原生 48kHz 與資料集匹配，且官方 zero-shot SIM 與 v2ProPlus 接近。**

---

## 七、實測後的修正

選型當時的判斷，經實際訓練後有一項需要修正：

| 選型時的判斷 | 實測結果 | 影響 |
|---|---|---|
| 「v4 LoRA 在 12 GB 下有約 4 GB 餘裕」（§一） | **不成立**。`grad_ckpt=false` 時整卡峰值達 11958/12282 MiB（97.3%），訓練在 epoch 2 假死且不丟 OOM；改 `grad_ckpt=true` 後峰值降到 4603 MiB | 12 GB 卡上 `grad_ckpt=true` 是**必要設定**，見 [`07_finetune.md`](07_finetune.md) §7、[`08_troubleshooting.md`](08_troubleshooting.md) PK-013 |

其餘判斷維持：
- **取樣率**：48 kHz 端到端一致。
- **TF32**：本機訓練全程未出現 CUDA 損壞，維持官方預設，見 [`08_troubleshooting.md`](08_troubleshooting.md) §4。
- **參考音訊**：已固定為 `wav/03536.wav`，見 [`05_dataset_split.md`](05_dataset_split.md) §3。

v4 LoRA 已達到評估標準，因此未啟用 v2ProPlus 備選。結果見 [`reports/finetune_report.md`](../reports/finetune_report.md)。

---

## 附錄 A：音訊品質實測

**目的**：確認訓練資料是否符合 GPT-SoVITS 官方 wiki 的「低音質」判定條件，判斷資料品質能不能作為選型的依據。

**方法**：
- 範圍：`data/ryza_train.list` 全數 491 段，涵蓋清單前、中、後段（164／164／163）。
- 工具：[`tools/audio_quality.py`](../tools/audio_quality.py)，標準庫加 numpy 的純 CPU FFT，不載入模型、不使用 GPU。
- 原始數據：[`reports/audio_quality.json`](../reports/audio_quality.json)。

**官方判定條件對照**：

| 條件 | 官方描述 | 量測結果 | 判定 |
|---|---|---|---|
| 16k 以下頻段缺失 | 壓縮造成低通 | -24.7 dB @16k，-60dB 頻寬中位數 23.5 kHz | 未觸發 |
| 壓縮陡降 | 低通截斷懸崖 | 18/491 檔 >18dB，中位數 10.7 dB | 未觸發 |
| 背景噪音 | 寬頻噪音 | 安靜段低/高頻 -34.3/-34.1 dB，0/491 寬頻特徵 | 未觸發 |

**LTAS 長時平均頻譜**（相對 0.5–4k 頻帶，dB）：

| 頻率 | 4 kHz | 8 kHz | 12 kHz | 14 kHz | 16 kHz | 18 kHz | 20 kHz |
|---|---|---|---|---|---|---|---|
| 相對能量 | -6.6 | -12.2 | -14.7 | -20.1 | -24.7 | -31.3 | -39.1 |

**結論**：
- 本批音訊**不符合**官方的「低音質」分類，資料品質不構成偏向任何版本的理由。
- 選型因此回到顯存路徑與取樣率（§一、§五）。高頻能量延伸到 20 kHz 以上，也支持採用原生 48 kHz 的 v4。

> 殘響尚無法由本工具精確量化。安靜段的語音染色輪廓只能排除「恆定寬頻背景噪音」，無法區分輕微空間感與無殘響。
> 「少量音訊帶輕微空間感」屬於這一類尚未量化的殘留風險。
