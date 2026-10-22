# Ryza Voice：萊莎日文聲線 TTS

從《Atelier Ryza 2》的遊戲語音，建立**ライザ（Ryza）日文聲線 TTS** 的完整流程：
解包遊戲音聲庫 → 找出ライザ的語音 → 以 ASR 建立訓練清單 → 以 GPT-SoVITS v4 LoRA 微調。

> 本 repo **不含遊戲音訊與模型權重**。需自備遊戲檔，並另行取得 GPT-SoVITS 與預訓練權重。
> 專案目標、範圍與評估標準見 [`docs/01_overview.md`](docs/01_overview.md)。

## 模塊

| 模塊 | 做什麼 | 主要工具 | 文件 |
|---|---|---|---|
| 語音解包 | 解出 KTSR 音聲庫，建立索引 | `vgmstream-cli`、`parse_ktsr.py` | [`02_unpack.md`](docs/02_unpack.md) |
| 說話者篩選 | 以說話者嵌入，在沒有角色名稱的主庫中找出ライザ | `spk_model.py`、`refine_labels.py` | [`03_speaker_id.md`](docs/03_speaker_id.md) |
| 轉譯資料集 | ASR、過濾、去重、專有名詞校正，產出訓練清單 | `screen_asr.py`、`transcribe.py` | [`04_transcript_pipeline.md`](docs/04_transcript_pipeline.md) |
| 資料切分 | 分層抽樣，切出 train／test／eval10 | `split_train_test.py` | [`05_dataset_split.md`](docs/05_dataset_split.md) |
| TTS 微調 | GPT-SoVITS v4 LoRA 的預處理、訓練、推論 | `run_preprocess.py`、`run_lora_train.py`、`stage_d_infer.py` | [`06_model_selection.md`](docs/06_model_selection.md)、[`07_finetune.md`](docs/07_finetune.md) |
| 評估 | 聲線相似度、可懂度 CER | `stage_d_eval.py`、`cer_report.py` | [`07_finetune.md`](docs/07_finetune.md) §7 |

所有工具的用途與輸入輸出見 [`tools/README.md`](tools/README.md)。

## 安裝

| 環境 | 用途 | 安裝方式 |
|---|---|---|
| 本 repo 的 `.venv` | 解包、篩選、轉譯、切分、評估 | `pip install -r requirements.txt` |
| GPT-SoVITS 的 `.venv` | 預處理、訓練、推論 | clone [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 並 checkout `48b1a01`，依其 `requirements.txt` 安裝 |

另需準備：
- **ffmpeg**：放在 GPT-SoVITS 的 `.venv\Scripts\`，見 [`07_finetune.md`](docs/07_finetune.md) §1.1。
- **預訓練權重**：見 [`07_finetune.md`](docs/07_finetune.md) §2。
- **遊戲資料**：讀取路徑 `D:\SteamLibrary\steamapps\common\Atelier Ryza 2`，全程唯讀。

兩個 venv 的 numpy 版本不同，不可混用。Windows 上的安裝偏差見 [`07_finetune.md`](docs/07_finetune.md) §1.2。

## 使用

以下是各模塊的入口指令。完整的執行順序、參數與注意事項，見 [`07_finetune.md`](docs/07_finetune.md) §0 的全流程總覽。

```bash
# 語音解包（完整指令見 docs/02_unpack.md §6）
tools/vgmstream/vgmstream-cli.exe -s 1 -S 0 -o "wav/?05s.wav" "<遊戲目錄>\Data\x64\Sound\VOICE.ktsl2asbin"

# 轉譯資料集：由既有 ASR 結果重產訓練清單（不重跑 ASR）
python tools/transcribe.py --pos-label tools/pos_label.py --asr-json tools/asr_screening.json --output data/ryza_train.list

# 資料切分
python tools/split_train_test.py --input data/ryza_train.list --train-out data/finetune_train.list \
  --test-out data/finetune_test.list --eval-out data/finetune_eval10.list --seed 42

# TTS 微調（在 GPT-SoVITS 目錄執行）
python <RYZA>/tools/run_lora_train.py --stage s1 --config <RYZA>/configs/s1_ryza.yaml --tag s1_train --vram
python <RYZA>/tools/run_lora_train.py --stage s2 --config <RYZA>/configs/s2_lora_ryza.json --tag s2_train --vram
```

說話者篩選、ASR 重跑、預處理、推論與評估的指令，見各模塊文件。

## 目錄結構

| 路徑 | 內容 |
|---|---|
| `docs/` | 專案概述與各模塊文件（見下方「文件」） |
| `tools/` | 全部工具，說明見 [`tools/README.md`](tools/README.md) |
| `configs/` | 訓練設定檔：正式訓練用 `s1_ryza.yaml`、`s2_lora_ryza.json`，另有 3 份冒煙測試用設定 |
| `data/` | 訓練清單與切分結果，說明見 [`data/README.md`](data/README.md) |
| `samples/` | 生成樣本 20 句，說明見 [`samples/README.md`](samples/README.md) |
| `reports/` | 各模塊的結果報告；`reports/logs/` 為訓練紀錄 |
| `voice_index.csv`、`battle_voice_index.csv` | 主庫與戰鬥庫的索引 |
| `ryza_main_index.csv`、`ryza_main_review.csv` | 說話者篩選的完整索引與待審清單 |
| `wav/`、`battle_voice/`、`ryza_battle/`、`ryza_main/` | 解出的音訊與分級結果。**版權，不進版控** |
| `requirements.txt` | 本 repo venv 的 Python 相依套件 |

## 文件

**`docs/`**

| 文件 | 內容 |
|---|---|
| [`01_overview.md`](docs/01_overview.md) | 專案目標、範圍、模塊架構、評估標準 |
| [`02_unpack.md`](docs/02_unpack.md) | 語音解包：KTSR 容器、加密、索引 |
| [`03_speaker_id.md`](docs/03_speaker_id.md) | 說話者篩選：嵌入、分級、驗證 |
| [`04_transcript_pipeline.md`](docs/04_transcript_pipeline.md) | 轉譯資料集：ASR、過濾、校正、訓練清單 |
| [`05_dataset_split.md`](docs/05_dataset_split.md) | 資料切分：分層抽樣、參考音訊 |
| [`06_model_selection.md`](docs/06_model_selection.md) | 基礎模型選型：為什麼是 v4 LoRA |
| [`07_finetune.md`](docs/07_finetune.md) | TTS 微調與推論：環境、預處理、訓練、推論、評估 |
| [`08_troubleshooting.md`](docs/08_troubleshooting.md) | 疑難排解：坑記錄、GPU 專區、症狀速查 |

**`reports/`**

| 文件 | 內容 |
|---|---|
| [`asr_report.md`](reports/asr_report.md) | 轉譯資料集的過濾統計、CER 抽查、專有名詞校正 |
| [`split_report.md`](reports/split_report.md) | 切分統計與驗證結果 |
| [`finetune_report.md`](reports/finetune_report.md) | 訓練過程、聲線相似度、可懂度、主觀試聽 |
| [`defect_triage.md`](reports/defect_triage.md) | 主觀試聽發現的缺陷與排查 |
| [`logs/stage_d_train.md`](reports/logs/stage_d_train.md) | 正式訓練紀錄 |

## 樣本與結果

- 生成樣本：[`samples/`](samples/README.md)
- 評估結果：[`reports/finetune_report.md`](reports/finetune_report.md)

## 授權

- 本 repo 的程式碼與文件採 MIT 授權，見 [`LICENSE`](LICENSE)。
- 解出的音聲內容版權屬 Koei Tecmo／Gust，**不在 MIT 授權範圍內**，僅供個人用途，不進版控；`samples/` 的生成樣本衍生自該音聲，僅供評估。
- vgmstream 為其作者所有（ISC-style license），見 [`tools/vgmstream/COPYING`](tools/vgmstream/COPYING)。
