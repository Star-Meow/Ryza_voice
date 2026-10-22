# TTS 微調與推論：GPT-SoVITS v4 LoRA

> 在已有 `data/finetune_{train,test,eval10}.list` 的前提下，從環境、模型準備、預處理、訓練到推論與評估的完整重現步驟。
> 為什麼選 v4 LoRA 見 [`06_model_selection.md`](06_model_selection.md)；遇到問題查 [`08_troubleshooting.md`](08_troubleshooting.md)。
>
> 路徑記號：`<GPT>` 為 GPT-SoVITS repo，`<RYZA>` 為本 repo。

## 0. 全流程總覽

從遊戲檔到生成樣本，依下表順序執行：

| 步驟 | 文件 | 產出 |
|---|---|---|
| 1. 環境 | 本文 §1 | 兩個 venv、ffmpeg |
| 2. 語音解包 | [`02_unpack.md`](02_unpack.md) | `wav/`（9,569 段） |
| 3. 說話者篩選 | [`03_speaker_id.md`](03_speaker_id.md) | `ryza_main/`、top500 |
| 4. 轉譯資料集 | [`04_transcript_pipeline.md`](04_transcript_pipeline.md) | `data/ryza_train.list`（491） |
| 5. 資料切分 | [`05_dataset_split.md`](05_dataset_split.md) | `data/finetune_{train,test,eval10}.list` |
| 6. 模型準備 | 本文 §2 | 底模權重、T0 轉檔 |
| 7. 預處理 | 本文 §3 | `2-name2text`／`4-cnhubert`／`5-wav32k`／`6-name2semantic` |
| 8. 冒煙測試與訓練 | 本文 §5 | `GPT_weights_v4`／`SoVITS_weights_v4` |
| 9. 推論 | 本文 §6 | `logs/ryza_v4_lora/infer_e16/*.wav` |
| 10. 評估 | 本文 §7 | 聲線相似度、CER |

步驟 2～5 使用的工具與參數見 [`tools/README.md`](../tools/README.md)。

---

## 1. 環境需求

| 項目 | 值 | 備註 |
|---|---|---|
| OS | Windows 10 19045 x64 | |
| Python | 3.10.11（`<GPT>\.venv`） | 獨立 venv，不繼承系統 site-packages |
| GPU | RTX 4070 SUPER（12 GB） | `torch.cuda.is_available()` = True，cu121 |
| GPT-SoVITS | 分支 `main`，commit `48b1a0169a28582a8984402f82cf438d3bfa6aca`（短碼 `48b1a01`） | 無本地修改（`git status` 全程乾淨） |
| **ffmpeg** | **9.0.2-essentials_build（gyan.dev 靜態版）** | **隱性系統依賴，見下** |

本 repo 與 `<GPT>` 各用獨立 venv：前者跑 ASR、ECAPA 與評估，後者跑預處理、訓練與推論。兩者的 numpy 版本不同（2.2.6 vs 1.26.4），不可混用。

### 1.1 ffmpeg（隱性系統依賴）

`tools/my_utils.py` 的 `load_audio`（`:16-37`）透過 `ffmpeg-python` 發起 subprocess，`cmd=["ffmpeg", "-nostdin"]`，**只靠 PATH 上的 `ffmpeg`**，無 Python fallback。缺 ffmpeg 時會 `FileNotFoundError`，且腳本的裸 `except` 會吞掉錯誤只印 traceback（見 §4 易錯點）。

```text
ffmpeg：9.0.2-essentials_build（gyan.dev 靜態版）
  下載：https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
  zip SHA256：60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba
  放置位置：<GPT>\.venv\Scripts\ffmpeg.exe 與 ffprobe.exe（不改名、不做 .cmd shim）
  原因：tools/my_utils.py 的 load_audio 只走 ffmpeg CLI；venv Scripts 需在 PATH 上
```

**驗證方式**（venv 啟動態）：

```bash
source <GPT>/.venv/Scripts/activate
python -c "import shutil; print(shutil.which('ffmpeg')); print(shutil.which('ffprobe'))"
ffmpeg -version | head -1
```

預期：兩路徑皆指向 `<GPT>\.venv\Scripts\`，版本 `9.0.2-essentials_build-www.gyan.dev`。`.venv\` 在 `<GPT>` 的 `.gitignore` 內，`git status` 不會出現。

### 1.2 與官方 requirements.txt 的偏差（Windows 必要調整）

`<GPT>` 的 venv 依官方 `requirements.txt` 安裝。以下 4 項因本機沒有 MSVC 編譯器而調整：

| 套件 | requirements.txt 指定 | 實際安裝 | 原因 |
|---|---|---|---|
| pyopenjtalk | `pyopenjtalk>=0.4.1` | **pyopenjtalk-prebuilt 0.3.0** | 原套件需 CMake + MSVC 從原始碼建置；本機無 MSVC 編譯器（`CMAKE_C_COMPILER not set`）。改用 API 相容的預編譯版本，`pyopenjtalk.g2p()` 與 `OPEN_JTALK_DICT_DIR` 均可用 |
| opencc | `--no-binary=opencc`（強制原始碼建置） | opencc 1.4.2（預編譯 wheel） | 同上，原始碼建置需編譯器；改用預編譯 wheel |
| jieba_fast | `jieba_fast` | **未安裝**（保留 `jieba 0.42.1`） | 無 Windows 預編譯 wheel，原始碼建置需編譯器。`text/cleaner.py` 採延遲 import（`__import__("text." + lang)`），`ja` 路徑只載入 `text.japanese`，不載入 `chinese.py`／`chinese2.py`，訓練不需要；**推論端則因上游 import 鏈仍需別名**，見 §6.2.2 |
| numpy | `numpy<2.0` | 1.26.4 | torch 安裝帶入 numpy 2.2.6 導致 pyopenjtalk-prebuilt 二進位不相容（`numpy.dtype size changed`）；降級至 1.26.4，同時符合 requirements.txt 的 `numpy<2.0` |

### 1.3 日文 G2P 驗證

安裝完成後，可用下列輸出確認日文 G2P 鏈正常（在 `<GPT>` 的 venv 內執行）：

```python
pyopenjtalk.g2p('こんにちは')
# -> k o N n i ch i w a

pyopenjtalk.g2p('錬金術の勉強は難しいけど、すごく楽しいんだ。')
# -> r e N k i N j u ts u n o b e N ky o o w a m u z u k a sh i i k e d o pau s u g o k U t a n o sh i i N d a

# 經 GPT-SoVITS 自身的路徑（text/cleaner.py → text/japanese.py）
clean_text('こんにちは、ライザです。', 'ja', 'v2')
# -> phones: k o [ N n i ch i w a , r a ] i z a d e s u .

clean_text('今日も錬金術の研究を頑張るよ！', 'ja', 'v2')
# -> ky o ] o m o UNK r e [ N k i ] ...
```

- OpenJTalk 字典會在首次執行時自動下載到 venv。
- 輸出中的 `[`／`]` 是韻律標記；`UNK` 來自 `#` 韻律邊界，屬正常（見 §8 #1、[PK-003](08_troubleshooting.md)）。

---

## 2. 模型準備

### 2.1 必須存在的權重

全部下載自 `huggingface.co/XXXXRT/GPT-SoVITS-Pretrained`，置於 `<GPT>\GPT_SoVITS\pretrained_models\`：

| 用途 | 相對路徑 | 大小 | 驗證 |
|---|---|---|---|
| SoVITS v4 底模（G） | `gsv-v4-pretrained/s2Gv4.pth` | **769,025,545 B** | `torch.load` OK，4 個頂層鍵 |
| v4 專用 vocoder | `gsv-v4-pretrained/vocoder.pth` | **57,781,109 B** | `torch.load` OK，194 鍵 |
| GPT／AR 底模 | `s1v3.ckpt` | **155,284,856 B** | `torch.load` OK，3 個頂層鍵 |
| BERT | `chinese-roberta-wwm-ext-large/` | `pytorch_model.bin` 651 MB ＋ `config.json` ＋ `tokenizer.json` | 三檔齊全；轉檔見 §2.2 |
| HuBERT | `chinese-hubert-base/` | `pytorch_model.bin` 189 MB ＋ `config.json` ＋ `preprocessor_config.json` | 三檔齊全；轉檔見 §2.2 |

**注意 `s1v3.ckpt` 直接位於 `pretrained_models/` 下，不在 `gsv-v4-pretrained/` 裡。** v4 沿用 v3 的 GPT 權重（`config.py:25`）。

**v4 發行包只附 G、不附 D**：`gsv-v4-pretrained/` 只有 `s2Gv4.pth` 與 `vocoder.pth`。`webui.py:177` 對 `s2Dv3`／`s2Dv4` 特別豁免缺檔檢查，而 `s2_train_v3_lora.py` 全程不讀 `pretrained_s2D`——故 config 裡的 `pretrained_s2D` 填官方預設的 `s2Dv4.pth`（檔案不存在）**無害**。

**刻意未下載**：`GPT_SoVITS/text/G2PWModel/`（589 MB，僅中文 G2P 用）。

路徑依 `config.py` 的 `pretrained_sovits_name["v4"]`、`pretrained_gpt_name["v4"]`，以及 `inference_webui.py:513`／`export_torch_script_v3v4.py:535` 的 vocoder 路徑，與官方約定一致。

### 2.2 T0：`.bin` → safetensors（必要）

**為什麼要做**：`transformers 4.57.6` 因 CVE-2025-32434 預設禁止以 `torch.load` 載入 `.bin`；本環境 torch 2.5.1+cu121 未達 2.6 門檻，而升級 torch 會破壞 cu121 綁定。實測錯誤：

```text
ValueError: ... CVE-2025-32434 ... require users to upgrade torch to at least v2.6
... does not apply when loading files with safetensors
```

`tools/t0_convert_safetensors.py` 只有兩個必填參數（`--bin` / `--out`）。**原始執行指令未記入任何檔案**，以下為依檔案版面還原之形式（標記為還原，非引用）：

```bash
python tools/t0_convert_safetensors.py \
  --bin <GPT>/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin \
  --out <GPT>/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/model.safetensors
# chinese-hubert-base 同理
```

**轉檔結果**：

| 項目 | chinese-roberta-wwm-ext-large | chinese-hubert-base |
|---|---|---|
| `.bin` SHA256 | `e53a693acc59ace251d143d068096ae0d7b79e4b1b503fa84c9dcf576448c1d8` | `24164f129c66499d1346e2aa55f183250c223161ec2770c0da3d3b08cf432d3c` |
| `.bin` 大小 | 651,225,145 B | 188,811,417 B |
| `.bin` 鍵數／dtype | 397（fp16×396 ＋ int64×1） | 211（全 fp16） |
| `model.safetensors` SHA256 | `ddb2d1e2aec45a149bb1b19213c0478cee464f9cc75531e70e673b5224ef4f05` | `25adc31d1889ff5d3da262189433bdc755f0ddb9f194342583dbd17e447daefe` |
| `model.safetensors` 大小 | **694,454,896 B** | **188,767,040 B** |

轉檔**不刪除也不修改** `.bin`，key 名稱不變；共用儲存的 tensor 以 `data_ptr` 偵測並從第二次出現起 clone（roberta 有 2 組：word_embeddings↔decoder.weight、bias↔decoder.bias；hubert 為 0 組）。驗證腳本 `tools/t0_validate.py` 會比對 key/dtype/shape 並附 negative fixture。

> ⚠️ **新環境注意**：下載 HuggingFace 預設格式通常只有 `pytorch_model.bin`。若轉檔後仍報 CVE 錯誤，檢查 `model.safetensors` 是否真的落在 `from_pretrained` 指向的目錄。

其他載入細節（皆不影響使用）：
- `chinese-roberta-wwm-ext-large` 的 `config.json` 指定 BERT 架構，`AutoModelForMaskedLM` 實際解析為 `BertForMaskedLM`。
- roberta 的 safetensors 含 `bert.embeddings.position_ids`（int64）。`from_pretrained` 會吸收，但手動 `load_state_dict(strict=False)` 會把它列為 unexpected key。
- `chinese-hubert-base` 為標準 transformers 命名（211 key），沒有 `weight_g`／`weight_v` 這類 weight_norm 參數。
- safetensors 比 `.bin` 略大（roberta 694 MB vs 651 MB），是格式的 header／alignment 差異。

### 2.3 推論端模型檔

`fast_langdetect` 的 `lid.176.bin` 也要事先放置，見 §6.2.1。

---

## 3. 預處理

驅動：`tools/run_preprocess.py`（不經 webui，直接以 venv Python 執行官方腳本）。

```bash
cd <GPT>
export PYTHONPATH="<GPT>;<GPT>\GPT_SoVITS" PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python <RYZA>/tools/run_preprocess.py --exp train --steps 1,2,3
python <RYZA>/tools/run_preprocess.py --exp eval10 --steps 1,2,3
```

全域環境變數（`run_preprocess.py` 的 `BASE_ENV`）：`version=v4`、`is_half=True`、`hz=25hz`、`_CUDA_VISIBLE_DEVICES=0`、`i_part=0`、`all_parts=1`（單 GPU 序列）。驅動程式會把 `.venv\Scripts` 加進子行程 PATH，`load_audio` 才能找到 ffmpeg。

| 步驟 | 腳本 | 輸入 | 輸出 |
|---|---|---|---|
| 1 | `1-get-text.py` | `data/finetune_{train,eval10}.list` | `2-name2text-{i_part}.txt` |
| 2 | `2-get-hubert-wav32k.py` | 同上 + wav | `4-cnhubert/*.pt`、`5-wav32k/*.wav` |
| 3 | `3-get-semantic.py` | `4-cnhubert/*.pt` | `6-name2semantic-{i_part}.tsv` |

### 3.1 版本推斷（v4 → v3 codepath）

`3-get-semantic.py` 不讀 `version` 環境變數（`:17` 已註解），而是依 `pretrained_s2G` 的檔案大小推斷（`:18-28`）：

```python
size = os.path.getsize(pretrained_s2G)
if size < 82978 * 1024:      version = "v1"
elif size < 100 * 1024 * 1024:  version = "v2"
elif size < 103520 * 1024:   version = "v1"
elif size < 700 * 1024 * 1024: version = "v2"
else:                        version = "v3"
```

`s2Gv4.pth` = 769,025,545 bytes > 700 MB → 推斷 `"v3"`。**無害**：v3/v4 共用同一個模型類別（`module.models.SynthesizerTrnV3`），v4 額外的 vocoder 在此階段不需要。log 顯示 `<All keys matched successfully>`。

### 3.2 合併分片

`all_parts=1` 時只有 `-0` 分片。webui 的合併流程（`webui.py:1213-1222`）為：加表頭 `item_name\tsemantic_audio` + 讀入各分片 + `os.remove(分片)` + 寫 LF 結尾。手動驅動時合併後保留 `-0` 原檔（內容已驗證等價）。

注意：官方腳本以 `"\n".join(lines1)` 寫 `-0` 分片（`3-get-semantic.py:117-118`），**最後一行後無換行**，故 `wc -l` 顯示 466 而非 467；合併時已正規化。`2-name2text.txt` 在 Windows 寫出為 CRLF（467 行皆含 `\r`），S1 訓練以 Python 讀取不受影響。

### 3.3 完成條件（PA 檢查）

預處理的結束碼不可信（見 §4），是否完成以下列檢查為準：

| 項目 | train（467） | eval10（10） |
|---|---|---|
| PA-1 2-name2text 行數／欄位 | 467 行、4 欄 | 10 行、4 欄 |
| PA-2 4-cnhubert `.pt` 數 | 467 | 10 |
| PA-3 5-wav32k 數＋取樣率 | 467、32000 Hz | 10、32000 Hz |
| PA-4 6-name2semantic 行數＋表頭 | 468 行、LF、0 CR | 11 行 |
| PA-5 五處鍵集合全等 | 全等 | 全等 |
| PA-6 HuBERT 無 NaN／形狀 768×frames | 抽 20 檔 | 全檢 10 檔 |
| PA-7 5-wav32k 總大小 | 179.9 MB | 3.4 MB |
| PA-8 3-bert 存在且為空 | 存在且為空（ja 不產 BERT） | 同 |
| PA-9 非 # UNK | 0 | 0 |
| PA-10 音素與 G2P 一致 | 0 mismatch | 0 mismatch |
| PA-11 版本推斷 | v3（無害）、載入正常 | 同 |

驗證腳本：`tools/stage_a_validate.py`（唯讀，結果寫到 `reports/logs/raw/stage_a_pa.json`，不進版控）。表中數值為預期結果。

---

## 4. 易錯點

1. **官方腳本的裸 `except` 會吞錯，exit code 恆為 0**：`1-get-text.py:102-103`、`2-get-hubert-wav32k.py:124-125`、`3-get-semantic.py:115-116` 皆印 traceback 後繼續。**結束碼不可作為成功依據**——預處理、冒煙測試與所有檢查都必須以「預期產出檔案數量與內容」為準（§3.3 與 §5.2 的檢查即此設計）。實例：ffmpeg 缺失時 467 檔全 `FileNotFoundError`（log 中 934 次 traceback = 467×2），exit code 仍是 0。
2. **ffmpeg 是隱性系統依賴**：`load_audio` 只走 ffmpeg CLI，無 Python fallback；必須在 PATH 上（`.venv\Scripts`）。`ffprobe` 未被 `load_audio` 使用，但一併放置以便排查。
3. **`load_audio` 的錯誤訊息被包裝**：`except` 重跑一次後 `raise RuntimeError("音频加载失败")`，看不到真正的 stderr。排查 ffmpeg 問題時先確認 `shutil.which('ffmpeg')`。
4. **`3-bert\` 在 ja 時確實為空**：符合預期，但第 1 步仍會載入 RoBERTa（`1-get-text.py:61-62`）——T0 的 safetensors 轉檔是必要的。
5. **2-get-hubert-wav32k.py 有增量續跑**：`name2go` 開頭 `if os.path.exists(hubert_path): return`。已存在的 `.pt` 會跳過；要全新重跑需先清空 `4-cnhubert`。
6. **PA-9 門檻只計非 # 來源**：`#` 是官方 G2P 的韻律邊界標記（T1 已證明），`unk_sources` 中 `#` 佔多數屬正常，不計入「非 # UNK 總數 ≤ 20、影響行數 ≤ 10」的停止條件。
7. **venv 隔離**：`<RYZA>` 與 `<GPT>` 各用獨立 venv（前者 ASR／ECAPA，後者訓練），numpy 版本不同（2.2.6 vs 1.26.4），不可混用。
8. **TF32**：`s2_train_v3_lora.py:44-45` 預設開啟。本機從未發生 TF32 相關損壞，依官方原版設定即可，見 [`08_troubleshooting.md`](08_troubleshooting.md) §4。

---

## 5. 訓練

### 5.1 執行環境與前置

| 項目 | 值 |
|---|---|
| CWD | **`<GPT>`**（`exp_dir` 為相對路徑，`config.py:137`；CWD 錯會寫錯位置） |
| Python | `<GPT>\.venv\Scripts\python.exe`（3.10.11、torch 2.5.1+cu121、numpy 1.26.4） |
| 必須設定的環境變數 | `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`、`PYTHONPATH=<GPT>;<GPT>\GPT_SoVITS`、`version=v4`、`is_half=True`、`hz=25hz`、`_CUDA_VISIBLE_DEVICES=0` |
| PATH | 必須含 `<GPT>\.venv\Scripts`（`load_audio` 靠它找 ffmpeg） |
| 驅動程式 | `<RYZA>\tools\run_lora_train.py`（已內建上述環境變數，勿手動組 `python` 命令） |

**前置：建立推論權重目錄。** `SoVITS_weights_v4`／`GPT_weights_v4` 在整個 GPT-SoVITS repo 中**只有 `webui.py:200-201` 會建立**；手動驅動繞過 webui 時若沒有這兩個目錄，`process_ckpt.py:37` 會 `FileNotFoundError`，而該錯誤被同檔 `:59` 的裸 `except` 吞成一行 INFO log、**exit code 仍為 0**（PK-010）。`run_lora_train.py` 已內建 preflight 自動建立。

### 5.2 冒煙測試

正式訓練前，先用 20 筆的獨立 exp 確認 S2、S1、resume、推論都能端到端跑通，並量測顯存與速度。
冒煙 exp 必須和正式 exp 分開：S2 會自動從 `logs_s2_*` 內最新的 `G_*.pth` 接續，共用 exp 會讓冒煙 checkpoint 被誤當成正式訓練的起點。

| 步驟 | 指令或工具 | 說明 |
|---|---|---|
| 啟動腳本自檢 | `python tools/run_lora_train_selftest.py` | 不載入模型，以小程式驗證環境變數、CWD、log 分檔與結束碼 |
| 建立冒煙 exp | `python tools/build_smoke_exp.py` | 從正式預處理產出取前 20 筆，建立 `logs/ryza_v4_lora_smoke`；路徑寫死在檔內 |
| 訓練 | 見下方指令 | 使用 `configs/*_smoke.*`：S2 跑 1 epoch 再 resume 到 2 epoch，S1 跑 1 epoch |
| 推論 | `python tools/infer_smoke.py --mode base`、`--mode smoke` | 分別以底模與冒煙權重各生成 3 句 |

```bash
cd <GPT>
python <RYZA>/tools/run_lora_train.py --stage s2 --config <RYZA>/configs/s2_lora_ryza_smoke.json    --tag s2_smoke        --watchdog-hours 0 --vram
python <RYZA>/tools/run_lora_train.py --stage s2 --config <RYZA>/configs/s2_lora_ryza_smoke_e2.json --tag s2_smoke_resume --watchdog-hours 0 --vram
python <RYZA>/tools/run_lora_train.py --stage s1 --config <RYZA>/configs/s1_ryza_smoke.yaml         --tag s1_smoke        --watchdog-hours 0 --vram
python <RYZA>/tools/infer_smoke.py --mode base
python <RYZA>/tools/infer_smoke.py --mode smoke
```

**通過條件**（結束碼不可信，一律檢查產出）：

| 對象 | 條件 |
|---|---|
| S2 | stderr 無 Traceback；`skipped_phone`、`skipped_dur` 皆為 0；loss 無 NaN／Inf；`G_233333333333.pth` 存在；`SoVITS_weights_v4` 出現 LoRA 權重（檔頭 `04`，約為底模的 10%）；resume 從 epoch 2 接續，且 `global_step` 不歸零 |
| S1 | 無 Traceback；`3-bert` 為空時仍能訓練；loss 無 NaN／Inf；`logs_s1_v4/ckpt` 只保留最新；`GPT_weights_v4/*-e*.ckpt` 可在 CPU 載入，頂層三鍵為 `config`／`info`／`weight` |
| 推論 | BERT 與 HuBERT 載入無例外；`[ACTUAL]` 行顯示的權重路徑符合預期（見 §6.1 的 custom 陷阱）；每句 48 kHz、時長 0.5～20 s、無 NaN、非全靜音；`tts_infer.yaml` 已還原 |
| 共通 | 無 `illegal instruction`、`CUBLAS` 錯誤；正式 exp `logs/ryza_v4_lora` 內沒有任何訓練產物；`<GPT>` 工作樹乾淨 |

冒煙測試的數據只用來估算正式訓練的耗時與顯存，不能代表正式訓練，因為樣本會被擴充到 100 筆（見 §5.4）。

### 5.3 訓練指令（先 S1，再 S2）

```bash
cd <GPT>
# S1（GPT／AR）— 先跑，顯存低、風險小
python <RYZA>\tools\run_lora_train.py --stage s1 \
  --config <RYZA>\configs\s1_ryza.yaml --tag s1_train --vram
# S2（SoVITS LoRA）— 顯存高
python <RYZA>\tools\run_lora_train.py --stage s2 \
  --config <RYZA>\configs\s2_lora_ryza.json --tag s2_train --vram
```

| 參數 | 說明 |
|---|---|
| `--watchdog-hours` | 8 小時上限（預設 8.0，**S2＋S1 合計**；`0` 關閉）。到點 `proc.terminate()`，不續跑、不自動刪檔 |
| `--stall-timeout-min` | 無 log 輸出門檻（預設 15 分鐘），到點終止並記錄。⚠️ 已知缺陷見 PK-014 |
| `--vram` | 經 `tools\vram_runner.py` 以 `runpy` 載入官方腳本，結束時印 `torch.cuda.max_memory_allocated()`。**官方腳本本身無此輸出**（grep 0 命中），且此法不修改任何官方檔 |
| `--no-gpu-sample` | 關閉 nvidia-smi 每 5 秒取樣 |
| `--tag` | log 檔名前綴，寫入 `<RYZA>\reports\logs\raw\` |

### 5.4 訓練參數

| 項目 | S2（SoVITS LoRA） | S1（GPT／AR） |
|---|---|---|
| config | `configs/s2_lora_ryza.json` | `configs/s1_ryza.yaml` |
| batch_size | 4 | 8 |
| **epochs** | **16** | **16** |
| **grad_ckpt** | **true（必要）** | 不適用 |
| lora_rank | 32 | 不適用 |
| log_interval | 20 | 不適用（Lightning 只在 epoch 層記錄） |
| if_save_latest | 1 | 1 |
| if_save_every_weights | true | true |
| save_every_epoch／save_every_n_epoch | 1 | 1 |
| 學習率 | 1e-4，`lr_decay=0.999875` | **固定 0.002**（見 §8 #5） |
| precision | fp16（GradScaler） | `16-mixed` |
| seed | 1234 | 1234 |
| step 數 | 119 step/epoch × 16 = **1,904** | 59 step/epoch × 16 = **944** |
| 實測耗時 | 928 s（8 ep 483 s ＋ 續訓 8 ep 445 s） | 184 s（98 s ＋ 86 s） |
| 顯存（`max_memory_allocated`） | 3,504–3,508 MiB | 4,642–4,705 MiB |
| 整卡峰值（nvidia-smi） | 4,603–5,093 MiB | 6,465–10,218 MiB |

> **兩個 dataset 都會自動擴充到至少 100 筆**（S2 為 `module/data_utils.py:546-551`，S1 為 `AR/data/dataset.py` 的同構寫法）。正式 467 > 100，故不擴充。
> **S2 每 epoch 的 step 數不等於「筆數 ÷ batch」**：`DistributedBucketSampler` 依長度分桶並補齊（`data_utils.py:994-1016`），所以 467 筆、batch 4 是 119 step，而不是 117。
> **S1 的實際 batch**由 `AR/data/data_module.py:51` 的 `max(min(batch_size, len(dataset)//4), 1)` 決定；467//4 = 116 > 8，故實際 batch 就是設定值 8。即使只有 20 筆（冒煙測試），資料集也會先擴充到 100（100//4 = 25 > 8），這個下限同樣不會生效。

#### 設定欄位依據

各欄位在官方腳本中的讀取處。修改 config 前請先確認：

**S2**（`s2_train_v3_lora.py` ＋ `webui.py:489-560` open1Ba）

| config 位置 | 讀取處 | 說明 |
|---|---|---|
| `train.batch_size` | `s2_train_v3_lora.py:101` | 4 |
| `train.log_interval` | `:330` `global_step % hps.train.log_interval` | 20 |
| `train.epochs` | `:229` `range(epoch_str, hps.train.epochs + 1)` | 16（見 §8 #6） |
| `train.if_save_latest` | `:345` | 1（固定檔名 G_233333333333.pth，每 epoch 覆寫） |
| `train.if_save_every_weights` | `:361` `== True`（須為 bool） | true |
| `train.save_every_epoch` | `:344` `epoch % hps.train.save_every_epoch` | **官方 `s2.json` 無此欄，open1Ba 才注入**；`s2_train_v3_lora.py:344` 直接讀取，缺欄會 AttributeError，故必補 1 |
| `train.lora_rank` | `:142,144` `hps.train.lora_rank` | 32 |
| `train.grad_ckpt` | `:319` `use_grad_ckpt=hps.train.grad_ckpt` | **true**（必要，見 §5.6） |
| `train.pretrained_s2G` | `:196-205` 載入底模 | `GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth`（存在） |
| `train.pretrained_s2D` | **不被 `s2_train_v3_lora.py` 讀取**（只有 webui 寫入它） | 官方 v4 預設路徑 `.../s2Dv4.pth`；**該檔不存在**，webui 的 `check_pretrained_is_exist` 對 `s2Dv3`／`s2Dv4` 特別豁免（`webui.py:177`），LoRA 訓練只訓 G 不訓 D，故無害 |
| `train.gpu_numbers` | `:9` `hps.train.gpu_numbers.replace("-", ",")` | "0" |
| `model.version` | `:96-97` 選 V4 的 DataLoader／Collate | **"v4"（硬性）** |
| `data.exp_dir`、`s2_ckpt_dir` | `:78,81,142`；`:142` save_root = `{exp_dir}/logs_s2_{version}_lora_{lora_rank}` | 皆 `logs/ryza_v4_lora`（exp_root="logs" 相對路徑，**CWD 必須是 `<GPT>`**，`config.py:137`） |
| `save_weight_dir` | `:375` 推論權重輸出目錄 | `SoVITS_weights_v4`（`config.py:60-67`） |
| `name` | `:375,379` 推論權重檔名前綴 `{name}_e{epoch}_s{step}_l{lora_rank}` | `ryza_v4_lora` |

**S1**（`s1_train.py` ＋ `webui.py:590-650` open1Bb）

| yaml 位置 | 讀取處 | 說明 |
|---|---|---|
| `train.batch_size` | `s1_train.py:101` 經 Trainer | 8 |
| `train.epochs` | `:112` `max_epochs` | 16（見 §8 #6） |
| `train.save_every_n_epoch` | `:105` `every_n_epochs` | 1（官方已有） |
| `train.if_save_latest` | `:40,52-58` my_model_ckpt：存新 ckpt 後刪舊的 | 1 |
| `train.if_save_every_weights` | `:62-84` 每 epoch 存 `{half_weights_save_dir}/{exp_name}-e{epoch}.ckpt`（半精度） | true |
| `train.half_weights_save_dir` | `:77` | `GPT_weights_v4`（`config.py:68-75`） |
| `train.exp_name` | `:78` 檔名前綴 | `ryza_v4_lora` |
| `train.if_dpo` | `t2s_lightning_module.py:45` `config["train"].get("if_dpo", False)` | **`s1_train.py` 不讀，但 lightning module 讀**；open1Bb 會注入，故補 false |
| `pretrained_s1`（頂層） | `t2s_lightning_module.py` 經 config 載入 | `GPT_SoVITS/pretrained_models/s1v3.ckpt`（`config.py:25` v4 沿用 v3） |
| `train_semantic_path`（頂層） | `s1_train.py:134` | `logs/ryza_v4_lora/6-name2semantic.tsv` |
| `train_phoneme_path`（頂層） | `:135` | `logs/ryza_v4_lora/2-name2text.txt` |
| `output_dir`（頂層） | `:88-91`；`output_dir/ckpt` 為 checkpoint 目錄 | `logs/ryza_v4_lora/logs_s1_v4` |
| `optimizer.warmup_steps` | `t2s_lightning_module.py:142` | 2000（官方預設；實際無效，見 §8 #5） |

### 5.5 產出

| 路徑 | 內容 |
|---|---|
| `GPT_weights_v4/ryza_v4_lora-e{1..16}.ckpt` | S1 推論權重，各 155,312,893 B（fp16，頂層 `weight`/`config`/`info`） |
| `SoVITS_weights_v4/ryza_v4_lora_e{1..16}_s*_l32.pth` | S2 LoRA 權重，各 75,550,062 B（檔頭 `04`、`lora_rank=32`） |
| `logs/ryza_v4_lora/logs_s2_v4_lora_32/G_233333333333.pth` | resume 用 checkpoint（`if_save_latest=1`，固定檔名每 epoch 覆寫） |
| `logs/ryza_v4_lora/logs_s1_v4/ckpt/epoch=*.ckpt` | Lightning checkpoint（`if_save_latest=1`，只留最新） |
| `<RYZA>\reports\logs\raw\{tag}.{out,err,gpu}`、`{tag}_summary.json` | 原始 log 與結構化摘要（受 `.gitignore` 忽略，不提交） |

**resume 行為**：S2 啟動時若 `logs_s2_v4_lora_32/` 內有 `G_*.pth`，會自動從其 `iteration` 接續（`utils.latest_checkpoint_path`，注意其排序用整條絕對路徑的數字）。S1 由 Lightning 從 `output_dir/ckpt/` 挑最新的接續。兩者都**不會**清理舊 checkpoint。

**S1 沒有逐步 loss**：Lightning module 的 `self.log` 未設 `on_step=True`，TensorBoard 只有 epoch 層的 tag，stdout 進度棒也只保留最後一步。要看 S1 的收斂只能看 epoch 層的 `total_loss_epoch`。

### 5.6 顯存

選型時「12 GB 有餘裕」的判斷**不成立**（見 [`06_model_selection.md`](06_model_selection.md) §七）。正式訓練（467 筆）的實測：

| grad_ckpt | 整卡峰值 | 結果 |
|---|---|---|
| `false` | **11958 MiB / 12282 MiB（97.3%）** | ❌ epoch 2 起進程假死（caching allocator 反覆 `cudaFree`／重試），**不拋 OOM** |
| `true` | **4603 MiB（37.5%）** | ✅ 16 epoch 順利完成 |

**`grad_ckpt=true` 是本機跑 v4 LoRA 的必要設定**，不是可選項。詳見 [`../reports/finetune_report.md`](../reports/finetune_report.md) §4-1 與 [`08_troubleshooting.md`](08_troubleshooting.md) PK-013。

**顯存不足時的調整順序**：**`grad_ckpt` → `batch_size` 4→2 → `lora_rank` 32→16**；梯度累積備案（本次未用）；不動層數。

---

## 6. 推論

### 6.1 產生樣本

```bash
cd <GPT>
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
PYTHONPATH="<GPT>;<GPT>\GPT_SoVITS" \
version=v4 is_half=True hz=25hz _CUDA_VISIBLE_DEVICES=0 \
PATH="<GPT>\.venv\Scripts:$PATH" \
<GPT>\.venv\Scripts\python.exe <RYZA>\tools\stage_d_infer.py \
  --s1-prefix ryza_v4_lora-e --s2-prefix ryza_v4_lora_e16 \
  --out-dir logs/ryza_v4_lora/infer_e16 \
  --summary-name stage_d_infer_summary_e16.json
```

| 項目 | 值 |
|---|---|
| S1 權重 | `GPT_weights_v4/ryza_v4_lora-e16.ckpt` |
| S2 權重 | `SoVITS_weights_v4/ryza_v4_lora_e16_s1904_l32.pth`（推論端自動先載底模再合併 LoRA，底模必須存在） |
| 參考音訊 | `<RYZA>\data\ref\ref.wav`（＝ `wav/03536.wav`） |
| **prompt text** | `よかった。また手伝えることがあったらいつでも言ってね。`（03536 原文；**勿用 `ref.txt`**，見 [`05_dataset_split.md`](05_dataset_split.md) §3） |
| 版本／裝置／精度 | `version=v4`、`device=cuda`、`is_half=True`（透過 `TTS_Config` 的 **`custom` 鍵**傳入） |
| 產出 | `logs/ryza_v4_lora/infer_e16/*.wav`，**48 kHz／mono／16-bit**（`logs/` 被 `.gitignore` 忽略） |

**推論參數**（`stage_d_infer.py` 內固定）：`top_k=15`、`top_p=1`、`temperature=1`、`text_split_method="cut1"`、`batch_size=1`、`batch_threshold=0.75`、`split_bucket=True`、`speed_factor=1.0`、`seed=1234`、`parallel_infer=False`、`repetition_penalty=1.35`、`sample_steps=32`、`super_sampling=False`。

> ⚠️ **`TTS_Config` 的 custom 陷阱**（`TTS.py:318`）：`self.configs = configs_.get("custom", configs_["v2"])`。自訂權重路徑**必須放在 `custom` 鍵之下**；放在頂層會被靜默忽略並退回 v2 預設權重。`stage_d_infer.py` 每次都會印出 `[ACTUAL]` 行證明實際生效的路徑。
> ⚠️ 推論會改寫 `GPT_SoVITS/configs/tts_infer.yaml`（`TTS.py:384-385`，每次 `init_vits_weights`／`init_t2s_weights` 都呼叫 `save_configs()`）。該檔是 **git 追蹤**檔，`stage_d_infer.py` 會在結束後比對 SHA256 並在改動時 `git checkout --` 還原。

**樣本內容**：(a) `data/finetune_eval10.list` 的 10 句原台詞（檔名沿用原編號 `eval10_NNNNN.wav`）；(b) 10 句遊戲語料庫中不存在的全新台詞（`new_01..new_10.wav`，平敘 4／疑問 2／興奮 2／悲傷 2，已以 grep `data/split_manifest.csv` 逐句驗證 0 命中）。全文見 `../reports/finetune_report.md` 附錄 A。

### 6.2 推論端隱性依賴

推論（`TTS_infer_pack.TTS`）在**模組層級**依賴下列兩樣東西，與語言無關，即使只做日文推論也會被擋。缺任一樣，整個推論管線都無法 import。

#### 6.2.1 fast_langdetect 語言偵測模型

| 項目 | 值 |
|---|---|
| 需求來源 | `GPT_SoVITS/text/LangSegmenter/langsegmenter.py:11` 把 `fast_langdetect` 的 `cache_dir` 指到 `GPT_SoVITS/pretrained_models/fast_langdetect/` |
| 放置路徑 | `<GPT>\GPT_SoVITS\pretrained_models\fast_langdetect\lid.176.bin` |
| 大小 | 131,266,198 bytes |
| SHA256 | `7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e` |
| 下載來源 | `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin`（即 `fast_langdetect/infer.py` 的 `FASTTEXT_LARGE_MODEL_URL`，檔名常數 `FASTTEXT_LARGE_MODEL_NAME`） |
| 套件版本 | `fast_langdetect>=0.3.1`（**已列於 `<GPT>\requirements.txt` 第 29 行**，非本專案新增） |
| 取得方式 | 一次性下載。`fast_langdetect` 只在使用「套件預設 cache 目錄」時自動建立目錄；對自訂目錄會直接拋 `FileNotFoundError` 並轉而嘗試連網下載 |

**替代做法（不需下載）**：套件本身內附 `resources/lid.176.ftz`（938 KB），可用 `LangDetectConfig(custom_model_path=...)` 指向它。冒煙測試時曾以此法離線驗證，正式推論改為放置正式模型檔。
注意：若走此路徑，必須在 `from text.LangSegmenter import LangSegmenter` **之後**才設定，因為 `langsegmenter.py:11` 會在 import 時把偵測器重新指回預設路徑。

#### 6.2.2 jieba_fast（不採用，但別名不可刪）

| 項目 | 值 |
|---|---|
| 決定 | jieba_fast 為中文專用分詞器，本專案 language=ja **不需要**，不安裝，也不列為待補齊的依賴（2026-10-05） |
| 現況 | venv 內**從未安裝**（`importlib.util.find_spec("jieba_fast")` = None），故無任何東西需要移除 |
| 宣告位置 | 僅 `<GPT>\requirements.txt` 第 26 行，且那是 **GPT-SoVITS 官方的上游相依**（第 27 行另有 `jieba`），非本專案新增——不修改官方檔案 |
| 為何裝不上（背景） | PyPI 不提供 `jieba_fast` 的任何 wheel，只有 sdist；安裝需 MSVC 14.0 編譯 C 擴充，本機無此工具鏈。套件內無純 Python fallback |

**⚠️ 但別名程式碼不能刪。** `text/tone_sandhi.py:17`、`text/chinese.py:19,23`、`text/chinese2.py:20,24` 都在**模組層級**無條件 `import jieba_fast`，而 `TTS_infer_pack/TextPreprocessor.py:13` 直接 `from text import chinese`，因此**整個推論管線連純日文也無法 import**：

```text
>>> from TTS_infer_pack.TTS import TTS
ModuleNotFoundError: No module named 'jieba_fast'
  File "GPT_SoVITS\text\tone_sandhi.py", line 17, in <module>
```

這不是「可選相依」，而是**上游 import 鏈強迫該模組必須存在**。目前的解法（`tools/infer_smoke.py`、`tools/stage_d_infer.py` 的 `install_jieba_fast_alias()`）把 `jieba_fast` 別名到**已安裝的 `jieba`**——而 `jieba` 本來就是官方第 27 行宣告、且日文推論路徑（`text/LangSegmenter/langsegmenter.py:5`）本來就在用的套件。**因此這個別名不為本專案增加任何新相依**，只是繞開上游對 jieba_fast 的硬性 import。

日文推論全程不呼叫 jieba（`cleaner.py` 的 `language_module_map["ja"] = "japanese"`），已用 20 句實證正確。

---

## 7. 評估

評估在 **`<RYZA>\.venv`** 執行（該 venv 才有 faster-whisper 與 speechbrain；`<GPT>\.venv` 兩者皆無）。

```bash
cd <RYZA>
HF_HUB_OFFLINE=1 .venv\Scripts\python.exe tools\stage_d_eval.py \
  --summary stage_d_infer_summary_e16.json --out stage_d_eval_e16.json

# 文字稿與專案既有 CER 工具交叉驗證
python tools\cer_report.py --input reports\cer_generated.csv --output reports\cer_gen.md
```

| 項目 | 做法 |
|---|---|
| 聲線相似度 | `tools/spk_model.py`（ECAPA-TDNN）算生成音訊對 **A_high 中心點**的 cosine。A_high 中心點＝`ryza_main/A_high/*.wav`（92 筆）對 `tools/embeddings.npy` 的 L2-normalized 均值再正規化，算法取自 `tools/refine_labels.py:18-28` |
| 基線 | **測試集真實音訊**對同一中心點的平均 cosine（`finetune_test.list` 24 筆＝0.8832；其子集 `finetune_eval10.list` 10 筆＝0.8908），門檻＝基線 × 90% |
| 可懂度 | faster-whisper `large-v3`（`local_files_only=True`、`HF_HUB_OFFLINE=1`），`language=ja`、`task=transcribe`、`vad_filter=True`、`beam_size=5`（沿用專案既有設定） |
| CER | **沿用專案既有實作** `tools/cer_report.py` 的 `core_text()` ＋ `levenshtein()`，以維持與 ASR 抽查（1.0%）可比。`core_text` 會濾掉 CJK／假名／長音符／英數以外的所有字元（含標點），故標點差不計入 |

評估結果、主觀試聽評語與缺陷分析見 [`../reports/finetune_report.md`](../reports/finetune_report.md) 與 [`../reports/defect_triage.md`](../reports/defect_triage.md)。

---

## 8. 已知偏差

| # | 偏差 | 說明 | 是否需處理 |
|---|---|---|---|
| 1 | **`#` → UNK** | 日文 G2P 的 `pyopenjtalk_g2p_prosody` 在 accent phrase 邊界插入 `#`（`text/japanese.py:248`），而 `symbols2.py:783-788` 的 732 個符號**不含 `#`**。train 467 行中 451 行含 UNK、共 1,776 個；eval10 35 個 | **否**。T1 已證明推論端走同一 `clean_text` 路徑，訓練／推論兩端一致 |
| 2 | **transformers 與 torch 版本衝突** | transformers 4.57.6 因 CVE-2025-32434 拒載 `.bin`，而 torch 2.5.1+cu121 未達 2.6 門檻；升級 torch 會破壞 cu121 綁定 | 已由 T0 轉檔解決（§2.2），不升級 torch |
| 3 | **`3-get-semantic.py` 把 v4 推斷成 v3** | 該腳本不讀 `version` 環境變數（`:17` 已註解），改依 `pretrained_s2G` 的**檔案大小**推斷；`s2Gv4.pth` 769,025,545 B > 700 MB → 判為 `"v3"` | **否**。v3/v4 共用 `SynthesizerTrnV3`，此階段不需要 v4 的 vocoder；log 顯示 `<All keys matched successfully>` |
| 4 | **jieba_fast 未安裝** | 中文專用分詞器，本專案 language=ja **不需要**，2026-10-05 決定不採用 | 否，但**行程內別名不可刪**（§6.2.2） |
| 5 | **S1 學習率恆為 0.002** | `AR/modules/lr_schedulers.py:58` 把 lr 硬寫成 `self.lr = lr = self.end_lr = 0.002`，覆蓋 warmup/cosine 公式；`:38-42` 的 `set_lr` 亦忽略傳入值。故 `optimizer.{lr,lr_init,lr_end,warmup_steps,decay_steps}` **五個欄位全部無效** | 接受，不修改官方排程器 |
| 6 | **epochs 8 → 16** | 原定 8，但 8 epoch 的成品聲線相似度僅達真實基線的 **86.9%**（門檻 90%）。續訓至 16 後為 **91.5%** | 已採 16。若要回到 8，把兩個 config 的 `epochs` 改回 8 即可，但相似度會未達標 |
| 7 | **`grad_ckpt` false → true** | 正式訓練時 `false` 導致顯存假死（§5.6） | 已採 true，屬必要設定 |
| 8 | **v4 無 D 模型** | 官方 v4 發行包只附 `s2Gv4.pth` 與 `vocoder.pth`；config 的 `pretrained_s2D` 指向不存在的 `s2Dv4.pth` | 否。`s2_train_v3_lora.py` 全程不讀該欄位 |
| 9 | **`data/ref/ref.txt` 與 `ref.wav` 不同步** | `ref.txt` 是 `00042` 的台詞，`ref.wav` 實際是 `03536` | 推論時直接傳 03536 原文，不讀 `ref.txt`（§6.1） |
| 10 | **推論端缺 `%`／`￥` 前處理** | 訓練端 `1-get-text.py:92` 會 `.replace("%","-").replace("￥",",")`，推論端 `TextPreprocessor` 沒有 | 否。本資料集 train／eval10 含這兩個字元的行數皆為 0；新台詞若含這兩個字元需自行替換 |

---

## 9. 疑難排解

依症狀查原因與處理方式，見 [`08_troubleshooting.md`](08_troubleshooting.md) §5「症狀速查」；各坑的完整記錄是同檔的 PK-001～PK-015。

---

## 10. 相關文件

| 文件 | 內容 |
|---|---|
| [`01_overview.md`](01_overview.md) | 專案目標、模塊架構、評估標準 |
| [`05_dataset_split.md`](05_dataset_split.md) | 切分與參考音訊 |
| [`06_model_selection.md`](06_model_selection.md) | 基礎模型選型、備選方案 |
| [`08_troubleshooting.md`](08_troubleshooting.md) | 坑記錄與症狀速查 |
| [`../tools/README.md`](../tools/README.md) | 工具目錄 |
| [`../reports/finetune_report.md`](../reports/finetune_report.md) | 訓練與評估結果（含生成樣本文字稿） |
| [`../reports/logs/stage_d_train.md`](../reports/logs/stage_d_train.md) | 正式訓練紀錄 |

---

## 附錄 A：上游程式碼關鍵行號

以下皆為 `<GPT>`（GPT-SoVITS @ `48b1a01`）內的位置。手動驅動官方腳本（不經 webui）時，這些是最容易踩到的地方。

| 事項 | 位置 | 說明 |
|---|---|---|
| 推論權重目錄只有 webui 會建立 | `webui.py:200-201` | `config.py` 全文無 `os.makedirs`（[PK-010](08_troubleshooting.md)） |
| `savee` 的裸 `except` | `process_ckpt.py:59-60` | `return traceback.format_exc()`，呼叫端只 `logger.info`，結束碼仍為 0 |
| `my_save2` 不建目錄 | `process_ckpt.py:37` | `open(path, "wb")` 前沒有 `makedirs` |
| S2 resume 分支 | `s2_train_v3_lora.py:177-189` | `utils.latest_checkpoint_path(save_root, "G_*.pth")`；**`:190` 的裸 `except` 會在 resume 失敗時靜默退回底模** |
| checkpoint 排序 | `utils.py:114` | 排序鍵是整條絕對路徑中的數字，不是檔名 |
| S2 樣本自動擴充 | `module/data_utils.py:546-551` | `min_num=100`，重複 `max(2, 100//leng)` 次 |
| S2 時長過濾 | `module/data_utils.py:588` | `if 54 > duration > 0.6`，範圍外的樣本會被剔除 |
| S2 分桶 | `module/data_utils.py:994-1016` | `DistributedBucketSampler` 依長度分桶並補齊 |
| S2 需 webui 注入的欄位 | `s2_train_v3_lora.py:344`、`:361` | `save_every_epoch`、`if_save_every_weights` 官方 `s2.json` 沒有，是 webui open1Ba 注入（`webui.py:527-528`）；手動執行不補會 AttributeError |
| S1 batch 下限 | `AR/data/data_module.py:51` | `max(min(batch_size, len(dataset)//4), 1)`，作用在已擴充的 dataset 上 |
| S1 BERT 缺檔補零 | `AR/data/dataset.py:223-237`、`:276-282` | 缺檔 → `bert_feature=None` → collate 時 `zero_()` |
| S1 `if_dpo` | `AR/models/t2s_lightning_module.py:45` | `s1_train.py` 不讀，但 lightning module 讀；webui open1Bb 會注入（`webui.py:622`） |
| S1 checkpoint 三鍵 | `s1_train.py:62-81` | `weight`（fp16）／`config`／`info` |
| S1 排程器硬寫常數 | `AR/modules/lr_schedulers.py:38-42`、`:58` | `self.lr = lr = self.end_lr = 0.002`（[PK-012](08_troubleshooting.md)） |
| 推論與訓練共用 G2P | `TTS_infer_pack/TextPreprocessor.py:15`、`:206-210` | 與 `1-get-text.py:20` 同一個 `clean_text`；version 由 `TTS_Config` 傳入（`TTS.py:331-333`、`:399-401`、`:1171`） |
| `#` 的來源 | `text/japanese.py:248`、`:267` | `g2p()` 預設 `with_prosody=True`，accent phrase 邊界插入 `#`；`symbols2.py:783-788` 的 732 個符號不含 `#` |
| `TTS.run` 是 generator | `TTS.py:997` | 呼叫端須 `next()` 取得 `(sr, audio)`，`api_v2.py:441` 即如此 |
| custom 陷阱 | `TTS.py:318` | `self.configs = configs_.get("custom", configs_["v2"])`；頂層傳入的鍵被靜默忽略 |
| `tts_infer.yaml` 被改寫 | `TTS.py:384-385`、`:590`、`:597` | 每次 `init_vits_weights`／`init_t2s_weights` 都會 `save_configs()` |
| `jieba_fast` 模組層級依賴 | `text/chinese.py:19-23`、`text/tone_sandhi.py:17`、`TTS_infer_pack/TextPreprocessor.py:13` | 日文推論也會被擋（[PK-011](08_troubleshooting.md)） |
| `fast_langdetect` 路徑 | `text/LangSegmenter/langsegmenter.py:11` | 自訂 `cache_dir` 不存在時會改為連網下載 |
| 官方腳本不輸出顯存 | 全 repo `grep -rn max_memory_allocated` 為 0 命中（排除 `.venv`） | 因此以 `tools/vram_runner.py` 量測 |
