# Part 2 環境與模型準備報告

## 執行摘要

於 `H:\git\GPT-SoVITS`（主分支 commit `48b1a01`）建立獨立 Python 3.10 虛擬環境，安裝 PyTorch cu121 與 GPT-SoVITS 全部依賴，下載 v4 預訓練權重，完成日文 G2P 驗證與 TF32 相關性記錄（來源標示為其他環境過往經驗，本機未發生）。

**本階段僅準備環境與權重，尚未開始訓練，未執行任何預處理腳本。**

---

## 環境資訊

| 項目 | 值 |
|---|---|
| 作業系統 | Windows 10 19045 x64 |
| Python | 3.10.11 |
| PyTorch | 2.5.1+cu121 |
| torchvision / torchaudio | 0.20.1+cu121 / 2.5.1+cu121 |
| CUDA（torch 綁定） | 12.1 |
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| VRAM | 12282 MiB |
| Driver | 591.86 |
| `torch.cuda.is_available()` | **True** |
| 已安裝套件數 | 201（完整清單見 `reports/env_pip_freeze.txt`） |
| GPT-SoVITS commit | `48b1a01`（Fix Fun-ASR-Nano Transformers requirement #2824） |

---

## 依賴清單

完整 pip freeze 見 [`env_pip_freeze.txt`](env_pip_freeze.txt)（201 行）。關鍵套件：

| 套件 | 版本 | 備註 |
|---|---|---|
| torch | 2.5.1+cu121 | `--index-url https://download.pytorch.org/whl/cu121` |
| numpy | 1.26.4 | 降級自 2.2.6（見下方偏差說明） |
| transformers | 4.57.6 | |
| peft | 0.17.1 | LoRA 訓練所需 |
| librosa | 0.10.2 | |
| pytorch-lightning | 2.6.6 | |
| onnxruntime-gpu | 1.23.2 | |
| funasr | 1.4.16 | |
| gradio | 4.44.1 | |

### 與 requirements.txt 的偏差（Windows 環境必要調整）

| 套件 | requirements.txt 指定 | 實際安裝 | 原因 |
|---|---|---|---|
| pyopenjtalk | `pyopenjtalk>=0.4.1` | **pyopenjtalk-prebuilt 0.3.0** | 原套件需 CMake + MSVC 從原始碼建置；本機無 MSVC 編譯器（`CMAKE_C_COMPILER not set`）。改用 API 相容的預編譯版本，`pyopenjtalk.g2p()` 與 `OPEN_JTALK_DICT_DIR` 均可用 |
| opencc | `--no-binary=opencc`（強制原始碼建置） | opencc 1.4.2（預編譯 wheel） | 同上，原始碼建置需編譯器；改用預編譯 wheel |
| jieba_fast | `jieba_fast` | **未安裝**（保留 `jieba 0.42.1`） | 無 Windows 預編譯 wheel，原始碼建置需編譯器。`text/cleaner.py` 採延遲 import（`__import__("text." + lang)`），`ja` 路徑只載入 `text.japanese`，不載入 `chinese.py`／`chinese2.py`，故日文訓練與推論不需要。若未來需中文路徑，須補裝 MSVC Build Tools 或建立 jieba 相容層 |
| numpy | `numpy<2.0` | 1.26.4 | torch 安裝帶入 numpy 2.2.6 導致 pyopenjtalk-prebuilt 二進位不相容（`numpy.dtype size changed`）；降級至 1.26.4，同時符合 requirements.txt 的 `numpy<2.0` |

---

## 模型清單與大小

全部自 `huggingface.co/XXXXRT/GPT-SoVITS-Pretrained` 下載，存放於 `H:\git\GPT-SoVITS\GPT_SoVITS\pretrained_models\`：

| 模型 | 路徑 | 大小 | 載入驗證 |
|---|---|---|---|
| SoVITS v4 底模 | `gsv-v4-pretrained/s2Gv4.pth` | 769,025,545 B（769 MB） | torch.load 成功（4 個頂層鍵） |
| Vocoder | `gsv-v4-pretrained/vocoder.pth` | 57,781,109 B（57.8 MB） | torch.load 成功（194 鍵） |
| GPT 底模 | `s1v3.ckpt` | 155,284,856 B（155 MB） | torch.load 成功（3 個頂層鍵） |
| BERT（文字特徵） | `chinese-roberta-wwm-ext-large/` | pytorch_model.bin 651 MB ＋ config.json 963 B ＋ tokenizer.json 269 KB | 三檔齊全 |
| HuBERT（語音特徵） | `chinese-hubert-base/` | pytorch_model.bin 189 MB ＋ config.json 1.4 KB ＋ preprocessor_config.json 212 B | 三檔齊全 |

路徑依 `config.py` 的 `pretrained_sovits_name["v4"]`、`pretrained_gpt_name["v4"]`，以及 `inference_webui.py:513`／`export_torch_script_v3v4.py:535` 的 vocoder 路徑，與官方約定一致。

---

## 日文 G2P 驗證

```python
pyopenjtalk.g2p('こんにちは')
# -> k o N n i ch i w a

pyopenjtalk.g2p('錬金術の勉強は難しいけど、すごく楽しいんだ。')
# -> r e N k i N j u ts u n o b e N ky o o w a m u z u k a sh i i k e d o pau s u g o k U t a n o sh i i N d a
```

經 GPT-SoVITS 自身程式碼路徑驗證（`text/cleaner.py` → `text/japanese.py`）：

```python
clean_text('こんにちは、ライザです。', 'ja', 'v2')
# -> phones: k o [ N n i ch i w a , r a ] i z a d e s u .

clean_text('今日も錬金術の研究を頑張るよ！', 'ja', 'v2')
# -> ky o ] o m o UNK r e [ N k i ] ...
```

- **pyopenjtalk 基礎 G2P：通過**（音素序列正確，OpenJTalk 字典自動下載至 venv）。
- **clean_text 日文全鏈：通過**（`ja` 路徑的 import 鏈完整可運作）。
- 待確認（不影響本階段門禁）：clean_text 輸出中可見 `[`／`]` 韻律標記與 `UNK`。`text/japanese.py` 的 g2p 對 prosody 版本以 `[1:-1]` 裁剪外層括號；WebUI 1Aa 實跑時傳入的 `version` 環境變數與符號表對應，將在訓練階段首次執行 1Aa 時確認，若出現大量 UNK 需回頭檢查 version 與符號表。

---

## TF32 記錄（不 patch）

本專案 `tools/spk_model.py:14-19` 記錄：**TF32 matmul 曾在其他環境（RTX 3060 / 當時 driver）造成 CUDA 記憶體損壞**（約 20–30 檔後 illegal instruction / CUBLAS_STATUS_INTERNAL_ERROR）。**本機（RTX 4070 SUPER）未再發生**，該記錄屬其他環境的過往經驗，非本機已確認的問題。

GPT-SoVITS 官方腳本的 TF32 設定位置：

| 位置 | 設定 | 與本專案記錄的關係 |
|---|---|---|
| `GPT_SoVITS/s2_train_v3_lora.py:44-45` | `allow_tf32 = True`（**預設開啟**） | v4 LoRA 訓練腳本；與過往 TF32 不穩記錄相關，本機是否重現待驗證 |
| `GPT_SoVITS/s2_train.py:44-45` | `allow_tf32 = True` | v1/v2/v2Pro/v2ProPlus 訓練腳本 |
| `GPT_SoVITS/s2_train_v3.py:44-45` | `allow_tf32 = True` | v3 訓練腳本 |
| `GPT_SoVITS/TTS_infer_pack/TTS.py:210` | `allow_tf32 = False` | 推論端已關閉 |

**處理方式**：依本階段約定**不 patch、僅記錄**。由於本機尚未發生此問題，優先以官方原版設定直接訓練；若訓練時出現 CUDA 記憶體損壞徵兆（illegal instruction / CUBLAS_STATUS_INTERNAL_ERROR），再於 `s2_train_v3_lora.py` 的 TF32 設定之後補上關閉（保留 cudnn 開啟，僅關 TF32 matmul 與 cudnn TF32；`spk_model.py` 連 `cudnn.enabled` 都關是推論專用做法，訓練不可照抄）。

---

## 門禁檢查

| 項目 | 結果 |
|---|---|
| .venv 建立，Python 3.10 | PASS（3.10.11） |
| `torch.cuda.is_available()` = True | PASS（RTX 4070 SUPER，cu121） |
| requirements.txt 安裝成功 | **PASS（附帶 4 項 Windows 偏差，見上表）** |
| pyopenjtalk 可用，日文 G2P 測試通過 | PASS（pyopenjtalk-prebuilt 0.3.0，pyopenjtalk.g2p 與 clean_text 全鏈） |
| s2Gv4.pth 存在且大小合理 | PASS（769 MB） |
| vocoder.pth 存在且大小合理 | PASS（57.8 MB） |
| BERT / HuBERT / s1v3.ckpt 存在 | PASS（651 MB / 189 MB / 155 MB） |
| finetune_train.list 467 行、路徑存在 | PASS（467 行，0 缺失，4 欄格式 0 異常） |
| TF32 記錄已完成（來源標示為其他環境過往經驗） | PASS（4 處位置，本階段不 patch） |

通過 9/9 項。

---

## 風險與待確認

1. **jieba_fast 缺失**：日文路徑不需要（已由 cleaner.py 延遲 import 設計驗證）；若日後需要中文（zh）或粵語（yue）路徑，`chinese.py`／`chinese2.py` 頂層 `import jieba_fast` 會直接失敗，屆時需安裝 MSVC Build Tools 或改用 jieba 相容層。
2. **TF32 相容性（觀察項）**：`s2_train_v3_lora.py:44-45` 預設開啟 TF32，與其他環境（RTX 3060）過往的 CUDA 記憶體損壞記錄相關。**本機（RTX 4070 SUPER）未再發生**，故不預先 patch，先以官方原版設定訓練；若訓練時出現 CUDA 損壞徵兆再補關閉（見上節）。
3. **pyopenjtalk-prebuilt 版本 0.3.0 < requirements 指定的 0.4.1**：API 相容且 G2P 功能驗證通過；若 0.4.1 有日文相關改進（如韻律標記處理），可能影響 1Aa 輸出品質，待訓練階段觀察。
4. **clean_text 的 `[`／`]` 與 UNK**：需在訓練階段首次執行 1Aa 時確認 version 環境變數與符號表對應是否正確。
5. **未下載 G2PWModel**（`GPT_SoVITS/text/G2PWModel/`，install.sh 另行下載 589 MB）：用於中文 G2P 加速；日文路徑不需要，未下載。若需中文路徑再補。
6. **v4 LoRA 實測顯存**：本階段不訓練，未量測。訓練階段需以 `torch.cuda.max_memory_allocated()` 確認，對照官方 8GB LoRA 門檻與本機 12GB 上限。
7. **環境隔離**：Ryza_voice 與 GPT-SoVITS 使用各自的 venv（前者用於 ASR／ECAPA 評估，後者用於訓練），numpy 版本不同（2.2.6 vs 1.26.4），不可混用。

---

## 聲明

**尚未開始訓練。** 本階段完成：venv 建立、CUDA 驗證、依賴安裝、v4 預訓練權重下載、日文 G2P 驗證、TF32 記錄（來源為其他環境過往經驗）。等待使用者確認後，才進入 GPT-SoVITS v4 LoRA 訓練配置。
