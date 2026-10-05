# GPT-SoVITS v4 LoRA 微調文件

> 本文為萊莎（ライザ）日文語音 v4 LoRA 微調的環境與流程文件。資料集準備見 [03-01_dataset.md](06_dataset.md)，選型分析見 [issue_part2.md](02_issue_part2.md)，G2 預處理驗收見 [../reports/G2_stage_a_report.md](../reports/G2_stage_a_report.md)。

---

## 1. 環境需求

| 項目 | 值 | 備註 |
|---|---|---|
| OS | Windows 10 19045 x64 | |
| Python | 3.10.11（`<GPT>\.venv`） | 獨立 venv，不繼承系統 site-packages |
| GPU | RTX 4070 SUPER（12 GB） | `torch.cuda.is_available()` = True，cu121 |
| GPT-SoVITS | main commit `48b1a01` | 無分支、無修改 |
| **ffmpeg** | **9.0.2-essentials_build（gyan.dev 靜態版）** | **隱性系統依賴，見下** |

### ffmpeg（隱性系統依賴）

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

---

## 2. 預處理三步（階段 A）

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

### 版本推斷（v4 → v3 codepath）

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

### 合併分片

`all_parts=1` 時只有 `-0` 分片。webui 的合併流程（`webui.py:1213-1222`）為：加表頭 `item_name\tsemantic_audio` + 讀入各分片 + `os.remove(分片)` + 寫 LF 結尾。手動驅動時合併後保留 `-0` 原檔（內容已驗證等價）。

注意：官方腳本以 `"\n".join(lines1)` 寫 `-0` 分片（`3-get-semantic.py:117-118`），**最後一行後無換行**，故 `wc -l` 顯示 466 而非 467；合併時已正規化。`2-name2text.txt` 在 Windows 寫出為 CRLF（467 行皆含 `\r`），S1 訓練以 Python 讀取不受影響。

---

## 3. 階段 A 狀態（2026-10-05，G2 已驗收）

| 項目 | train（467） | eval10（10） |
|---|---|---|
| PA-1 2-name2text 行數／欄位 | ✅ 467 行、4 欄 | ✅ 10 行、4 欄 |
| PA-2 4-cnhubert `.pt` 數 | ✅ 467 | ✅ 10 |
| PA-3 5-wav32k 數＋取樣率 | ✅ 467、32000 Hz | ✅ 10、32000 Hz |
| PA-4 6-name2semantic 行數＋表頭 | ✅ 468 行、LF、0 CR | ✅ 11 行 |
| PA-5 五處鍵集合全等 | ✅ | ✅ |
| PA-6 HuBERT 無 NaN／形狀 768×frames | ✅ 抽 20 檔 | ✅ 全檢 10 檔 |
| PA-7 5-wav32k 總大小 | 179.9 MB | 3.4 MB |
| PA-8 3-bert 存在且為空 | ✅（ja 不產 BERT） | ✅ |
| PA-9 非 # UNK | ✅ 0 | ✅ 0 |
| PA-10 音素與 G2P 一致 | ✅ 0 mismatch | ✅ 0 mismatch |
| PA-11 版本推斷 | v3（無害）、載入正常 | 同 |

完整結果：`reports/logs/raw/stage_a_pa.json`（被 `.gitignore` 忽略，不提交）。驗證腳本：`tools/stage_a_validate.py`（唯讀，只寫 json）。

---

## 4. 易錯點

1. **官方腳本的裸 `except` 會吞錯，exit code 恆為 0**：`1-get-text.py:102-103`、`2-get-hubert-wav32k.py:124-125`、`3-get-semantic.py:115-116` 皆印 traceback 後繼續。**結束碼不可作為成功依據**——階段 C 冒煙測試與所有驗收必須以「預期產出檔案數量與內容」為準（本專案 PA／SM 檢查表即此設計）。實例：ffmpeg 缺失時 467 檔全 `FileNotFoundError`（log 中 934 次 traceback = 467×2），exit code 仍是 0。
2. **ffmpeg 是隱性系統依賴**：`load_audio` 只走 ffmpeg CLI，無 Python fallback；必須在 PATH 上（`.venv\Scripts`）。`ffprobe` 未被 `load_audio` 使用，但一併放置以便排查。
3. **`load_audio` 的錯誤訊息被包裝**：`except` 重跑一次後 `raise RuntimeError("音频加载失败")`，看不到真正的 stderr。排查 ffmpeg 問題時先確認 `shutil.which('ffmpeg')`。
4. **`3-bert\` 在 ja 時確實為空**：符合預期，但第 1 步仍會載入 RoBERTa（`1-get-text.py:61-62`）——T0 的 safetensors 轉檔是必要的。
5. **2-get-hubert-wav32k.py 有增量續跑**：`name2go` 開頭 `if os.path.exists(hubert_path): return`。已存在的 `.pt` 會跳過；要全新重跑需先清空 `4-cnhubert`。
6. **PA-9 門檻只計非 # 來源**：`#` 是官方 G2P 的韻律邊界標記（T1 已證明），`unk_sources` 中 `#` 佔多數屬正常，不計入「非 # UNK 總數 ≤ 20、影響行數 ≤ 10」的停止條件。
7. **venv 隔離**：`<RYZA>` 與 `<GPT>` 各用獨立 venv（前者 ASR／ECAPA，後者訓練），numpy 版本不同（2.2.6 vs 1.26.4），不可混用。
8. **TF32**：`s2_train_v3_lora.py:44-45` 預設開啟。**TF32 損壞是其他環境（RTX 3060）的過往異常，本機（RTX 4070 SUPER）從未發生，本地訓練不需要理會**——依官方原版設定跑即可，不 patch、不預防、不監控。記錄僅供未來移地至其他環境時應急參考（見 `docs/09_pitfalls.md` §4）。

---

## 5. 後續階段

- **階段 B**：訓練配置與啟動腳本 ✅ 已完成。
- **階段 C**：冒煙測試 ✅ 已完成（驗收以產出檔案數量與內容為準，不可只看 exit code，§4.1）。
- **階段 D**：正式訓練與品質評估 ✅ 已完成（2026-10-05，結果見 `../reports/finetune_report.md`）。
- v4 LoRA 顯存門檻 8 GB（官方），本機 12 GB **實測不足**：見 §7。

---

## 6. 推論端隱性依賴（階段 D 補齊）

推論（`TTS_infer_pack.TTS`）在**模組層級**依賴兩樣東西，與語言無關——即使只做日文推論也會被擋。

### 6.1 fast_langdetect 語言偵測模型（已就位）

| 項目 | 值 |
|---|---|
| 需求來源 | `GPT_SoVITS/text/LangSegmenter/langsegmenter.py:11` 把 `fast_langdetect` 的 `cache_dir` 指到 `GPT_SoVITS/pretrained_models/fast_langdetect/` |
| 放置路徑 | `<GPT>\GPT_SoVITS\pretrained_modelsast_langdetect\lid.176.bin` |
| 大小 | 131,266,198 bytes |
| SHA256 | `7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e` |
| 下載來源 | `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin`（即 `fast_langdetect/infer.py` 的 `FASTTEXT_LARGE_MODEL_URL`，檔名常數 `FASTTEXT_LARGE_MODEL_NAME`） |
| 套件版本 | `fast_langdetect>=0.3.1`（**已列於 `<GPT>
equirements.txt` 第 29 行**，非本專案新增） |
| 取得方式 | 一次性下載。`fast_langdetect` 只在使用「套件預設 cache 目錄」時自動建立目錄；對自訂目錄會直接拋 `FileNotFoundError` 並轉而嘗試連網下載 |

**替代做法（不需下載）**：套件本身內附 `resources/lid.176.ftz`（938 KB），可用 `LangDetectConfig(custom_model_path=...)` 指向它。階段 C 曾以此法離線驗證，階段 D 改為放置正式模型檔。
注意：若走此路徑，必須在 `from text.LangSegmenter import LangSegmenter` **之後**才設定，因為 `langsegmenter.py:11` 會在 import 時把偵測器重新指回預設路徑。

### 6.2 jieba_fast（**永久不採用**，2026-10-05 PM 決定）

| 項目 | 值 |
|---|---|
| PM 決定 | jieba_fast 為中文專用分詞器，本專案 language=ja **不需要**，**永久不安裝**，不再列為待補齊的依賴 |
| 現況 | venv 內**從未安裝**（`importlib.util.find_spec("jieba_fast")` = None），故無任何東西需要移除 |
| 宣告位置 | 僅 `<GPT>equirements.txt` 第 26 行，且那是 **GPT-SoVITS 官方的上游相依**（第 27 行另有 `jieba`），非本專案新增——不修改官方檔案 |
| 為何裝不上（背景） | PyPI 不提供 `jieba_fast` 的任何 wheel，只有 sdist；安裝需 MSVC 14.0 編譯 C 擴充，本機無此工具鏈。套件內無純 Python fallback |

**⚠️ 但別名程式碼不能刪。** `text/tone_sandhi.py:17`、`text/chinese.py:19,23`、`text/chinese2.py:20,24` 都在**模組層級**無條件 `import jieba_fast`，而 `TTS_infer_pack/TextPreprocessor.py:13` 直接 `from text import chinese`，因此**整個推論管線連純日文也無法 import**：

```text
>>> from TTS_infer_pack.TTS import TTS
ModuleNotFoundError: No module named 'jieba_fast'
  File "GPT_SoVITS	ext	one_sandhi.py", line 17, in <module>
```

這不是「可選相依」，而是**上游 import 鏈強迫該模組必須存在**。目前的解法（`tools/infer_smoke.py`、`tools/stage_d_infer.py` 的 `install_jieba_fast_alias()`）把 `jieba_fast` 別名到**已安裝的 `jieba`**——而 `jieba` 本來就是官方第 27 行宣告、且日文推論路徑（`text/LangSegmenter/langsegmenter.py:5`）本來就在用的套件。**因此這個別名不為本專案增加任何新相依**，只是繞開上游對 jieba_fast 的硬性 import。

日文推論全程不呼叫 jieba（`cleaner.py` 的 `language_module_map["ja"] = "japanese"`），已用 20 句實證正確。

---

## 7. v4 LoRA 顯存：實測結論（階段 D）

階段 C 的「12 GB 有餘裕」判斷**不成立**。正式訓練（467 筆）的實測：

| grad_ckpt | 整卡峰值 | 結果 |
|---|---|---|
| `false` | **11958 MiB / 12282 MiB（97.3%）** | ❌ epoch 2 起進程假死（caching allocator 反覆 `cudaFree`／重試），**不拋 OOM** |
| `true` | **4603 MiB（37.5%）** | ✅ 16 epoch 順利完成 |

**`grad_ckpt=true` 是本機跑 v4 LoRA 的必要設定**，不是可選項。詳見 `../reports/finetune_report.md` §3、§4-1 與 `09_pitfalls.md` PK-013。
