# GPT-SoVITS v4 LoRA 微調文件

> 本文為萊莎（ライザ）日文語音 v4 LoRA 微調的環境與流程文件。資料集準備見 [06_dataset.md](06_dataset.md)，選型分析見 [02_issue_part2.md](02_issue_part2.md)，G2 預處理驗收見 [../reports/G2_stage_a_report.md](../reports/G2_stage_a_report.md)。

## 0. 章節導覽（重現工作流順序）

issue Part 2 要求「從資料準備到推論的完整重現步驟，他人照做可重現」。**請依下表順序閱讀／執行**；本文件的章節編號沿用歷史演进，非工作流順序。

| 工作流步驟 | 本文件章節 | 產出 | 對應 issue 要求 |
|---|---|---|---|
| 0. 環境 | [§1 環境需求](#1-環境需求) | 兩個 venv、ffmpeg | — |
| 1. 解包語料 | [§8 資料準備](#8-資料準備解包--ryza-trainlist) | `wav/`（9,569 段） | — |
| 2. 說話者篩選＋ASR | [§8.2 篩選漏斗](#82-說話者篩選--asr-漏斗-9569--491) | `data/ryza_train.list`（491） | Part 1 |
| 3. 切分 | [§9 切分重現](#9-切分重現-467--24--10) | `finetune_{train,test,eval10}.list` | Part 2「保留 5% 測試集」 |
| 4. 模型準備 | [§10 模型準備](#10-模型準備權重--safetensors-轉檔) | 4 組權重 ＋ T0 轉檔 | Part 2「基礎模型指定 GPT-SoVITS」 |
| 5. 預處理 | [§2 預處理三步](#2-預處理三步階段-a) | `2-name2text`／`4-cnhubert`／`5-wav32k`／`6-name2semantic` | — |
| 6. **訓練** | [§11 訓練重現](#11-訓練重現) | `GPT_weights_v4`／`SoVITS_weights_v4` | Part 2「兩階段皆須 finetune」 |
| 7. **推論** | [§12 推論重現](#12-推論重現) | `logs/ryza_v4_lora/infer_e16/*.wav` | Part 2「生成樣本 (a)(b)」 |
| 8. 評估 | [§12.3 評估重現](#123-評估重現) | `reports/finetune_report.md` | Part 2「聲線相似度／可懂度」 |
| — | [§13 已知偏差](#13-已知偏差) | — | — |
| — | [§14 常見問題](#14-常見問題速查) | — | — |

**踩坑紀錄全文見 [09_pitfalls.md](09_pitfalls.md)**（PK-001～PK-014）。

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
| 放置路徑 | `<GPT>\GPT_SoVITS\pretrained_models\fast_langdetect\lid.176.bin` |
| 大小 | 131,266,198 bytes |
| SHA256 | `7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e` |
| 下載來源 | `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin`（即 `fast_langdetect/infer.py` 的 `FASTTEXT_LARGE_MODEL_URL`，檔名常數 `FASTTEXT_LARGE_MODEL_NAME`） |
| 套件版本 | `fast_langdetect>=0.3.1`（**已列於 `<GPT>\requirements.txt` 第 29 行**，非本專案新增） |
| 取得方式 | 一次性下載。`fast_langdetect` 只在使用「套件預設 cache 目錄」時自動建立目錄；對自訂目錄會直接拋 `FileNotFoundError` 並轉而嘗試連網下載 |

**替代做法（不需下載）**：套件本身內附 `resources/lid.176.ftz`（938 KB），可用 `LangDetectConfig(custom_model_path=...)` 指向它。階段 C 曾以此法離線驗證，階段 D 改為放置正式模型檔。
注意：若走此路徑，必須在 `from text.LangSegmenter import LangSegmenter` **之後**才設定，因為 `langsegmenter.py:11` 會在 import 時把偵測器重新指回預設路徑。

### 6.2 jieba_fast（**永久不採用**，2026-10-05 PM 決定）

| 項目 | 值 |
|---|---|
| PM 決定 | jieba_fast 為中文專用分詞器，本專案 language=ja **不需要**，**永久不安裝**，不再列為待補齊的依賴 |
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

## 7. v4 LoRA 顯存：實測結論（階段 D）

階段 C 的「12 GB 有餘裕」判斷**不成立**。正式訓練（467 筆）的實測：

| grad_ckpt | 整卡峰值 | 結果 |
|---|---|---|
| `false` | **11958 MiB / 12282 MiB（97.3%）** | ❌ epoch 2 起進程假死（caching allocator 反覆 `cudaFree`／重試），**不拋 OOM** |
| `true` | **4603 MiB（37.5%）** | ✅ 16 epoch 順利完成 |

**`grad_ckpt=true` 是本機跑 v4 LoRA 的必要設定**，不是可選項。詳見 `../reports/finetune_report.md` §3、§4-1 與 `09_pitfalls.md` PK-013。

---

## 8. 資料準備（解包 → ryza_train.list）

### 8.1 從遊戲解包語音

來源路徑 `D:\SteamLibrary\steamapps\common\Atelier Ryza 2` **唯讀**，全程不寫入。容器格式為 Koei Tecmo 的 KTSR「Sound Ring」，主庫音訊本體是 **KOVS 容器包 Ogg Vorbis** 的兩層混淆（`03_unpack.md:93-98`）。

工具：**vgmstream r2117**，預編譯版位於 `tools/vgmstream/vgmstream-cli.exe`。

| 庫 | 來源檔 | 輸出 | 產出 |
|---|---|---|---|
| VOICE 主庫 | `VOICE.ktsl2asbin`（同名 `.ktsl2stbin` 自動帶入） | `wav/` | **9,569** WAV、13.76 小時、4.5 GB、48 kHz/16 bit/mono |
| 戰鬥庫 | `014_battle_SP_character`、`015_battle_character` | `battle_voice/` | **646** WAV（4-bit ADPCM，6 個 stereo） |

```bash
# 主庫（必須指向 .ktsl2asbin）
tools/vgmstream/vgmstream-cli.exe -s 1 -S 0 -o "wav/?05s.wav" \
  "D:\SteamLibrary\steamapps\common\Atelier Ryza 2\Data\x64\Sound\VOICE.ktsl2asbin"

# 戰鬥庫（?n 為 cue 名稱，經 LCG XOR 解密，seed 0xAF4905A9 @ 0x0c）
tools/vgmstream/vgmstream-cli.exe -s 1 -S 0 -o "battle_voice/015_?n.wav" \
  "D:\SteamLibrary\steamapps\common\Atelier Ryza 2\Data\x64\Sound\015_battle_character.ktsl2asbin"
```

參數：`-s 1` 輸出所有 subsong；`-S 0` 不輸出 subsong 索引資訊；`?05s` 五位數索引、`?n` cue 名稱。

> ⚠️ **此指令會覆寫既有輸出目錄**（`03_unpack.md:118`）。`voice_index.csv` 的 `cue_index = N` 對應 `wav/(N+1):05d.wav`（vgmstream 的 subsong 編號從 1 起算）。

### 8.2 說話者篩選 + ASR 漏斗（9,569 → 491）

```text
9,569  wav/ 全部解包音檔
  ↓ ECAPA-TDNN 說話者嵌入（tools/spk_model.py），9570×192 @ tools/embeddings.npy
  ↓ 錨點 wav/00030.wav + wav/00042.wav，consensus_score = 兩者 cosine 平均
  ↓ LinearSVC(C=1) 決策邊界 svm_dec > -0.433
分層：A_high 92 / B_likely 383 / C_possible 728 / D_flagged 165 / B_review 40 / wrong 22
  ↓ tools/select_samples.py --limit 500 --copy-dir ryza_main/top500
500   top500 序列
  −3   top500 內人工確認非萊莎者（0365 / 0442 / 0484）
=497  tools/pos_label.py
  ↓ ASR（faster-whisper large-v3）＋ 6 條過濾規則 → 觸發 0 次
  ↓ DUPLICATE_TEXT（core 文字跨檔重複 ≥2）→ 4 組 10 筆，排除 6 筆
=491  data/ryza_train.list
```

**重跑 ASR 的完整指令**（`05_transcript_pipeline.md:234-266`）：

```bash
# 步驟 3：從 top500 造冊（497）
python tools/build_pos_label.py --source-dir ryza_main/top500 --output tools/pos_label.py

# 步驟 4+5：ASR，門檻 ≥2 生效 → json 497 筆（6 筆 EXCLUDE）
.venv/Scripts/python.exe tools/screen_asr.py \
  --pos-label tools/pos_label.py --audio-root wav \
  --raw-out tools/asr_screening.json

# 步驟 6：排除 6 筆，pos_label 與 json 同步為 491
python tools/build_pos_label.py --source-dir ryza_main/top500 \
  --exclude-from tools/asr_screening.json --prune-raw tools/asr_screening.json \
  --output tools/pos_label.py

# 步驟 7：產清單（純標準庫，不重跑 ASR）
python tools/transcribe.py --pos-label tools/pos_label.py \
  --asr-json tools/asr_screening.json --output data/ryza_train.list

# 驗證：三處一致
python tools/validate_pos_label.py --source-dir ryza_main/top500 \
  --pos-label tools/pos_label.py --neg-label tools/neg_labels.py \
  --asr-json tools/asr_screening.json --train-list data/ryza_train.list --audio-root wav
```

**產出格式**：`音檔路徑|ryza|ja|文字`，**相對路徑**，491 行。

> **專有名詞校正**：`tools/apply_glossary.py` 套用 6 條規則（15 hits / 14 檔，`wav/02580.wav` 命中兩條），重算 core 後再由 `transcribe.py` 重產清單。

> `export_ryza.py`、`final_index.py`、`score_ryza.py`、`refine_labels.py` **無 argparse**，路徑寫死在模組層級，必須在 repo 根目錄以無參數執行。

---

## 9. 切分重現（467 / 24 / 10）

```bash
.venv/Scripts/python.exe tools/split_train_test.py \
  --input data/ryza_train.list \
  --train-out data/finetune_train.list \
  --test-out data/finetune_test.list \
  --eval-out data/finetune_eval10.list \
  --manifest data/split_manifest.csv \
  --ref-dir data/ref \
  --report reports/split_report.md \
  --seed 42
```

> ⚠️ `06_dataset.md:103-111` 另記錄一版省略 `--manifest`／`--ref-dir` 並加上 `--audio-root wav` 的指令。兩者產出相同（前者兩項有預設值），但 `data/ryza_train.list` 存的是**相對路徑**，而預設 `--audio-root` 已是 repo 根目錄，**加 `wav` 會解析成 `<root>/wav/wav/xxxxx.wav`**。以本節指令為準。

### 9.1 分層抽樣設計

兩層分層，seed 固定 42：

**第一層＝句末標點分類**（`split_train_test.py:58-70`）

| 類別 | 判斷 | test 配額 | eval10 配額 |
|---|---|---|---|
| `Q` | 結尾 `?` `？` | 4 | 2 |
| `EXCL` | 結尾 `!` `！` | 2 | 1 |
| `ELL` | 結尾 `…` 或 `...` | 3 | 1 |
| `DECL` | 其餘 | 15 | 6 |
| | **合計** | **24** | **10** |

**第二層＝各類別內的時長三分位**：依 `(duration, utt_id)` 排序，桶索引 `= (pos*3)//len(items)`，`d0` 最短、`d1` 中、`d2` 最長。配額依桶大小以**最大餘數法**分配，再由 `random.Random(seed)` 依固定 `CLASS_ORDER = [DECL, EXCL, ELL, Q]` 抽樣。

manifest 的 `bucket` 欄即 `f'{class}-d{bucket_index}'`，例如 `DECL-d1` = 平敘類、中時長三分位；`Q-d0` = 疑問類、最短三分位。

### 9.2 產出與門檻

| 檔案 | 內容 |
|---|---|
| `data/finetune_train.list` | **467** 行，四欄，**絕對路徑** |
| `data/finetune_test.list` | **24** 行 |
| `data/finetune_eval10.list` | **10** 行（`eval10` 是 test 的子集） |
| `data/split_manifest.csv` | 491 列 ＋ 表頭，11 欄：`utt_id,audio_path,speaker,duration,text,split,bucket,md5,source_rel,class,eval10` |
| `data/ref/ref.wav` | 參考音訊（來源見 §9.3） |
| `reports/split_report.md` | 切分報告 |

**寫檔前的 11 項門檻全部通過才會落盤**（`split_train_test.py:279-318`，任一失敗 `sys.exit(1)`）：train=467、test=24、eval10=10、**train ∩ test = ∅**、train ∪ test = 491、eval10 ⊆ test、輸入 SHA256 未變、參考音訊取自 train、逐列路徑存在且四欄齊全、speaker=ryza、lang=ja、文字非空且不含 `|`、無重複路徑。

時長統計：train 49.11 分、test 2.36 分、eval10 0.92 分、全集 51.47 分；MD5 重複組 **0**。

### 9.3 參考音訊的選法

規則（`split_train_test.py:300-310`）：在 **train** 內依原始清單順序，取第一個 `4.0 ≤ duration ≤ 8.0` 且峰值 `< 32767` 者；退而求其次放寬為 `3.0–10.0`。

本次選中 `wav/00042.wav`（清單第 3 行、utt_id `00042`、7.12 s、峰值 31488/32767）。

> ⚠️ **`data/ref/ref.txt` 與 `ref.wav` 不同步**：`ref.txt` 的內容是 `00042` 的台詞，而磁碟上的 `ref.wav` 實際已被覆蓋為 `wav/03536.wav`（md5 `1f63d4fceb5bdebb1b556e37c8331276`）。**推論時的 prompt text 必須用 `wav/03536.wav` 的原文**：`よかった。また手伝えることがあったらいつでも言ってね。`（見 §12.1）。`ref.wav` 本身被 `.gitignore` 忽略，`ref.txt` 則為追蹤檔。

---

## 10. 模型準備（權重 + safetensors 轉檔）

### 10.1 必須存在的權重

全部下載自 `huggingface.co/XXXXRT/GPT-SoVITS-Pretrained`，置於 `<GPT>\GPT_SoVITS\pretrained_models\`：

| 用途 | 相對路徑 | 大小 | 驗證 |
|---|---|---|---|
| SoVITS v4 底模（G） | `gsv-v4-pretrained/s2Gv4.pth` | **769,025,545 B** | `torch.load` OK，4 個頂層鍵 |
| v4 專用 vocoder | `gsv-v4-pretrained/vocoder.pth` | **57,781,109 B** | `torch.load` OK，194 鍵 |
| GPT／AR 底模 | `s1v3.ckpt` | **155,284,856 B** | `torch.load` OK，3 個頂層鍵 |
| BERT | `chinese-roberta-wwm-ext-large/` | 見 §10.2 | — |
| HuBERT | `chinese-hubert-base/` | 見 §10.2 | — |

**注意 `s1v3.ckpt` 直接位於 `pretrained_models/` 下，不在 `gsv-v4-pretrained/` 裡。** v4 沿用 v3 的 GPT 權重（`config.py:25`）。

**v4 發行包只附 G、不附 D**：`gsv-v4-pretrained/` 只有 `s2Gv4.pth` 與 `vocoder.pth`。`webui.py:177` 對 `s2Dv3`／`s2Dv4` 特別豁免缺檔檢查，而 `s2_train_v3_lora.py` 全程不讀 `pretrained_s2D`——故 config 裡的 `pretrained_s2D` 填官方預設的 `s2Dv4.pth`（檔案不存在）**無害**。

**刻意未下載**：`GPT_SoVITS/text/G2PWModel/`（589 MB，僅中文 G2P 用）。

### 10.2 T0：`.bin` → safetensors（必要）

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

**驗收數據**（`G1_t0_t1_report.md` §2）：

| 項目 | chinese-roberta-wwm-ext-large | chinese-hubert-base |
|---|---|---|
| `.bin` SHA256 | `e53a693acc59ace251d143d068096ae0d7b79e4b1b503fa84c9dcf576448c1d8` | `24164f129c66499d1346e2aa55f183250c223161ec2770c0da3d3b08cf432d3c` |
| `.bin` 大小 | 651,225,145 B | 188,811,417 B |
| `.bin` 鍵數／dtype | 397（fp16×396 ＋ int64×1） | 211（全 fp16） |
| `model.safetensors` SHA256 | `ddb2d1e2aec45a149bb1b19213c0478cee464f9cc75531e70e673b5224ef4f05` | `25adc31d1889ff5d3da262189433bdc755f0ddb9f194342583dbd17e447daefe` |
| `model.safetensors` 大小 | **694,454,896 B** | **188,767,040 B** |

轉檔**不刪除也不修改** `.bin`，key 名稱不變；共用儲存的 tensor 以 `data_ptr` 偵測並從第二次出現起 clone（roberta 有 2 組：word_embeddings↔decoder.weight、bias↔decoder.bias；hubert 為 0 組）。驗證腳本 `tools/t0_validate.py` 會比對 key/dtype/shape 並附 negative fixture。

> ⚠️ **新環境注意**：下載 HuggingFace 預設格式通常只有 `pytorch_model.bin`。若轉檔後仍報 CVE 錯誤，檢查 `model.safetensors` 是否真的落在 `from_pretrained` 指向的目錄。

### 10.3 fast_langdetect 模型檔

見 §6.1（階段 D 已放置，`lid.176.bin`，131,266,198 B，SHA256 `7e69ec54…62764e`）。

---

## 11. 訓練重現

### 11.1 執行環境與前置

| 項目 | 值 |
|---|---|
| CWD | **`<GPT>`**（`exp_dir` 為相對路徑，`config.py:137`；CWD 錯會寫錯位置） |
| Python | `<GPT>\.venv\Scripts\python.exe`（3.10.11、torch 2.5.1+cu121、numpy 1.26.4） |
| 必須設定的環境變數 | `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`、`PYTHONPATH=<GPT>;<GPT>\GPT_SoVITS`、`version=v4`、`is_half=True`、`hz=25hz`、`_CUDA_VISIBLE_DEVICES=0` |
| PATH | 必須含 `<GPT>\.venv\Scripts`（`load_audio` 靠它找 ffmpeg） |
| 驅動程式 | `<RYZA>\tools\run_lora_train.py`（已內建上述環境變數，勿手動組 `python` 命令） |

**前置：建立推論權重目錄。** `SoVITS_weights_v4`／`GPT_weights_v4` 在整個 GPT-SoVITS repo 中**只有 `webui.py:200-201` 會建立**；手動驅動繞過 webui 時若沒有這兩個目錄，`process_ckpt.py:37` 會 `FileNotFoundError`，而該錯誤被同檔 `:59` 的裸 `except` 吞成一行 INFO log、**exit code 仍為 0**（PK-010）。`run_lora_train.py` 已內建 preflight 自動建立。

### 11.2 訓練指令（先 S1，再 S2）

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

### 11.3 訓練參數

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
| 學習率 | 1e-4，`lr_decay=0.999875` | **固定 0.002**（見 §13） |
| precision | fp16（GradScaler） | `16-mixed` |
| seed | 1234 | 1234 |
| step 數 | 119 step/epoch × 16 = **1,904** | 59 step/epoch × 16 = **944** |
| 實測耗時 | 928 s（8 ep 483 s ＋ 續訓 8 ep 445 s） | 184 s（98 s ＋ 86 s） |
| 顯存（`max_memory_allocated`） | 3,504–3,508 MiB | 4,642–4,705 MiB |
| 整卡峰值（nvidia-smi） | 4,603–5,093 MiB | 6,465–10,218 MiB |

> **兩個 dataset 都會自動擴充到至少 100 筆**（S2 為 `module/data_utils.py:546-551`）。正式 467 > 100，故不擴充。
> **S1 的實際 batch**由 `AR/data/data_module.py:51` 的 `max(min(batch_size, len(dataset)//4), 1)` 決定；467//4 = 116 > 8，故實際 batch 就是設定值 8。計畫 §10.1 曾預期 20 筆時會被壓到 5，實際上因資料集先被擴充到 100 而不會發生。

### 11.4 產出

| 路徑 | 內容 |
|---|---|
| `GPT_weights_v4/ryza_v4_lora-e{1..16}.ckpt` | S1 推論權重，各 155,312,893 B（fp16，頂層 `weight`/`config`/`info`） |
| `SoVITS_weights_v4/ryza_v4_lora_e{1..16}_s*_l32.pth` | S2 LoRA 權重，各 75,550,062 B（檔頭 `04`、`lora_rank=32`） |
| `logs/ryza_v4_lora/logs_s2_v4_lora_32/G_233333333333.pth` | resume 用 checkpoint（`if_save_latest=1`，固定檔名每 epoch 覆寫） |
| `logs/ryza_v4_lora/logs_s1_v4/ckpt/epoch=*.ckpt` | Lightning checkpoint（`if_save_latest=1`，只留最新） |
| `<RYZA>\reports\logs\raw\{tag}.{out,err,gpu}`、`{tag}_summary.json` | 原始 log 與結構化摘要（受 `.gitignore` 忽略，不提交） |

**resume 行為**：S2 啟動時若 `logs_s2_v4_lora_32/` 內有 `G_*.pth`，會自動從其 `iteration` 接續（`utils.latest_checkpoint_path`，注意其排序用整條絕對路徑的數字）。S1 由 Lightning 從 `output_dir/ckpt/` 挑最新的接續。兩者都**不會**清理舊 checkpoint。

### 11.5 顯存不足時的調整順序

**`grad_ckpt` → `batch_size` 4→2 → `lora_rank` 32→16**；梯度累積備案（本次未用）；不動層數。

本機實測：`grad_ckpt=false` 時整卡峰值達 **11958/12282 MiB（97.3%）**並在 epoch 2 假死（**不拋 OOM**，是 caching allocator 反覆 `cudaFree`／重試）；改 `true` 後峰值 **4603 MiB**。詳見 §7 與 PK-013。

---

## 12. 推論重現

### 12.1 產生參考音訊

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
| **prompt text** | `よかった。また手伝えることがあったらいつでも言ってね。`（03536 原文；**勿用 `ref.txt`**，見 §9.3） |
| 版本／裝置／精度 | `version=v4`、`device=cuda`、`is_half=True`（透過 `TTS_Config` 的 **`custom` 鍵**傳入） |
| 產出 | `logs/ryza_v4_lora/infer_e16/*.wav`，**48 kHz／mono／16-bit**（`logs/` 被 `.gitignore` 忽略） |

**推論參數**（`stage_d_infer.py` 內固定）：`top_k=15`、`top_p=1`、`temperature=1`、`text_split_method="cut1"`、`batch_size=1`、`batch_threshold=0.75`、`split_bucket=True`、`speed_factor=1.0`、`seed=1234`、`parallel_infer=False`、`repetition_penalty=1.35`、`sample_steps=32`、`super_sampling=False`。

> ⚠️ **`TTS_Config` 的 custom 陷阱**（`TTS.py:318`）：`self.configs = configs_.get("custom", configs_["v2"])`。自訂權重路徑**必須放在 `custom` 鍵之下**；放在頂層會被靜默忽略並退回 v2 預設權重。`stage_d_infer.py` 每次都會印出 `[ACTUAL]` 行證明實際生效的路徑。
> ⚠️ 推論會改寫 `GPT_SoVITS/configs/tts_infer.yaml`（`TTS.py:384-385`，每次 `init_vits_weights`／`init_t2s_weights` 都呼叫 `save_configs()`）。該檔是 **git 追蹤**檔，`stage_d_infer.py` 會在結束後比對 SHA256 並在改動時 `git checkout --` 還原。

**樣本內容**：(a) `data/finetune_eval10.list` 的 10 句原台詞（檔名沿用原編號 `eval10_NNNNN.wav`）；(b) 10 句遊戲語料庫中不存在的全新台詞（`new_01..new_10.wav`，平敘 4／疑問 2／興奮 2／悲傷 2，已以 grep `data/split_manifest.csv` 逐句驗證 0 命中）。全文見 `../reports/finetune_report.md` 附錄 A。

### 12.2 推論前置依賴

見 §6：fast_langdetect 模型檔（已放置）與 jieba_fast 別名（**不可刪**）。兩者缺任一，整個推論管線無法 import。

### 12.3 評估重現

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
| CER | **沿用專案既有實作** `tools/cer_report.py` 的 `core_text()` ＋ `levenshtein()`，以維持與 Part 1（1.0% 基線）可比。`core_text` 會濾掉 CJK／假名／長音符／英數以外的所有字元（含標點），故標點差不計入 |

**本次結果**：相似度 0.8152 / 門檻 0.8017 ＝ **91.5%** ✅；CER 逐句平均 **2.99%**、整體加權 2.62% ✅。完整數據與逐句明細見 `../reports/finetune_report.md`。

> **主觀試聽評語（聲線／發音／語氣自然度與明顯缺陷）尚待提交**，本次未產出。

---

## 13. 已知偏差

| # | 偏差 | 說明 | 是否需處理 |
|---|---|---|---|
| 1 | **`#` → UNK** | 日文 G2P 的 `pyopenjtalk_g2p_prosody` 在 accent phrase 邊界插入 `#`（`text/japanese.py:248`），而 `symbols2.py:783-788` 的 732 個符號**不含 `#`**。train 467 行中 451 行含 UNK、共 1,776 個；eval10 35 個 | **否**。T1 已證明推論端走同一 `clean_text` 路徑，訓練／推論兩端一致 |
| 2 | **transformers 與 torch 版本衝突** | transformers 4.57.6 因 CVE-2025-32434 拒載 `.bin`，而 torch 2.5.1+cu121 未達 2.6 門檻；升級 torch 會破壞 cu121 綁定 | 已由 T0 轉檔解決（§10.2），不升級 torch |
| 3 | **`3-get-semantic.py` 把 v4 推斷成 v3** | 該腳本不讀 `version` 環境變數（`:17` 已註解），改依 `pretrained_s2G` 的**檔案大小**推斷；`s2Gv4.pth` 769,025,545 B > 700 MB → 判為 `"v3"` | **否**。v3/v4 共用 `SynthesizerTrnV3`，此階段不需要 v4 的 vocoder；log 顯示 `<All keys matched successfully>` |
| 4 | **`<GPT>` 根目錄 `=0.4.1`** | 計畫預期的「工作樹僅 =0.4.1」；該檔已在階段 A 期間消失，現為**完全乾淨** | 否，僅記錄 |
| 5 | **jieba_fast 未安裝** | 中文專用分詞器，本專案 language=ja **不需要**，2026-10-05 經 PM 決定永久不採用 | 否，但**行程內別名不可刪**（§6.2） |
| 6 | **S1 學習率恆為 0.002** | `AR/modules/lr_schedulers.py:58` 把 lr 硬寫成 `self.lr = lr = self.end_lr = 0.002`，覆蓋 warmup/cosine 公式；`:38-42` 的 `set_lr` 亦忽略傳入值。故 `optimizer.{lr,lr_init,lr_end,warmup_steps,decay_steps}` **五個欄位全部無效** | 接受，不修改官方排程器 |
| 7 | **epochs 8 → 16** | PM 原定案 8，但 8 epoch 的成品聲線相似度僅達真實基線的 **86.9%**（門檻 90%）。續訓至 16 後為 **91.5%** | 已採 16。若要回到 8，把兩個 config 的 `epochs` 改回 8 即可，但相似度會未達標 |
| 8 | **`grad_ckpt` false → true** | 正式訓練時 `false` 導致顯存假死（§7、§11.5） | 已採 true，屬必要設定 |
| 9 | **v4 無 D 模型** | 官方 v4 發行包只附 `s2Gv4.pth` 與 `vocoder.pth`；config 的 `pretrained_s2D` 指向不存在的 `s2Dv4.pth` | 否。`s2_train_v3_lora.py` 全程不讀該欄位 |
| 10 | **`data/ref/ref.txt` 與 `ref.wav` 不同步** | `ref.txt` 是 `00042` 的台詞，`ref.wav` 實際是 `03536` | 推論時直接傳 03536 原文，不讀 `ref.txt`（§12.1） |
| 11 | **`data/06_dataset.md` 的切分指令與腳本 docstring 不一致** | 前者省略 `--manifest`／`--ref-dir` 並多給 `--audio-root wav`；因 `ryza_train.list` 存相對路徑，額外的 `wav` 會解析成 `<root>/wav/wav/…` | 以 §9 的指令為準 |
| 12 | **`README.md` 與 `08_tools.md` 的版控敘述過時** | 兩處仍稱 `docs/07_finetune.md`、`09_pitfalls.md` 及 5 個 tools 腳本「尚未納入版控」，但 `git ls-files` 顯示皆已追蹤 | 文件更新項，非流程問題 |

---

## 14. 常見問題速查

| 症狀 | 原因 | 處理 |
|---|---|---|
| 預處理 exit code 0，但 `4-cnhubert\`／`5-wav32k\` 為空 | 官方腳本裸 `except` 吞錯（`1-get-text.py:102-103` 等）。缺 ffmpeg 時 467 檔全 `FileNotFoundError`，exit code 仍 0 | **不要看 exit code**，看產出檔數量。查 `shutil.which('ffmpeg')`（PK-001、PK-004） |
| `FileNotFoundError: 'SoVITS_weights_v4/xxx.pth'`，但 log 只印一行 INFO | `process_ckpt.savee` 的裸 `except` 把 traceback 當回傳字串，呼叫端只 `logger.info` | 建立 `SoVITS_weights_v4`／`GPT_weights_v4`（`run_lora_train.py` 已自動做）（PK-010） |
| 訓練進入 epoch 2 後停滯，GPU util 100%、顯存不動、無 OOM | 顯存接近上限，caching allocator 陷入 `cudaFree`／重試迴圈 | 開 `grad_ckpt=true`；判別法：util 100% ＋ 顯存平 ＋ CPU 時間持續上升＝allocator 迴圈（真死鎖時 util 會掉 0）（PK-013） |
| 訓練「無輸出」但 watchdog 沒反應 | watchdog 看 `.out`/`.err` **檔案大小**，tqdm 在非 tty 時緩衝約 8 KB，週期性 flush 會把計時器歸零 | 目前**未解**。應改為監控 `<exp_dir>/train.log`（PK-014） |
| `ModuleNotFoundError: No module named 'jieba_fast'` | 上游 import 鏈無條件要求該模組存在，與語言無關 | 不可刪別名；用 `install_jieba_fast_alias()`（§6.2、PK-011） |
| `FileNotFoundError: fast-langdetect: Cache directory not found` | `langsegmenter.py:11` 指向的 cache 目錄不存在，套件轉而嘗試連網下載 | 放置 `lid.176.bin`（§6.1、§10.3） |
| 自訂權重路徑沒生效，跑出來是底模 | `TTS_Config` 只讀 `custom` 鍵，頂層傳入被靜默忽略（`TTS.py:318`） | 包成 `{"custom": {...}}`，並印出實際生效路徑核對（§12.1） |
| `git status` 出現 `tts_infer.yaml` | 推論的 `save_configs()` 每次都會改寫它 | 推論後 `git checkout --` 該單檔（§12.1） |
| S2 續訓時挑錯 checkpoint | `utils.latest_checkpoint_path` 的排序鍵是**整條絕對路徑**的所有數字，不是檔名 | 確認 `logs_s2_*/` 內只有預期的 `G_*.pth` |
| `transformers` 報 CVE-2025-32434 | `.bin` 載入被禁 | 執行 T0 轉檔（§10.2、PK-002） |
| 顯存取樣讀值小於 torch 自報值 | nvidia-smi 每 5 秒取樣，短輪次會漏掉峰值 | 以 `torch.cuda.max_memory_allocated()` 為準 |
| `tools/select_samples.py --output ryza_train.list` 寫錯位置 | 該腳本 docstring 的 `--output` 範例已過時 | 正式清單是 `data/ryza_train.list`，由 `transcribe.py` 產出（§8.2） |

---

## 15. 相關文件

| 文件 | 內容 |
|---|---|
| [`01_issue.md`](01_issue.md) | issue 全文（Part 1 ASR、Part 2 finetune 的驗收要件） |
| [`02_issue_part2.md`](02_issue_part2.md) | 基礎模型選型分析、fallback 方案 |
| [`03_unpack.md`](03_unpack.md) | 語音解包細節（KTSR/KOVS、vgmstream 參數） |
| [`04_speaker_id.md`](04_speaker_id.md) | 說話者識別與分層 |
| [`05_transcript_pipeline.md`](05_transcript_pipeline.md) | ASR pipeline 與過濾規則 |
| [`06_dataset.md`](06_dataset.md) | 資料集與切分說明 |
| [`08_tools.md`](08_tools.md) | 工具腳本說明 |
| [`09_pitfalls.md`](09_pitfalls.md) | 踩坑紀錄全文（PK-001～PK-014） |
| [`../reports/split_report.md`](../reports/split_report.md) | 切分驗收（11/11 通過） |
| [`../reports/G1_t0_t1_report.md`](../reports/G1_t0_t1_report.md) | T0 轉檔驗收、T1 UNK 分析 |
| [`../reports/G2_stage_a_report.md`](../reports/G2_stage_a_report.md) | 預處理驗收（PA-1～PA-13） |
| [`../reports/G4_stage_c_report.md`](../reports/G4_stage_c_report.md) | 冒煙測試驗收 |
| [`../reports/finetune_report.md`](../reports/finetune_report.md) | **訓練與品質評估報告**（含文字稿附錄 A） |
| [`../reports/logs/stage_d_train.md`](../reports/logs/stage_d_train.md) | 訓練 log 歸檔 |
