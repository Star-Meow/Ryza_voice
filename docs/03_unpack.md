# 語音解包（Part 1 ①）

> 本文是 `Ryza_voice` 專案第一階段的完整記錄：如何從遊戲檔案取出音訊、兩種語音庫的結構差異、以及容器格式與加密的逆向結果。
> 專案總覽見 [`README.md`](../README.md)，後續階段見 [`speaker_id.md`](04_speaker_id.md) 與 [`transcript_pipeline.md`](05_transcript_pipeline.md)。

---

## 1. 資料來源與授權前提

來源為 `D:\SteamLibrary\steamapps\common\Atelier Ryza 2`，**全程唯讀，未對遊戲目錄做任何寫入或修改**。
音聲資料版權屬 Koei Tecmo / Gust，僅供個人用途；解出的音訊因版權與體積因素**不納入版控**。

格式為光榮特庫摩自有的 **KTSR「Sound Ring」** 封裝，vgmstream 有官方支援（KTSC/KTSR/KOVS）。

## 2. 兩種語音庫

| | VOICE 主庫 | 戰鬥庫 |
|---|---|---|
| 輸出目錄 | `wav/` | `battle_voice/` |
| 來源檔 | `VOICE.ktsl2asbin`（+ 同名 `.ktsl2stbin`） | `014_battle_SP_character`、`015_battle_character` |
| 檔數 | **9,569** 個 WAV，13.76 小時，4.5 GB | **646** 個 WAV |
| 音訊規格 | 48 kHz / 16 bit / mono | 4-bit ADPCM，多為 mono、6 個立體聲 |
| cue 名稱 | **全空（0/9569）** | 已解密，即遊戲內名稱 |
| 佔全案台詞 | 95.7% | 4.3% |

### 為什麼主庫沒有名稱

主庫 9,569 個 cue 的名稱欄位經**全數逐一掃描確認為空**，只留下遞增的數值 `sound_id`。
來源名稱只存在於遊戲腳本／事件資料（**PAK 封裝**）內，音聲庫本身不保存。

因此主庫只能以編號順序輸出，**無法從音聲庫本身分辨哪一軌屬於哪個角色**——這是 Part 1 ② 必須改用說話者嵌入比對的直接原因（見 [`speaker_id.md`](04_speaker_id.md)）。

> **待解項目**：若要從編號還原角色，需進一步解包 PAK 封裝的事件腳本，比對腳本引用的語音 cue。目前未做。

### 戰鬥庫為何有名稱

戰鬥庫是**裸 KTSR**（檔案直接以 `KTSR` 開頭，非 KTSC 容器），其 cue 名稱以 LCG XOR 加密存放於 config 與 sound 區塊中。解密後即可直接看出角色歸屬，這也是戰鬥庫被拿來當說話者嵌入**校準來源**的原因。

## 3. 戰鬥庫的 `ryza_battle/`

`ryza_battle/` 是從戰鬥庫節選的 **48 個ライザ戰鬥語音**，檔名即遊戲內 cue 名稱：

```
015_SE_BCH_001_RYZA01.wav
015_SE_BCH_003_RYZA_BLAZE_RUSH03.wav
014_SE_BSP_060_RYZA_WAND01_a.wav
014_SE_BSP_063_RYZA_CHARGE01.wav
```

- `SE_BCH_*` = 戰鬥語音；`SE_BSP_*` = 必殺技語音
- 前綴數字為來源庫編號（014 / 015）
- **`_a` / `_b` / `_c` 後綴是 vgmstream 對重複名稱的自動編號**，非遊戲原始命名的一部分

**音訊驗證**：全部 48 檔已驗證為有效的 16-bit PCM WAV。原始編碼為 Microsoft 4-bit ADPCM，解成 WAV 後取樣率**保留原值**（47976–48019 Hz 之間浮動，並非整數 48000），多數單聲道、6 個立體聲——皆屬正常，非轉換錯誤。

## 4. 容器格式與加密

### 4.1 KTSC 容器（主庫）

`VOICE.ktsl2asbin` 是 KTSC 容器：

```
0x00 'KTSC'  0x04 ver  0x08 cue_count  0x0c table1_off  0x10 table2_off
table1：逐 cue 的 sound_id
table2：逐 cue 的 subentry 偏移
```

subentry 為 KTSR 結構：`0x00 'KTSR'`、`0x04 audio_id`，自 `0x40` 起為區塊串接。
區塊格式為 `magic(4, 大端) + total_size(4, 小端，含區塊表頭)`，其中
`0xBD888C36` 為 config 區塊（`+0x08 stream_id`、`+0x0c flags`、`+0x28 name_off`）。

解析工具：[`tools/parse_ktsr.py`](../tools/parse_ktsr.py)，產出 `voice_index.csv`。

### 4.2 裸 KTSR（戰鬥庫）

```
0x0b platform  0x0c audio_id（LCG 解密種子）  0x40 起為區塊
  0xBD888C36 config：+0x08 stream_id、+0x0c flags、+0x28 name_off（flags & 0x0200 時加密）
  0xC5CCCB70 sound： +0x08 sound_id、 +0x0c flags、+0x18 name_off（flags & 0x0008 時加密）
```

cue 名稱的 LCG XOR 加密（標準 rand）：

```python
seed = 0x343FD * seed + 0x269EC3      # 每次逐位元組推進
plain[i] = enc[i] ^ ((seed >> 16) & 0xFF)
```

種子取自各庫檔案偏移 `0x0c` 的 `audio_id`，本作固定為 **`0xAF4905A9`**。

解析工具：[`tools/parse_bare_ktsr.py`](../tools/parse_bare_ktsr.py)，產出 `battle_voice_index.csv`。

### 4.3 KOVS 雙層混淆（主庫音訊本體）

主庫音訊是 **KOVS 容器包裹的 Ogg Vorbis**，兩層混淆：

1. 每個 KOVS 區塊有 **0x20 位元組標頭**（`KOVS` magic、`block_size`、`loop`、`通道數`），Ogg 串流由 `+0x20` 開始。
2. Ogg 串流的**前 0x100 位元組與自身偏移做 XOR**（`buf[i] ^= (i & 0xFF)`，i 自 Ogg 串流起算），之後為標準 Ogg。

還原工具：[`tools/kovs_to_ogg.py`](../tools/kovs_to_ogg.py)（`--cue N` 可直接用索引）。
**驗證結果**：解碼後的 PCM 與 vgmstream 輸出的 WAV **位元級完全一致**。

格式細節可對照 [`tools/ktsr_ref.c`](../tools/ktsr_ref.c)（vgmstream `ktsr.c` 原始碼）。

## 5. 索引欄位對應

`voice_index.csv` 逐 cue 記錄 `sound_id`、旗標、取樣率、樣本數，以及 KOVS 區塊在 `.ktsl2stbin` 內的偏移與大小。

**對應關係**：`voice_index.csv` 中 `cue_index = N` 對應 `wav/(N+1):05d.wav`
（vgmstream 的 subsong 編號為 **1 起算**，故需 +1）。

**驗證**：
- cue 0 的 `num_samples`（223043）與 vgmstream 第 1 軌、以及 `wav/00001.wav` 的幀數完全一致
- 200 個抽樣 cue 的 stbin 偏移全數指向有效的 KOVS 區塊，大小總和佔 stbin 的 **99.9%**

## 6. 重新解出

如需從遊戲檔案重新解出（**會覆寫現有輸出目錄，執行前請確認**）：

```bash
# 主庫（注意：須指向 .ktsl2asbin，vgmstream 會自動找同名的 .ktsl2stbin）
tools/vgmstream/vgmstream-cli.exe -s 1 -S 0 -o "wav/?05s.wav" \
  "D:\SteamLibrary\steamapps\common\Atelier Ryza 2\Data\x64\Sound\VOICE.ktsl2asbin"

# 戰鬥庫（?n 為 cue 名稱）
tools/vgmstream/vgmstream-cli.exe -s 1 -S 0 -o "battle_voice/015_?n.wav" \
  "D:\SteamLibrary\steamapps\common\Atelier Ryza 2\Data\x64\Sound\015_battle_character.ktsl2asbin"
```

- `-s 1`：輸出所有 subsong
- `-S 0`：不輸出 subsong 索引資訊
- `-o`：輸出樣式；`?05s` 為 5 位數字編號、`?n` 為 cue 名稱

**vgmstream 版本**：`r2117`。此版本字串取自 `tools/vgmstream/vgmstream-cli.exe` 本身——
隨附的 [`tools/vgmstream/README.md`](../tools/vgmstream/README.md) 是上游原版，**不含版本資訊**，
完整用法見 [`tools/vgmstream/USAGE.md`](../tools/vgmstream/USAGE.md)，授權見 [`tools/vgmstream/COPYING`](../tools/vgmstream/COPYING)。

## 7. 相關產出

| 檔案 | 內容 |
|---|---|
| `voice_index.csv` | 主庫 9,570 行索引（表頭 + 9,569 cue） |
| `battle_voice_index.csv` | 戰鬥庫 1,094 行索引 |
| `wav/` | 主庫 WAV，9,569 檔（版權，不進版控） |
| `battle_voice/` | 戰鬥庫 WAV，646 檔（版權，不進版控） |
| `ryza_battle/` | 節選的 48 個ライザ戰鬥語音（版權，不進版控） |