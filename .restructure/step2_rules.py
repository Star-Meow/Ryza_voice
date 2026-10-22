"""Step 2 定案後的去向（依 target_outline.md 定案版）。

build_inventory.py 先比對本檔的 STEP2_RULES，再比對舊的 RULES；
STEP2_FILE_DEFAULTS 覆寫整檔預設，STEP2_ADDS 取代舊的 ADDS。
"""
BASE = "https://github.com/Star-Meow/Ryza_voice/blob/262addab90ce98bc8229741ce2353c0d6fe8a8a8"
STAGE = "開發過程紀錄（階段回報）：技術內容已移到 docs/07、docs/08；原文見整理前的固定版本 commit 262adda"
WORDING = "措辭：去除 Part／階段／PM／計畫字樣，或更新連結與章節編號"
PK = "坑記錄 append-only：只把階段／PM／計畫字樣改為中性敘述，證據改為固定版本連結；現象、根因、處置不變"

D02, D03, D04, D05 = "docs/02_unpack.md", "docs/03_speaker_id.md", "docs/04_transcript_pipeline.md", "docs/05_dataset_split.md"
D06, D07, D08, TR = "docs/06_model_selection.md", "docs/07_finetune.md", "docs/08_troubleshooting.md", "tools/README.md"
ASR, FT = "reports/asr_report.md", "reports/finetune_report.md"

STEP2_FILE_DEFAULTS = {
    f: ("過程紀錄", "delete", "", STAGE) for f in [
        "reports/01_env_model_prep.md", "reports/G1_t0_t1_report.md", "reports/G2_stage_a_report.md",
        "reports/G3_stage_b_report.md", "reports/G4_stage_c_report.md",
        "reports/smoke_test_stage_a.md", "reports/smoke_test_stage_c.md"]
}

R = []


def r(f, pat, cat, act, dest="", reason=""):
    R.append((f, pat, cat, act, dest, reason))


# ---------- README：只留入口 ----------
r("README.md", r"^Atelier Ryza", "專案說明", "edit", "README.md#Ryza Voice：萊莎日文聲線 TTS", "標題與簡介改為專案本身，不用 Part 1／2 框架")
r("README.md", r"^專案流程總覽", "專案說明", "merge", f"docs/01_overview.md#3. 模塊架構", "流程與結果移到 01 概述；README 只留模塊表。人工確認：未命中的關鍵字為結果數字（CER、A_high 92 等），已在 asr_report／03")
r("README.md", r"^Part 1 ①", "流程", "merge", f"{D02}#2. 兩種語音庫", "方法細節屬模塊文件，02 已有")
r("README.md", r"^Part 1 ②", "流程", "merge", f"{D03}#7.1 分級目錄（`ryza_main/`）", "分級表屬模塊文件，03 已有")
r("README.md", r"^Part 1 ③", "流程", "merge", f"{D04}#7. 七步資料流", "方法細節屬模塊文件，04 已有")
r("README.md", r"^3\.1 ASR", "結果", "merge", f"{ASR}#1. 模型與參數", "ASR 參數與 CER 屬結果報告；CER 定義已加在 asr_report 檔頭。人工確認：language=ja 與資料漏斗在 asr_report §1、§2 以表格呈現")
r("README.md", r"^3\.2 ", "流程", "merge", f"{D04}#8. 三處一致性", "三處一致性屬模塊文件，04 已有")
r("README.md", r"^工具入口", "專案說明", "merge", "README.md#模塊", "工具入口併入模塊表的「主要工具」欄；完整清單在 tools/README")
r("README.md", r"^文件地圖", "專案說明", "edit", "README.md#文件", "依新結構重排")
r("README.md", r"^目錄結構", "專案說明", "edit", "", "精簡；加入 data／tools README")
r("README.md", r"^授權", "專案說明", "edit", "", "連結 LICENSE，註明不涵蓋音訊")

# ---------- docs/02（原 03_unpack）----------
r("docs/03_unpack.md", r"^語音解包", "流程", "edit", f"{D02}#語音解包", WORDING)
r("docs/03_unpack.md", r"^為什麼主庫沒有名稱", "流程", "edit", "", WORDING)

# ---------- docs/03（原 04_speaker_id）----------
r("docs/04_speaker_id.md", r"^說話者篩選", "流程", "edit", f"{D03}#說話者篩選：說話者嵌入比對", WORDING)
r("docs/04_speaker_id.md", r"^1\. 動機|^4\.2 ", "流程", "edit", "", WORDING)

# ---------- docs/04（原 05_transcript_pipeline）----------
r("docs/05_transcript_pipeline.md", r"^ASR 轉譯與訓練清單建構", "流程", "edit", f"{D04}#轉譯資料集：ASR 與訓練清單建構", WORDING)
r("docs/05_transcript_pipeline.md", r"^1\. 三個核心檔案|^7\. 七步資料流", "流程", "edit", "", WORDING + "；correction_diff 改指 asr_report 附錄 A")
r("docs/05_transcript_pipeline.md", r"^10\. 歷史執行紀錄", "過程紀錄", "delete", "",
  "當次執行紀錄（一次性動作），不是可重跑的流程；Whisper 原始輸出的位置已記於 asr_report 附錄 B")

# ---------- docs/05（原 06_dataset）----------
r("docs/06_dataset.md", r"^Part 2：GPT-SoVITS 微調", "流程", "edit", f"{D05}#資料切分", "內容收斂為資料切分，標題依內容命名")
r("docs/06_dataset.md", r"^一、定位與目標", "流程", "merge", f"{D05}#1. 目標", "Part 框架移除；目標改為切分目標。人工確認：未命中的為 Part 1 產出敘述，已在 01 概述")
r("docs/06_dataset.md", r"^三、資料切分$|^3\.1 ", "流程", "merge", f"{D05}#2. 分層抽樣設計", "以 07 §9.1 較完整的版本為準；「・・・」不屬 ELL（split_train_test.py:67）。人工確認：分層與配額表在 05 §2 以 07 的版本呈現")
r("docs/06_dataset.md", r"^3\.2 |^3\.3 ", "流程", "merge", f"{D05}#4. 產出與驗證", "以 07 §9.2 為準；驗證為 11 項")
r("docs/06_dataset.md", r"^3\.4 ", "流程", "merge", f"{D05}#5. 執行指令", "改用正確指令（不加 --audio-root wav）")

# ---------- docs/06（原 02_issue_part2）----------
r("docs/02_issue_part2.md", r"^GPT-SoVITS Part 2 模型選型", "設計", "move", f"{D06}#基礎模型選型：GPT-SoVITS 版本與微調路線", "標題去 Part 框架")
r("docs/02_issue_part2.md", r"^五、選型結論|^六、決策鏈摘要", "設計", "move", f"{D06}#<same>", "改名；結論改指評估標準、決策鏈補 code fence")

# ---------- docs/07 ----------
for pat, dest, act in [
    (r"^GPT-SoVITS v4 LoRA 微調文件", "TTS 微調與推論：GPT-SoVITS v4 LoRA", "edit"),
    (r"^0\. 章節導覽", "0. 全流程總覽", "edit"),
    (r"^1\. 環境需求", "1. 環境需求", "edit"),
    (r"^ffmpeg（隱性系統依賴）", "1.1 ffmpeg（隱性系統依賴）", "move"),
    (r"^2\. 預處理三步", "3. 預處理", "move"),
    (r"^版本推斷", "3.1 版本推斷（v4 → v3 codepath）", "move"),
    (r"^合併分片", "3.2 合併分片", "move"),
    (r"^3\. 階段 A 狀態", "3.3 完成條件（PA 檢查）", "move"),
    (r"^4\. 易錯點", "4. 易錯點", "edit"),
    (r"^6\. 推論端隱性依賴", "6.2 推論端隱性依賴", "move"),
    (r"^6\.1 fast_langdetect", "6.2.1 fast_langdetect 語言偵測模型", "move"),
    (r"^6\.2 jieba_fast", "6.2.2 jieba_fast（不採用，但別名不可刪）", "move"),
    (r"^7\. v4 LoRA 顯存", "5.6 顯存", "move"),
    (r"^10\. 模型準備", "2. 模型準備", "move"),
    (r"^10\.1 ", "2.1 必須存在的權重", "move"),
    (r"^10\.2 ", "2.2 T0：`.bin` → safetensors（必要）", "move"),
    (r"^10\.3 ", "2.3 推論端模型檔", "move"),
    (r"^11\. 訓練重現", "5. 訓練", "move"),
    (r"^11\.1 ", "5.1 執行環境與前置", "move"),
    (r"^11\.2 ", "5.3 訓練指令（先 S1，再 S2）", "move"),
    (r"^11\.3 ", "5.4 訓練參數", "move"),
    (r"^11\.4 ", "5.5 產出", "move"),
    (r"^11\.5 ", "5.6 顯存", "merge"),
    (r"^12\. 推論重現", "6. 推論", "move"),
    (r"^12\.1 ", "6.1 產生樣本", "move"),
    (r"^12\.2 ", "6.2 推論端隱性依賴", "merge"),
    (r"^12\.3 ", "7. 評估", "move"),
    (r"^13\. 已知偏差", "8. 已知偏差", "move"),
]:
    r("docs/07_finetune.md", pat, "流程", act, f"{D07}#{dest}", "依工作流順序重新編號；去除階段／issue／PM 字樣")
r("docs/07_finetune.md", r"^15\. 相關文件", "流程", "move", f"{D07}#10. 相關文件",
  "依工作流順序重新編號。人工確認：未命中的為已改名或已刪除的檔名，改連到新檔名")
r("docs/07_finetune.md", r"^8\. 資料準備", "流程", "delete", "", "前段模塊不在 07 重寫；§0 全流程總覽改連到 02～05")
r("docs/07_finetune.md", r"^8\.1 ", "流程", "merge", f"{D02}#6. 重新解出", "解包指令以 02 為唯一來源。人工確認：未命中的為檔案數與檔案大小，02 §2 已有")
r("docs/07_finetune.md", r"^8\.2 ", "流程", "merge", f"{D04}#9. 執行順序", "ASR 漏斗與指令以 04 為唯一來源")
r("docs/07_finetune.md", r"^9\. 切分重現", "流程", "merge", f"{D05}#5. 執行指令", "切分以 05 為唯一來源")
r("docs/07_finetune.md", r"^9\.1 ", "流程", "move", f"{D05}#2. 分層抽樣設計", "切分以 05 為唯一來源")
r("docs/07_finetune.md", r"^9\.2 ", "流程", "move", f"{D05}#4. 產出與驗證", "切分以 05 為唯一來源")
r("docs/07_finetune.md", r"^9\.3 ", "流程", "move", f"{D05}#3. 參考音訊", "切分以 05 為唯一來源")
r("docs/07_finetune.md", r"^14\. 常見問題", "參考", "move", f"{D08}#5. 症狀速查", "症狀速查屬疑難排解；07 §9 只連過去")

# ---------- tools/README（原 docs/08_tools）----------
r("docs/08_tools.md", r"^8\. 尚未納入版控", "參考", "merge", f"{TR}#5. TTS 微調與推論",
  "「未納入版控」為事實錯誤；工具併入分類表。人工確認：未命中的是 `tools/` 路徑前綴，tools/README 以相對檔名列出")
for pat, dest, act in [
    (r"^工具總覽", "tools/", "edit"),
    (r"^1\. Part 1 ①", "1. 語音解包", "move"),
    (r"^2\. Part 1 ②", "2. 說話者篩選", "move"),
    (r"^2\.1 |^2\.2 ", "<same>", "edit"),
    (r"^3\. Part 1 ③", "3. 轉譯資料集", "move"),
    (r"^4\. 資料切分與品質量測", "4. 資料切分", "merge"),
    (r"^5\. 模型權重轉檔與 UNK 分析", "7. 測試與驗證", "merge"),
    (r"^6\. 生成產物", "8. 生成產物（不進版控）", "move"),
    (r"^7\. 附帶資源", "9. 附帶資源", "move"),
    (r"^8\. 尚未納入版控", "5. TTS 微調與推論", "merge"),
]:
    r("docs/08_tools.md", pat, "參考", act, f"{TR}#{dest}", "依慣例移到 tools/README.md；依模塊分組並補齊全部工具")

# ---------- docs/08（原 09_pitfalls）----------
r("docs/09_pitfalls.md", r"^訓練坑記錄", "參考", "edit", f"{D08}#疑難排解（坑記錄）", "改用疑難排解的通用名稱；檔頭去階段字樣")
r("docs/09_pitfalls.md", r"^PK-011", "參考", "edit", f"{D08}#PK-011 上游 import 鏈強迫依賴 `jieba_fast`（不採用，別名不可刪）", PK)
r("docs/09_pitfalls.md", r"^1\. 欄位定義|^2\. 追加規則|^觸發本節", "參考", "edit", "", WORDING)
r("docs/09_pitfalls.md", r"^PK-0", "參考", "edit", "", PK)
r("docs/09_pitfalls.md", r"^本機顯存實測彙總", "參考", "edit", f"{D08}#本機顯存實測彙總（冒煙測試 ＋ 正式訓練）", WORDING)
r("docs/09_pitfalls.md", r"^6\. 階段 C 追加", "過程紀錄", "merge", f"{D07}#5.4 訓練參數",
  "#5 batch 推導移到 07 §5.4。人工確認：#4 已在 PK-011，#6 已在 08 §4，其餘為「計畫」側敘述")

# ---------- reports（刪除檔的技術內容）----------
r("reports/01_env_model_prep.md", r"^與 requirements\.txt 的偏差", "流程", "move", f"{D07}#1.2 與官方 requirements.txt 的偏差（Windows 必要調整）", "Windows 依賴偏差屬重現資訊")
r("reports/01_env_model_prep.md", r"^模型清單與大小", "流程", "merge", f"{D07}#2.1 必須存在的權重", "權重表屬重現資訊")
r("reports/01_env_model_prep.md", r"^TF32", "參考", "merge", f"{D08}#TF32 —— 其他環境過往異常，本機無需理會", "TF32 統一記於 08 §4。人工確認：未命中的為環境版本字串，08 §4 以檔案行號記錄")
r("reports/G1_t0_t1_report.md", r"^2\. T0 明細表", "流程", "merge", f"{D07}#2.2 T0：`.bin` → safetensors（必要）", "T0 轉檔結果屬重現資訊。人工確認：未命中的為逐檢查項明細（T0-1～T0-8），07 §2.2 保留 SHA256 與大小")
r("reports/G2_stage_a_report.md", r"^1\. 結果總表", "流程", "merge", f"{D07}#3.3 完成條件（PA 檢查）", "PA 檢查改為預處理完成條件。人工確認：PA-12／PA-13 為當次工作樹狀態，不屬完成條件")
r("reports/G2_stage_a_report.md", r"^7\. docs/07", "流程", "merge", f"{D07}#1.1 ffmpeg（隱性系統依賴）", "已寫入 07 §1.1")
r("reports/G3_stage_b_report.md", r"^3\. 讀碼依據|^S2（|^S1（", "流程", "move", f"{D07}#設定欄位依據", "設定欄位為何這樣填，屬重現參考；值更新為最終設定")
r("reports/G4_stage_c_report.md", r"^1\. 結果總表", "流程", "merge", f"{D07}#5.2 冒煙測試", "冒煙測試改為 07 的執行方式與通過條件。人工確認：未命中的為當次量測值，顯存已記於 08 §4")
r("reports/smoke_test_stage_c.md", r"^1\. 測試輪次對照表", "流程", "merge", f"{D07}#5.2 冒煙測試", "指令與輪次設定移到 07 §5.2。人工確認：未命中的為當次時間戳與秒數")
r("reports/smoke_test_stage_c.md", r"^5\.1 顯存|^6\. GPU 異常表", "參考", "merge", f"{D08}#本機顯存實測彙總（冒煙測試 ＋ 正式訓練）", "顯存實測統一記於 08 §4。人工確認：未命中的為當次時間戳")
r("reports/smoke_test_stage_c.md", r"^5\.5 SM-S1-5", "參考", "merge", f"{D08}#PK-012 S1 的學習率排程器被上游硬寫成常數 0.002", "LR 鎖死結論已在 PK-012。人工確認：四種長度的逐步數值表留在 PK-012 證據所連的固定版本")

r("reports/audio_quality.md", r".*", "設計", "merge", f"{D06}#附錄 A：音訊品質實測", "選型佐證併入 06 附錄 A；「V2Pro 優先」結論依最終選型修正")
r("reports/correction_diff.md", r"^專有名詞校正前後對照表", "結果", "merge", f"{ASR}#附錄 A：專有名詞校正前後對照", "併入 asr_report 附錄 A；CSV 保留")
r("reports/correction_diff.md", r"^規則命中統計", "結果", "move", f"{ASR}#規則命中統計", "併入 asr_report 附錄 A")
r("reports/correction_diff.md", r"^14 段校正明細", "結果", "move", f"{ASR}#14 段校正明細", "併入 asr_report 附錄 A")

# ---------- reports（保留檔）----------
r(ASR, r"^ASR 語音轉文字報告", "結果", "edit", f"{ASR}#ASR 轉譯報告", "檔頭加 CER 定義框；標題去階段字樣")
r(ASR, r"^1\. 模型與參數", "結果", "edit", "", WORDING)
r(ASR, r"^4\. 專有名詞", "結果", "edit", f"{ASR}#4. 專有名詞校正", "刪除驗收宣告；ボース 規則依據說明對齊 §5")
r(ASR, r"^附錄：相關檔案", "結果", "edit", f"{ASR}#附錄 B：相關檔案", "correction_diff 改指附錄 A")

r("reports/defect_triage.md", r"^主觀缺陷排查報告|^修正後的根因層級|^歸納表|^相關文件", "結果", "edit", "", "PM／Agent 改為試聽者／覆核的中性稱呼；連結更新")
r("reports/defect_triage.md", r"^【唯讀覆核】候選一", "結果", "edit", "reports/defect_triage.md#覆核結果：候選一可大幅收斂", "「唯讀覆核」改為「覆核結果」")
r("reports/defect_triage.md", r"^【唯讀覆核】標註只影響", "結果", "edit", "reports/defect_triage.md#覆核結果：標註只影響切分，但正因如此讓評測對情緒是盲的", "「唯讀覆核」改為「覆核結果」")
r("reports/defect_triage.md", r"^【唯讀覆核】兩點澄清", "結果", "edit", "reports/defect_triage.md#覆核結果：兩點澄清", "「唯讀覆核」改為「覆核結果」")
r("reports/defect_triage.md", r"^最有力的實證：Agent", "結果", "edit", "reports/defect_triage.md#最有力的實證：文字標註被試聽結果推翻", "Agent 改為中性敘述")

r(FT, r"^階段 D 回報", "結果", "edit", f"{FT}#TTS 微調與評估報告", "刪除回報頭（計畫 §15、分支、issue）")
r(FT, r"^1\. 結果表|^2\.1 |^2\.3 |^2\.4 |^4-2 |^4-3 |^5\.2\.1 |^6-1 |^6-3 |^6-4 |^6-5 |^8\. 無法判定|^附錄 A", "結果", "edit", "", WORDING + "；「與計畫是否相符」欄改為備註")
r(FT, r"^A-3", "結果", "edit", "", "ボオス 的錯誤性質更正為長音「オ」脫落（原誤記為促音）")
r(FT, r"^5\.3 ", "結果", "edit", "", "ボオス 表格更正：原文與轉寫寫錯、錯誤性質改為長音脫落")
r(FT, r"^6-6 ", "結果", "edit", f"{FT}#6-6 未能驗證的事項", WORDING)
r(FT, r"^9\. 重現步驟", "結果", "edit", f"{FT}#9. 重現方式與原始資料", "重現步驟以 docs/07 為唯一來源；本節只留原始資料位置")
r(FT, r"^9\.1 ", "流程", "merge", f"{D07}#6.2.1 fast_langdetect 語言偵測模型", "重現步驟以 docs/07 為唯一來源")
r(FT, r"^9\.2 ", "流程", "merge", f"{D07}#5.3 訓練指令（先 S1，再 S2）", "重現步驟以 docs/07 為唯一來源")
r(FT, r"^9\.3 ", "流程", "merge", f"{D07}#6.1 產生樣本", "重現步驟以 docs/07 為唯一來源")
r(FT, r"^9\.4 ", "流程", "merge", f"{D07}#7. 評估", "重現步驟以 docs/07 為唯一來源")
r(FT, r"^9\.5 ", "結果", "merge", f"{FT}#9. 重現方式與原始資料", "原始資料位置保留")

r("reports/logs/stage_d_train.md", r"^階段 D 訓練 log", "結果", "edit", "reports/logs/stage_d_train.md#正式訓練紀錄", "標題與檔頭去階段／計畫字樣")
r("reports/logs/stage_d_train.md", r"^1\. 執行環境|^3\.1 ", "結果", "edit", "", WORDING)

r("reports/split_report.md", r"^Part 2 資料切分報告", "結果", "edit", "reports/split_report.md#資料切分報告", "標題去 Part 框架")
r("reports/split_report.md", r"^門禁檢查", "結果", "edit", "reports/split_report.md#驗證結果", "「門禁」改為通用的驗證結果")
r("reports/split_report.md", r"^風險與待確認", "結果", "edit", "reports/split_report.md#已知限制", "刪除已結的待確認事項")

r("samples/README.md", r"^生成樣本", "專案說明", "edit", "samples/README.md#生成樣本", "「供 reviewer」改為中性描述")
r("samples/README.md", r"^\(b\) 全新台詞", "專案說明", "edit", "samples/README.md#(b) 全新台詞 10 句 — 遊戲語料庫中不存在，語氣依句末標點標註", "語氣標註來源寫清楚")
r("samples/README.md", r"^評估結果", "專案說明", "edit", "samples/README.md#評估", "評估改為連結；刪除錯誤的「1 epoch 級別」")

r(FT, r"^2\.2 |^3\. GPU 異常表", "結果", "keep", "", "各輪摘要與 GPU 異常是結果的一部分，保留；完整紀錄在 logs/stage_d_train")
r("reports/logs/stage_d_train.md", r"^2\.3 ", "結果", "keep", "", "116 s/it 是假死期間 tqdm 的瞬時讀值，與 finetune_report 的整輪平均不衝突，不需修改")

STEP2_RULES = R

STEP2_ADDS = [
    ("README.md#模塊", "入口：六個模塊與主要工具"),
    ("README.md#安裝", "入口：兩個 venv 與外部依賴"),
    ("README.md#使用", "入口：各模塊的入口指令"),
    ("README.md#樣本與結果", "入口：只放連結"),
    ("data/README.md#data/", "慣例位置的資料說明（dataset card）"),
    ("data/README.md#檔案", "各資料檔的格式、行數、產生工具"),
    ("data/README.md#使用注意", "絕對路徑、測試集、class 欄、ref 不同步"),
    ("docs/01_overview.md#專案概述", "補上 01：issue 需求的專案化版本"),
    ("docs/01_overview.md#1. 目標", "專案概述"),
    ("docs/01_overview.md#2. 範圍與限制", "專案概述"),
    ("docs/01_overview.md#3. 模塊架構", "專案概述"),
    ("docs/01_overview.md#4. 評估標準", "門檻的唯一出處"),
    ("docs/01_overview.md#5. 實際執行與原需求的差異", "專案概述"),
    (f"{D06}#七、實測後的修正", "取代「待驗證項」：記錄實測推翻的判斷"),
    (f"{D07}#9. 疑難排解", "連到 08 §5 症狀速查"),
    (f"{TR}#6. 評估", "依模塊分組"),
    (f"{ASR}#5. 已知限制", "承接 issue_compliance §7／§8"),
]
