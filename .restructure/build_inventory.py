"""從 docs-baseline 產生文件章節清單（inventory.csv）。

每個 heading 章節一列；沒有 heading 的檔案以段落切分。
category／action／dest／reason 依下方 RULES 填入，之後可直接手改 CSV。
重跑會覆蓋 CSV，所以手改之後請改 RULES 而不是重跑。

用法：python -I .restructure/build_inventory.py
"""
import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".restructure" / "inventory.csv"
BASELINE = "docs-baseline"
FIELDS = ["id", "src_file", "heading", "level", "lines", "hash",
          "category", "action", "dest", "reason"]

ISSUE1 = "https://github.com/Star-Meow/Ryza_voice/issues/1"
PR6 = "https://github.com/Star-Meow/Ryza_voice/pull/6"
DEV_LOG_PENDING = "過程紀錄；Step 3 決定是否併入 reports/dev_log.md，去除停止點／放行／branch 狀態"

# 檔案層級預設：(category, action, dest, reason)
FILE_DEFAULTS = {
    "README.md": ("專案說明", "keep", "", ""),
    "docs/01_issue.md": ("協作產物", "delete", "",
                         f"issue #1 原文，以 GitHub 為準（{ISSUE1}）；README 改連結至 issue"),
    "docs/02_issue_part2.md": ("設計", "move", "docs/02_model_selection.md#<same>",
                               "改名：內容是選型設計，不是 issue"),
    "docs/03_unpack.md": ("流程", "keep", "", ""),
    "docs/04_speaker_id.md": ("流程", "keep", "", ""),
    "docs/05_transcript_pipeline.md": ("流程", "keep", "", ""),
    "docs/06_dataset.md": ("流程", "keep", "", ""),
    "docs/07_finetune.md": ("流程", "keep", "", ""),
    "docs/08_tools.md": ("參考", "keep", "", ""),
    "docs/09_pitfalls.md": ("參考", "keep", "", ""),
    "reports/01_env_model_prep.md": ("過程紀錄", "edit", "", DEV_LOG_PENDING),
    "reports/G1_t0_t1_report.md": ("過程紀錄", "edit", "", DEV_LOG_PENDING),
    "reports/G2_stage_a_report.md": ("過程紀錄", "edit", "", DEV_LOG_PENDING),
    "reports/G3_stage_b_report.md": ("過程紀錄", "edit", "", DEV_LOG_PENDING),
    "reports/G4_stage_c_report.md": ("過程紀錄", "edit", "", DEV_LOG_PENDING),
    "reports/smoke_test_stage_a.md": ("過程紀錄", "edit", "", DEV_LOG_PENDING),
    "reports/smoke_test_stage_c.md": ("過程紀錄", "edit", "", DEV_LOG_PENDING),
    "reports/asr_filter_report.md": ("結果", "delete", "",
                                     "已被 asr_report.md §2 取代（docs/05 §10 已註明）；"
                                     "所引根目錄 ryza_train.list 已於 672e4e5 移除"),
    "reports/asr_report.md": ("結果", "keep", "", ""),
    "reports/audio_quality.md": ("設計", "merge",
                                 "docs/02_model_selection.md#附錄：音訊品質實測",
                                 "選型的佐證，併入 02 附錄；「V2Pro 優先」結論依最終選型修正"),
    "reports/correction_diff.md": ("結果", "keep", "", ""),
    "reports/defect_triage.md": ("結果", "keep", "", ""),
    "reports/finetune_report.md": ("結果", "keep", "", ""),
    "reports/issue_compliance.md": ("協作產物", "delete", "",
                                    f"對 issue #1 的合規自我稽核（審查產物）；需求與驗收討論在 {ISSUE1}、{PR6}"),
    "reports/issue_reply_part1.md": ("協作產物", "delete", "",
                                     f"issue #1 的進度回報 comment 草稿，已張貼於 {ISSUE1}"),
    "reports/logs/stage_d_train.md": ("結果", "keep", "", ""),
    "reports/pr_description.md": ("協作產物", "delete", "",
                                  f"PR #6 內文，已在 {PR6}"),
    "reports/split_report.md": ("結果", "keep", "", ""),
    "samples/README.md": ("專案說明", "keep", "", ""),
}

# 章節層級覆寫：(檔案, heading 正則, category, action, dest, reason)
# 先比對到的優先。dest 用「檔案#heading」；<same> 表示沿用原 heading。
RULES = [
    # ---------- README ----------
    ("README.md", r"^需求與文件對照", "協作產物", "delete", "",
     f"協作內容；reviewer 指示 README 只寫專案本身，連結問題應原地編輯 issue comment（{ISSUE1}）"),
    ("README.md", r"^Part 1：ASR 轉譯$", "協作產物", "delete", "", f"同上（需求對照子節，{ISSUE1}）"),
    ("README.md", r"^Part 2：GPT-SoVITS finetune$", "協作產物", "delete", "", f"同上（需求對照子節，{ISSUE1}）"),
    ("README.md", r"^Atelier Ryza", "專案說明", "edit", "", "Part 2 由「進行中」改為已完成並列結果；需求來源改連結 issue #1"),
    ("README.md", r"^專案流程總覽", "專案說明", "edit", "", "Part 2 列補方法與結果"),
    ("README.md", r"^3\.1 ASR", "專案說明", "edit", "", "CER 註明 reference＝人工校對 50 段、hypothesis＝Whisper 原始輸出"),
    ("README.md", r"^文件地圖", "專案說明", "edit", "", "依新結構更新；列出預留的協作檔"),
    ("README.md", r"^授權", "專案說明", "edit", "", "連結新增的 LICENSE；註明 LICENSE 僅涵蓋程式碼"),
    ("README.md", r"^目錄結構", "專案說明", "edit", "", "加入 LICENSE"),
    # ---------- docs/02 ----------
    ("docs/02_issue_part2.md", r"^七、待驗證項", "過程紀錄", "delete", "",
     "待驗證項已全數解決（顯存已實測見 07 §7、TF32 本機未發生見 09 §4、參考音訊已定見 07 §9.3）"),
    # ---------- Step 1 連帶修改（連結指向已刪除／改名檔） ----------
    ("docs/07_finetune.md", r"^GPT-SoVITS v4 LoRA 微調文件", "流程", "edit", "", "選型連結改指 02_model_selection.md"),
    ("docs/09_pitfalls.md", r"^4\. GPU 不穩專區", "參考", "edit", "", "issue 引用改連結 issue #1"),
    ("reports/finetune_report.md", r"^6-2 ", "結果", "edit", "", "02 連結改名"),
    ("reports/finetune_report.md", r"^基礎模型（本專案", "結果", "edit", "",
     "吸收 pr_description 的 commit hash 說明與上游 URL"),
    # ---------- docs/03 ----------
    ("docs/03_unpack.md", r"^語音解包", "流程", "edit", "", "修正連結文字中的舊檔名"),
    # ---------- docs/04 ----------
    ("docs/04_speaker_id.md", r"^2\. 嵌入 pipeline", "流程", "edit", "",
     "刪除「尚未納入版控」（已追蹤）；TF32 段縮為一行並連到 09 §4"),
    # ---------- docs/05 ----------
    ("docs/05_transcript_pipeline.md", r"^1\.5 ", "流程", "edit",
     "docs/05_transcript_pipeline.md#1.5 為何走 `top500/` 而非 `A_high/ + B_likely/`",
     "PM 字樣改中性敘述；移除對 issue_compliance 的連結與稽核口吻；吸收 v00596 來源說明"),
    ("docs/05_transcript_pipeline.md", r"^4\.1 Issue", "流程", "edit",
     "docs/05_transcript_pipeline.md#4.1 必要過濾規則", "標題去 issue 口吻"),
    ("docs/05_transcript_pipeline.md", r"^4\.2 額外", "流程", "edit",
     "docs/05_transcript_pipeline.md#4.2 附加過濾規則", "標題去 issue 口吻"),
    # ---------- docs/06 ----------
    ("docs/06_dataset.md", r"^二、模型選型$", "設計", "merge", "docs/02_model_selection.md#五、選型結論",
     "與 02 重複；06 改為一行連結"),
    ("docs/06_dataset.md", r"^2\.1 版本篩選", "設計", "merge", "docs/02_model_selection.md#背景：版本篩選準則",
     "與 02 重複"),
    ("docs/06_dataset.md", r"^2\.2 ", "設計", "merge", "docs/02_model_selection.md#二、版本規格與官方基準",
     "與 02 重複"),
    ("docs/06_dataset.md", r"^2\.3 ", "設計", "merge", "docs/02_model_selection.md#五、選型結論",
     "與 02 重複；「預期相近低顯存」已被實測推翻"),
    ("docs/06_dataset.md", r"^一、定位與目標", "流程", "edit", "", "去除「依 Issue 要求」口吻"),
    ("docs/06_dataset.md", r"^3\.1 ", "流程", "edit", "", "ELL 規則與 07 §9.1 對齊；補時長三分位分層"),
    ("docs/06_dataset.md", r"^3\.3 ", "流程", "edit", "", "「五項驗證」改為 11 項"),
    ("docs/06_dataset.md", r"^3\.4 ", "流程", "merge", "docs/07_finetune.md#9. 切分重現（467 / 24 / 10）",
     "指令有誤（--audio-root wav 會變 wav/wav）；以 07 §9 為唯一指令來源"),
    # ---------- docs/07 ----------
    ("docs/07_finetune.md", r"^0\. 章節導覽", "流程", "edit", "", "刪除「對應 issue 要求」欄；PK 範圍改 001～015"),
    ("docs/07_finetune.md", r"^1\. 環境需求", "流程", "edit", "", "吸收 01_env 的依賴偏差表"),
    ("docs/07_finetune.md", r"^3\. 階段 A 狀態", "過程紀錄", "edit",
     "docs/07_finetune.md#3. 預處理完成條件", "去除「G2 已驗收」敘事，保留可檢查的完成條件"),
    ("docs/07_finetune.md", r"^5\. 後續階段", "過程紀錄", "delete", "",
     "階段進度清單，已全數完成；結果見 finetune_report"),
    ("docs/07_finetune.md", r"^6\.2 jieba_fast", "流程", "edit",
     "docs/07_finetune.md#6.2 jieba_fast（不採用）", "PM 字樣改中性敘述"),
    ("docs/07_finetune.md", r"^8\.2 ", "流程", "edit", "", "PM 字樣改中性敘述"),
    ("docs/07_finetune.md", r"^10\.1 ", "流程", "edit", "", "吸收 01_env 模型清單"),
    ("docs/07_finetune.md", r"^11\.3 ", "流程", "edit", "", "吸收 G3 §3 欄位依據表、09 §6 的 batch 推導"),
    ("docs/07_finetune.md", r"^12\.1 ", "流程", "edit", "docs/07_finetune.md#12.1 產生樣本", "標題與內容不符"),
    ("docs/07_finetune.md", r"^13\. 已知偏差", "流程", "edit", "",
     "PM 字樣改中性；「主觀評語尚待提交」過時；#11 路徑 data/→docs/；刪 #12 待辦"),
    ("docs/07_finetune.md", r"^15\. 相關文件", "流程", "edit", "", "PK 範圍；被刪檔的連結"),
    # ---------- docs/08 ----------
    ("docs/08_tools.md", r"^工具總覽", "參考", "edit", "", "刪除 L4 版控說明"),
    ("docs/08_tools.md", r"^3\. ", "參考", "edit", "", "「五項驗證」改為 11 項"),
    ("docs/08_tools.md", r"^6\. 生成產物", "參考", "edit", "", "「体」改「體」"),
    ("docs/08_tools.md", r"^8\. 尚未納入版控", "參考", "merge",
     "docs/08_tools.md#5. Part 2：訓練、推論與評估",
     "「未納入版控」為事實錯誤（全已追蹤）；所列工具併入分類表，並補齊遺漏工具"),
    # ---------- docs/09 ----------
    ("docs/09_pitfalls.md", r"^訓練坑記錄", "參考", "edit", "", "更新檔頭定位（不再是「供階段 B、C」）"),
    ("docs/09_pitfalls.md", r"^PK-011", "參考", "edit",
     "docs/09_pitfalls.md#PK-011 上游 import 鏈強迫依賴 `jieba_fast`（不採用）",
     "PM 字樣改中性；「不下載」與實際處置對齊"),
    ("docs/09_pitfalls.md", r"^PK-015", "參考", "edit", "", "移到 PK-014 之後（append-only 順序）"),
    ("docs/09_pitfalls.md", r"^TF32", "參考", "edit", "", "PM 字樣改中性；成為 TF32 的唯一完整記載"),
    ("docs/09_pitfalls.md", r"^5\. 與前因", "過程紀錄", "delete", "",
     "與「前因」比對的過程敘事；內容已在 PK-002／003／005 與 §4"),
    ("docs/09_pitfalls.md", r"^6\. 階段 C 追加", "過程紀錄", "merge", "docs/07_finetune.md#11.3 訓練參數",
     "#5 batch=8 推導移到 07 §11.3；#4 已在 PK-011；#6 為「計畫」側記錄誤差"),
    ("docs/09_pitfalls.md", r"^7\. 階段 D 追加", "過程紀錄", "delete", "",
     "重複：#7→PK-013／07 §7，#8→finetune_report §4-2，#9→PK-011"),
    # ---------- reports/01_env ----------
    ("reports/01_env_model_prep.md", r"^與 requirements\.txt 的偏差", "流程", "move",
     "docs/07_finetune.md#1. 環境需求", "Windows 依賴偏差屬重現資訊"),
    ("reports/01_env_model_prep.md", r"^模型清單與大小", "流程", "merge",
     "docs/07_finetune.md#10.1 必須存在的權重", "權重表屬重現資訊"),
    ("reports/01_env_model_prep.md", r"^TF32", "參考", "merge", "docs/09_pitfalls.md#TF32 —— 其他環境過往異常，本機無需理會",
     "TF32 統一記於 09 §4"),
    # ---------- reports/G1..G4、smoke ----------
    ("reports/G1_t0_t1_report.md", r"^7\. 目前工作樹", "過程紀錄", "delete", "",
     "branch／untracked 快照，相關檔案已於 0b7c743 進版控"),
    ("reports/G1_t0_t1_report.md", r"^2\. T0 明細表", "流程", "merge", "docs/07_finetune.md#10.2 T0：`.bin` → safetensors（必要）",
     "SHA256 表屬重現資訊（07 §10.2 已有轉檔流程）"),
    ("reports/G2_stage_a_report.md", r"^5\. 目前工作樹", "過程紀錄", "delete", "", "branch／untracked 快照，已過時"),
    ("reports/G2_stage_a_report.md", r"^`<GPT>`", "過程紀錄", "delete", "", "同上"),
    ("reports/G2_stage_a_report.md", r"^`<RYZA>`", "過程紀錄", "delete", "", "同上"),
    ("reports/G2_stage_a_report.md", r"^7\. docs/07", "流程", "merge", "docs/07_finetune.md#ffmpeg（隱性系統依賴）",
     "已寫入 07 §1 ffmpeg"),
    ("reports/G3_stage_b_report.md", r"^3\. 讀碼依據", "流程", "move", "docs/07_finetune.md#11.3 訓練參數",
     "設定欄位為何這樣填，屬重現參考"),
    ("reports/G3_stage_b_report.md", r"^S2（", "流程", "move", "docs/07_finetune.md#11.3 訓練參數", "同上"),
    ("reports/G3_stage_b_report.md", r"^S1（", "流程", "move", "docs/07_finetune.md#11.3 訓練參數", "同上"),
    ("reports/G3_stage_b_report.md", r"^6\. 停止點", "過程紀錄", "delete", "", "等待放行的狀態，已過時"),
    ("reports/G4_stage_c_report.md", r"^2\.1 ", "過程紀錄", "delete", "", "「皆未 commit」快照，已過時"),
    ("reports/G4_stage_c_report.md", r"^7\. 停止點", "過程紀錄", "delete", "", "等待放行的狀態，已過時"),
    ("reports/smoke_test_stage_a.md", r"^`<GPT>`", "過程紀錄", "delete", "", "branch／untracked 快照，已過時"),
    ("reports/smoke_test_stage_a.md", r"^`<RYZA>`", "過程紀錄", "delete", "", "同上"),
    ("reports/smoke_test_stage_c.md", r"^8\. 實際訓練", "過程紀錄", "delete", "",
     "空白模板從未填寫；實際結果在 finetune_report 與 logs/stage_d_train"),
    ("reports/smoke_test_stage_c.md", r"^8\.\d ", "過程紀錄", "delete", "", "同上（空白）"),
    # ---------- reports/asr_report ----------
    ("reports/asr_report.md", r"^ASR 語音轉文字報告", "結果", "edit", "",
     "檔頭加 CER 定義框：reference＝人工校對 50 段，hypothesis＝Whisper 原始輸出（reviewer 建議 highlight）"),
    ("reports/asr_report.md", r"^資料漏斗", "結果", "edit", "", "吸收 WRONG 23 vs 3 說明（來自 issue_reply_part1）"),
    ("reports/asr_report.md", r"^Issue 過濾規則", "結果", "edit", "reports/asr_report.md#必要過濾規則觸發統計",
     "標題去 issue 口吻"),
    ("reports/asr_report.md", r"^額外過濾", "結果", "edit", "reports/asr_report.md#附加過濾", "標題去 issue 口吻"),
    ("reports/asr_report.md", r"^抽查方法", "結果", "edit", "",
     "「留空＝無異常」改為「reference 與 whisper 相同＝已審判定正確」；消除「計算基準」歧義"),
    ("reports/asr_report.md", r"^結果$", "結果", "edit", "", "刪除「非預設值」防衛措辭"),
    ("reports/asr_report.md", r"^4\. 專有名詞", "結果", "edit", "", "刪除「驗收要件全數滿足」宣告"),
    # ---------- reports/defect_triage ----------
    ("reports/defect_triage.md", r"^本文件的定位", "結果", "edit", "", "刪除依「計畫 §15」拆檔的說明"),
    ("reports/defect_triage.md", r"^我發現了什麼", "結果", "edit", "reports/defect_triage.md#現象", "「我」指涉不明，改中性"),
    ("reports/defect_triage.md", r"^我判斷可能是哪部分錯", "結果", "edit", "reports/defect_triage.md#可能成因", "同上"),
    ("reports/defect_triage.md", r"^我想怎麼排查", "結果", "edit", "reports/defect_triage.md#排查方式", "同上"),
    ("reports/defect_triage.md", r"^【唯讀覆核】", "結果", "edit", "", "內文「我」改中性"),
    ("reports/defect_triage.md", r"^結論$", "結果", "edit", "", "刪除「若 PM 授權」；P-3 掃描標為未執行"),
    # ---------- reports/finetune_report ----------
    ("reports/finetune_report.md", r"^階段 D 回報", "結果", "edit", "reports/finetune_report.md#Part 2 訓練與品質評估報告",
     "標題去回報口吻；刪除「計畫 §15 回報格式」"),
    ("reports/finetune_report.md", r"^2\.2 ", "結果", "edit", "", "縮為摘要並連到 logs/stage_d_train"),
    ("reports/finetune_report.md", r"^3\. GPU 異常表", "結果", "edit", "", "縮為摘要並連到 logs/stage_d_train"),
    ("reports/finetune_report.md", r"^5\.3 ", "結果", "edit", "", "ボース 規則與附錄 A 不一致"),
    ("reports/finetune_report.md", r"^結論（PM 原文）", "結果", "edit", "reports/finetune_report.md#結論（試聽原文）",
     "PM 字樣改中性"),
    ("reports/finetune_report.md", r"^7\. 與計畫不符處", "結果", "edit", "reports/finetune_report.md#7. 與原定設定的差異",
     "「計畫」不在 repo；刪除 session 敘事"),
    ("reports/finetune_report.md", r"^10\. 版本與 commit", "結果", "edit",
     "reports/finetune_report.md#10. 版本與 commit hash 記錄", "標題去 issue 口吻"),
    ("reports/finetune_report.md", r"^本專案的產出 commit", "結果", "edit",
     "reports/finetune_report.md#本專案的產出 commit", "「尚未 push、尚未開 PR」過時（PR #6 已合併）"),
    ("reports/finetune_report.md", r"^11\. 停止點", "過程紀錄", "delete", "",
     "等待驗收的狀態，已過時；epochs 16 採認狀態併入 §7"),
    ("reports/finetune_report.md", r"^9\.5 ", "結果", "edit", "", "修正 \\r 被吃掉的路徑"),
    # ---------- reports/issue_compliance ----------
    ("reports/issue_compliance.md", r"^1\.4 ", "流程", "merge",
     "docs/05_transcript_pipeline.md#1.5 為何走 `top500/` 而非 `A_high/ + B_likely/`",
     "v00596 的來源與保留理由；05 §1.5 注意事項 3 已有，補足缺漏"),
    ("reports/issue_compliance.md", r"^7\. 需 PM 裁示", "結果", "merge", "reports/asr_report.md#5. 已知限制",
     "#1 人工試聽無逐筆紀錄、#2 cer_sample 無審閱欄、#3 ボース 規則無人工依據、#5 --prune-raw 抹除排除原因 → 已知限制；"
     "#7 指標對語氣盲視已在 defect_triage P-2。"
     "人工確認：未命中的關鍵字為已結案項（#4 v00596→docs/05 §1.5、#6 samples 對照表、#8 PR #6）與文件行號引用，不需搬"),
    ("reports/issue_compliance.md", r"^8\. 判定為", "結果", "merge", "reports/asr_report.md#5. 已知限制", "同上（缺哪些紀錄）"),
    ("reports/issue_compliance.md", r"^6\. 交叉檢查", "協作產物", "delete", "",
     f"所列過期宣稱於 Step 2／3 在原處修正（#4→asr_report §3、#5／#8→asr_filter_report 刪除、#7→08 §8）；稽核脈絡見 {ISSUE1}"),
    # ---------- reports/issue_reply_part1 ----------
    ("reports/issue_reply_part1.md", r"^目前進度", "結果", "merge", "reports/asr_report.md#資料漏斗",
     "唯一獨有內容為 WRONG 23 vs 3 口徑說明；其餘與 asr_report／correction_diff 重複"),
    ("reports/issue_reply_part1.md", r"^實作決策", "流程", "merge", "docs/05_transcript_pipeline.md#6. 專有名詞校正",
     "詞彙表規則取捨理由"),
    # ---------- reports/pr_description ----------
    ("reports/pr_description.md", r"^關於 commit hash", "結果", "move",
     "reports/finetune_report.md#10. 版本與 commit hash 記錄", "commit hash 記錄方式說明，pr_description 獨有"),
    ("reports/pr_description.md", r"^未解決事項", "協作產物", "delete", "",
     f"「PR 尚未建立」過時（已合併 {PR6}）；人工紀錄缺口併入 asr_report §5；P-1～3 在 defect_triage"),
    ("reports/pr_description.md", r"^已結案的決策", "協作產物", "delete", "",
     f"已記錄於 docs/05 §1.5（v00596）與 docs/07 §6.2（jieba_fast）；原文在 {PR6}"),
    # ---------- split_report ----------
    ("reports/split_report.md", r"^執行摘要", "結果", "edit", "", "刪除「尚未開始訓練」"),
    ("reports/split_report.md", r"^參考音訊", "結果", "edit", "", "00042 改為實際使用的 03536，註明變更"),
    ("reports/split_report.md", r"^風險與待確認", "結果", "edit", "", "刪除「待確認」的已結事項"),
    ("reports/split_report.md", r"^下一步", "過程紀錄", "delete", "", "「等待使用者確認後開始訓練」已過時"),
    # ---------- logs ----------
    ("reports/logs/stage_d_train.md", r"^階段 D 訓練 log", "結果", "edit", "", "刪除「依計畫 §12.4」"),
    ("reports/logs/stage_d_train.md", r"^2\.3 ", "結果", "edit", "", "釐清 s/it 與 finetune_report 的差異"),
    # ---------- samples ----------
    ("samples/README.md", r"^生成樣本", "專案說明", "edit", "", "「供 reviewer」改中性"),
    ("samples/README.md", r"^評估結果", "專案說明", "edit", "", "修正「1 epoch 級別」（實為 16 epoch）"),
]

# 計畫新增的章節（反向檢查用）：(dest, reason)
ADDS = [
    ("README.md#Part 2：GPT-SoVITS v4 LoRA 微調", "與 Part 1 ①②③ 同格式的 Part 2 短節"),
    ("README.md#安裝與執行", "通用 README 應有的入口，連到 07"),
    ("docs/02_model_selection.md#附錄：音訊品質實測", "承接 reports/audio_quality.md"),
    ("docs/08_tools.md#5. Part 2：訓練、推論與評估", "承接 08 §8 與遺漏工具"),
    ("reports/asr_report.md#5. 已知限制", "承接 issue_compliance §7／§8"),
]

HEADING = re.compile(r"^(#{1,4})\s+(.*?)\s*$")


def git_show(path):
    r = subprocess.run(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT,
                       capture_output=True, check=True)
    return r.stdout.decode("utf-8")


def baseline_md_files():
    r = subprocess.run(["git", "ls-tree", "-r", "--name-only", BASELINE], cwd=ROOT,
                       capture_output=True, check=True, text=True)
    return [p for p in r.stdout.splitlines()
            if p.endswith(".md") and not p.startswith("tools/vgmstream/")]


def norm(text):
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def split_sections(text):
    """回傳 [(level, heading, start, end, body)]；行號 1 起算，忽略 code fence 內的 #。"""
    lines = text.splitlines()
    marks, fence = [], False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        m = None if fence else HEADING.match(line)
        if m:
            marks.append((i, len(m.group(1)), m.group(2)))
    if not marks:  # 無 heading：以空行分段
        out, start = [], None
        for i, line in enumerate(lines + [""], 1):
            if line.strip() and start is None:
                start = i
            elif not line.strip() and start is not None:
                body = "\n".join(lines[start - 1:i - 1])
                out.append((0, f"¶ {lines[start - 1][:30]}", start, i - 1, body))
                start = None
        return out
    out = []
    if marks[0][0] > 1 and "".join(lines[:marks[0][0] - 1]).strip():
        out.append((0, "(前言)", 1, marks[0][0] - 1, "\n".join(lines[:marks[0][0] - 1])))
    for k, (start, lvl, head) in enumerate(marks):
        end = marks[k + 1][0] - 1 if k + 1 < len(marks) else len(lines)
        out.append((lvl, head, start, end, "\n".join(lines[start:end])))
    return out


def classify(path, heading):
    for f, pat, cat, act, dest, reason in RULES:
        if f == path and re.search(pat, heading):
            return cat, act, dest, reason
    cat, act, dest, reason = FILE_DEFAULTS[path]
    return cat, act, dest, reason


def file_key(path):
    return "samples" if path == "samples/README.md" else Path(path).stem


def main():
    rows = []
    for path in baseline_md_files():
        if path not in FILE_DEFAULTS:
            sys.exit(f"缺少檔案預設：{path}")
        for n, (lvl, head, start, end, body) in enumerate(split_sections(git_show(path)), 1):
            cat, act, dest, reason = classify(path, head)
            dest = dest.replace("<same>", head)
            rows.append({
                "id": f"{file_key(path)}-{n:03d}", "src_file": path, "heading": head,
                "level": lvl, "lines": f"{start}-{end}",
                "hash": hashlib.sha1(norm(body).encode()).hexdigest()[:10],
                "category": cat, "action": act, "dest": dest, "reason": reason,
            })
    for n, (dest, reason) in enumerate(ADDS, 1):
        rows.append({"id": f"ADD-{n:03d}", "src_file": "", "heading": "", "level": "",
                     "lines": "", "hash": "", "category": "新增", "action": "add",
                     "dest": dest, "reason": reason})
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
