背景
目前 repo 已完成語音解包與萊莎候選語音篩選（A_high / B_likely / C_possible …），
但尚無文字稿，也沒有任何 TTS 模型。本 issue 分兩階段：先以 ASR 取得部分文字稿，
再以 GPT-SoVITS 為基礎模型進行 finetune，驗證「以遊戲語音建立萊莎聲線 TTS」的可行性。

Part 1：ASR 轉譯（5%）
範圍
轉譯主語音庫約 5%（≈ 478 段），範圍為 ryza_main/A_high/（92）＋ ryza_main/B_likely/（386）。
B_likely 尚未全數人工確認，轉譯前須逐一試聽確認為萊莎；非萊莎者移至 wrong/
並更新 tools/neg_labels.py。確認後不足 478 段者，可由 C_possible 中 svm_dec 最高者依序遞補。
要求
使用 Whisper large-v3（或其他 ASR 模型，須於 PR 說明選擇理由），language=ja。
新增 tools/transcribe.py：不得寫死本機絕對路徑（如 D:\...），路徑以參數或設定檔傳入。
過濾以下片段並記錄排除原因：時長 < 2 秒或 > 15 秒、非說話聲（喘氣、吶喊）、
ASR 輸出為空、重複或明顯幻覺。
驗收要件

產出 GPT-SoVITS 格式清單 data/ryza_train.list，每行為 音檔路徑|ryza|ja|文字。

有效轉譯段數 ≥ 450 段（扣除過濾後），且全數經人工確認為萊莎。

產出 reports/asr_report.md，內容須包含：
使用的模型與參數（beam size、temperature 等）
總段數、過濾段數及原因統計、有效總時長
品質抽查：隨機抽 50 段人工校對，記錄 Whisper 原始輸出的 CER（字元錯誤率）

專有名詞（人名、地名、鍊金術用語）已人工校正；最終清單以校正後文字為準。

新增 requirements.txt，列出本階段所需套件與版本。


Part 2：GPT-SoVITS finetune
要求
基礎模型指定為 GPT-SoVITS，須在 PR 中記錄使用的版本與 commit hash。
訓練前自 ryza_train.list 保留 5%（約 24 段）作為測試集，不得用於訓練。
SoVITS 與 GPT 兩個階段皆須 finetune，記錄 epoch、batch size 等訓練參數。
驗收要件

新增 docs/finetune.md：從資料準備到推論的完整重現步驟，他人照做可重現。

提交訓練設定檔與訓練 log（不含模型權重）。

生成樣本：
(a) 測試集 10 句：以原台詞文字生成，與原音並列
(b) 全新台詞 10 句：不存在於遊戲中的日文句子，涵蓋平敘、疑問、興奮、悲傷語氣

聲線相似度：以既有 ECAPA-TDNN pipeline（tools/spk_model.py）計算生成音訊與
A_high 中心點的 cosine similarity；生成樣本平均值須 ≥ 測試集真實音訊平均值的 90%。

可懂度：以 Whisper 轉寫生成音訊，與輸入文字比對，平均 CER ≤ 10%。

產出 reports/finetune_report.md，彙整上述數據，並附主觀試聽評語
（聲線、發音、語氣自然度，各列出明顯缺陷）。

訓練過程可穩定完成；若遇到既有的 GPU 不穩問題，須記錄處理方式。