# -*- coding: utf-8 -*-
"""§10.1 冒煙 exp 建置：從正式產出過濾前 20 條，建立 logs\\ryza_v4_lora_smoke。

不重新預處理；用複製不用 symlink（Windows 權限問題）。
S2 resume 會自動偵測 logs_s2_* 下最新的 G_*.pth，故冒煙必須獨立 exp，
冒煙 checkpoint 才不會被正式訓練誤當 resume 起點（PE-3）。
"""
import json
import os
import shutil

GPT = r"H:\git\GPT-SoVITS"
RYZA = r"H:\git\Ryza_voice"
SRC = os.path.join(GPT, "logs", "ryza_v4_lora")
DST = os.path.join(GPT, "logs", "ryza_v4_lora_smoke")
TRAIN_LIST = os.path.join(RYZA, "data", "finetune_train.list")
N = 20

# 前 20 條的 key（list 順序）
keys = []
with open(TRAIN_LIST, "r", encoding="utf-8") as f:
    for line in f.read().strip("\n").split("\n")[:N]:
        keys.append(os.path.basename(line.split("|")[0]))
assert len(keys) == N, len(keys)

os.makedirs(DST, exist_ok=True)
for sub in ("3-bert", "4-cnhubert", "5-wav32k"):
    os.makedirs(os.path.join(DST, sub), exist_ok=True)

# 2-name2text.txt：過濾前 20 條（保留 4 欄格式）
rows_out = []
with open(os.path.join(SRC, "2-name2text.txt"), "r", encoding="utf-8") as f:
    src_rows = [l for l in f.read().strip("\n").split("\n") if l.strip()]
by_key = {}
for r in src_rows:
    by_key[r.split("\t")[0]] = r
missing = []
for k in keys:
    if k not in by_key:
        missing.append(k)
    else:
        rows_out.append(by_key[k])
assert not missing, missing
with open(os.path.join(DST, "2-name2text.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(rows_out) + "\n")
with open(os.path.join(DST, "2-name2text-0.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(rows_out) + "\n")

# 6-name2semantic.tsv：保留表頭 + 前 20 條
with open(os.path.join(SRC, "6-name2semantic.tsv"), "r", encoding="utf-8") as f:
    sem_lines = [l for l in f.read().strip("\n").split("\n") if l.strip()]
header, sem_rows = sem_lines[0], sem_lines[1:]
sem_by_key = {r.split("\t")[0]: r for r in sem_rows}
sem_out = [header] + [sem_by_key[k] for k in keys if k in sem_by_key]
assert len(sem_out) == N + 1, len(sem_out)
for path in (os.path.join(DST, "6-name2semantic.tsv"), os.path.join(DST, "6-name2semantic-0.tsv")):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sem_out) + "\n")

# 複製對應的 4-cnhubert *.pt 與 5-wav32k（含 .wav 副檔名）
copied_pt, copied_wav = [], []
for k in keys:
    pt_src = os.path.join(SRC, "4-cnhubert", k + ".pt")
    pt_dst = os.path.join(DST, "4-cnhubert", k + ".pt")
    shutil.copy2(pt_src, pt_dst)
    copied_pt.append(k)
    wav_src = os.path.join(SRC, "5-wav32k", k)
    wav_dst = os.path.join(DST, "5-wav32k", k)
    shutil.copy2(wav_src, wav_dst)
    copied_wav.append(k)

summary = {
    "src": SRC,
    "dst": DST,
    "n": N,
    "keys": keys,
    "name2text_rows": len(rows_out),
    "semantic_rows_incl_header": len(sem_out),
    "hubert_pt_copied": len(copied_pt),
    "wav32k_copied": len(copied_wav),
    "3_bert": "created empty (ja behavior)",
}
with open(os.path.join(RYZA, "reports", "logs", "raw", "smoke_exp_setup.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
