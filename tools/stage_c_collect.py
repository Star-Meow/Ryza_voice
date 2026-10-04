# -*- coding: utf-8 -*-
"""階段 C 數據彙整：loss 趨勢、每 step 秒數、顯存，以及 S1 checkpoint 的 CPU 載入驗證。

驗收一律以產出檔案內容為準（§4.1），不以 exit code 為準。
loss 取值來源：
  - S2：logs\\ryza_v4_lora_smoke\\train.log 的 `[loss, global_step, lr]`（log_interval=1）
  - S1：Lightning 的 TensorBoard events（venv 已裝 tensorboard 2.21.0，不需安裝）
"""
import json
import os
import re
import sys

GPT = r"H:\git\GPT-SoVITS"
RYZA = r"H:\git\Ryza_voice"
SMOKE = os.path.join(GPT, "logs", "ryza_v4_lora_smoke")
RAW = os.path.join(RYZA, "reports", "logs", "raw")
S2_LOG = os.path.join(SMOKE, "train.log")

# train.log: 2026-10-05 05:43:59,713\tname\tINFO\t[0.0216..., 0, 9.99875e-05]
S2_RE = re.compile(r"^(\S+ \S+)\t\w+\tINFO\t\[([0-9.eE+-]+), (\d+), ([0-9.eE+-]+)\]$")
EPOCH_RE = re.compile(r"^(\S+ \S+)\t\w+\tINFO\t(?:Train Epoch: (\d+) \[0%\]|====> Epoch: (\d+))")


def parse_s2():
    """回傳 [(ts_float, loss, step, lr)]，依 global_step 排序。"""
    rows = []
    with open(S2_LOG, encoding="utf-8") as f:
        for line in f:
            m = S2_RE.match(line.rstrip("\n"))
            if m:
                rows.append((m.group(1), float(m.group(2)), int(m.group(3)), float(m.group(4))))
    rows.sort(key=lambda r: r[2])
    return rows


def epoch_spans():
    """回傳 {epoch: (起 ts, 訖 ts)}，用於算每 epoch 的訓練迴圈時間。"""
    spans, cur = {}, None
    with open(S2_LOG, encoding="utf-8") as f:
        for line in f:
            m = EPOCH_RE.match(line.rstrip("\n"))
            if not m:
                continue
            ts, start_ep, end_ep = m.group(1), m.group(2), m.group(3)
            if start_ep:
                cur = (int(start_ep), ts)
            elif end_ep and cur and cur[0] == int(end_ep):
                spans[end_ep] = (cur[1], ts)
                cur = None
    return spans


def parse_s1_tb():
    """從 TensorBoard events 取 train_loss / train_lr 的逐步值。"""
    root = os.path.join(SMOKE, "logs_s1_v4", "logs_s1_v4")
    if not os.path.isdir(root):
        return {"error": "no events dir: %s" % root}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    found = None
    for dirpath, _dirs, files in os.walk(root):
        if any(f.startswith("events.out.tfevents") for f in files):
            found = dirpath
            break
    if not found:
        return {"error": "no events file under %s" % root}
    ea = EventAccumulator(found)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {"events_dir": found, "tags": tags}
    # S1 的 lightning module 只在 epoch 層記錄（total_loss_epoch / lr_epoch ...），
    # 逐步值只出現在進度棒上，故把全部 scalar tag 都收下來。
    for tag in tags:
        out[tag] = [[e.step, e.value] for e in ea.Scalars(tag)]
    return out


def verify_s1_ckpt():
    """SM-S1-3：以 CPU 載入推論權重，確認頂層三鍵 weight / config / info。"""
    import torch

    d = os.path.join(GPT, "GPT_weights_v4")
    files = sorted(f for f in os.listdir(d) if f.startswith("ryza_v4_lora_smoke-e") and f.endswith(".ckpt"))
    if not files:
        return {"error": "no ryza_v4_lora_smoke-e*.ckpt in %s" % d}
    path = os.path.join(d, files[-1])
    obj = torch.load(path, map_location="cpu", weights_only=False)
    info = {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "top_level_keys": sorted(obj.keys()),
        "has_weight": "weight" in obj,
        "has_config": "config" in obj,
        "has_info": "info" in obj,
        "info_value": obj.get("info"),
        "n_weight_tensors": len(obj.get("weight", {})),
        "weight_dtype": str(next(iter(obj["weight"].values())).dtype) if obj.get("weight") else None,
        "config_keys": sorted(obj.get("config", {}).keys()) if isinstance(obj.get("config"), dict) else None,
    }
    return info


def verify_s2_lora_weights():
    """SM-S2-5：檔頭 04、大小遠小於底模、lora_rank 頂層鍵。"""
    import torch

    from process_ckpt import load_sovits_new  # 需 PYTHONPATH 含 GPT_SoVITS

    d = os.path.join(GPT, "SoVITS_weights_v4")
    out = []
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        with open(p, "rb") as fh:
            head2 = fh.read(2)
        obj = load_sovits_new(p)
        out.append({
            "file": f,
            "size_bytes": os.path.getsize(p),
            "header2": head2.decode("ascii", "replace"),
            "top_level_keys": sorted(obj.keys()),
            "lora_rank": obj.get("lora_rank"),
            "info": obj.get("info"),
            "n_weight_tensors": len(obj.get("weight", {})),
            "has_enc_q": any("enc_q" in k for k in obj.get("weight", {})),
        })
    return out


def main():
    s2 = parse_s2()
    spans = epoch_spans()
    res = {
        "s2": {
            "n_records": len(s2),
            "first": s2[0] if s2 else None,
            "mid": s2[len(s2) // 2] if s2 else None,
            "last": s2[-1] if s2 else None,
            "epoch1_steps": [r for r in s2 if r[2] < 26],
            "epoch2_steps": [r for r in s2 if r[2] >= 26],
            "any_nan_inf": any(r[1] != r[1] or r[1] in (float("inf"), float("-inf")) for r in s2),
            "epoch_spans": spans,
        },
        "s1": parse_s1_tb(),
    }
    for tag in ("s2_smoke", "s2_smoke_resume", "s1_smoke"):
        p = os.path.join(RAW, "%s_summary.json" % tag)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                res.setdefault("summaries", {})[tag] = json.load(f)
    try:
        res["s1_ckpt"] = verify_s1_ckpt()
    except Exception as exc:
        res["s1_ckpt"] = {"error": repr(exc)}
    try:
        res["s2_lora_weights"] = verify_s2_lora_weights()
    except Exception as exc:
        res["s2_lora_weights"] = {"error": repr(exc)}

    out = os.path.join(RAW, "stage_c_metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    sys.exit(main())