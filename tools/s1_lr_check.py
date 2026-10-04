# -*- coding: utf-8 -*-
"""SM-S1-5 學習率檢查（D11）：以實測取代推論。

計畫 §11.3 的待驗證推論是「warmup_steps=2000 是否使正式訓練全程停留在 warmup」。
本腳本直接跑 WarmupCosineLRSchedule，量測：
  - 冒煙（13 step）最後一個 step 的實際 LR
  - 正式（467 樣本 / batch 8 ≈ 59 step per epoch × 20 epoch ≈ 1180 step）最後一個 step 的實際 LR
並與「若排程器正常」��理論值對照。

注意：AR/modules/lr_schedulers.py:58 把 lr 硬寫成 0.002，
本腳本即為驗證該行為在冒煙與正式兩種長度下都成立。
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(r"H:\git\GPT-SoVITS", "GPT_SoVITS"))
from AR.modules.lr_schedulers import WarmupCosineLRSchedule  # noqa: E402

RYZA = r"H:\git\Ryza_voice"
CFG = {
    "lr": 0.01,
    "lr_init": 1.0e-05,
    "lr_end": 0.0001,
    "warmup_steps": 2000,
    "decay_steps": 40000,
}


def run(total_steps, warmup_steps=2000):
    p = torch.nn.Parameter(torch.zeros(2, 2))
    opt = torch.optim.SGD([p], lr=CFG["lr"])
    sched = WarmupCosineLRSchedule(
        opt,
        init_lr=CFG["lr_init"],
        peak_lr=CFG["lr"],
        end_lr=CFG["lr_end"],
        warmup_steps=warmup_steps,
        total_steps=CFG["decay_steps"],
    )
    traj = []
    for step in range(1, total_steps + 1):
        sched.step()
        traj.append(opt.param_groups[0]["lr"])
    return traj


def theoretical(step, warmup_steps=2000):
    """若排程器照 :45-56 的公式運作，該 step 應該是什麼值。"""
    if step < warmup_steps:
        return CFG["lr_init"] + (CFG["lr"] - CFG["lr_init"]) / warmup_steps * (step - 1)
    if step > CFG["decay_steps"]:
        return CFG["lr_end"]
    import math

    ratio = (step - warmup_steps) / (CFG["decay_steps"] - warmup_steps)
    return CFG["lr_end"] + 0.5 * (1 + math.cos(math.pi * ratio)) * (CFG["lr"] - CFG["lr_end"])


def summarize(name, steps):
    traj = run(steps)
    uniq = sorted(set(traj))
    return {
        "case": name,
        "total_steps": steps,
        "actual_lr_first": traj[0],
        "actual_lr_last": traj[-1],
        "actual_lr_distinct_values": uniq,
        "actual_lr_is_constant": len(uniq) == 1,
        "theoretical_lr_last_if_normal": theoretical(steps),
        "warmup_fraction_elapsed": steps / CFG["warmup_steps"],
    }


def main():
    cases = [summarize("smoke (dataset 100 / batch 8 -> 13 step)", 13),
             summarize("formal 8 epoch (467/8=59 x 8 = 472 step)", 472),
             summarize("formal 20 epoch (59 x 20 = 1180 step)", 1180),
             summarize("warmup boundary (2000 step)", 2000)]
    out = {
        "config_optimizer": CFG,
        "source": "GPT_SoVITS/AR/modules/lr_schedulers.py:38-62",
        "hard_lock_line": "AR/modules/lr_schedulers.py:58 -> self.lr = lr = self.end_lr = 0.002",
        "cases": cases,
    }
    path = os.path.join(RYZA, "reports", "logs", "raw", "s1_lr_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()