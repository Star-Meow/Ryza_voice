# -*- coding: utf-8 -*-
"""以 runpy 載入官方訓練腳本，並在結束時印出 torch 的顯存峰值。

官方 s2_train_v3_lora.py / s1_train.py 全 repo 都沒有 torch.cuda.max_memory_allocated()
的輸出（grep 0 命中），無法直接取得「PyTorch 配置的顯存」；整卡用量只能靠 nvidia-smi 外部取樣，
兩者不可混為一談。

用法（由 run_lora_train.py --vram 組裝後呼叫，勿手動跑）：
    python vram_runner.py GPT_SoVITS/s2_train_v3_lora.py --config <cfg>

用 runpy 而非修改官方腳本，是為了守住 §4.2「不改官方腳本核心邏輯」；工作目錄仍須為 <GPT>。
"""
import atexit
import runpy
import sys

MIB = 1024 * 1024


def _report():
    try:
        import torch

        if not torch.cuda.is_available():
            print("[VRAM] cuda unavailable", flush=True)
            return
        alloc = torch.cuda.max_memory_allocated() / MIB
        reserved = torch.cuda.max_memory_reserved() / MIB
        print("[VRAM] max_memory_allocated_mib=%.1f" % alloc, flush=True)
        print("[VRAM] max_memory_reserved_mib=%.1f" % reserved, flush=True)
    except Exception as exc:  # 報告失敗不可影響訓練結束碼
        print("[VRAM] report_failed=%r" % (exc,), flush=True)


atexit.register(_report)

if len(sys.argv) < 2:
    print("usage: vram_runner.py <script.py> [args...]", flush=True)
    sys.exit(2)

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")