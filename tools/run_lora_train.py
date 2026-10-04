# -*- coding: utf-8 -*-
"""階段 B/C 啟動腳本：S2 LoRA 與 S1 訓練（不經 webui）。

沿用 tools/run_preprocess.py 風格：
- §4.3 全域環境變數明確設定、CWD = <GPT>
- stdout/stderr 分檔保存，記錄結束碼與耗時，非 0 即停
- .venv\\Scripts 加進子行程 PATH（load_audio 需要 ffmpeg）

額外內建（§10.4）：
- 8 小時 watchdog：到點終止進程，不續跑、不自動刪除任何檔案（正式訓練用；
  --watchdog-hours 0 關閉；冒煙不受限，但需傳可設定參數以利驗收 PE-8）
- 每 5 秒取樣整張 GPU 顯存用量（nvidia-smi），基準約 1455 MiB

階段 C 新增（§11.1）：
- 停滯 watchdog：--stall-timeout-min（預設 15）無 log 輸出即終止進程並記錄
- --vram：以 tools/vram_runner.py 為入口，用 runpy 載入官方腳本（不修改官方檔），
  結束時印 torch.cuda.max_memory_allocated() / max_memory_reserved()
- preflight 建立推論權重目錄：官方僅由 webui.py:200-201 建立，手動驅動缺此步會讓
  process_ckpt.savee 的 open() 失敗，而該失敗被同檔 :59 的裸 except 吞成一行 log，
  exit code 仍為 0（見 docs/pitfalls.md PK-010）
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

GPT = r"H:\git\GPT-SoVITS"
RYZA = r"H:\git\Ryza_voice"
PY = os.path.join(GPT, ".venv", "Scripts", "python.exe")
VRAM_RUNNER = os.path.join(RYZA, "tools", "vram_runner.py")
RAW_LOGS = os.path.join(RYZA, "reports", "logs", "raw")

# §4.1 允許寫入；webui.py:200-201 是官方唯一的建立者
WEIGHT_DIRS = ("SoVITS_weights_v4", "GPT_weights_v4")
VRAM_RE = re.compile(r"\[VRAM\] max_memory_(allocated|reserved)_mib=([0-9.]+)")

# §4.3 全域環境（每步都設）
BASE_ENV = {
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONPATH": GPT + ";" + os.path.join(GPT, "GPT_SoVITS"),
    "version": "v4",
    "is_half": "True",
    "hz": "25hz",
    "_CUDA_VISIBLE_DEVICES": "0",
}

STAGES = {
    "s2": {
        "script": "GPT_SoVITS/s2_train_v3_lora.py",
        "arg": "--config",
        "config_default": os.path.join(RYZA, "configs", "s2_lora_ryza.json"),
    },
    "s1": {
        "script": "GPT_SoVITS/s1_train.py",
        "arg": "--config_file",
        "config_default": os.path.join(RYZA, "configs", "s1_ryza.yaml"),
    },
}


def make_env(config_path):
    env = dict(BASE_ENV)
    scripts_dir = os.path.join(GPT, ".venv", "Scripts")
    env["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
    env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    env["USERPROFILE"] = os.environ.get("USERPROFILE", "")
    env["TEMP"] = os.environ.get("TEMP", "")
    env["TMP"] = os.environ.get("TMP", "")
    return env


def sample_gpu(stop_event, out_path, interval=5.0):
    """每 interval 秒以 nvidia-smi 查詢整張卡的顯存用量，寫入 out_path。"""
    nvidia_smi = shutil.which("nvidia-smi")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("timestamp,used_mib,total_mib\n")
        while not stop_event.is_set():
            try:
                out = subprocess.check_output(
                    [nvidia_smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, timeout=15,
                ).decode("ascii", "ignore").strip()
                ts = datetime.datetime.now().isoformat(timespec="seconds")
                f.write("%s,%s\n" % (ts, out.replace(" ", "")))
                f.flush()
            except Exception:
                pass
            stop_event.wait(interval)


def preflight():
    """建立推論權重目錄。缺了會讓 savee 失敗並被裸 except 吞掉（PK-010）。"""
    created = []
    for d in WEIGHT_DIRS:
        p = os.path.join(GPT, d)
        if not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)
            created.append(d)
    return created


def parse_vram(out_path, err_path):
    """從 stdout/stderr 抓 vram_runner 印的 torch 顯存峰值（MiB）。"""
    found = {}
    for path in (out_path, err_path):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = VRAM_RE.search(line)
                if m:
                    found[m.group(1)] = float(m.group(2))
    return found


def run_stage(stage, config_path, tag, watchdog_hours, gpu_sample, stall_minutes, use_vram):
    spec = STAGES[stage]
    env = make_env(config_path)
    os.makedirs(RAW_LOGS, exist_ok=True)
    out_path = os.path.join(RAW_LOGS, "%s.out" % tag)
    err_path = os.path.join(RAW_LOGS, "%s.err" % tag)
    gpu_path = os.path.join(RAW_LOGS, "%s.gpu" % tag)

    if use_vram:
        cmd = [PY, "-s", VRAM_RUNNER, spec["script"], spec["arg"], config_path]
    else:
        cmd = [PY, "-s", spec["script"], spec["arg"], config_path]
    print("CMD: %s" % " ".join('"%s"' % c if " " in c else c for c in cmd), flush=True)
    print("CWD : %s" % GPT, flush=True)

    created = preflight()
    if created:
        print("preflight: 建立權重目錄 %s" % created, flush=True)

    t0 = time.time()
    ts0 = datetime.datetime.now().isoformat(timespec="seconds")

    stop_evt = threading.Event()
    gpu_thread = threading.Thread(target=sample_gpu, args=(stop_evt, gpu_path), daemon=True)
    if gpu_sample:
        gpu_thread.start()

    killed_reason = None

    with open(out_path, "wb") as fout, open(err_path, "wb") as ferr:
        proc = subprocess.Popen(cmd, cwd=GPT, env=env, stdout=fout, stderr=ferr)
        deadline = t0 + watchdog_hours * 3600 if watchdog_hours and watchdog_hours > 0 else None

        def watchdog():
            if deadline is None:
                return
            while proc.poll() is None:
                if time.time() >= deadline:
                    print("!! watchdog: %s 小時到，終止進程（不自動刪除任何檔案）" % watchdog_hours, flush=True)
                    proc.terminate()
                    return
                time.sleep(10)

        wd = threading.Thread(target=watchdog, daemon=True)
        if deadline is not None:
            wd.start()

        def stall_watchdog():
            """§11.1：無 log 輸出超過 stall_minutes 分鐘即終止。"""
            nonlocal killed_reason
            if not stall_minutes or stall_minutes <= 0:
                return
            last = -1
            last_change = time.time()
            while proc.poll() is None:
                time.sleep(15)
                if proc.poll() is not None:
                    return
                try:
                    size = os.path.getsize(out_path) + os.path.getsize(err_path)
                except OSError:
                    continue
                if size != last:
                    last, last_change = size, time.time()
                    continue
                idle_min = (time.time() - last_change) / 60.0
                if idle_min >= stall_minutes:
                    killed_reason = "stall_no_log_%.1fmin" % idle_min
                    print("!! stall watchdog: %.1f 分鐘無新 log，終止進程" % idle_min, flush=True)
                    proc.terminate()
                    return

        st = threading.Thread(target=stall_watchdog, daemon=True)
        if stall_minutes and stall_minutes > 0:
            st.start()
        code = proc.wait()

    stop_evt.set()
    if gpu_sample:
        gpu_thread.join(timeout=10)
    t1 = time.time()
    ts1 = datetime.datetime.now().isoformat(timespec="seconds")

    peak_mib = None
    if os.path.exists(gpu_path):
        with open(gpu_path, encoding="utf-8") as f:
            vals = [int(l.split(",")[1]) for l in f if "," in l and l.split(",")[1].isdigit()]
        peak_mib = max(vals) if vals else None

    vram = parse_vram(out_path, err_path) if use_vram else {}

    result = {
        "stage": stage,
        "config": config_path,
        "script": spec["script"],
        "cmd": " ".join(cmd),
        "cwd": GPT,
        "start": ts0,
        "end": ts1,
        "wall_seconds": round(t1 - t0, 1),
        "exit_code": code,
        "stdout_log": out_path,
        "stderr_log": err_path,
        "gpu_log": gpu_path,
        "gpu_peak_mib": peak_mib,
        "vram_peak_alloc_mib": vram.get("allocated"),
        "vram_peak_reserved_mib": vram.get("reserved"),
        "watchdog_hours": watchdog_hours,
        "stall_timeout_min": stall_minutes,
        "killed_reason": killed_reason,
        "weight_dirs_created": created,
        "vram_runner": VRAM_RUNNER if use_vram else None,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    with open(os.path.join(RAW_LOGS, "%s_summary.json" % tag), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--config", default=None, help="預設使用 <RYZA>/configs 下對應的正式 config")
    ap.add_argument("--tag", default=None, help="log 檔名前綴（預設 = stage）")
    ap.add_argument("--watchdog-hours", type=float, default=8.0, help="0 = 關閉（冒煙用）；正式訓練 8 小時上限")
    ap.add_argument("--no-gpu-sample", action="store_true")
    ap.add_argument("--stall-timeout-min", type=float, default=15.0,
                    help="§11.1 無 log 輸出的逾時門檻（分鐘）；0 = 關閉")
    ap.add_argument("--vram", action="store_true",
                    help="經 tools/vram_runner.py 載入官方腳本，結束時印 torch 顯存峰值")
    args = ap.parse_args()

    config_path = args.config or STAGES[args.stage]["config_default"]
    if not os.path.exists(config_path):
        print("!! config 不存在：%s" % config_path, flush=True)
        sys.exit(2)
    tag = args.tag or args.stage

    r = run_stage(args.stage, config_path, tag, args.watchdog_hours, not args.no_gpu_sample,
                  args.stall_timeout_min, args.vram)
    if r["exit_code"] != 0:
        print("!! %s exit_code=%s，停止（§14）" % (tag, r["exit_code"]), flush=True)
    sys.exit(0 if r["exit_code"] == 0 else 1)


if __name__ == "__main__":
    main()
