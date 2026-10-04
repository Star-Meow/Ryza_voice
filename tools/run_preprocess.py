# -*- coding: utf-8 -*-
"""階段 A 預處理驅動：以 venv Python 直接執行官方三步腳本（不經 webui）。

§4.3 全域環境變數在此明確設定；每步 stdout/stderr 分檔保存，記錄結束碼與耗時；
任一步驟非 0 結束即停止（§14）。單 GPU 序列執行，不可並行。
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

GPT = r"H:\git\GPT-SoVITS"
RYZA = r"H:\git\Ryza_voice"
PY = os.path.join(GPT, ".venv", "Scripts", "python.exe")
PRE = os.path.join(GPT, "GPT_SoVITS", "pretrained_models")
SCRIPTS = os.path.join(GPT, "GPT_SoVITS", "prepare_datasets")
RAW_LOGS = os.path.join(RYZA, "reports", "logs", "raw")

EXPS = {
    "train": {
        "exp_name": "ryza_v4_lora",
        "inp_text": os.path.join(RYZA, "data", "finetune_train.list"),
        "opt_dir": os.path.join(GPT, "logs", "ryza_v4_lora"),
    },
    "eval10": {
        "exp_name": "ryza_v4_lora_eval10",
        "inp_text": os.path.join(RYZA, "data", "finetune_eval10.list"),
        "opt_dir": os.path.join(GPT, "logs", "ryza_v4_lora_eval10"),
    },
}

# §4.3 全域環境（每步都設）
BASE_ENV = {
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONPATH": GPT + ";" + os.path.join(GPT, "GPT_SoVITS"),
    "version": "v4",
    "is_half": "True",
    "hz": "25hz",
    "_CUDA_VISIBLE_DEVICES": "0",
    "inp_wav_dir": "",
    "i_part": "0",
    "all_parts": "1",
}

# 各步驟額外環境變數
STEP_ENV = {
    "1": {"bert_pretrained_dir": os.path.join(PRE, "chinese-roberta-wwm-ext-large")},
    "2": {"cnhubert_base_dir": os.path.join(PRE, "chinese-hubert-base")},
    "3": {
        "pretrained_s2G": os.path.join(PRE, "gsv-v4-pretrained", "s2Gv4.pth"),
        "s2config_path": os.path.join(GPT, "GPT_SoVITS", "configs", "s2.json"),
    },
}
STEP_SCRIPT = {
    "1": "1-get-text.py",
    "2": "2-get-hubert-wav32k.py",
    "3": "3-get-semantic.py",
}


def run_step(exp_key: str, step: str) -> dict:
    exp = EXPS[exp_key]
    env = dict(BASE_ENV)
    env.update(STEP_ENV[step])
    env["exp_name"] = exp["exp_name"]
    env["inp_text"] = exp["inp_text"]
    env["opt_dir"] = exp["opt_dir"]
    # 繼承必要的系統變數（PATH 等），其餘由上面明確覆蓋
    # .venv\Scripts 必須在 PATH 上，官方腳本的 load_audio 才能找到 ffmpeg CLI
    scripts_dir = os.path.join(GPT, ".venv", "Scripts")
    env["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
    env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    env["USERPROFILE"] = os.environ.get("USERPROFILE", "")
    env["TEMP"] = os.environ.get("TEMP", "")
    env["TMP"] = os.environ.get("TMP", "")

    os.makedirs(RAW_LOGS, exist_ok=True)
    tag = f"{exp_key}_step{step}"
    out_path = os.path.join(RAW_LOGS, f"{tag}.out")
    err_path = os.path.join(RAW_LOGS, f"{tag}.err")
    script = os.path.join(SCRIPTS, STEP_SCRIPT[step])

    t0 = time.time()
    ts0 = datetime.datetime.now().isoformat(timespec="seconds")
    with open(out_path, "wb") as fout, open(err_path, "wb") as ferr:
        proc = subprocess.Popen(
            [PY, script],
            cwd=GPT,
            env=env,
            stdout=fout,
            stderr=ferr,
        )
        code = proc.wait()
    t1 = time.time()
    ts1 = datetime.datetime.now().isoformat(timespec="seconds")

    result = {
        "exp": exp_key,
        "exp_name": exp["exp_name"],
        "step": step,
        "script": STEP_SCRIPT[step],
        "start": ts0,
        "end": ts1,
        "wall_seconds": round(t1 - t0, 1),
        "exit_code": code,
        "stdout_log": out_path,
        "stderr_log": err_path,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXPS))
    ap.add_argument("--steps", default="1,2,3", help="逗號分隔，預設 1,2,3")
    args = ap.parse_args()

    summary_path = os.path.join(RAW_LOGS, f"{args.exp}_summary.json")
    results = []
    for step in args.steps.split(","):
        step = step.strip()
        r = run_step(args.exp, step)
        results.append(r)
        if r["exit_code"] != 0:
            print(f"!! step {step} exit_code={r['exit_code']}，停止（§14）", flush=True)
            break
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"exp": args.exp, "results": results}, f, ensure_ascii=False, indent=2)
    all_ok = all(r["exit_code"] == 0 for r in results)
    print(f"=== {args.exp} all_ok={all_ok} ===", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
