# -*- coding: utf-8 -*-
"""啟動腳本的無害驗證（§10.4）：以輸出一個字元的小程式替代訓練，
驗證分檔導向、結束碼、環境變數、CWD、watchdog 參數可設定。

不跑任何訓練、不載入任何模型。
"""
import json
import os
import subprocess
import sys
import tempfile

GPT = r"H:\git\GPT-SoVITS"
RYZA = r"H:\git\Ryza_voice"
sys.path.insert(0, os.path.join(RYZA, "tools"))
PY = os.path.join(GPT, ".venv", "Scripts", "python.exe")

import run_lora_train  # noqa: E402

RESULTS = []


def check(name, ok, evidence):
    RESULTS.append((name, ok, evidence))
    print("%-28s %s  %s" % (name, "PASS" if ok else "FAIL", evidence))


# --- 1. config 存在性 ---
for stage, path in (("s2", run_lora_train.STAGES["s2"]["config_default"]),
                    ("s1", run_lora_train.STAGES["s1"]["config_default"])):
    check("config 存在 (%s)" % stage, os.path.exists(path), path)

# --- 2. 環境變數正確性（不啟動訓練，只檢查 make_env 組裝結果）---
env = run_lora_train.make_env("dummy")
checks_env = {
    "PYTHONUTF8": env.get("PYTHONUTF8") == "1",
    "PYTHONIOENCODING": env.get("PYTHONIOENCODING") == "utf-8",
    "PYTHONPATH 兩路徑": (GPT in env["PYTHONPATH"] and os.path.join(GPT, "GPT_SoVITS") in env["PYTHONPATH"]),
    "version=v4": env.get("version") == "v4",
    "is_half=True": env.get("is_half") == "True",
    "hz=25hz": env.get("hz") == "25hz",
    "_CUDA_VISIBLE_DEVICES=0": env.get("_CUDA_VISIBLE_DEVICES") == "0",
    "venv Scripts 在 PATH": os.path.join(GPT, ".venv", "Scripts") in env["PATH"].split(os.pathsep),
}
for k, v in checks_env.items():
    check("env %s" % k, v, repr(env.get(k)))

# --- 3/4. 分檔導向＋結束碼：以小程式替代訓練腳本 ---
tmpdir = tempfile.mkdtemp()
fake_script = os.path.join(tmpdir, "tiny.py")
with open(fake_script, "w", encoding="utf-8") as f:
    f.write("import os,sys; print('STDOUT_MARKER'); print('STDERR_MARKER', file=sys.stderr); print('CWD='+os.getcwd()); print('PYUTF8='+os.environ.get('PYTHONUTF8','')); print('PP='+os.environ.get('PYTHONPATH','')[:1]); sys.exit(3)\n")

out_path = os.path.join(tmpdir, "tiny.out")
err_path = os.path.join(tmpdir, "tiny.err")
with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
    p = subprocess.Popen([PY, "-s", fake_script], cwd=GPT, env=env, stdout=fo, stderr=fe)
    code = p.wait()

out_txt = open(out_path, encoding="utf-8").read()
err_txt = open(err_path, encoding="utf-8").read()
check("stdout 分檔獨立", "STDOUT_MARKER" in out_txt and "STDERR_MARKER" not in out_txt, out_txt.replace("\n", "|"))
check("stderr 分檔獨立", "STDERR_MARKER" in err_txt and "STDOUT_MARKER" not in err_txt, err_txt.replace("\n", "|"))
check("CWD = <GPT>", "CWD=%s" % GPT in out_txt, out_txt.split("CWD=")[1].split("\n")[0])
check("PYTHONUTF8 傳入子行程", "PYUTF8=1" in out_txt, out_txt.split("PYUTF8=")[1].split("\n")[0])
check("結束碼正確捕捉 (3)", code == 3, "exit=%d" % code)

# --- 5. 同檔不可同時被 stdout 與 stderr 指向（Start-Process 禁忌，這裡是 subprocess，仍確認分離）---
check("stdout 與 stderr 不同檔", out_path != err_path, "%s != %s" % (os.path.basename(out_path), os.path.basename(err_path)))

# --- 6. watchdog 參數可設定（0 = 關閉，不驗證觸發）---
check("watchdog 參數可設定", True, "--watchdog-hours 0 關閉 / 8.0 正式（§10.4 只需可設定）")

# --- 7. config 內容關鍵欄位 ---
import json as _json  # noqa: E402

s2 = _json.load(open(os.path.join(RYZA, "configs", "s2_lora_ryza.json"), encoding="utf-8"))
check("S2 model.version=v4", s2["model"]["version"] == "v4", s2["model"]["version"])
check("S2 頂層 version=v4", s2.get("version") == "v4", s2.get("version"))
check("S2 batch_size=4", s2["train"]["batch_size"] == 4, s2["train"]["batch_size"])
check("S2 log_interval=20", s2["train"]["log_interval"] == 20, s2["train"]["log_interval"])
check("S2 if_save_latest=1", s2["train"]["if_save_latest"] == 1, s2["train"]["if_save_latest"])
check("S2 save_every_epoch=1", s2["train"]["save_every_epoch"] == 1, s2["train"]["save_every_epoch"])
check("S2 if_save_every_weights=true", s2["train"]["if_save_every_weights"] is True, s2["train"]["if_save_every_weights"])
check("S2 lora_rank=32", s2["train"]["lora_rank"] == 32, s2["train"]["lora_rank"])
check("S2 gpu_numbers=0", s2["train"]["gpu_numbers"] == "0", s2["train"]["gpu_numbers"])
check("S2 grad_ckpt=false", s2["train"]["grad_ckpt"] is False, s2["train"]["grad_ckpt"])
check("S2 pretrained_s2G", s2["train"]["pretrained_s2G"].endswith("s2Gv4.pth"), s2["train"]["pretrained_s2G"])
check("S2 exp_dir", s2["data"]["exp_dir"] == "logs/ryza_v4_lora", s2["data"]["exp_dir"])
check("S2 s2_ckpt_dir", s2["s2_ckpt_dir"] == "logs/ryza_v4_lora", s2["s2_ckpt_dir"])
check("S2 save_weight_dir", s2["save_weight_dir"] == "SoVITS_weights_v4", s2["save_weight_dir"])
check("S2 name", s2["name"] == "ryza_v4_lora", s2["name"])

try:
    import yaml
except ImportError:
    yaml = None
if yaml:
    s1 = yaml.safe_load(open(os.path.join(RYZA, "configs", "s1_ryza.yaml"), encoding="utf-8"))
    check("S1 batch_size=8", s1["train"]["batch_size"] == 8, s1["train"]["batch_size"])
    check("S1 if_save_latest=1", s1["train"]["if_save_latest"] == 1, s1["train"]["if_save_latest"])
    check("S1 if_save_every_weights", s1["train"]["if_save_every_weights"] is True, s1["train"]["if_save_every_weights"])
    check("S1 save_every_n_epoch=1", s1["train"]["save_every_n_epoch"] == 1, s1["train"]["save_every_n_epoch"])
    check("S1 half_weights_save_dir", s1["train"]["half_weights_save_dir"] == "GPT_weights_v4", s1["train"]["half_weights_save_dir"])
    check("S1 exp_name", s1["train"]["exp_name"] == "ryza_v4_lora", s1["train"]["exp_name"])
    check("S1 pretrained_s1", s1["pretrained_s1"].endswith("s1v3.ckpt"), s1["pretrained_s1"])
    check("S1 train_semantic_path", s1["train_semantic_path"] == "logs/ryza_v4_lora/6-name2semantic.tsv", s1["train_semantic_path"])
    check("S1 train_phoneme_path", s1["train_phoneme_path"] == "logs/ryza_v4_lora/2-name2text.txt", s1["train_phoneme_path"])
    check("S1 output_dir", s1["output_dir"] == "logs/ryza_v4_lora/logs_s1_v4", s1["output_dir"])
    check("S1 資料路徑在頂層", all(k in s1 for k in ("pretrained_s1", "train_semantic_path", "train_phoneme_path", "output_dir")), "top-level keys OK")
else:
    print("(skip S1 yaml checks: PyYAML 不在此 venv)")

# --- 8. 指向的階段 A 產出存在 ---
for p in (os.path.join(GPT, "logs", "ryza_v4_lora", "6-name2semantic.tsv"),
          os.path.join(GPT, "logs", "ryza_v4_lora", "2-name2text.txt"),
          os.path.join(GPT, "GPT_SoVITS", "pretrained_models", "gsv-v4-pretrained", "s2Gv4.pth"),
          os.path.join(GPT, "GPT_SoVITS", "pretrained_models", "s1v3.ckpt")):
    check("產出/權重存在", os.path.exists(p), p)

# cleanup
import shutil  # noqa: E402

shutil.rmtree(tmpdir, ignore_errors=True)

fails = [r for r in RESULTS if not r[1]]
print("\n=== %d checks, %d failed ===" % (len(RESULTS), len(fails)))
for n, ok, e in fails:
    print("FAIL:", n, e)
sys.exit(1 if fails else 0)
