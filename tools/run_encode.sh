#!/bin/bash
# GPU-only 分段編碼驅動：CUDA_LAUNCH_BLOCKING=1 序列化核心，
# 每段獨立進程，崩潰（隨機驅動不穩）重試最多 3 次，不退回 CPU。
set -u
CHUNK=500
N=9569
mkdir -p tools/chunks
rm -f tools/chunks/*.npy

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for ((start=1; start<=N; start+=CHUNK)); do
  end=$((start+CHUNK)); [ $end -gt $((N+1)) ] && end=$((N+1))
  out="tools/chunks/$(printf '%05d' $start).npy"
  ok=0
  for try in 1 2 3 4; do
    if python tools/encode_chunk.py $start $end "$out" >>tools/encode_log.txt 2>&1; then
      ok=1; break
    fi
    echo "FAIL $start-$end try $try, retrying on GPU" >&2
  done
  if [ $ok -eq 1 ]; then echo "OK   $start-$end (try $try)"; else echo "FATAL $start-$end"; fi
done
echo "ALL CHUNKS DONE"
