#!/usr/bin/env bash
# Preflight for the GPU VM: hardware, disk, tools, network, and whether the installed vLLM has what config.yaml uses.
# Run from the repo root before and after installing: bash scripts/vm_check.sh | tee reports/vm_check.txt
if [ -f .vm_env ]; then source .vm_env; fi  # machine-specific exports such as HF_HOME
echo "== $(date -u +%FT%TZ) on $(hostname)"

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv 2>&1
nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*"
echo "processes on the GPU (should be none):"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1

echo "== disk (need ~60 GB: two models ~40 GB in the HF cache, venv with vLLM ~10 GB, datasets and outputs)"
HF=${HF_HOME:-$HOME/.cache/huggingface}
echo "all filesystems (a bigger one can hold the models: export HF_HOME=/that/disk/hf):"
df -h -x tmpfs -x devtmpfs -x squashfs -x overlay 2>/dev/null
echo "block devices (a disk without a mount point is not usable yet):"
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT 2>/dev/null
echo "HF cache: $HF"
du -sh "$HF" "$HOME/.cache/uv" .venv 2>/dev/null
mkdir -p "$HF" 2>/dev/null
FREE_GB=$(df -BG --output=avail "$HF" 2>/dev/null | tail -1 | tr -dc 0-9)
if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt 45 ]; then
  echo "WARNING: only ${FREE_GB} GB free where the models go ($HF); Qwen (19 GB) + Gemma (20 GB) need ~45 GB with headroom"
fi

echo "== memory and CPUs"
free -g 2>/dev/null | head -2
echo "CPUs: $(nproc 2>/dev/null)"

echo "== tools"
for t in python3 uv tmux git curl; do printf '%-8s %s\n' "$t" "$(command -v "$t" || echo MISSING)"; done

echo "== network (HTTP status; 000 = unreachable: set a proxy or HF_ENDPOINT)"
for u in https://huggingface.co https://pypi.org https://github.com; do
  printf '%-24s %s\n' "$u" "$(curl -sI --max-time 10 -o /dev/null -w '%{http_code}' "$u")"
done

echo "== models already downloaded"
ls "$HF/hub" 2>/dev/null | grep -i -E "qwen3.5-9b|gemma-4-31b" || echo "none yet"

echo "== vLLM"
if [ -x .venv/bin/vllm ]; then
  .venv/bin/python - <<'EOF'
import torch, vllm
print(f"vllm {vllm.__version__} | torch {torch.__version__} | CUDA {torch.version.cuda} | GPU visible: {torch.cuda.is_available()}")
try:
    from vllm.reasoning import ReasoningParserManager
    names = set(getattr(ReasoningParserManager, "reasoning_parsers", {}) or getattr(ReasoningParserManager, "lazy_parsers", {}))
    for p in ("qwen3", "gemma4"):
        print(f"reasoning parser {p}: {'ok' if p in names else 'MISSING'}")
except Exception as e:
    print(f"could not list reasoning parsers ({e}); check `vllm serve --help=all`")
EOF
  HELP=$(.venv/bin/vllm serve --help=all 2>/dev/null || .venv/bin/vllm serve --help 2>/dev/null)
  for flag in --language-model-only --reasoning-parser --async-scheduling --speculative-config --limit-mm-per-prompt; do
    if grep -q -- "$flag" <<<"$HELP"; then echo "flag $flag: ok"; else echo "flag $flag: MISSING"; fi
  done
else
  echo "not installed yet: uv venv --python 3.12 && uv pip install -e '.[dev,vm]'"
fi
