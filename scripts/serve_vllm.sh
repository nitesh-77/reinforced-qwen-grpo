#!/usr/bin/env bash
# Serve one baseline model on an AMD ROCm GPU for the eval harness (OpenAI API on :PORT).
#   scripts/serve_vllm.sh MODEL [PORT] [GPU_MEM_UTIL] [EXTRA vllm args...]
#   scripts/serve_vllm.sh Qwen/Qwen3.5-2B 8010 0.25
# Stop: docker rm -f smol-eval
set -euo pipefail
MODEL=${1:?model id}; PORT=${2:-8010}; UTIL=${3:-0.25}; shift $(( $# < 3 ? $# : 3 ))
source "$(dirname "${BASH_SOURCE[0]}")/gpu_runtime.sh"
IMAGE=${SMOL_VLLM_IMAGE:?Set SMOL_VLLM_IMAGE to a pinned ROCm vLLM image}
docker rm -f smol-eval >/dev/null 2>&1 || true
docker run -d --name smol-eval "${GPU_DOCKER_ARGS[@]}" --ipc host -p "${PORT}:8000" \
  -e HF_HUB_OFFLINE=1 -v "${HF_HOME_HOST:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  --entrypoint python3 "$IMAGE" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" --gpu-memory-utilization "$UTIL" \
  --max-model-len 32768 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder "$@"
echo "waiting for http://127.0.0.1:${PORT}/v1/models ..."
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then echo "ready after $((i*5))s"; exit 0; fi
  if ! docker ps -q --filter name=smol-eval | grep -q .; then echo "container exited:"; docker logs --tail 60 smol-eval; exit 1; fi
  sleep 5
done
echo "not ready after 15 min"; docker logs --tail 60 smol-eval; exit 1
