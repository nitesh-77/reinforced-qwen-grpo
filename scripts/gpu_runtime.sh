#!/usr/bin/env bash
if [[ "${SMOL_GPU_BACKEND:-rocm}" != rocm ]]; then
  echo "This launcher supports only AMD ROCm GPUs." >&2
  return 1
fi
GPU_DOCKER_ARGS=(--device /dev/kfd --device /dev/dri -e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}")
gpu_device_groups=()
for gpu_device in /dev/kfd /dev/dri/renderD*; do
  [[ -e "$gpu_device" ]] || continue
  gpu_group=$(stat -c '%g' "$gpu_device") || return 1
  if [[ " ${gpu_device_groups[*]} " != *" $gpu_group "* ]]; then
    gpu_device_groups+=("$gpu_group")
    GPU_DOCKER_ARGS+=(--group-add "$gpu_group")
  fi
done
unset gpu_device_groups gpu_device gpu_group
