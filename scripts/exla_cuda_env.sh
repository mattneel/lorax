#!/usr/bin/env bash
set -euo pipefail

print_python_nvidia_lib_dirs() {
  if ! command -v python3 >/dev/null 2>&1; then
    return 0
  fi

  python3 - <<'PY'
from pathlib import Path
base = Path.home() / ".local" / "lib"
if base.exists():
    for path in sorted(base.glob("python*/site-packages/nvidia/*/lib")):
        if path.is_dir():
            print(path)
PY
}

build_candidate_lib_dirs() {
  local wsl_lib_dir="${WSL_LIB_DIR:-/usr/lib/wsl/lib}"

  echo "${CUDA_LIB_DIR}"
  echo "/usr/local/cuda/lib64"
  echo "/usr/local/cuda/targets/x86_64-linux/lib"
  echo "/usr/lib/x86_64-linux-gnu"
  echo "${wsl_lib_dir}"
  print_python_nvidia_lib_dirs
}

find_first_lib_dir_for() {
  local lib_name="$1"
  while IFS= read -r dir; do
    [[ -z "${dir}" ]] && continue
    if [[ -f "${dir}/${lib_name}" ]]; then
      echo "${dir}"
      return 0
    fi
  done < <(build_candidate_lib_dirs | awk '!seen[$0]++')

  return 1
}

missing_required_cuda_libs() {
  local required_libs=(
    "libcudart.so.12"
    "libnvrtc.so.12"
    "libcublas.so.12"
    "libcublasLt.so.12"
    "libcufft.so.11"
    "libcusolver.so.11"
    "libcudnn.so.9"
    "libnccl.so.2"
    "libnvJitLink.so.12"
  )
  local missing=()
  local lib=""

  for lib in "${required_libs[@]}"; do
    if ! find_first_lib_dir_for "${lib}" >/dev/null; then
      missing+=("${lib}")
    fi
  done

  if [[ "${#missing[@]}" -gt 0 ]]; then
    printf '%s\n' "${missing[@]}"
    return 1
  fi

  return 0
}

install_user_cuda12_libs() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required for user-space CUDA library installation." >&2
    return 1
  fi

  echo "==> Installing user-space CUDA12 runtime libs via pip"
  python3 -m pip install --user --upgrade \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-nccl-cu12
}

configure_exla_cuda_env() {
  local install_missing_cuda_libs="${INSTALL_MISSING_CUDA_LIBS:-0}"
  local wsl_lib_dir="${WSL_LIB_DIR:-/usr/lib/wsl/lib}"
  local missing_libs=""
  local candidate_dirs=()
  local lib_dir=""

  export XLA_TARGET="${XLA_TARGET:-cuda12}"
  export CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda/targets/x86_64-linux/lib}"

  if [[ ! -f "${CUDA_LIB_DIR}/libcudart.so" ]]; then
    echo "Expected ${CUDA_LIB_DIR}/libcudart.so but it was not found." >&2
    echo "Set CUDA_LIB_DIR to your CUDA runtime lib path and rerun." >&2
    return 1
  fi

  if ! missing_libs="$(missing_required_cuda_libs 2>/dev/null)"; then
    if [[ "${install_missing_cuda_libs}" == "1" ]]; then
      install_user_cuda12_libs
      if ! missing_libs="$(missing_required_cuda_libs 2>/dev/null)"; then
        echo "Missing required CUDA libs after installation attempt:" >&2
        echo "${missing_libs}" >&2
        return 1
      fi
    else
      echo "Missing required CUDA libs:" >&2
      echo "${missing_libs}" >&2
      echo "Run ./scripts/setup_exla_cuda12.sh to install user-space CUDA12 libs." >&2
      return 1
    fi
  fi

  export NCCL_LIB_DIR="${NCCL_LIB_DIR:-$(find_first_lib_dir_for libnccl.so.2)}"

  while IFS= read -r lib_dir; do
    [[ -z "${lib_dir}" ]] && continue
    candidate_dirs+=("${lib_dir}")
  done < <(build_candidate_lib_dirs | awk '!seen[$0]++')

  if [[ "${#candidate_dirs[@]}" -eq 0 ]]; then
    echo "Could not derive CUDA library search paths." >&2
    return 1
  fi

  local joined_dirs
  joined_dirs="$(IFS=:; echo "${candidate_dirs[*]}")"

  # WSL2/CUDA runtime hint:
  # EXLA NIF may fail to resolve CUDA symbols unless libcudart is preloaded.
  export LD_LIBRARY_PATH="${joined_dirs}:${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="${CUDA_LIB_DIR}/libcudart.so${LD_PRELOAD:+:${LD_PRELOAD}}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script is meant to be sourced, not executed directly." >&2
  exit 1
fi
