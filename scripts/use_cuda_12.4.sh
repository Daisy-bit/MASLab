#!/usr/bin/env bash
# 在当前 shell 中切换到 CUDA 12.4（仅影响本终端）
# 用法: source scripts/use_cuda_12.4.sh

CUDA_12_4="/GLOBALFS/sysu_chchen_1/cuda-12.4"
export CUDA_HOME="$CUDA_12_4"
export PATH="$CUDA_12_4/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$CUDA_12_4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# 必须让 pip 自带的 nvjitlink 优先于 CUDA_HOME/lib64，否则 torch 2.5+cu12 可能报
# libcusparse.so.12: undefined symbol __nvJitLinkComplete_12_4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=prepend_nvjitlink_ld_path.sh
source "${SCRIPT_DIR}/prepend_nvjitlink_ld_path.sh"
echo "CUDA_HOME=$CUDA_HOME (using CUDA 12.4; nvjitlink prepended if found)"
