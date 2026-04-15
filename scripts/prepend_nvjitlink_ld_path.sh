#!/usr/bin/env bash
# 将当前 Python 环境里 pip 安装的 nvidia-nvjitlink 的 lib 目录放到 LD_LIBRARY_PATH 最前，
# 避免系统/CUDA_HOME 里较旧的 libnvJitLink 先于 torch 的 libcusparse 加载，导致
# undefined symbol: __nvJitLinkComplete_12_4
#
# 用法（需在已激活的 conda/venv 下，且 python 指向该环境）:
#   source scripts/prepend_nvjitlink_ld_path.sh

_py="${PYTHON:-python}"
NVJ_LIB="$($_py -c "
import os
import site
for sp in site.getsitepackages():
    p = os.path.join(sp, 'nvidia', 'nvjitlink', 'lib')
    if os.path.isdir(p):
        print(p)
        break
" 2>/dev/null || true)"
if [[ -n "${NVJ_LIB}" ]]; then
    export LD_LIBRARY_PATH="${NVJ_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
