#!/bin/bash
# 使用 vLLM 部署 Qwen2.5-1.5B-Instruct 作为 OpenAI 兼容 API 服务
# 参考: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
#
# 前置要求:
#   pip install vllm
#
# 用法:
#   bash serve_qwen25_1.5b_inst.sh                    # 默认配置启动（使用 GPU 0）
#   bash serve_qwen25_1.5b_inst.sh --gpu 1            # 指定使用 GPU 1
#   bash serve_qwen25_1.5b_inst.sh --port 9000        # 指定端口
#   bash serve_qwen25_1.5b_inst.sh --gpu-memory-utilization 0.9  # 调整 GPU 显存
#
# 也可通过环境变量指定显卡: VLLM_CUDA_DEVICES=1 bash serve_qwen25_1.5b_inst.sh
#
# 启动后 API 地址: http://localhost:8084/v1
# 测试: python test_vllm_client.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${SCRIPT_DIR}/../qwen_1.5b_instruct"
PORT="${VLLM_PORT:-8084}"
API_KEY="${VLLM_API_KEY:-token-abc123}"
# 指定使用的显卡，默认 GPU 0；可通过 --gpu 或环境变量 VLLM_CUDA_DEVICES 设置
CUDA_DEVICES="${VLLM_CUDA_DEVICES:-1}"

# 解析额外参数（传递给 vllm serve）
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpu)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# 限制 vLLM 只使用指定的显卡
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# 指定 API 请求时使用的模型名（与 addition/run_full_pipeline 等客户端一致）
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen25-1.5b-instruct}"

echo "=========================================="
echo "vLLM OpenAI 兼容 API 服务"
echo "=========================================="
echo "模型: ${MODEL}"
echo "显卡: GPU ${CUDA_DEVICES}"
echo "端口: ${PORT}"
echo "API 地址: http://localhost:${PORT}/v1"
echo "API Key: ${API_KEY}"
echo "API 模型名: ${SERVED_MODEL_NAME}"
echo "=========================================="

# 检查 vllm 是否安装
if ! python -c "import vllm" 2>/dev/null; then
    echo "错误: vllm 未安装。请运行: pip install vllm"
    exit 1
fi

exec vllm serve "${MODEL}" \
    --dtype auto \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --gpu-memory-utilization 0.2 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    "${EXTRA_ARGS[@]}"
