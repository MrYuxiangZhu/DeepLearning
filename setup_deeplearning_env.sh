#!/bin/bash

# 指定需要激活的 Conda 环境名称。
CONDA_ENV_NAME="deep_learning_py11"

# 指定 pip 镜像源，这里使用清华源加速下载。
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# 开启严格模式：
# -e 表示命令失败立即退出，
# -u 表示使用未定义变量时报错，
# -o pipefail 表示管道中任一命令失败都算失败。
set -euo pipefail

# 打印开始信息，便于用户观察脚本执行状态。
echo "====================================="
echo "Starting dependency installation. Current target environment: ${CONDA_ENV_NAME}"
echo "====================================="

# 定义依赖列表，后续通过循环逐个安装，结构更清晰，也更容易维护。
# 说明：以下已覆盖 video_human_animal_detector.py 等示例所需库——
# torch、torchvision、transformers、pillow、tqdm、opencv-python（其中 opencv-python 专用于 OpenCV 视频读写）。
PACKAGES=(
  "absl-py==2.3.1"
  "accelerate==1.12.0"
  "aiohappyeyeballs==2.6.1"
  "aiohttp==3.13.3"
  "aiosignal==1.4.0"
  "annotated-types==0.7.0"
  "anyio==4.12.1"
  "attrs==25.4.0"
  "certifi==2026.1.4"
  "charset-normalizer==3.4.4"
  "datasets==4.5.0"
  "deepspeed==0.18.4"
  "dill==0.4.0"
  "einops==0.8.1"
  "filelock==3.20.3"
  "frozenlist==1.8.0"
  "fsspec==2025.10.0"
  "grpcio==1.76.0"
  "h11==0.16.0"
  "hf-xet==1.2.0"
  "hjson==3.1.0"
  "httpcore==1.0.9"
  "httpx==0.28.1"
  "huggingface-hub==0.36.0"
  "idna==3.11"
  "Jinja2==3.1.6"
  "Markdown==3.10"
  "MarkupSafe==3.0.3"
  "mpmath==1.3.0"
  "msgpack==1.1.2"
  "multidict==6.7.0"
  "multiprocess==0.70.18"
  "networkx==3.6.1"
  "ninja==1.13.0"
  "numpy==2.4.1"
  "opencv-python==4.10.0.84"
  "nvidia-cublas-cu12==12.8.4.1"
  "nvidia-cuda-cupti-cu12==12.8.90"
  "nvidia-cuda-nvrtc-cu12==12.8.93"
  "nvidia-cuda-runtime-cu12==12.8.90"
  "nvidia-cudnn-cu12==9.10.2.21"
  "nvidia-cufft-cu12==11.3.3.83"
  "nvidia-cufile-cu12==1.13.1.3"
  "nvidia-curand-cu12==10.3.9.90"
  "nvidia-cusolver-cu12==11.7.3.90"
  "nvidia-cusparse-cu12==12.5.8.93"
  "nvidia-cusparselt-cu12==0.7.1"
  "nvidia-nccl-cu12==2.27.5"
  "nvidia-nvjitlink-cu12==12.8.93"
  "nvidia-nvshmem-cu12==3.3.20"
  "nvidia-nvtx-cu12==12.8.90"
  "packaging==25.0"
  "pandas==2.3.3"
  "pillow==12.1.0"
  "propcache==0.4.1"
  "protobuf==3.20.3"
  "psutil==7.2.1"
  "py-cpuinfo==9.0.0"
  "pyarrow==22.0.0"
  "pydantic==2.12.5"
  "pydantic_core==2.41.5"
  "python-dateutil==2.9.0.post0"
  "pytz==2025.2"
  "PyYAML==6.0.3"
  "regex==2026.1.15"
  "requests==2.32.5"
  "safetensors==0.7.0"
  "sentencepiece==0.2.1"
  "six==1.17.0"
  "sympy==1.14.0"
  "tensorboard==2.20.0"
  "tensorboard-data-server==0.7.2"
  "tokenizers==0.22.2"
  "torch==2.9.1"
  "torchvision==0.24.1"
  "tqdm==4.67.1"
  "transformers==4.57.5"
  "triton==3.5.1"
  "typing-inspection==0.4.2"
  "typing_extensions==4.15.0"
  "tzdata==2025.3"
  "urllib3==2.6.3"
  "Werkzeug==3.1.5"
  "xxhash==3.6.0"
  "yarl==1.22.0"
)

# 优先尝试激活 Conda 环境，保证依赖安装到目标环境中。
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
  echo "Successfully activated Conda environment: ${CONDA_ENV_NAME}"
else
  echo "Conda not detected, will use the current system Python environment."
fi

# 优先使用当前环境中的 python，再回退到 python3。
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Error: neither python nor python3 was found."
  exit 1
fi

# 使用 python -m pip 可以确保安装到当前激活环境对应的解释器中。
PIP_CMD=("${PYTHON_BIN}" "-m" "pip")

# 打印环境信息，便于确认当前实际使用的是哪个解释器。
echo
echo "Using Python interpreter: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "Using pip index: ${PIP_INDEX_URL}"
echo
echo "Starting dependency installation..."

# 先升级 pip，避免过旧版本导致依赖解析失败。
"${PIP_CMD[@]}" install --upgrade pip -i "${PIP_INDEX_URL}"

# 按顺序逐个安装依赖，便于出错时快速定位具体是哪个包失败。
for package in "${PACKAGES[@]}"; do
  echo
  echo "Installing ${package} ..."
  "${PIP_CMD[@]}" install "${package}" -i "${PIP_INDEX_URL}"
done

# 如果需要安装 DeepSpeedExamples 中的可编辑包，可取消下面注释后再执行。
# "${PIP_CMD[@]}" install -e \
#   "git+https://github.com/microsoft/DeepSpeedExamples.git@6a3d817ab345dc853d0a742a5b0ecf373123dc34#egg=deepspeed_chat&subdirectory=applications/DeepSpeed-Chat" \
#   -i "${PIP_INDEX_URL}"

# 安装完成后，输出核心依赖版本用于核对。
echo
echo "All dependencies installed successfully!"
echo "Verifying core dependency versions:"
"${PIP_CMD[@]}" show deepspeed | awk '/^Version:/{print "deepspeed Version:", $2}'
"${PIP_CMD[@]}" show transformers | awk '/^Version:/{print "transformers Version:", $2}'
"${PIP_CMD[@]}" show torch | awk '/^Version:/{print "torch Version:", $2}'
"${PIP_CMD[@]}" show opencv-python | awk '/^Version:/{print "opencv-python Version:", $2}'

# 输出结束标识。
echo
echo "====================================="
echo "Dependency installation script completed!"
echo "====================================="
