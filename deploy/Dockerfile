# syntax=docker/dockerfile:1
ARG VLLM_IMAGE=vllm/vllm-openai:v0.29.0
FROM ${VLLM_IMAGE}

# --- CUDA 13.0 build toolchain ---
# The base image has nvcc and runtime libraries, but not all development headers
# and linker symlinks needed to compile MXFP6 and the native extensions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends cuda-toolkit-13-0 \
    && rm -rf /var/lib/apt/lists/*
# Use CUDA 13.0 and target SM120 for all native extensions.
ARG MAX_JOBS=8
ENV MAX_JOBS=${MAX_JOBS} CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0a

# --- Python build tools ---
# Keep the base image's vLLM and PyTorch versions; install only build tooling.
COPY deploy/requirements-runtime.txt /opt/mach/deploy/requirements-runtime.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-deps -r /opt/mach/deploy/requirements-runtime.txt

# --- MXFP6 kernels ---
# Build the pinned MXFP6/CUTLASS sources against the base image's PyTorch ABI.
COPY deploy/install.py deploy/build-mxfp6.py /opt/mach/deploy/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 /opt/mach/deploy/build-mxfp6.py \
    && mv /opt/mach-build/mxfp6-source.json /opt/mach/ \
    && rm -rf /opt/mach-build

# --- Dense native extensions ---
# Compile lossless prefill, owner prefill, and fused AllReduce/RMSNorm/MXFP8.
COPY native/lossless_prefill /opt/mach/native/lossless_prefill
COPY native/owner_prefill /opt/mach/native/owner_prefill
COPY native/ar_norm /opt/mach/native/ar_norm
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 /opt/mach/deploy/install.py --native-only --cuda-home /usr/local/cuda \
    && rm -rf /opt/mach-build /opt/mach/native/*/build /opt/mach/native/*/*.egg-info

# --- Mach Python runtime and dependency patches ---
# Install Mach, validate dependency versions, and patch vLLM/FlashInfer.
COPY pyproject.toml README.md LICENSE NOTICE /opt/mach/
COPY src /opt/mach/src
WORKDIR /opt/mach
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 deploy/install.py --runtime-only \
    && rm -rf /opt/mach/build /opt/mach/src/*.egg-info

# --- Serving entry point ---
ENV OMP_NUM_THREADS=1
ENTRYPOINT ["vllm-mach-serve"]
