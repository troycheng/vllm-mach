# syntax=docker/dockerfile:1
ARG VLLM_IMAGE=vllm/vllm-openai:v0.29.0
FROM ${VLLM_IMAGE}
# Use the base image's CUDA 13.0 toolkit for all native extensions.
ARG MAX_JOBS=8
ENV MAX_JOBS=${MAX_JOBS} CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0a
COPY deploy/requirements-runtime.txt /opt/mach/deploy/requirements-runtime.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-deps -r /opt/mach/deploy/requirements-runtime.txt
COPY deploy/install.py deploy/build-mxfp6.py /opt/mach/deploy/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 /opt/mach/deploy/build-mxfp6.py \
    && mv /opt/mach-build/mxfp6-source.json /opt/mach/ \
    && rm -rf /opt/mach-build
COPY native/lossless_prefill /opt/mach/native/lossless_prefill
COPY native/owner_prefill /opt/mach/native/owner_prefill
COPY native/ar_norm /opt/mach/native/ar_norm
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 /opt/mach/deploy/install.py --native-only --cuda-home /usr/local/cuda \
    && rm -rf /opt/mach-build /opt/mach/native/*/build /opt/mach/native/*/*.egg-info
COPY pyproject.toml README.md LICENSE NOTICE /opt/mach/
COPY src /opt/mach/src
WORKDIR /opt/mach
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 deploy/install.py --runtime-only \
    && rm -rf /opt/mach/build /opt/mach/src/*.egg-info
ENV OMP_NUM_THREADS=1
ENTRYPOINT ["vllm-mach-serve"]
