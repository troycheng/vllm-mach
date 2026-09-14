# syntax=docker/dockerfile:1
# cuda132 is a named build context containing the public CUDA 13.2 toolkit.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.29.0
FROM ${VLLM_IMAGE}
COPY --from=cuda132 / /opt/cuda-13.2/
ARG MAX_JOBS=8
ENV MAX_JOBS=${MAX_JOBS} CUDA_HOME=/opt/cuda-13.2 TORCH_CUDA_ARCH_LIST=12.0a
COPY deploy/requirements-runtime.txt /opt/mach/deploy/requirements-runtime.txt
RUN python3 -m pip install --no-deps -r /opt/mach/deploy/requirements-runtime.txt
COPY deploy/install.py /opt/mach/deploy/install.py
COPY profiles/exllamav3-1.5.0 /opt/mach/profiles/exllamav3-1.5.0
RUN python3 /opt/mach/deploy/install.py --native-only --native-part exllamav3 --cuda-home /opt/cuda-13.2 --lossless-cuda-home /usr/local/cuda
COPY native/exl3_m32 /opt/mach/native/exl3_m32
RUN python3 /opt/mach/deploy/install.py --native-only --native-part exl3_m32 --cuda-home /opt/cuda-13.2 --lossless-cuda-home /usr/local/cuda
COPY native/exl3_temporal_m24 /opt/mach/native/exl3_temporal_m24
RUN python3 /opt/mach/deploy/install.py --native-only --native-part exl3_temporal_m24 --cuda-home /opt/cuda-13.2 --lossless-cuda-home /usr/local/cuda
COPY native/lossless_prefill /opt/mach/native/lossless_prefill
RUN python3 /opt/mach/deploy/install.py --native-only --native-part lossless_prefill --cuda-home /opt/cuda-13.2 --lossless-cuda-home /usr/local/cuda
COPY native/owner_prefill /opt/mach/native/owner_prefill
RUN python3 /opt/mach/deploy/install.py --native-only --native-part owner_prefill --cuda-home /opt/cuda-13.2 --lossless-cuda-home /usr/local/cuda
COPY deploy/build-mxfp6.py /opt/mach/deploy/build-mxfp6.py
RUN python3 /opt/mach/deploy/build-mxfp6.py
COPY . /opt/mach
WORKDIR /opt/mach
# Docker builds do not expose a GPU; the profile still verifies its JIT target.
RUN FLASHINFER_CUDA_ARCH_LIST=12.0f python3 deploy/install.py --runtime-only --cuda-home /opt/cuda-13.2 --lossless-cuda-home /usr/local/cuda
ENV OMP_NUM_THREADS=1
ENTRYPOINT ["python3", "/opt/mach/deploy/serve.py"]
