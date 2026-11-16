import pytest


@pytest.fixture
def default_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device=device))) as config:
        yield config
