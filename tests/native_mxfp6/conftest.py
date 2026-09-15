import pytest


@pytest.fixture
def default_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()) as config:
        yield config
