"""Public LM-head aliases must select one greedy-and-sampling policy."""

import importlib
import json
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "vllm_mach.mxfp6.serve",
        "vllm_mach.mxfp8.serve",
        "vllm_mach.mxfp8.two_b.serve",
    ],
)
@pytest.mark.parametrize(
    "flags,enabled",
    [
        ([], True),
        (["--hybrid-head"], True),
        (["--nvfp4-lm-head"], True),
        (["--lm-head-sampling"], True),
        (["--no-hybrid-head"], False),
        (["--no-nvfp4-lm-head"], False),
        (["--no-lm-head-sampling"], False),
        (["--nvfp4-lm-head", "--no-lm-head-sampling"], False),
        (["--no-hybrid-head", "--lm-head-sampling"], True),
    ],
)
def test_cli_aliases_control_the_same_switch(
    monkeypatch, capsys, module_name, flags, enabled
):
    module = importlib.import_module(module_name)
    if module_name == "vllm_mach.mxfp6.serve":
        monkeypatch.setattr(
            sys, "argv", ["serve", "--model", "/models/model", "--dry-run", *flags]
        )
        module.main()
        env = json.loads(capsys.readouterr().out)["environment"]
        assert env["VLLM_HYBRID_NVFP4_LM_HEAD"] == str(int(enabled))
        assert env["VLLM_MACH_LM_HEAD_SAMPLING"] == str(int(enabled))
    else:
        class Configured(Exception):
            pass

        selected = []

        def configure(directory, **options):
            selected.append(options["hybrid_head"])
            raise Configured

        # Stop before checkpoint/runtime validation; parsing still uses the
        # public launcher, and separate profile tests verify environment wiring.
        monkeypatch.setattr(module, "configure_environment", configure)
        monkeypatch.setattr(
            sys, "argv", ["serve", "/models/model", "--run-dir", "/tmp/mach-run", *flags]
        )
        with pytest.raises(Configured):
            module.main()
        assert selected == [enabled]
