"""Local, bounded worker diagnostics for serving qualification.

The worker extension installs a Unix socket after warmup. Audit wrappers are
present only during explicitly unscored requests and are removed before timing.
No arbitrary callable, pickle, network listener or GPU synchronization is used.
"""

import collections
import functools
import json
import os
import socketserver
import threading
import types
from pathlib import Path


class RouteProbe:
    """Named worker extension; importing this module installs the lifecycle hook."""


def install(worker):
    hybrid_class = None
    if os.environ.get("VLLM_MACH_PROFILE") != "qwen35-4b-block-fp8-v1":
        from vllm_mach.mxfp6.hybrid_nvfp4_lm_head import HybridNvfp4LmHead

        hybrid_class = HybridNvfp4LmHead

    runner = worker.model_runner
    model = getattr(runner.model, "language_model", runner.model)
    head = model.lm_head
    kernel_types = collections.Counter()
    for layer in model.modules():
        method = getattr(layer, "quant_method", None)
        if method is not None:
            kernel_types[type(method).__name__] += 1
        kernel = getattr(layer, "kernel", None)
        if kernel is not None:
            kernel_types[type(kernel).__name__] += 1
    counts = collections.Counter()
    originals = {}

    def snapshot():
        state = getattr(head, "_hybrid_nvfp4_lm_head_state", None)
        if state is None:
            state = getattr(runner, "_mach_sampling_sidecar", None)
        if state is None:
            for name in ("_mach_2b_hybrid_head", "_mach_mxfp8_full_head_state"):
                attached = getattr(worker, name, {})
                state = attached.get("state", attached.get("fp4"))
                if state is not None:
                    break
        manager = runner.cudagraph_manager
        proxy = getattr(runner, "_mach_sampling_proxy", None)
        proxy_state = None if proxy is None else proxy._get_hybrid_lm_head_state(head)
        features = {}
        for name, value in vars(worker).items():
            if (
                name.startswith("_mach")
                and isinstance(value, dict)
                and "stats" in value
            ):
                features[name] = value["stats"]
        return {
            "rank": worker.rank,
            "pid": os.getpid(),
            "audit_active": bool(originals),
            "counts": dict(counts),
            "head_dtype": str(head.weight.dtype),
            "head_shape": list(head.weight.shape),
            "sidecar": None
            if state is None
            else {
                "backend": state.backend,
                "candidates": state.candidates,
                "max_rows": state.max_rows,
                "can_use_failure_counts": dict(state.can_use_failure_counts),
            },
            "sampling": dict(getattr(runner, "_mach_sampling_stats", {})),
            "sampling_can_use_failures": None
            if proxy_state is None
            else dict(proxy_state.can_use_failure_counts),
            "model_kernel_types": dict(kernel_types),
            "profile_head_stats": features,
            "graphs": {
                mode.name: sorted(d.num_tokens for d in descs)
                for mode, descs in manager._capture_descs.items()
            },
        }

    def audit_on():
        if originals:
            raise RuntimeError("Audit already enabled")
        counts.clear()
        originals["sample"] = runner.sample
        if hybrid_class is not None:
            originals.update(
                coarse=hybrid_class.coarse_logits, refine=hybrid_class.refine_logits
            )

        def coarse(state, hidden, *args, **kwargs):
            result = originals["coarse"](state, hidden, *args, **kwargs)
            counts[f"coarse_m{hidden.shape[0]}_k{state.candidates}"] += 1
            return result

        def refine(hidden, weight, ids, *args, **kwargs):
            result = originals["refine"](hidden, weight, ids, *args, **kwargs)
            counts[f"refine_m{hidden.shape[0]}_k{ids.shape[-1]}"] += 1
            return result

        def sample(this, hidden, batch, grammar):
            prefill = bool(batch.is_prefilling_np.any())
            rows = batch.num_reqs
            before = sum(
                v
                for k, v in counts.items()
                if k.startswith("coarse_") and k.endswith("_k128")
            )
            result = originals["sample"](hidden, batch, grammar)
            after = sum(
                v
                for k, v in counts.items()
                if k.startswith("coarse_") and k.endswith("_k128")
            )
            route = "prefill_or_mixed" if prefill else "decode"
            counts[f"{route}_m{rows}"] += 1
            if after > before:
                counts[f"{route}_hybrid_m{rows}"] += 1
            else:
                counts[f"{route}_complete_head_m{rows}"] += 1
            return result

        if hybrid_class is not None:
            hybrid_class.coarse_logits = coarse
            hybrid_class.refine_logits = staticmethod(refine)
        runner.sample = types.MethodType(sample, runner)

    def audit_off():
        if not originals:
            raise RuntimeError("Audit not enabled")
        runner.sample = originals["sample"]
        if hybrid_class is not None:
            hybrid_class.coarse_logits = originals["coarse"]
            hybrid_class.refine_logits = staticmethod(originals["refine"])
        originals.clear()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request = json.loads(self.rfile.readline(4096))
                op = request["op"]
                if op == "audit_on":
                    audit_on()
                elif op == "audit_off":
                    audit_off()
                elif op == "set_sampling":
                    os.environ["VLLM_MACH_LM_HEAD_SAMPLING"] = str(
                        int(bool(request["enabled"]))
                    )
                elif op == "snapshot":
                    pass
                else:
                    raise ValueError("Unknown diagnostic command")
                response = snapshot()
            except (
                KeyError,
                ValueError,
                RuntimeError,
                TypeError,
                AttributeError,
                OSError,
            ) as exc:
                response = {"error": str(exc)}
            self.wfile.write(json.dumps(response).encode() + b"\n")

    directory = Path(os.environ["MACH_BENCH_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = f"/tmp/mach-route-{os.getpid()}.sock"
    server = socketserver.UnixStreamServer(path, Handler)
    worker._mach_bench_route_server = server
    os.chmod(path, 0o600)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    (directory / f"rank-{worker.rank}.json").write_text(
        json.dumps({"socket": path, **snapshot()})
    )


def install_lifecycle():
    from vllm.v1.worker.gpu_worker import Worker

    original = Worker.compile_or_warm_up_model

    @functools.wraps(original)
    def compiled(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        install(self)
        return result

    Worker.compile_or_warm_up_model = compiled


if os.environ.get("MACH_BENCH_PROBE_DIR"):
    install_lifecycle()
