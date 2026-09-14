"""Explicit local-only diagnostics for a disposable Mach acceptance service."""
import importlib.metadata as metadata
import json


class WorkerExtension:
    def mach_status(self):
        import torch
        from vllm_mach.exl3.rank64 import ATTRIBUTE
        model = self.model_runner.model
        selected = [getattr(m, ATTRIBUTE) for m in model.modules() if hasattr(m, ATTRIBUTE)]
        owner = [getattr(m, "_owner_prefill_state") for m in model.modules() if hasattr(m, "_owner_prefill_state")]
        states = []
        for s in owner:
            states.append({"rank": s.rank, "replica_bytes": s.replica_bytes,
                           "total_forwards": s.total_forwards, "verified_forwards": s.verified_forwards,
                           "checks": s.checks})
        return {"rank": torch.distributed.get_rank(), "versions": {p: metadata.version(p) for p in
                ("vllm", "torch", "exllamav3", "flashinfer-python", "b12x", "mxfp6-sm120", "vllm-mach")},
                "selected_layers": len(selected), "rank64_graph_rows": sorted(set().union(*(s.graph_rows for s in selected))),
                "norm_verified": all(s.norm_verified for s in selected), "owner": states,
                "kv_blocks": self.cache_config.num_gpu_blocks}

    def mach_snapshot(self):
        from rank64_probe import snapshot_worker_inputs
        return snapshot_worker_inputs(self)

    def mach_rank64_probe(self):
        from rank64_probe import validate_worker
        return validate_worker(self, require_real_inputs=True)


class Middleware:
    METHODS = {"/__mach/status": "mach_status", "/__mach/snapshot": "mach_snapshot",
               "/__mach/rank64": "mach_rank64_probe"}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        method = self.METHODS.get(scope.get("path"))
        if scope["type"] != "http" or method is None:
            return await self.app(scope, receive, send)
        if scope.get("client", ("",))[0] not in ("127.0.0.1", "::1") or scope["method"] != "POST":
            status, result = 403, {"error": "local POST only"}
        else:
            result = {"results": await scope["app"].state.engine_client.collective_rpc(
                method=method, timeout=600, args=(), kwargs={})}
            status = 200
        body = json.dumps(result, allow_nan=False).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})
