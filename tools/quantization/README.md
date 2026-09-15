# Teacher-forced diagnostics

`teacher_decode.py` retains the original physical-M32 protocol: replace token
selection only, keep raw logits unchanged, reinstate the last prompt token as
an unscored first output, and pad continuations so all 32 requests stay synchronized.
`chunked_teacher.py` supports the historical 32×128 partial-prefill protocol.

The current runner is [`../fidelity_native_mxfp6.py`](../fidelity_native_mxfp6.py).
Neither hook is installed by the serving launcher. Compatibility checks use
sampler structure and actual request/token/batch behavior.

The obsolete EXL3/rank64 calibration tools remain available in Git history.
