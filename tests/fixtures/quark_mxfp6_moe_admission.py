# Source-extracted native Quark admission fixture; no runtime dependency.
from __future__ import annotations

class QuarkOCP_MX_MoEMethod(QuarkMoEMethod):
    supported_activation_quant_keys = [
        kMxfp4Dynamic,
        kMxfp6E3M2Dynamic,
        kMxfp6E2M3Dynamic,
        kFp8DynamicTensorSym,
        kFp8StaticTensorSym,
        None,
    ]

    supported_weight_quant_keys = [
        kMxfp4Static,
        kMxfp6E3M2Static,
        kMxfp6E2M3Static,
    ]

    def __init__(
        self,
        moe: FusedMoEConfig,
        weight_quant_key: QuantKey,
        activation_quant_key: QuantKey | None,
    ):
        # The native W6A8 checkpoint uses per-group E8M0 activation scales.
        # 0.33's stock OCP class rejects that key before reaching our backend.
        # Admit only the existing native E3M2/MXFP8 pair, on its supported device;
        # keep class-wide stock admission and all other combinations unchanged.
        native_mxfp8 = (
            weight_quant_key == kMxfp6E3M2Static
            and activation_quant_key == kMxfp8Dynamic
            and is_mxfp6_sm120_moe_available()
        )
        if native_mxfp8:
            self.supported_activation_quant_keys = [
                *self.supported_activation_quant_keys, kMxfp8Dynamic
            ]
        super().__init__(moe, weight_quant_key, activation_quant_key)
        self.weight_dtype = next(
            dtype
            for dtype, quant_key in _WEIGHT_QUANT_KEY_MAP.items()
            if quant_key == weight_quant_key
        )
        if native_mxfp8 or activation_quant_key in {
            kFp8DynamicTensorSym, kFp8StaticTensorSym
        }:
            # This is the preexisting native scheme label, not a conversion of
            # the QuantKey: activation_quant_key remains per-group kMxfp8Dynamic.
            self.input_dtype: str | None = "fp8"
        elif activation_quant_key is None:
            self.input_dtype = None
        else:
            self.input_dtype = next(
                dtype
                for dtype, quant_key in _ACTIVATION_QUANT_KEY_MAP.items()
                if quant_key == activation_quant_key
            )

        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(
            self.input_dtype, self.weight_dtype
        )

        if self.ocp_mx_scheme is None:
            raise ValueError(
                f"Unsupported OCP MX dtype combination for MoE: "
                f"input_dtype={self.input_dtype}, weight_dtype={self.weight_dtype}. "
                f"Supported MoE weight dtypes are mxfp4, mxfp6_e3m2, and mxfp6_e2m3."
            )

        # TODO(bowenbao): refactor and introduce backends for other OCP MX schemes,
        # use kernel abstraction for all OCP MX MOE implementations.
        self.mxfp4_backend: Mxfp4MoeBackend = Mxfp4MoeBackend.NONE
        self.experts_cls: type[mk.FusedMoEExperts] | None = None
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs (W4A8, TRITON backends)
        self.w13_precision_config = None
        self.w2_precision_config = None

        self.static_input_scales = activation_quant_key == kFp8StaticTensorSym

        self.use_mxfp6_sm120_moe = (
            self.ocp_mx_scheme == "w_mxfp6_e3m2_a_fp8"
            and not self.static_input_scales
            and is_mxfp6_sm120_moe_available()
        )
        if self.use_mxfp6_sm120_moe:
            from vllm_mach.mxfp6.moe import Mxfp6Sm120Experts

            self.experts_cls = Mxfp6Sm120Experts
            self.model_type = getattr(
                get_current_vllm_config().model_config.hf_config,
                "model_type",
                None,
            )
            logger.info_once(
                "Using native mxfp6-sm120 grouped MoE backend for %s",
                self.ocp_mx_scheme,
            )
            return

        # Select backend based on OCP MX scheme
        if self.ocp_mx_scheme == "w_mxfp4":
            # W4A16: weight-only MXFP4
            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)
        elif self.ocp_mx_scheme == "w_mxfp4_a_fp8" and self.static_input_scales:
            # W4A8: MXFP4 weights + static FP8 activations
            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(
                moe, activation_key=kFp8StaticTensorSym
            )
        elif self.ocp_mx_scheme == "w_mxfp4_a_mxfp4":
            # W4A4: MXFP4 weights + MXFP4 activations
            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(
                moe, activation_key=kMxfp4Dynamic
            )

        # Validation for unsupported schemes
        if any(
            self.ocp_mx_scheme.endswith(a_scheme)
            for a_scheme in ["a_mxfp4", "a_mxfp6_e3m2", "a_mxfp6_e2m3"]
        ):
            if self.static_input_scales:
                raise NotImplementedError(
                    "QuarkOCP_MX_MoEMethod with static input scales is currently "
                    f"not implemented for OCP MX scheme {self.ocp_mx_scheme}. "
                    "Please open an issue."
                )
        elif self.ocp_mx_scheme.endswith("a_fp8") and not self.static_input_scales:
            raise NotImplementedError(
                "QuarkOCP_MX_MoEMethod with dynamic input scales is currently "
                f"not implemented for OCP MX scheme {self.ocp_mx_scheme}. "
                "Please open an issue."
            )

        self.model_type = getattr(
            get_current_vllm_config().model_config.hf_config, "model_type", None
        )

        # If no native backend available, use emulation.
        if self.mxfp4_backend is Mxfp4MoeBackend.NONE:
            self.mxfp4_backend = Mxfp4MoeBackend.EMULATION
            self.experts_cls = None

        if self.experts_cls is None:
            self.experts_cls = backend_to_kernel_cls(self.mxfp4_backend)[0]

        logger.info_once(
            f"Using {self.mxfp4_backend.value} backend for {self.ocp_mx_scheme}"
        )
