"""The CP plan must retain only FI scratch and rebind live data on every call."""
from math import prod
from types import SimpleNamespace
from unittest.mock import patch

from vllm_mach.mxfp8.two_b import gdn_prefill_plan as plan


class Tensor:
    device = "cuda:0"

    def __init__(self, ptr, shape, dtype="float32", strides=None, values=()):
        self.ptr, self.shape, self.dtype = ptr, tuple(shape), dtype
        self.strides = strides or tuple(prod(shape[i+1:]) for i in range(len(shape)))
        self.values = values

    ndim = property(lambda self: len(self.shape))
    nbytes = property(lambda self: prod(self.shape) * (2 if self.dtype == "bfloat16" else 4))

    def data_ptr(self):
        return self.ptr

    def stride(self):
        return self.strides

    def numel(self):
        return prod(self.shape)

    def as_strided(self, shape, stride):
        return Tensor(self.ptr, shape, self.dtype, stride, self.values)


class Stream:
    def __init__(self, stream):
        self.cuda_stream, self.waited = stream, []

    def wait_stream(self, stream):
        self.waited.append(stream)


class Worker:
    def __init__(self):
        self.stream, self.capturing = Stream(7), False
        cuda = SimpleNamespace(current_stream=lambda device: self.stream,
                               is_current_stream_capturing=lambda: self.capturing)
        api = SimpleNamespace(Tensor=Tensor, cuda=cuda, bfloat16="bfloat16", float32="float32")
        self.buffers = {("gdn_cp_sm120_"+name, Tensor.device): Tensor((i+1)*1024,(128,))
                        for i, name in enumerate(plan.BUFFERS)}
        self.outputs, self.observed = [], []
        self.bad_abi, self.stock_count = False, 0
        with patch.object(plan, "_recordable", return_value=(self.trace, self.observed)):
            self.adapter = plan._Adapter(self.stock, self.buffers, Tensor.device, api)

    def launch(self, *args):
        self.outputs.append(tuple((v.ptr, v.shape, v.stride(), v.values)
                                  if isinstance(v, Tensor) else v for v in args))

    def trace(self, *args, **kw):
        sources = args[:8] + (kw["initial_state"],)
        owners = tuple(self.buffers.values())
        for arity, binding, scratch in zip(plan.ARITY, plan.BINDINGS, plan.WORKSPACE):
            values = [0] * arity
            for i, role in binding.items():
                tensor = sources[role]
                values[i] = tensor.as_strided((tensor.numel(),),(1,)) if role in (5,6) else tensor
            for i, role in scratch.items():
                values[i] = owners[role]
            values[-1] = self.stream.cuda_stream
            if self.bad_abi and arity==10:
                values[4] = Tensor(1234,(1,))
            self.launch(*values)
            self.observed.append((self.launch, tuple(values)))

    def stock(self,*args,**kw):
        self.stock_count += 1

    def call(self, base, offsets, **kw):
        t = offsets[-1]
        def f(i, shape, dtype):
            return Tensor(base+256*i,shape,dtype)
        q = [f(i,(t,16,128),"bfloat16") for i in range(4)]
        states = [f(i,(2,16,128,128),"float32") for i in (4,5)]
        gates = [f(i,(t,16),"float32") for i in (6,7)]
        cu = Tensor(base+2048,(3,),"int32",values=offsets)
        self.adapter.dispatch(q[0],states[0],q[1],q[2],q[3],*gates,cu,0.088,
                              initial_state=states[1],max_seqlen=t,**kw)


def test_offsets_views_and_workspace_replacement():
    w=Worker()
    w.call(16000,(0,5,100))
    w.call(32000,(0,80,100))
    assert w.adapter.counts=={"primed":1,"replays":1}
    assert w.outputs[4][3][-1]==(0,80,100)
    assert w.outputs[4][3][0]==32000+2048
    assert w.outputs[4][1][:3]==(32000+256*7,(1600,),(1,))
    assert w.outputs[7][6][0]==32000+256*4 and w.outputs[7][8][0]==32000+256*5
    name, old=next(iter(w.buffers.items()))
    new = Tensor(8192,(256,))
    w.buffers[name] = new
    w.call(48000,(0,29,100))
    assert w.adapter.counts=={"primed":2,"replays":1}
    assert w.outputs[8][2][0]==new.ptr!=old.ptr
    assert w.adapter.peak_bytes==sum(w.nbytes for w in w.buffers.values())


def test_fallback_once_for_unknown_abi_and_call_options():
    w=Worker()
    w.call(16000,(0,5,100),unexpected=True)
    w.capturing=True
    w.call(16000,(0,5,100))
    assert w.stock_count==2 and not w.outputs
    w.capturing=False
    w.bad_abi=True
    w.call(16000,(0,5,100))
    assert w.adapter.reason=="unrecognized compiled ABI"
    assert len(w.outputs)==4 and w.stock_count==2
    w.call(32000,(0,80,100))
    assert w.stock_count==3 and len(w.outputs)==4


def test_other_stream_retires_plan_and_uses_stock():
    w=Worker()
    w.call(16000,(0,5,100))
    first=w.stream
    w.stream=Stream(8)
    w.call(32000,(0,80,100))
    assert w.stream.waited==[first] and w.stock_count==1 and not w.adapter.plan
    w.stream=first
    w.call(32000,(0,80,100))
    assert w.stock_count==2 and len(w.outputs)==4


def test_changed_flashinfer_source_keeps_original_callable():
    import sys
    from vllm_mach.stack import BASE_PACKAGES

    def cp_delta_rule_dsl_sm120(*args, **kwargs):
        return args, kwargs

    fi = SimpleNamespace(cp_delta_rule_dsl_sm120=cp_delta_rule_dsl_sm120, __file__=__file__)
    utils = SimpleNamespace(__file__=__file__)
    worker = SimpleNamespace(device=SimpleNamespace(index=0),
                             _mach_2b_contract={"production_throughput_profile": True, "ssm_dtype": "float32"})
    modules = {"torch": SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda dev: (12,0))),
               "flashinfer": SimpleNamespace(gdn_prefill=fi,utils=utils)}
    with patch.dict(BASE_PACKAGES, {"vllm": "0.34.0"}), patch.dict(sys.modules, modules), \
            patch.object(plan, "_adapter", None), patch.object(plan, "_reason", ""):
        assert not plan.install(worker)
        assert plan.receipt()=={"installed":False,"reason":"different FlashInfer or GPU"}
        assert fi.cp_delta_rule_dsl_sm120 is cp_delta_rule_dsl_sm120
