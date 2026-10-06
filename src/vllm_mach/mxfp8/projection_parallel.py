# SPDX-License-Identifier: Apache-2.0
"""Closed QKVZ/BA fork/join in whitelisted lowered Runner.call modules.

No CUDA work at import. Complete profiles initialize after projection model
preparation, then verify capture after vLLM compilation. Arithmetic stays in
the two original opaque operations; ordinal zero stays serial.
"""
import collections
import functools
import hashlib
import tempfile
from pathlib import Path

_AOT_INSTALLED = False
_AOT_DIRECTORY = None
_READY = False
_AUX = None
_OWNER = None
_PAIRS = {}
_COUNTS = collections.Counter()
_MODULES = {}
ACTIVE_LAYERS = frozenset(range(1,24))
REQUIRED_ROWS = (32,64)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dispatch(x, qw, qs, layer_id, bw, private_bw, qfirst):
    import torch
    if not _READY or type(layer_id) is not int or layer_id not in _PAIRS:
        raise RuntimeError('unprepared private QKVZ/BA pair')
    expected = _PAIRS[layer_id]
    observed = (qw.data_ptr(), qs.data_ptr(), bw.data_ptr(), private_bw.data_ptr())
    if observed != expected['pointers'] or any(ptr % 16 for ptr in observed):
        raise RuntimeError('QKVZ/BA pair weight identity or alignment differs')
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != 2560 or not x.is_contiguous():
        raise RuntimeError('QKVZ/BA original input contract changed')
    m = int(x.shape[0])
    capture = torch.cuda.is_current_stream_capturing()
    parallel = m in REQUIRED_ROWS and layer_id in ACTIVE_LAYERS
    _COUNTS[('capture' if capture else 'eager', m, layer_id, parallel)] += 1
    qop = torch.ops.vllm.mach_mx8_dual_qkvz_both.default
    bop = torch.ops.mx8_ba_only.small_n.default
    if parallel:
        origin = torch.cuda.current_stream(x.device)
        _AUX.wait_stream(origin)
        with torch.cuda.stream(_AUX):
            ba = bop(x,bw,private_bw)
        qkvz = qop(x,qw,qs,layer_id)
        origin.wait_stream(_AUX)
    elif qfirst:
        qkvz = qop(x,qw,qs,layer_id)
        ba = bop(x,bw,private_bw)
    else:
        ba = bop(x,bw,private_bw)
        qkvz = qop(x,qw,qs,layer_id)
    return qkvz,ba


def install_aot(cache_directory=None):
    """Load reviewed AOT source rewrites; leave generated kernel text untouched."""
    global _AOT_INSTALLED, _AOT_DIRECTORY
    if _AOT_INSTALLED:
        if cache_directory is not None and Path(cache_directory).resolve() != _AOT_DIRECTORY:
            raise RuntimeError("projection AOT cache directory frozen after installation")
        return False
    from torch._inductor.codecache import PyCodeCache
    from .projection_transform import transform_source
    old = PyCodeCache.load_by_key_path.__func__
    directory = (Path(cache_directory).resolve() if cache_directory is not None
                 else Path(tempfile.mkdtemp(prefix="vllm-mach-aot-")))
    directory.mkdir(parents=True,exist_ok=True)
    _AOT_DIRECTORY = directory

    def load(cls,key,path,linemap=None,attrs=None,*,set_sys_modules=None):
        text = Path(path).read_text()
        if 'mach_mx8_dual_qkvz_both.default' in text and 'class Runner' in text:
            changed,audit = transform_source(text)
            if changed == text:
                if audit.get('pairs') != [] or audit.get('skipped_layer_ids') != [0]:
                    raise RuntimeError('unexpected unchanged target AOT module')
                sid=hashlib.sha256(text.encode()).hexdigest()
                audit.update(parent_key=key,parent_sha256=sid,original_layer0_preserved=True)
                _MODULES[sid]=audit
                return old(cls,key,path,linemap,attrs,set_sys_modules=set_sys_modules)
            sid = hashlib.sha256(changed.encode()).hexdigest()
            destination = directory/('aot_'+sid+'.py')
            if destination.exists():
                if destination.read_text() != changed:
                    raise RuntimeError('projection AOT hash collision')
            else:
                destination.write_text(changed)
            audit.update(parent_key=key,parent_sha256=sha(path),candidate_sha256=sid)
            _MODULES[sid] = audit
            return old(cls,sid,str(destination),None,attrs,set_sys_modules=set_sys_modules)
        return old(cls,key,path,linemap,attrs,set_sys_modules=set_sys_modules)

    PyCodeCache.load_by_key_path = classmethod(load)
    _AOT_INSTALLED = True
    return True


def prepare_model(worker, *, cache_directory=None):
    """After dual+BA preparation and before profiling/capture: prepare all24 pairs."""
    global _READY,_AUX,_OWNER
    import torch
    from . import dual
    if _READY or _OWNER is not None or len(dual._TAGGED_NAMES) != 24:
        raise RuntimeError('prepared native dual/BA predecessors and one worker required')
    modules = dict(worker.model_runner.model.named_modules())
    for name in dual._TAGGED_NAMES:
        q = modules[name]
        b = modules[name.rsplit('.',1)[0]+'.in_proj_ba']
        layer_id = getattr(q,'_mx8_dual_qkvz_both_id',None)
        if type(layer_id) is not int or layer_id in _PAIRS or not 0 <= layer_id < 24:
            raise RuntimeError('combined layer tag changed: '+name)
        private = getattr(b,'_mx8_ba_private_weight',None)
        if (private is None or b.bias is not None or tuple(b.weight.shape) != (64,2560) or
                b.weight.dtype != torch.bfloat16 or tuple(q.weight.shape) != (12288,2560)):
            raise RuntimeError('BA/QKVZ tensor contract changed')
        pointers = (q.weight.data_ptr(),q.weight_scale.data_ptr(),b.weight.data_ptr(),private.data_ptr())
        if any(ptr % 16 for ptr in pointers):
            raise RuntimeError('projection pair weights/scales must be16-byte aligned')
        _PAIRS[layer_id]={'name':name,'pointers':pointers}
    if set(_PAIRS) != set(range(24)):
        raise RuntimeError('missing current24 projection pairs')
    # Initialize the auxiliary stream and original BA cuBLAS workspace.
    _AUX = torch.cuda.Stream(device=worker.device)
    origin = torch.cuda.current_stream(worker.device)
    _AUX.wait_stream(origin)
    with torch.cuda.stream(_AUX):
        for m in REQUIRED_ROWS:
            x = torch.zeros((m,2560),device=worker.device,dtype=torch.bfloat16)
            warmed = torch.nn.functional.linear(x,b.weight)
    origin.wait_stream(_AUX)
    torch.cuda.synchronize(worker.device)
    del x,warmed
    _OWNER = worker
    _READY = True
    install_aot(cache_directory)
    return inspect_worker(worker)


def verify_capture(worker):
    """After compilation: verify both row counts and all23 parallel ordinals."""
    if not _READY or worker is not _OWNER or not _MODULES:
        raise RuntimeError('projection pair AOT lowering did not execute for this worker')
    for m in REQUIRED_ROWS:
        covered={k[2] for k,v in _COUNTS.items()
                 if k[0]=='capture' and k[1]==m and k[3] and v>0}
        if covered != ACTIVE_LAYERS:
            raise RuntimeError(f'incomplete M{m} pair capture: {covered}')
    return inspect_worker(worker)


def inspect_worker(worker=None):
    if worker is not None and _OWNER is not None and worker is not _OWNER:
        raise RuntimeError('projection pair state belongs to another worker')
    return {'ready':_READY,'mode':'parallel','source_sha256':sha(__file__),
            'transform_sha256':sha(Path(__file__).with_name('projection_transform.py')),
            'aot_modules':list(_MODULES.values()),
            'counts':[{'phase':k[0],'m':k[1],'layer_id':k[2],'parallel':k[3],'calls':v}
                      for k,v in sorted(_COUNTS.items())],
            'pairs':{str(k):{'name':v['name'],'pointers':list(v['pointers'])} for k,v in _PAIRS.items()},
            'auxiliary_stream':int(_AUX.cuda_stream) if _AUX is not None else None,
            'required_capture_rows':list(REQUIRED_ROWS),
            'active_layer_ids':sorted(ACTIVE_LAYERS),'preserved_layer_ids':[0],
            'm64_variant':'native_dual_qkvz64_pipereg_cfg2',
            'stream_contract':'Every call forks from and rejoins its origin; graph replay encodes both dependencies.',
            'counts_are_python_construction_only':True}


def install_hook(*, cache_directory=None):
    """Optional composed Worker wrapper, installed only by the complete profile."""
    from vllm.v1.worker.gpu_worker import Worker
    if getattr(Worker,'_mach_mxfp8_projection_parallel',False):
        return False
    old_load,old_compile = Worker.load_model,Worker.compile_or_warm_up_model
    @functools.wraps(old_load)
    def load_model(self,*args,**kwargs):
        result = old_load(self,*args,**kwargs)
        prepare_model(self,cache_directory=cache_directory)
        return result
    @functools.wraps(old_compile)
    def compile_or_warm_up_model(self,*args,**kwargs):
        result = old_compile(self,*args,**kwargs)
        verify_capture(self)
        return result
    Worker.load_model,Worker.compile_or_warm_up_model=load_model,compile_or_warm_up_model
    Worker._mach_mxfp8_projection_parallel=True
    return True
