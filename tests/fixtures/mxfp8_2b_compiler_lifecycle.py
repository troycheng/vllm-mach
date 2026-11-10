"""Actual pinned product constructor/direct/precompile CPU lifecycle."""
import ast,copy,hashlib,json,pickle,sys,types,os
from pathlib import Path
R=Path(__file__).resolve().parents[2];H=Path(sys.argv[1]);H.mkdir(parents=True,exist_ok=False);sys.path[:0]=[str(R/'src'),str(R/'tools'),str(R/'tools/quantization')]
os.environ['VLLM_MACH_PROFILE']='qwen35-2b-mxfp8-champion-v1'
os.environ['VLLM_MACH_MXFP8_SSM_DTYPE']='float32'
os.environ['VLLM_MACH_MXFP8_MODE']='quality';os.environ['VLLM_MACH_MXFP8_QUALITY_ROWS']='32'
import torch,triton
assert str(torch.__version__)=='2.14.0+cu130'and triton.__version__=='3.8.0'
from vllm_mach.mxfp8.two_b.serve import configure_environment
configure_environment(H/'product-cpu-config',quality_rows=32,ssm_dtype='float32',attention_backend='TRITON_ATTN',hybrid_head=True)
from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu=lambda:None
def forbidden(*a,**kw):raise AssertionError('CUDA initialization attempted')
torch.cuda._lazy_init=forbidden
from torch._inductor.codecache import PyCodeCache
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codegen.triton import TritonScheduling
from vllm_mach.mxfp8.compile_choices import canonical_function_sha256 as canonical
raw=(R/'tests/fixtures/mxfp8_2b_ordered_direct.py').read_text();baseline=PyCodeCache.load(raw);original=baseline.triton_
origcfg=[dict(kwargs=dict(x.kwargs),num_warps=x.num_warps,num_stages=x.num_stages,num_ctas=x.num_ctas)for x in original.configs]
async_sources=[]
def gpu_boundary(self,name,source,*a,**kw):async_sources.append(source);return getattr(PyCodeCache.load(source),name)
AsyncCompile.triton=gpu_boundary
from vllm_mach.mxfp8.two_b import profile,ordered_reduction as h
assert profile.install_worker_hook()is True
from vllm_mach.gdn_prep import verify_binding
verify_binding()
class IR:
 def __init__(self,source):self.source=source
 def generate_kernel_code_from_nodes(self,nodes,benchmark_kernel=False):assert nodes==[]and benchmark_kernel;return self.source
 def benchmark_codegened_module(self,module,*a,**kw):return module
def direct(source):return TritonScheduling.benchmark_fused_nodes(IR(source),[])
module=direct(raw);obj=module.triton_;assert obj is not original and h._TEMPORARY and not h._OBJECTS
assert h.rewrite(h.rewrite(raw,'triton_'),'triton_')==h.rewrite(raw,'triton_')
try:PyCodeCache.load(raw)
except RuntimeError:pass
else:raise AssertionError('Unclassified temporary constructor accepted')
keep=[obj]
for case in json.loads((R/'tests/fixtures/mxfp8_2b_ordered_sources.json').read_text()):
 result=AsyncCompile().triton(case['name'],case['source']);result=getattr(result,'_original',result);keep.append(result)
 assert canonical(result.fn.src)==case['after']
for bad in [raw.replace('1e-06','2e-06'),raw.replace("'enable_fp_fusion': True","'enable_fp_fusion': False")]:
 try:direct(bad)
 except (RuntimeError,AssertionError):pass
 else:raise AssertionError('Changed direct math/flags accepted')
# Actual pinned Torch precompile finalization; only GPU compile/load are stubbed.
def compile_boundary(self,cfg):return types.SimpleNamespace(config=cfg,kernel=types.SimpleNamespace(hash='CPU_NO_BINARY'),compile_meta=self._create_compile_meta(cfg))
for obj in keep:
 obj._precompile_config=types.MethodType(compile_boundary,obj);saved=copy.deepcopy(obj._mach_2b_norm_constructor_configs);obj.precompile(warm_cache_only=True);assert obj.configs is None and obj.compile_results and obj._mach_2b_norm_constructor_configs==saved
 obj.launchers=[types.SimpleNamespace(config=x.config)for x in obj.compile_results]
 body=canonical(obj.fn.src);expected=next(v['selected']for v in h.policy()['targets'].values()if v['after_sha256']==body)
 chosen=next(x for x in obj.launchers if h._config(x.config)==expected);obj.launchers=[chosen];obj._prune_compile_results_to_launcher(chosen)
receipt=pickle.loads(pickle.dumps(h.receipt))(require_complete=True);json.dumps(receipt);assert len({x['body_sha256']for x in receipt['actual']})==4 and receipt['temporary_benchmark_constructors']>=1
assert all(x['current_configs']is None and x['actual_launcher_configs'] and x['constructor_configs']==x['actual_launcher_configs'] for x in receipt['actual']);assert not torch.cuda.is_initialized()
# A cold guard must reject an unqualified actual winner and preserve enough
# source/config/launcher evidence to diagnose it without another model load.
obj=keep[1];good=obj.launchers
obj.launchers=[types.SimpleNamespace(config=triton.Config({'XBLOCK':999},num_warps=8))]
try:
 try:h.receipt(require_complete=True)
 except RuntimeError:pass
 else:raise AssertionError('Unknown actual selected config accepted')
finally:obj.launchers=good
diagnostic=Path(os.environ['VLLM_MACH_MXFP8_RUN_DIR'])/('ordered-norm-diagnostic-'+str(os.getpid())+'.json')
snapshot=json.loads(diagnostic.read_text());assert all(x['size_hints'] and x['compiler_flags']=={'enable_fp_fusion':True,'disable_ftz':False,'launch_pdl':False,'native_matmul':False} for x in snapshot['actual']);assert snapshot['actual'] and any(x['source']and x['constructor_configs']and x['launchers'][0]['kwargs']=={'XBLOCK':999}for x in snapshot['actual'])

p=H/'PRODUCT_LIFECYCLE_CPU.json';p.write_text(json.dumps({'passed':True,'actual_pinned_Torch':True,'direct_benchmark_and_async_source_paths':True,'temporary_not_model_coverage':True,'actual_precompile_configs_None':True,'serialized_receipt':receipt,'GPU_initialized':False},indent=2)+'\n');print(p)
