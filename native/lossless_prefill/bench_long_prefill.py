#!/usr/bin/env python3
"""TP2 prefill boundary probe for a config-frozen mach_lossless_prefill_long build.

This is a local AR+residual+GemmaRMSNorm screen.  It records rank-max CUDA-event
cost only; it neither starts a service nor makes an E2E claim.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, os, sys, traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import torch
import torch.distributed as dist

H, TP, MAX_TOKENS, PATTERN = 5120, 2, 4128, 1
ROTATE_BYTES, WARMUPS, ROUNDS = 320 * 1024 * 1024, 20, 8
CASES = ("finite_rank_distinct", "signed_zero_residual_neg_zero", "both_ranks_negative_zero", "input_span15", "input_span16", "nan_inf")

@dataclass(frozen=True)
class Config:
    ms: tuple[int, ...]
    library: str
    library_sha256: str
    capture_root: str
    capture_selection: dict[int, dict[str, Any]]
    path: Path
    sha256: str

@dataclass
class Fixture:
    name: str; source: str; m: int; x: torch.Tensor; residual: torch.Tensor; gamma: torch.Tensor; eps: float; receipt: dict[str, Any]
@dataclass
class Arm:
    name: str; workspace: Any; stream: torch.cuda.Stream; dispatch: Callable[[Fixture, torch.Tensor, torch.Tensor, torch.Tensor], None]
@dataclass
class Bank:
    graph: torch.cuda.CUDAGraph; x: torch.Tensor; restore: torch.Tensor; norm: torch.Tensor; fixture: str

def sha(p: Path) -> str:
    d=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): d.update(b)
    return d.hexdigest()
def eq(a:torch.Tensor,b:torch.Tensor,label:str)->None:
    a,b=a.view(torch.int16),b.view(torch.int16)
    if not torch.equal(a,b):
        bad=torch.nonzero(a!=b,as_tuple=False)[0].tolist()
        raise AssertionError(f"{label}: int16 BF16 mismatch at {tuple(bad)}; no tolerance path")
def ws_bytes(m:int)->int:return m*H*4+(m*H//256)*4
def oneshot(m:int)->bool:return m<=3276
def root_payload(p:Path)->Path:return p/"payloads" if (p/"payloads").is_dir() else p

def config_path_default()->Path:return Path(__file__).resolve().parent.parent/"candidate_config.json"
def read_config(p:Path)->Config:
    x=json.loads(p.read_text())
    required={"M","library_sha256","library","capture_root","capture_selection","namespace"}
    if not required <= set(x): raise ValueError(f"candidate config missing {sorted(required-set(x))}")
    if x["namespace"]!="mach_lossless_prefill_long": raise ValueError("candidate config namespace differs")
    ms=tuple(sorted({int(m) for m in x["M"]}))
    if not ms or len(ms)!=len(x["M"]) or any(m<=0 or m>MAX_TOKENS for m in ms): raise ValueError(f"invalid M list {x['M']}")
    lib=str(x["library"]); digest=str(x["library_sha256"])
    if Path(lib).name!=lib or not lib.endswith(".so") or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest): raise ValueError("invalid library name/SHA")
    capture=str(x["capture_root"])
    if not capture.startswith("/"): raise ValueError("capture_root must be an absolute mounted-source path")
    raw=x["capture_selection"]
    if not isinstance(raw,dict): raise ValueError("capture_selection must map target M strings to explicit fixture specifications")
    selection={}
    for key,item in raw.items():
        try:m=int(key)
        except Exception as exc:raise ValueError(f"invalid capture-selection M key {key!r}") from exc
        if m not in ms or not isinstance(item,dict):raise ValueError(f"invalid capture selection at M{m}")
        kind=item.get("kind")
        if kind=="actual":
            if set(item)!={"kind","stage"} or int(item["stage"]) not in (4,16,24,32):raise ValueError(f"actual M{m} needs only kind/stage with a valid score stage")
            selection[m]={"kind":"actual","stage":int(item["stage"])}
        elif kind=="prefix_proxy":
            schema=item.get("source_schema", "d1")
            if schema=="d1":
                if set(item)!={"kind","source_m","source_stage"}:raise ValueError(f"D1 prefix proxy M{m} needs kind/source_m/source_stage only")
                source_m=int(item["source_m"]);source_stage=int(item["source_stage"])
                if source_m not in ms or m>=source_m or source_stage not in (4,16,24,32):raise ValueError(f"invalid D1 prefix proxy M{m}: {item}")
                selection[m]={"kind":"prefix_proxy","source_schema":"d1","source_m":source_m,"source_stage":source_stage}
            elif schema=="legacy_long":
                if set(item)!={"kind","source_schema","source_root","source_m","source_stage"}:raise ValueError(f"legacy_long prefix proxy M{m} needs kind/source_schema/source_root/source_m/source_stage only")
                source_root=str(item["source_root"]);source_m=int(item["source_m"]);source_stage=int(item["source_stage"])
                # The sole approved legacy parent is call0 M3890.  Stage 32 is
                # provenance only: that capture predates D1's per-stage schema.
                if not source_root.startswith("/") or source_m!=3890 or source_stage!=32 or m>=source_m:raise ValueError(f"invalid legacy_long prefix proxy M{m}: {item}")
                selection[m]={"kind":"prefix_proxy","source_schema":"legacy_long","source_root":source_root,"source_m":source_m,"source_stage":source_stage}
            else:raise ValueError(f"prefix proxy M{m} has invalid source_schema {schema!r}")
        else:raise ValueError(f"capture selection M{m} has invalid kind {kind!r}")
    return Config(ms,lib,digest,capture,selection,p.resolve(),sha(p))

def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-config",type=Path,default=config_path_default())
    p.add_argument("--candidate-library",required=True,type=Path)
    p.add_argument("--capture-dir",type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--synthetic-only",action="store_true",help="validate six synthetic inputs per configured M only")
    p.add_argument("--validate-only",action="store_true")
    p.add_argument("--pdl",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--timeout-seconds",type=int,default=900)
    p.add_argument("--timing-rounds",type=int,default=ROUNDS)
    return p.parse_args()
def init(a):
    local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local)
    dist.init_process_group("gloo",init_method="env://",timeout=dt.timedelta(seconds=a.timeout_seconds))
    if dist.get_world_size()!=TP or local not in (0,1):raise RuntimeError("requires torchrun --nproc_per_node=2 on CUDA 0,1")
    return dist.get_rank(),torch.device(f"cuda:{local}")
def new_out(p:Path,r:int):
    err=None
    if r==0:
        try:p.mkdir(parents=True,exist_ok=False)
        except Exception as e:err=str(e)
    x=[err];dist.broadcast_object_list(x,0)
    if x[0]:raise RuntimeError(f"output-dir must be new: {x[0]}")
    dist.barrier()

def tensor_from(meta:dict[str,Any],root:Path,name:str,shape:tuple[int,...],dev:torch.device)->tuple[torch.Tensor,dict[str,str]]:
    item=meta.get("files",{}).get(name,{})
    fp=root/str(item.get("file", ""))
    if not fp.is_file() or sha(fp)!=item.get("sha256"):raise ValueError(f"capture hash mismatch {fp}")
    t=torch.load(fp,map_location="cpu",weights_only=True)
    if not isinstance(t,torch.Tensor) or t.dtype!=torch.bfloat16 or not t.is_contiguous() or tuple(t.shape)!=shape:raise ValueError(f"bad BF16 contiguous tensor {fp}: expected {shape}")
    return t.to(dev),{"file":fp.name,"sha256":str(item["sha256"])}
def load_capture(root:Path,rank:int,m:int,stage:int,dev:torch.device)->Fixture:
    # D1 sample selection is explicit.  Never infer a real fixture from a
    # filename prefix or from a larger M tensor.
    mp=root/f"rank{rank}_c{stage}_m{m}_sample0.json"
    if not mp.is_file():raise FileNotFoundError(f"selected D1 capture missing: {mp}")
    meta=json.loads(mp.read_text())
    identity=(int(meta.get("rank",-1)),int(meta.get("stage",-1)),int(meta.get("selected_m",-1)),int(meta.get("sample_index",-1)))
    if identity!=(rank,stage,m,0):raise ValueError(f"bad D1 identity {mp}: {identity}")
    if tuple(meta.get("input_shape",()))!=(m,H) or tuple(meta.get("residual_shape",()))!=(m,H) or tuple(meta.get("gamma_shape",()))!=(H,):raise ValueError(f"bad D1 companion shape {mp}")
    ts={};files={}
    for name,shape in (("input",(m,H)),("residual",(m,H)),("gamma",(H,))):ts[name],files[name]=tensor_from(meta,root,name,shape,dev)
    eps=float(meta.get("eps"));
    if not math.isfinite(eps) or eps<=0:raise ValueError(f"bad eps in {mp}")
    return Fixture(f"capture_c{stage}_m{m}_sample0","d1_actual_capture",m,ts["input"],ts["residual"],ts["gamma"],eps,{"kind":"d1_actual_capture","source_schema":"d1","rank":rank,"stage":stage,"m":m,"sample":0,"metadata_file":mp.name,"metadata_sha256":sha(mp),"files":files,"instrumented_diagnostic_not_e2e":bool(meta.get("instrumented_diagnostic_not_e2e")),"selection_rule":meta.get("selection_rule")})
def load_legacy_long_capture(root:Path,rank:int,m:int,stage:int,dev:torch.device)->Fixture:
    # The legacy source has no D1 c/stage filename.  It is permitted only as
    # M3890/call0 parent provenance for an explicit smaller-M prefix proxy.
    if (m,stage)!=(3890,32):raise ValueError(f"unsupported legacy_long parent M/stage: {(m,stage)}")
    root=root_payload(root);mp=root/f"rank{rank}_large_call0.json"
    if not mp.is_file():raise FileNotFoundError(f"selected legacy_long parent missing: {mp}")
    meta=json.loads(mp.read_text())
    identity=(int(meta.get("rank",-1)),int(meta.get("selected_m",-1)),int(meta.get("call",-1)))
    if identity!=(rank,m,0):raise ValueError(f"bad legacy_long identity {mp}: {identity}")
    if meta.get("dtype")!="torch.bfloat16" or tuple(meta.get("shape",()))!=(m,H) or tuple(meta.get("residual_shape",()))!=(m,H) or tuple(meta.get("gamma_shape",()))!=(H,):raise ValueError(f"bad legacy_long shape/dtype {mp}")
    ts={};files={}
    for name,shape in (("input",(m,H)),("residual",(m,H)),("gamma",(H,))):ts[name],files[name]=tensor_from(meta,root,name,shape,dev)
    eps=float(meta.get("eps"));
    if not math.isfinite(eps) or eps<=0:raise ValueError(f"bad eps in {mp}")
    return Fixture(f"legacy_long_m{m}_call0","legacy_long_capture_parent",m,ts["input"],ts["residual"],ts["gamma"],eps,{"kind":"legacy_long_capture_parent","source_schema":"legacy_long","rank":rank,"stage":stage,"m":m,"call":0,"metadata_file":mp.name,"metadata_sha256":sha(mp),"files":files,"legacy_version_input_proxy_parent":True,"not_a_d1_actual_capture":True,"group":meta.get("group"),"norm_instance":meta.get("norm_instance")})
def derived_fixture(parent:Fixture,m:int)->Fixture:
    if m>=parent.m: raise ValueError(f"cannot derive M{m} from M{parent.m}")
    # Explicit config-only BF16 [0:M) proxy.  It has no claim of being a
    # matched service input for its target M and is never relabelled as actual.
    x=parent.x[:m].contiguous();residual=parent.residual[:m].contiguous();schema=parent.receipt["source_schema"]
    return Fixture(f"derived_prefix_m{m}_from_{schema}_s{parent.receipt['stage']}_m{parent.m}","explicit_contiguous_prefix_proxy",m,x,residual,parent.gamma,parent.eps,{"kind":"explicit_contiguous_prefix_proxy","source_schema":schema,"m":m,"source_m":parent.m,"source_stage":parent.receipt["stage"],"rank":parent.receipt["rank"],"source_metadata_file":parent.receipt["metadata_file"],"source_metadata_sha256":parent.receipt["metadata_sha256"],"input_rows":"[0:M)","residual_rows":"[0:M)","gamma":"same BF16 tensor","legacy_version_input_proxy":schema=="legacy_long","not_a_real_matched_service_input":True})
def captures(path:Path,rank:int,dev:torch.device,cfg:Config)->tuple[dict[int,Fixture],dict[int,Fixture]]:
    root=root_payload(path)
    if not cfg.capture_selection:raise ValueError("non-synthetic run needs nonempty explicit capture_selection")
    source_cache={}
    def source(schema:str,m:int,stage:int,source_root:Path|None=None)->Fixture:
        key=(schema,str(source_root) if source_root is not None else "",m,stage)
        if key not in source_cache:
            source_cache[key]=load_capture(root,rank,m,stage,dev) if schema=="d1" else load_legacy_long_capture(source_root,rank,m,stage,dev)
        return source_cache[key]
    actual={};derived={}
    for m,spec in sorted(cfg.capture_selection.items()):
        if spec["kind"]=="actual":actual[m]=source("d1",m,spec["stage"])
        elif spec["source_schema"]=="d1":derived[m]=derived_fixture(source("d1",spec["source_m"],spec["source_stage"]),m)
        else:derived[m]=derived_fixture(source("legacy_long",spec["source_m"],spec["source_stage"],Path(spec["source_root"])),m)
    local={"selection":{str(m):cfg.capture_selection[m] for m in sorted(cfg.capture_selection)},"actual_ms":sorted(actual),"proxy_ms":sorted(derived),"fixture_shapes":{str(m):{"shape":list(f.x.shape),"residual_shape":list(f.residual.shape),"gamma_shape":list(f.gamma.shape),"dtype":str(f.x.dtype),"source_kind":f.receipt["kind"]} for m,f in {**actual,**derived}.items()}}
    allr=[None,None];dist.all_gather_object(allr,local)
    if allr[0]!=allr[1]:raise RuntimeError(f"ranks have incompatible explicit capture selections: {allr}")
    return actual,derived

def synthetic(m:int,rank:int,dev:torch.device,case:str)->Fixture:
    gen=torch.Generator(device=dev).manual_seed(20260908+m+rank*131)
    if case=="finite_rank_distinct":x=(torch.randn((m,H),device=dev,generator=gen,dtype=torch.float32)*(.004+rank*.001)).bfloat16()
    elif case=="signed_zero_residual_neg_zero":x=torch.full((m,H),-0.0 if rank==0 else 0.0,device=dev,dtype=torch.bfloat16)
    elif case=="both_ranks_negative_zero":x=torch.full((m,H),-0.0,device=dev,dtype=torch.bfloat16)
    elif case in ("input_span15","input_span16"):
        span=15 if case.endswith("15") else 16;ex=torch.arange(128,129+span,device=dev,dtype=torch.int16)
        bits=((ex.repeat((256+len(ex)-1)//len(ex))[:256]<<7)|(torch.arange(256,device=dev,dtype=torch.int16)&127)).to(torch.int16)
        x=bits.view(torch.bfloat16).repeat(m*H//256).reshape(m,H).clone()
        if rank:x=(x.view(torch.int16)|torch.tensor(-32768,device=dev,dtype=torch.int16)).view(torch.bfloat16)
    elif case=="nan_inf":
        bits=torch.full((m,H),0x3f80,device=dev,dtype=torch.int16);flat=bits.view(-1);flat[0::257]=0x7f80;flat[1::257]=0x7fc1;flat[2::257]=-128;x=bits.view(torch.bfloat16)
    else:raise AssertionError(case)
    residual=(torch.randn((m,H),device=dev,generator=gen,dtype=torch.float32)*.01).bfloat16()
    if case in ("signed_zero_residual_neg_zero","both_ranks_negative_zero"):residual.fill_(-0.)
    gamma=(1+torch.randn((H,),device=dev,generator=gen,dtype=torch.float32)*.01).bfloat16()
    return Fixture(case,"synthetic",m,x.contiguous(),residual.contiguous(),gamma.contiguous(),1e-6,{"kind":"synthetic","case":case,"m":m,"rank_distinct":True,"residual_contains_negative_zero":case in ("signed_zero_residual_neg_zero","both_ranks_negative_zero"),"both_input_ranks_negative_zero":case=="both_ranks_negative_zero"})

def make_workspace(rank:int,max_m:int):
    import flashinfer.comm
    w=flashinfer.comm.create_allreduce_fusion_workspace(backend="trtllm",world_size=TP,rank=rank,max_token_num=MAX_TOKENS,hidden_dim=H,dtype=torch.bfloat16,group=dist.group.WORLD)
    md=getattr(w,"metadata",{})
    if getattr(w,"backend",None)!="trtllm" or not hasattr(w,"workspace_tensor") or int(md.get("buffer_size",0))<ws_bytes(max_m):raise RuntimeError(f"bad workspace {md}")
    if int(md.get("max_token_num",MAX_TOKENS))<MAX_TOKENS or int(md.get("hidden_dim",H))!=H:raise RuntimeError(f"wrong workspace metadata {md}")
    return w
def make_arms(rank:int,pdl:bool,max_m:int)->dict[str,Arm]:
    import flashinfer.comm
    w={n:make_workspace(rank,max_m) for n in ("reference","candidate")};s={n:torch.cuda.Stream() for n in w}
    def ref(f:Fixture,x,o,n):flashinfer.comm.allreduce_fusion(x,workspace=w["reference"],pattern=PATTERN,launch_with_pdl=pdl,trigger_completion_at_end=True,output=None,residual_out=o,norm_out=n,residual_in=f.residual,rms_gamma=f.gamma,rms_eps=f.eps,use_oneshot=oneshot(f.m),fp32_acc=True,weight_bias=1.)
    def cand(f:Fixture,x,o,n):torch.ops.mach_lossless_prefill_long.run(x,f.residual,f.gamma,w["candidate"].workspace_tensor,o,n,rank,int(w["candidate"].metadata["buffer_size"]),f.eps,1.,pdl)
    return {"reference":Arm("reference",w["reference"],s["reference"],ref),"candidate":Arm("candidate",w["candidate"],s["candidate"],cand)}
def eager(a:Arm,f:Fixture,alias:bool)->tuple[torch.Tensor,torch.Tensor]:
    x=f.x.clone();o=x if alias else torch.empty_like(x);n=torch.empty_like(x);a.stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(a.stream):a.dispatch(f,x,o,n)
    a.stream.synchronize();return o,n
def validate(arms:dict[str,Arm],f:Fixture)->dict[str,Any]:
    dist.barrier();ref=eager(arms["reference"],f,False);dist.barrier();got=eager(arms["candidate"],f,False);eq(got[0],ref[0],f"{f.name} residual");eq(got[1],ref[1],f"{f.name} norm")
    dist.barrier();ref=eager(arms["reference"],f,True);dist.barrier();got=eager(arms["candidate"],f,True);eq(got[0],ref[0],f"{f.name} alias residual");eq(got[1],ref[1],f"{f.name} alias norm")
    return {"fixture":f.name,"source":f.source,"m":f.m,"alias_input_residual_out":True,"residual_int16_equal":True,"norm_int16_equal":True}
def graph(a:Arm,f:Fixture)->Bank:
    x=f.x.clone();restore=f.x.clone();n=torch.empty_like(x);a.stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(a.stream):a.dispatch(f,x,x,n)
    a.stream.synchronize();x.copy_(restore);a.stream.wait_stream(torch.cuda.current_stream(x.device));g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=a.stream):a.dispatch(f,x,x,n)
    return Bank(g,x,restore,n,f.name)
def changed_graph(arms:dict[str,Arm],first:Fixture,changed:Fixture)->dict[str,Any]:
    banks={}
    for arm in arms.values():dist.barrier();banks[arm.name]=graph(arm,first)
    for name,b in banks.items():b.x.copy_(changed.x);arms[name].stream.wait_stream(torch.cuda.current_stream(b.x.device))
    for name,b in banks.items():
        with torch.cuda.stream(arms[name].stream):b.graph.replay()
        arms[name].stream.synchronize()
    eq(banks["candidate"].x,banks["reference"].x,f"changed graph residual M{first.m}");eq(banks["candidate"].norm,banks["reference"].norm,f"changed graph norm M{first.m}")
    return {"m":first.m,"changed_from":first.name,"changed_to":changed.name,"changed_input_graph":True,"residual_int16_equal":True,"norm_int16_equal":True}
def shared_workspace_sequence(rank:int,dev:torch.device,pdl:bool,one:Fixture,two:Fixture,arms:dict[str,Arm],max_m:int)->dict[str,Any]:
    """No host sync/Gloo barrier occurs between the five PDL enqueues."""
    import flashinfer.comm
    w=make_workspace(rank,max_m);stream=torch.cuda.Stream();m32=synthetic(32,rank,dev,"finite_rank_distinct")
    pending=[]
    for label,fixture,candidate in (("reference_M32_before",m32,False),("candidate_oneshot",one,True),("reference_twoshot",two,False),("candidate_twoshot",two,True),("reference_M32_after",m32,False)):
        pending.append((label,fixture,candidate,fixture.x.clone(),torch.empty_like(fixture.x)))
    stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream):
        for _,fixture,candidate,x,norm in pending:
            if candidate:torch.ops.mach_lossless_prefill_long.run(x,fixture.residual,fixture.gamma,w.workspace_tensor,x,norm,rank,int(w.metadata["buffer_size"]),fixture.eps,1.,pdl)
            else:flashinfer.comm.allreduce_fusion(x,workspace=w,pattern=PATTERN,launch_with_pdl=pdl,trigger_completion_at_end=True,output=None,residual_out=x,norm_out=norm,residual_in=fixture.residual,rms_gamma=fixture.gamma,rms_eps=fixture.eps,use_oneshot=oneshot(fixture.m),fp32_acc=True,weight_bias=1.)
    stream.synchronize();dist.barrier()
    for label,fixture,_,x,norm in pending:
        dist.barrier();ref=eager(arms["reference"],fixture,True);eq(x,ref[0],f"shared {label} residual");eq(norm,ref[1],f"shared {label} norm")
    eq(pending[0][3],pending[-1][3],"shared M32 before/after residual");eq(pending[0][4],pending[-1][4],"shared M32 before/after norm")
    d=getattr(w,"destroy",None)
    if d:d()
    return {"sequence":[x[0] for x in pending],"completion_at_end":True,"pdl":pdl,"no_host_sync_or_gloo_between_enqueues":True,"each_result_independent_reference_int16_equal":True,"m32_before_after_int16_equal":True,"no_retry":True}
def warmup(a:Arm,f:Fixture):
    x=f.x.clone();n=torch.empty_like(x);a.stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(a.stream):
        for _ in range(WARMUPS):x.copy_(f.x);a.dispatch(f,x,x,n)
    a.stream.synchronize();dist.barrier()
def timed(arms:dict[str,Arm],f:Fixture,rounds:int):
    if rounds!=ROUNDS:raise ValueError(f"base schedule is fixed at ABBA x{ROUNDS}")
    nb=math.ceil(ROTATE_BYTES/(f.x.numel()*f.x.element_size()))
    # Each ABBA round touches two cold banks per arm.  Preserve the frozen
    # ABBA×8 order, repeating whole eight-round epochs only when a small-M
    # input needs more than sixteen addresses to reach the 320MiB rotation.
    epochs=math.ceil(nb/(2*ROUNDS));effective_rounds=ROUNDS*epochs
    banks={}
    for arm in arms.values():
        warmup(arm,f);banks[arm.name]=[]
        for _ in range(nb):dist.barrier();banks[arm.name].append(graph(arm,f))
    out=[]
    for rnd in range(effective_rounds):
        for oi,name in enumerate(("reference","candidate","candidate","reference")):
            bank_index=(2*rnd+(0 if oi<2 else 1))%nb
            dist.barrier();arm=arms[name];b=banks[name][bank_index];arm.stream.wait_stream(torch.cuda.current_stream(b.x.device));st,en=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(arm.stream):b.x.copy_(b.restore);st.record();b.graph.replay();en.record()
            en.synchronize();out.append({"round":rnd,"epoch":rnd//ROUNDS,"round_within_epoch":rnd%ROUNDS,"order_index":oi,"arm":name,"bank":bank_index,"fixture":b.fixture,"event_us":float(st.elapsed_time(en))*1000.,"banks_required":nb,"base_abba_rounds":ROUNDS,"abba_epochs":epochs,"effective_rounds":effective_rounds,"rotation_bytes_target":ROTATE_BYTES})
    return out
def driver_version():
    try:
        value=torch.cuda.cudart().cudaDriverGetVersion();return int(value[-1] if isinstance(value,tuple) else value)
    except Exception:return getattr(torch.cuda,"driver_version","unavailable")
def fi_sos()->dict[str,str]:
    ps=set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        z=line.split()
        if z and z[-1].endswith(".so") and "trtllm_comm" in z[-1]:ps.add(Path(z[-1]))
    if not ps:raise RuntimeError("loaded FI trtllm_comm .so not found")
    return {str(p):sha(p) for p in sorted(ps)}
def wreceipt(w):
    md=dict(getattr(w,"metadata",{}));keys=("buffer_size","max_token_num","hidden_dim","tp_size","tp_rank")
    return {"backend":getattr(w,"backend",None),"metadata":{k:md[k] for k in keys if k in md},"requested":{"max_token_num":MAX_TOKENS,"hidden_dim":H,"dtype":"torch.bfloat16"}}

def main():
    a=parse();rank=None;arms={}
    try:
        if not a.synthetic_only and not a.capture_dir:raise ValueError("--capture-dir is required for the one-lifecycle synthetic+capture probe")
        cfg=read_config(a.candidate_config)
        rank,dev=init(a);new_out(a.output_dir,rank)
        if not a.candidate_library.is_file():raise FileNotFoundError(a.candidate_library)
        if a.candidate_library.name!=cfg.library:raise RuntimeError(f"library basename differs from config: {a.candidate_library.name} != {cfg.library}")
        if sha(a.candidate_library)!=cfg.library_sha256:raise RuntimeError("candidate .so SHA differs from candidate_config")
        torch.ops.load_library(str(a.candidate_library))
        if not hasattr(torch.ops,"mach_lossless_prefill_long") or not all(hasattr(torch.ops.mach_lossless_prefill_long,n) for n in ("run","info")):raise RuntimeError("mach_lossless_prefill_long run/info missing")
        caps=None if a.synthetic_only else captures(a.capture_dir,rank,dev,cfg)
        arms=make_arms(rank,a.pdl,max(cfg.ms))
        syn={m:[synthetic(m,rank,dev,k) for k in CASES] for m in cfg.ms}
        syn_validation=[validate(arms,f) for fs in syn.values() for f in fs]
        graph_validation=[changed_graph(arms,fs[0],fs[3]) for fs in syn.values()]
        actual,derived=({}, {}) if caps is None else caps
        selected={**actual,**derived}
        cap_validation=[] if caps is None else [validate(arms,f) for f in selected.values()]
        # A real selected fixture is preferred for each topology; when D1 did
        # not capture that topology, the six-case synthetic leg remains an
        # exact protocol check but is recorded as synthetic, never as service data.
        one=next((selected[m] for m in sorted(selected,reverse=True) if oneshot(m)),None)
        two=next((selected[m] for m in sorted(selected,reverse=True) if not oneshot(m)),None)
        if one is None:one=next((syn[m][0] for m in cfg.ms if oneshot(m)),None)
        if two is None:two=next((syn[m][0] for m in cfg.ms if not oneshot(m)),None)
        if one is None or two is None:raise ValueError("candidate M must include >=1 one-shot and >=1 two-shot shape for mandatory mixed-PDL validation")
        shared=shared_workspace_sequence(rank,dev,a.pdl,one,two,arms,max(cfg.ms))
        shared["mixed_fixture_sources"]={"oneshot":{"fixture":one.name,"source":one.source,"m":one.m},"twoshot":{"fixture":two.name,"source":two.source,"m":two.m},"selected_capture_only":caps is not None,"synthetic_leg_is_protocol_only":one.source=="synthetic" or two.source=="synthetic"}
        timed_fixtures={} if caps is None else selected
        samples={} if (caps is None or a.validate_only) else {str(m):timed(arms,timed_fixtures[m],a.timing_rounds) for m in sorted(timed_fixtures)}
        props=torch.cuda.get_device_properties(dev);info={str(m):[int(v) for v in torch.ops.mach_lossless_prefill_long.info(syn[m][0].x)] for m in cfg.ms}
        rec={"rank":rank,"gpu":{"name":props.name,"uuid":str(getattr(props,"uuid","unavailable")),"capability":[props.major,props.minor],"sm_count":props.multi_processor_count,"driver":driver_version()},"candidate_config":{"path":str(cfg.path),"sha256":cfg.sha256,"M":list(cfg.ms),"capture_root":cfg.capture_root,"capture_selection":{str(m):cfg.capture_selection[m] for m in sorted(cfg.capture_selection)}},"candidate_library":{"path":str(a.candidate_library.resolve()),"sha256":sha(a.candidate_library),"namespace":"mach_lossless_prefill_long"},"loaded_flashinfer_trtllm_comm":fi_sos(),"workspace":{n:wreceipt(z.workspace) for n,z in arms.items()},"synthetic_validation":syn_validation,"changed_input_graph":graph_validation,"capture_validation":cap_validation,"captures":None if caps is None else {"actual_selected":{str(m):actual[m].receipt for m in sorted(actual)},"explicit_contiguous_prefix_proxies":{str(m):derived[m].receipt for m in sorted(derived)},"selection_is_config_explicit":True},"shared_workspace_sequence":shared,"candidate_info":info,"samples":samples}
        gathered=[None,None] if rank==0 else None;dist.gather_object(rec,gathered,dst=0)
        if rank==0:
            rankmax={}
            for m in sorted({int(k) for r in gathered for k in r["samples"]}):
                merged=[]
                for x,y in zip(gathered[0]["samples"].get(str(m),[]),gathered[1]["samples"].get(str(m),[]),strict=True):
                    key=("round","order_index","arm","bank","fixture")
                    if tuple(x[k] for k in key)!=tuple(y[k] for k in key):raise RuntimeError(f"rank schedule mismatch M{m}")
                    merged.append({**{k:x[k] for k in key},"rank_event_us":[x["event_us"],y["event_us"]],"rankmax_event_us":max(x["event_us"],y["event_us"])})
                rankmax[str(m)]=merged
            out={"scope":"local exact AR+residual+GemmaRMSNorm boundary only; no service/GEMM/E2E claim","configuration":{"tp":TP,"dtype":"bfloat16","H":H,"max_token_num":MAX_TOKENS,"pattern":PATTERN,"reference":"installed FI oneshot=(M<=3276), FP32acc, completion-at-end, weight_bias=1","candidate":"mach_lossless_prefill_long.run","alias":"input=residual_out","pdl":a.pdl,"timing":"rank-max CUDA events; restore before event; Gloo excluded","timed_schedule":f"ABBA x{ROUNDS} epochs=ceil(banks_required/{2*ROUNDS}); warmup {WARMUPS}; banks=ceil(320MiB/inputbytes) per explicitly selected actual/proxy M"},"script_sha256":sha(Path(__file__)),"by_rank":gathered,"rankmax_event_us":rankmax}
            (a.output_dir/"summary.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
        dist.barrier()
    except BaseException:
        if rank is not None:
            try:(a.output_dir/f"FAILURE_rank{rank}.json").write_text(json.dumps({"rank":rank,"exception":traceback.format_exc(),"script_sha256":sha(Path(__file__))},indent=2)+"\n")
            except Exception:pass
        raise
    finally:
        for arm in arms.values():
            d=getattr(arm.workspace,"destroy",None)
            if d:
                try:d()
                except Exception:pass
        if dist.is_initialized():dist.destroy_process_group()
if __name__=="__main__":
    try:main()
    except Exception as e:
        print(f"bench_prefill_direct failed: {e}",file=sys.stderr,flush=True);traceback.print_exc();raise
