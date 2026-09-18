"""Minimal CUDA compiler A/B for Qwen3.5-27B TP2 owner primitives.

Build each arm in a fresh process (CUDA_HOME must be set before importing Torch):
  CUDA_HOME=/path/to/cuda python bench_toolkit.py build --tag cu130 --out /tmp/owner-ab
Run both arms in the SAME processes, alternating AB/BA, with separate workspaces:
  CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc-per-node=2 \
    bench_toolkit.py run --out /tmp/owner-ab --tags cu132 cu130
No model weights/GEMMs: synthetic BF16 inputs, PDL on, CUDA Graph timings.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

SOURCES = ('owner.cu', 'local.cu', 'ragged.cu', 'ragged_local.cu')


def build(args):
    from importlib.metadata import distribution, version
    from torch.utils.cpp_extension import load, CUDA_HOME
    assert version('flashinfer-python') == '0.6.18'
    source = Path(__file__).resolve().parent
    target = args.out.resolve() / args.tag
    target.mkdir(parents=True, exist_ok=True)
    # Only namespaces change, allowing both compilers' identical kernels to coexist.
    paths = []
    digests = {}
    for path in sorted(source.glob('*')):
        if path.suffix not in ('.cu', '.cuh'):
            continue
        text = path.read_text()
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (target / path.name).write_text(text.replace('mach_owner', f'{args.tag}_mach_owner'))
    root = Path(distribution('flashinfer-python').locate_file('flashinfer/data'))
    flags = ['-O3', '-std=c++17', '-arch=sm_120f', '-use_fast_math', '-DNDEBUG',
             '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
             '-U__CUDA_NO_HALF2_OPERATORS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__']
    flags += ['-DFLASHINFER_ENABLE_' + x for x in
              ('F16', 'BF16', 'FP8_E4M3', 'FP8_E5M2', 'FP8_E8M0', 'FP4_E2M1')]
    for filename in SOURCES:
        name = f'{args.tag}_{Path(filename).stem}'
        folder = target / name
        folder.mkdir(exist_ok=True)
        paths.append(load(name=name, sources=[str(target / filename)],
                          extra_include_paths=[str(root / p) for p in
                                               ('include', 'spdlog/include', 'cutlass/include')],
                          extra_cflags=['-O3', '-std=c++17'], extra_cuda_cflags=flags,
                          build_directory=str(folder), is_python_module=False))
    (target / 'build.json').write_text(json.dumps(dict(
        libraries=paths, cuda_home=CUDA_HOME, flags=flags, source_sha256=digests,
        nvcc=subprocess.check_output([str(Path(CUDA_HOME) / 'bin/nvcc'), '--version'], text=True)
    ), indent=2))


def run(args):
    import torch
    import torch.distributed as dist
    import flashinfer.comm
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=180))
    assert dist.get_world_size() == 2
    config = json.loads(args.config.read_text())['text_config']
    assert (config['hidden_size'], config['intermediate_size'], config['num_hidden_layers']) == (5120, 17408, 64)
    assert config['rms_norm_eps'] == 1e-6
    assert torch.cuda.get_device_properties(rank).multi_processor_count == 170
    metadata, workspaces = {}, {}
    for tag in args.tags:
        metadata[tag] = json.loads((args.out / tag / 'build.json').read_text())
        for lib in metadata[tag]['libraries']:
            torch.ops.load_library(lib)
        workspaces[tag] = flashinfer.comm.create_allreduce_fusion_workspace(
            backend='trtllm', world_size=2, rank=rank, max_token_num=4128,
            hidden_dim=5120, dtype=torch.bfloat16, group=dist.group.WORLD)
    assert metadata[args.tags[0]]['source_sha256'] == metadata[args.tags[1]]['source_sha256']
    results = []
    def equal(a, b):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), 'compiler output mismatch'
    def random(shape):
        return torch.randn(shape, device='cuda', dtype=torch.bfloat16) * 0.1
    for m in args.rows:
        p = (m + 255) // 256 * 128
        n = p if rank == 0 else m - p
        own = slice(rank * p, rank * p + n)
        for operation in ('reduce_norm', 'local_norm', 'gather_mx8', 'gather_ba'):
            torch.manual_seed(20260918 + rank)
            gamma = random((5120,))
            if operation in ('reduce_norm', 'local_norm'):
                rows = m if operation == 'reduce_norm' else n
                x, y, residual = [random((rows, 5120)) for _ in range(3)]
                shapes = ((rows, 5120), (rows, 5120))
                dtype = torch.bfloat16
            else:
                # Match runtime transport packet sizes, including 128-row scale atoms.
                if operation == 'gather_mx8':
                    sizes = (n * 5120, ((n + 127) // 128) * 128 * 160)
                    shapes = ((2 * p * 5120,), (2 * p * 160,))
                else:
                    sizes = (n * 48 * 2, n * 48 * 2)
                    shapes = ((2 * p * 96,), (2 * p * 96,))
                x, y = [torch.full((s,), 31 + rank * 71 + i, device='cuda', dtype=torch.uint8)
                        for i, s in enumerate(sizes)]
                dtype = torch.uint8
            outputs, dispatches, graphs, eager = {}, {}, {}, {}
            shared_out = tuple(torch.zeros(shape, device='cuda', dtype=dtype) for shape in shapes)
            for tag in args.tags:
                ws = workspaces[tag]
                out = shared_out
                outputs[tag] = out
                base = f'{tag}_mach_owner' + ('' if m == 4096 else '_ragged')
                if operation == 'local_norm':
                    op = getattr(torch.ops, base + '_local').ordered_sum_norm
                    call_args = (x, y, residual, gamma, *out, 1e-6, 1.0, True)
                elif operation == 'reduce_norm':
                    op = getattr(torch.ops, base).reduce_owner
                    call_args = (x, residual, gamma, ws.workspace_tensor, *out, rank)
                    if m != 4096:
                        call_args += (p,)
                    call_args += (int(ws.metadata['buffer_size']), 1e-6, 1.0, True)
                else:
                    op = getattr(getattr(torch.ops, base), 'gather_mx8' if m == 4096 else 'gather_mx8_padded')
                    call_args = (x, y, ws.workspace_tensor, *out, rank, int(ws.metadata['buffer_size']), True)
                dispatches[tag] = lambda op=op, a=call_args: op(*a)
            # Eager plus changing-input graph correctness, including both ranks.
            for tag in args.tags:
                dist.barrier()
                for _ in range(5):
                    dispatches[tag]()
                torch.cuda.synchronize()
                eager[tag] = tuple(t.clone() for t in outputs[tag])
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph):
                    for _ in range(args.replays):
                        dispatches[tag]()
                graphs[tag] = graph
            for change in range(3):
                if change:
                    if dtype == torch.bfloat16:
                        x.mul_(-0.75)
                    else:
                        x.bitwise_xor_(17)
                for tag in args.tags:
                    dist.barrier()
                    graphs[tag].replay()
                    torch.cuda.synchronize()
                    outputs[tag] = tuple(t.clone() for t in shared_out)
                if operation.startswith('gather'):
                    for tag in args.tags:
                        for i, actual in enumerate(outputs[tag]):
                            packet = actual.numel() // 2
                            for peer in range(2):
                                peer_n = p if peer == 0 else m - p
                                valid = (peer_n * 5120 if i == 0 else ((peer_n + 127) // 128) * 128 * 160) if operation == 'gather_mx8' else peer_n * 96
                                value = 31 + peer * 71 + i
                                if i == 0 and change % 2:
                                    value ^= 17
                                expected = torch.zeros(packet, device='cuda', dtype=torch.uint8)
                                expected[:valid] = value
                                equal(actual[peer * packet:(peer + 1) * packet], expected)
                if change == 0:
                    for tag in args.tags:
                        for a, b in zip(outputs[tag], eager[tag]):
                            equal(a[own] if operation == 'reduce_norm' else a,
                                  b[own] if operation == 'reduce_norm' else b)
                for a, b in zip(outputs[args.tags[0]], outputs[args.tags[1]]):
                    equal(a[own] if operation == 'reduce_norm' else a,
                          b[own] if operation == 'reduce_norm' else b)
            samples = {tag: [] for tag in args.tags}
            for iteration in range(args.samples):
                for tag in args.tags[::1 if iteration % 2 == 0 else -1]:
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[tag].replay()
                    end.record()
                    end.synchronize()
                    value = torch.tensor(start.elapsed_time(end) * 1000 / args.replays)
                    dist.all_reduce(value, op=dist.ReduceOp.MAX)
                    samples[tag].append(value.item())
            medians = {tag: statistics.median(values) for tag, values in samples.items()}
            record = dict(rows=m, operation=operation, median_us=medians, samples_us=samples,
                          candidate_change_pct=100 * (medians[args.tags[1]] / medians[args.tags[0]] - 1),
                          compiler_bitwise_equal=True)
            results.append(record)
            if rank == 0:
                print(json.dumps({k: v for k, v in record.items() if k != 'samples_us'}), flush=True)
            del graphs, dispatches, outputs, eager
    if rank == 0:
        report = dict(model_config=str(args.config), hidden=5120, tp=2, layers=64,
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      driver=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,driver_version', '--format=csv,noheader'], text=True),
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      methodology='Synthetic inputs; identical input/output addresses between arms; separate workspaces; PDL; CUDA Graph; alternating AB/BA; rank-max events. No GEMM or end-to-end claim.',
                      replays=args.replays, builds=metadata, results=results)
        (args.out / args.result).write_text(json.dumps(report, indent=2))
    dist.barrier()
    for ws in workspaces.values():
        ws.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build', 'run'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag')
    parser.add_argument('--tags', nargs=2, default=['cu132', 'cu130'])
    parser.add_argument('--config', type=Path, default=Path('/data1/models/Qwen3.5-27B-MXFP6/config.json'))
    parser.add_argument('--rows', type=int, nargs='+', default=[512, 2048, 3000, 4096])
    parser.add_argument('--samples', type=int, default=60)
    parser.add_argument('--replays', type=int, default=100)
    parser.add_argument('--result', default='results.json')
    args = parser.parse_args()
    assert args.samples >= 2 and args.replays >= 1
    if args.mode == 'build':
        assert args.tag and args.tag.isidentifier()
        build(args)
    else:
        assert all(512 <= m <= 4096 for m in args.rows)
        run(args)
