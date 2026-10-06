"""CPU protocol, streaming and aggregation checks; no service or GPU required."""
import argparse
import ast
import asyncio
import contextlib
import io
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

FILE=Path(__file__).resolve().parents[1]/'src/vllm_mach/mxfp8/benchmark.py'
def load():
    spec=importlib.util.spec_from_file_location('mach_benchmark_cpu',FILE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def successful(index=0,*,ttft=.1,tpot=.01,latency=10.09):
    return {'request_index':index,'success':True,'ttft_s':ttft,'tpot_s':tpot,
            'latency_s':latency,'itl_s':[.01,.02],
            'usage':{'prompt_tokens':3000,'completion_tokens':1000,'total_tokens':4000}}

class BenchmarkContracts(unittest.TestCase):
    def setUp(self):self.bench=load()

    def test_request_generation_and_three_phase_parameters(self):
        for c in self.bench.POINTS:
            args=self.bench.phase_args('http://example.test','profile',c,'screen')
            contract,prompts=self.bench.make_contract(args)
            self.assertEqual((args.num_prompts,args.output_tokens,args.warmup_requests,args.warmup_output_tokens),
                             (self.bench.REQUEST_COUNTS[c],1000,c,128))
            self.assertEqual(contract['arrival_offsets_s'],[0.0]*args.num_prompts)
            self.assertEqual(contract['sampling'],{'temperature':0.0,'top_k':-1,'top_p':1.0,
                'presence_penalty':0.0,'ignore_eos':True,'per_request_seed_base':2026092400})
            rng=random.Random(20260924)
            expected=[[rng.randrange(1000,240000) for _ in range(3000)] for _ in range(args.num_prompts)]
            self.assertEqual(prompts,expected)
            warm=self.bench.phase_args('http://example.test','profile',c,'prewarm')
            self.assertEqual((warm.num_prompts,warm.output_tokens,warm.warmup_requests,
                              warm.contract_seed,warm.request_seed_base,warm.warmup_output_tokens),
                             (c,256,0,20261003,2026100300,32))
        self.assertEqual(sum(self.bench.REQUEST_COUNTS.values()),1040)

    def test_original_generator_sse_and_percentile_source_are_unchanged(self):
        expected={'percentile':'2ec557aabf3ad76cdf9917d68a75796cfc00e637ae87d36df31e4f638f5f614f',
                  'make_contract':'0b608d3b181ecd69e815359946312dffbfd4441b82d1a225ab09fffc185bb7d1',
                  'request_one':'731d2adb9528eacebf6104bd7ede466155cdb845db68abb0e161c4eccf9291ef'}
        tree=ast.parse(FILE.read_text())
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in expected:
                digest=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
                self.assertEqual(digest,expected[node.name])

    def test_successful_only_aggregation_and_original_quantile_formula(self):
        failed={'success':False,'usage':{'prompt_tokens':99999,'completion_tokens':99999}}
        agg=self.bench.aggregate_requests([successful(ttft=.1,tpot=.01),successful(1,ttft=.3,tpot=.03),failed],4)
        self.assertEqual((agg['completed'],agg['requested'],agg['prompt_tokens'],agg['completion_tokens']),(2,3,6000,2000))
        self.assertEqual(agg['output_throughput_tokens_per_s'],500)
        self.assertEqual(agg['total_throughput_tokens_per_s'],2000)
        self.assertEqual(agg['mean_ttft_ms'],200)
        self.assertAlmostEqual(agg['p99_ttft_ms'],298)
        self.assertAlmostEqual(agg['p99_tpot_ms'],29.8)
        self.assertEqual(self.bench.percentile([],0.99),None)
        self.assertEqual(self.bench.aggregate_requests([failed],1)['mean_ttft_ms'],None)

    def test_only_all_six_successful_points_are_complete_and_pool_durations(self):
        points=[]
        for c in self.bench.POINTS:
            requests=[successful(i) for i in range(self.bench.REQUEST_COUNTS[c])]
            points.append({'concurrency':c,'success':True,'screen':{
                'requests':requests,'aggregate':self.bench.aggregate_requests(requests,c)}})
        full=self.bench.summarize_points(points,self.bench.POINTS)
        self.assertTrue(full['complete'])
        self.assertEqual(full['aggregate']['duration_s'],sum(self.bench.POINTS))
        self.assertEqual(full['aggregate']['completion_tokens'],1040000)
        self.assertAlmostEqual(full['aggregate']['output_throughput_tokens_per_s'],1040000/sum(self.bench.POINTS))
        smoke=self.bench.summarize_points(points[:1],(4,))
        self.assertFalse(smoke['complete']);self.assertEqual(smoke['selection'],'smoke_subset')
        points[2]['success']=False
        self.assertFalse(self.bench.summarize_points(points,self.bench.POINTS)['complete'])

    def test_sse_split_utf8_payload_and_original_admission_timing(self):
        events=[{'choices':[{'text':'A'}]},{'choices':[{'text':'δ'}]},
                {'choices':[],'usage':{'prompt_tokens':3,'completion_tokens':3}}]
        data=''.join('data: '+json.dumps(event,ensure_ascii=False)+'\n\n' for event in events).encode()
        split=data.index('δ'.encode())+1
        class Content:
            async def iter_any(self):
                yield data[:split];yield data[split:]
        class Response:
            status=200;content=Content()
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
        payload=[]
        class Session:
            def post(self,url,json):payload.append((url,json));return Response()
        with patch.object(self.bench.time,'time',return_value=1000),patch.object(
                self.bench.time,'perf_counter',side_effect=[10,11,12,13,16]):
            row=asyncio.run(self.bench.request_one(Session(),asyncio.Semaphore(1),
                'http://example.test/v1/completions','profile',[1,2,3],3,2026092400,0))
        self.assertTrue(row['success']);self.assertEqual(row['response_text'],'Aδ')
        self.assertEqual((row['queue_wait_s'],row['ttft_s'],row['tpot_s'],row['latency_s']),(1,1,2,5))
        self.assertEqual(row['itl_s'],[1])
        self.assertEqual(payload[0][1],{'model':'profile','prompt':[1,2,3],'max_tokens':3,
            'temperature':0.0,'top_k':-1,'top_p':1.0,'presence_penalty':0.0,
            'seed':2026092400,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}})

    def test_internal_warmup_is_audited_and_excluded_from_duration(self):
        events=[]
        class Session:
            def __init__(self,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
        aiohttp=types.ModuleType('aiohttp');aiohttp.ClientSession=Session
        aiohttp.ClientTimeout=lambda **kwargs:None;aiohttp.TCPConnector=lambda **kwargs:None
        async def request(session,semaphore,url,model,prompt,output,seed,index,**kwargs):
            events.append((output,seed,index))
            row=successful(index);row['usage']['completion_tokens']=output;return row
        args=self.bench.phase_args('http://example.test','profile',4,'screen')
        with patch.dict(sys.modules,{'aiohttp':aiohttp}),patch.object(self.bench,'request_one',request),patch.object(
                self.bench.time,'perf_counter',side_effect=[10,15,20]),patch.object(self.bench.time,'time',return_value=1000):
            result=asyncio.run(self.bench.run_phase(args))
        self.assertEqual(events[:4],[(128,2026092400+100000+i,i) for i in range(4)])
        self.assertEqual(events[4:],[(1000,2026092400+i,i) for i in range(40)])
        self.assertEqual(result['aggregate']['duration_s'],5)
        self.assertEqual(result['aggregate']['completion_tokens'],40000)
        self.assertEqual(len(result['warmup']['results']),4)
        self.assertTrue(all(row['excluded_from_scoring'] for row in result['warmup']['results']))

    def test_failed_artifacts_json_and_canonical_point_validation(self):
        safe=self.bench.json_safe({'ttft_s':math.nan,'nested':[math.inf]})
        self.assertEqual(safe,{'ttft_s':None,'nested':[None]})
        with tempfile.TemporaryDirectory() as directory:
            argv=['--base-url','http://example.test','--model','profile','--outdir',directory+'/new']
            self.assertEqual(self.bench.parse_args(argv+['--points','4','16']).points,[4,16])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.bench.parse_args(argv+['--points','16','4'])
        self.assertEqual(self.bench.canonical_hash({'b':2,'a':1}),self.bench.canonical_hash({'a':1,'b':2}))

if __name__=='__main__':unittest.main()
