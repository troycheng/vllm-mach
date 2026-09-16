"""Collect matched serving stages; keep the P0 decode workload separate."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',action='append',required=True,help='LABEL=RESULTS/ARM')
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('tp2-serving.json'))
    a=p.parse_args()
    stages=[]
    contracts={}
    for item in a.stage:
        label,path=item.split('=',1);root=Path(path)
        points=[]
        for c in (4,16,24,32):
            source=root/f'c{c}.json';data=json.loads(source.read_text())
            contract=data['contract'];count=5*c
            assert contract['input_tokens']==3000 and contract['output_tokens']==1000
            assert contract['num_prompts']==count and contract['max_concurrency']==c
            if c in contracts:assert contract==contracts[c]
            else:contracts[c]=contract
            stats=data['aggregate']
            assert stats['completed']==stats['requested']==count
            assert stats['prompt_tokens']==count*3000 and stats['completion_tokens']==count*1000
            assert all(r['success'] for r in data['requests'])
            assert data['warmup']==dict(requests=min(32,count),output_tokens=128)
            points.append(dict(concurrency=c,aggregate=stats,raw_path=str(source),
                raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        stages.append(dict(label=label,points=points,launch=json.loads((root/'launch.json').read_text())))
    a.output.write_text(json.dumps(dict(schema='mach-tp2-serving/v1',contracts=contracts,stages=stages),indent=2)+'\n')


if __name__=='__main__':main()
