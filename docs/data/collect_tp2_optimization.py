"""Collect TP2 decode and freshly scored M4/M32 fidelity for one stage."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'tools'))
from summarize_tp2_decode import collect
from collect_native_comparison import bootstrap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--fidelity-prefix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--control',type=Path)
    p.add_argument('--control-profile',type=Path)
    p.add_argument('--control-label',default='Matched fusion-off control')
    a=p.parse_args()
    if bool(a.control) != bool(a.control_profile):
        p.error('--control and --control-profile must be supplied together')
    data=dict(schema='mach-tp2-optimization/v1',label=a.label,decode=collect(a.baseline,a.profile),fidelity={})
    if a.control:
        control=collect(a.control,a.control_profile)
        for key in ('config','extension_library_sha256','devices','input_tokens','output_tokens','repeats'):
            assert control['contract'][key] == data['decode']['contract'][key], key
        data['matched_control']=dict(label=a.control_label,decode=control)
    for m,filename in [(4,'gdn-m4-fidelity.json'),(32,'native-fidelity.json')]:
        reference=json.loads((HERE/filename).read_text())
        root=Path(str(a.fidelity_prefix)+f'-m{m}')
        assert json.loads((root/'COMPLETE.json').read_text())['target_tokens']==10479
        records=json.loads((root/'records.json').read_text())
        assert [r['id'] for r in records]==[r['id'] for r in reference['queries']]
        errors=[]
        for r,q in zip(records,reference['queries'],strict=True):
            assert len(r['gold_logprobs'])==len(q['bf16_logprobs'])
            errors.append(float(np.abs(np.array(r['gold_logprobs'])-q['bf16_logprobs']).mean()))
        data['fidelity'][str(m)]=dict(mae=float(np.mean(errors)),ci95=bootstrap(errors),query_mae=errors,
            matches_previous_default=([r['gold_logprobs'] for r in records]==reference['runs']['gdn']['gold_logprobs']),
            repeat=json.loads((root/'repeat.json').read_text()),
            records_sha256=hashlib.sha256((root/'records.json').read_bytes()).hexdigest(),
            contract=json.loads((root/'contract.json').read_text()))
    a.output.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    main()
