"""Check native C++ backend and ONNX without creating DDS communication objects."""
import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from cpp_runtime import prepare
prepare()
import numpy as np
import onnxruntime as ort
import wmrl_transport


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('deploy/reports/environment_cpp_board.json'))
    a=p.parse_args()
    meta=json.loads(a.policy.with_suffix('.json').read_text())
    assert meta['schema_version']==1 and meta['normalization_in_onnx']
    opt=ort.SessionOptions();opt.intra_op_num_threads=1
    session=ort.InferenceSession(str(a.policy),sess_options=opt,providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].name=='observation' and session.get_inputs()[0].shape==[1,384]
    assert session.get_outputs()[0].name=='action' and session.get_outputs()[0].shape==[1,29]
    values={'observation':np.zeros((1,384),np.float32)};ms=[]
    for i in range(110):
        begin=time.perf_counter();action=session.run(['action'],values)[0]
        if i>=10:ms.append((time.perf_counter()-begin)*1000)
        assert np.isfinite(action).all()
    assert np.percentile(ms,99)<20
    libs=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/libddsc' in line})
    assert libs and all(str(Path(p).resolve()).startswith('/usr/local/lib/') for p in libs),libs
    result=dict(status='passed',backend='cpp',python=sys.executable,architecture=platform.machine(),
                module=wmrl_transport.__file__,dds_libraries=libs,
                packages={n:importlib.metadata.version(n) for n in ['numpy','onnxruntime']},
                inference_p99_ms=float(np.percentile(ms,99)),network_used=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
