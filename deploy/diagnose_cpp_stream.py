"""Read-only real LowState subscription via C++; no LowCmd publisher exists."""
import argparse
import json
import time
from pathlib import Path
from cpp_runtime import prepare
prepare()
import numpy as np
from deploy_g1_cpp import Controller


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',required=True);p.add_argument('--net',default='eth0')
    p.add_argument('--seconds',type=float,default=15);p.add_argument('--output',default='deploy/reports/live_cpp_readonly_board.json')
    a=p.parse_args();c=Controller(a.policy,net=a.net,read_only=True)
    times=[];inferences=[];ages=[];ticks=[]
    try:
        c.wait_connected();start=time.perf_counter();deadline=start;n0=c.bridge.snapshot()['received']
        while time.perf_counter()-start<a.seconds:
            t=time.perf_counter();d=c.bridge.snapshot();times.append(t);ages.append(d['age']*1000);ticks.append(d['tick'])
            c.policy.run(None,{'observation':np.zeros((1,384),np.float32)})
            inferences.append((time.perf_counter()-t)*1000)
            deadline+=.02;time.sleep(max(0.,deadline-time.perf_counter()))
        final=c.bridge.snapshot();elapsed=time.perf_counter()-start
        def summary(v):return dict(median=float(np.median(v)),p99=float(np.percentile(v,99)),maximum=float(np.max(v)))
        result=dict(read_only=True,backend='cpp',receive_hz=(final['received']-n0)/elapsed,invalid_crc=final['invalid_crc'],
                    sent=final['sent'],snapshot_and_inference_ms=summary(inferences),
                    control_interval_ms=summary(np.diff(times)*1000),native_state_age_ms=summary(ages),unique_control_ticks=len(set(ticks)))
        assert final['sent']==0
        path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:c.close()


if __name__=='__main__':main()
