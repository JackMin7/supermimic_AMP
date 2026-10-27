"""C++ backend tests; domain 71/lo only, no real robot control."""
import argparse
import json
import struct
import time
from pathlib import Path
from cpp_runtime import prepare
prepare()
import numpy as np
from wmrl_transport import TestPeer
from deploy_g1_cpp import Controller,Stop


def wait(f,seconds=3):
    end=time.monotonic()+seconds
    while not f():
        if time.monotonic()>end:raise TimeoutError('Native DDS test timeout')
        time.sleep(.0005)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',required=True);p.add_argument('--trace',required=True)
    p.add_argument('--output',default='deploy/reports/cpp_validation_board.json')
    a=p.parse_args();tr=np.load(a.trace)
    np.testing.assert_allclose([0.],[0.])  # Warm lazy assertion imports before watchdog starts.
    peer=TestPeer()
    def frame(k=0,buttons=0,valid=True,axes=None):
        r=bytearray(40);struct.pack_into('<H',r,2,buttons)
        if axes is None:
            x,y,z=tr['command'];axes=(-y,-z/1.57,x/3)
        for off,v in zip((4,8,20),axes):struct.pack_into('<f',r,off,float(v))
        peer.state(tr['q'][k].tolist(),tr['dq'][k].tolist(),tr['quat'][k].tolist(),tr['gyro'][k].tolist(),bytes(r),valid)
    def connect():
        c=Controller(a.policy,net='lo',domain=71,velocity_scale=(3.,1.,1.57))
        end=time.monotonic()+3
        while not c.bridge.snapshot()['have']:
            frame();time.sleep(.01)
            if time.monotonic()>end:raise TimeoutError('Discovery failed')
        return c
    c=connect();observations=[];actions=[];targets=[];commands=[];step_ms=[]
    try:
        c.reset()
        for k in range(len(tr['action'])):
            if k%250==0:print('Replay',k,flush=True)
            previous=c.bridge.snapshot()['tick'];frame(k)
            wait(lambda:c.bridge.snapshot()['tick']!=previous)
            old=peer.last()['count'];t=time.perf_counter();c.step();step_ms.append((time.perf_counter()-t)*1000)
            wait(lambda:peer.last()['count']>old)
            cmd=peer.last();assert cmd['crc_valid'] and cmd['mode_pr']==0 and cmd['mode_machine']==5
            np.testing.assert_allclose(cmd['kp'],c.kp,rtol=1e-6);np.testing.assert_allclose(cmd['kd'],c.kd,rtol=1e-6)
            observations.append(c.last_observation[0].copy());actions.append(c.a_prev.copy());targets.append(cmd['q']);commands.append(cmd)
            c.a_prev=tr['action'][k].copy()
        np.testing.assert_allclose(observations,tr['observation'],atol=3e-5,rtol=1e-5)
        np.testing.assert_allclose(actions,tr['action'],atol=2e-4,rtol=1e-4)
        np.testing.assert_allclose(targets,tr['target'],atol=1e-4,rtol=1e-4)
        joystick=[]
        for axes,expected in [((0,0,0),(0,0,0)),((0,0,1),(3,0,0)),((0,0,-1),(-1.5,0,0)),((1,0,0),(0,-1,0)),((0,1,0),(0,0,-1.57)),((.005,.005,.005),(0,0,0))]:
            previous=c.bridge.snapshot()['tick'];frame(axes=axes);wait(lambda:c.bridge.snapshot()['tick']!=previous)
            c.step();np.testing.assert_allclose(c.command,expected,atol=1e-6);joystick.append(list(expected))
        # Reject bad CRC without replacing the last good state.
        before=c.bridge.snapshot();frame(valid=False);wait(lambda:c.bridge.snapshot()['invalid_crc']>before['invalid_crc'])
        assert c.bridge.snapshot()['tick']==before['tick']
        frame();peer.start();time.sleep(.02)
        c.reset();start=time.perf_counter();deadline=start;timings=[];intervals=[];previous=start
        n0=c.bridge.snapshot()['received']
        while time.perf_counter()-start<10:
            t=time.perf_counter();intervals.append((t-previous)*1000);previous=t
            c.step();timings.append((time.perf_counter()-t)*1000)
            deadline+=.02;time.sleep(max(0.,deadline-time.perf_counter()))
        rate=(c.bridge.snapshot()['received']-n0)/(time.perf_counter()-start)
        # Keep native input streaming while Python deliberately issues no commands.
        time.sleep(.16)
        assert c.bridge.snapshot()['stopped']
        assert 'command stale' in c.bridge.snapshot()['reason']
        wait(lambda:all(x==0 for x in peer.last()['kp']))
        assert all(x>0 for x in peer.last()['kd'])
        try:c.step()
        except Stop:pass
        else:raise AssertionError('Watchdog did not latch')
        peer.stop();c.close()
        checks=['bad_crc_rejected','python_stall_native_damping']
        # New controllers reset latches; existing robot controller is never touched.
        for label,button in [('B',9),('SELECT',3),('state_timeout',None),('frozen_tick',None)]:
            c=connect();c.step()
            if button is not None:frame(buttons=1<<button);time.sleep(.02);frame()
            elif label=='frozen_tick':
                begin=time.monotonic()
                while time.monotonic()-begin<.13:peer.repeat();time.sleep(.002)
                assert c.bridge.snapshot()['duplicates']>0
            else:time.sleep(.13)
            wait(lambda:c.bridge.snapshot()['stopped'])
            wait(lambda:all(x==0 for x in peer.last()['kp']))
            assert all(x>0 for x in peer.last()['kd'])
            checks.append(label);c.close()
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        result=dict(status='passed',domain=71,interface='lo',steps=len(actions),
                    observation_abs_max=float(np.max(np.abs(np.asarray(observations)-tr['observation']))),
                    action_abs_max=float(np.max(np.abs(np.asarray(actions)-tr['action']))),
                    target_abs_max=float(np.max(np.abs(np.asarray(targets)-tr['target']))),
                    stream_hz=rate,stream_control_p99_ms=float(np.percentile(timings,99)),
                    stream_interval_max_ms=max(intervals[1:]),checks=checks,joystick_cases=joystick,hardware_tested=False)
        assert result['stream_hz']>950
        assert result['stream_control_p99_ms']<20
        out.write_text(json.dumps(result,indent=2)+'\n')
        out.with_name('cpp_commands_for_crc.json').write_text(json.dumps(commands))
        print(json.dumps(result,indent=2))
    finally:
        c.close();peer.close()


if __name__=='__main__':main()
