"""C++ SDK must use its matching /usr/local ddscxx + ddsc, not Python DDS 0.10.2."""
import os
from pathlib import Path
import sys


def prepare():
    if not Path('/usr/local/lib/libddscxx.so').exists():
        raise RuntimeError('Board C++ SDK runtime not installed')
    paths=[p for p in os.environ.get('LD_LIBRARY_PATH','').split(':')
           if p and '$' not in p and 'cyclonedds' not in p and p!='/usr/local/lib']
    wanted=':'.join(['/usr/local/lib']+paths)
    if os.environ.get('LD_LIBRARY_PATH')!=wanted or 'CYCLONEDDS_HOME' in os.environ:
        env=os.environ.copy();env['LD_LIBRARY_PATH']=wanted;env.pop('CYCLONEDDS_HOME',None)
        print('[DDS C++] Selecting matched native SDK runtime.',flush=True)
        os.execve(sys.executable,[sys.executable]+sys.argv,env)
