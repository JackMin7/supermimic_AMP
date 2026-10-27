# Python ONNX + C++ SDK2 communication

Board entrypoint (manual hardware run; not executed during development):

```bash
conda activate robomimic
cd ~/supermimic_amp/current
python deploy/deploy_g1_cpp.py --net eth0 --policy deploy/wmrl/policy.onnx
```

No shell launcher or manual exports required. `cpp_runtime.py` selects the board's
matching `/usr/local/lib/libddscxx.so` and `libddsc.so` before importing the native
module. Python DDS's `/opt/cyclonedds-0.10.2` is NOT mixed into this process.
The environment is changed for this process only. Python ONNX inference is unchanged.

## Reference and scope

The designated reference is
`~/supermimic_amp/RoboMimic_Deploy_bjc/deploy_real/deploy_real.py` (identical to
`~/RoboMimic_Deploy_bjc/deploy_real/deploy_real.py`). That reference imports
**unitree_sdk2py**, not unitree_cpp. Its receive callback stores the latest state
and updates the remote; it does not compute Python CRC on every incoming state.
The original WMRL controller added this per-state check, measured at ~1.87 ms on
board, limiting receive throughput to ~540 Hz for an approximately 1 kHz stream.

This native extension preserves the reference's DDS topics, HG message types,
PR mode, mode_machine feedback, 29 SDK motor indices, pelvis wxyz quaternion,
gyro, and joystick axes (`ly,-lx,-rx`). It moves DDS deserialization, CRC, receive
buffering and send serialization into C++; it does not import RoboJuDo's controller,
call MotionSwitcher, or reproduce another policy's observation dimensions/gains.
WMRL keeps its own 4-frame observation, ONNX normalization and exported parameters.

The reference's FSM has START for POS_RESET and R1+A for LOCO. The existing WMRL
operator workflow remains START -> nominal pose -> A -> policy; SELECT/B stop.
This is explicit, not a claim that every reference FSM detail is identical.

## Native behavior

- Constructor subscribes, optionally creates a publisher, but sends no commands.
- Read-only mode creates no LowCmd publisher at all.
- Receive callback validates CRC, latches B/SELECT, and overwrites the latest state
  without the Python GIL or a Python queue. Duplicate ticks don't refresh age.
- Python sends one desired position command per 20 ms policy period.
- Native watchdog checks every 5 ms. After first command, no fresh state or no new
  Python command for >100 ms latches a stop. It attempts damping for about 1 second
  using exported KD, KP=0, tau=0; it never keeps replaying the previous policy target.
- DDS Write executes outside the state mutex. A separate writer mutex orders policy
  and watchdog sends. Callbacks never need a Python lock.
- Closing a read-only or never-started controller does not initiate motor commands.
- SIGKILL/power/network loss cannot guarantee damping; independent robot protection
  remains required. No automatic motion-controller release or ownership checking.

## Validation commands

Native loopback only, domain 71 / lo:

```bash
python deploy/verify_cpp.py --policy deploy/wmrl/policy.onnx --trace deploy/wmrl/trace.npz
```

Tests: 1000-frame observations/actions/targets, 1 kHz sustained native publishing
with 50 Hz policy commands, bad CRC rejection, joystick axes/deadzone, Python stall,
B/SELECT latch, stale state, frozen ticks, damping. Native LowCmd CRCs were additionally
cross-checked against the independent Python SDK CRC implementation on all 1000 commands.

Read-only real stream, no motor command publisher:

```bash
python deploy/diagnose_cpp_stream.py --net eth0 --policy deploy/wmrl/policy.onnx
```

Board results: replay action error <=9.54e-7, target error <=4.65e-7 rad;
1 kHz loopback control P99 1.53 ms; real stream ~999.9 Hz, snapshot+inference P99
0.75 ms, largest observed 20 ms loop interval 20.23 ms, native state age P99 1.22 ms.
Reports: `deploy/reports/cpp_validation_board.json`, `live_cpp_readonly_board.json`.
These are software/interface and read-only tests. Real-robot ONNX closed-loop behavior
has not been validated; improving throughput does not prove all physical instability fixed.

## Build

Built against the existing board SDK headers/static library and matching native DDS
libraries in /usr/local, with robomimic's Python 3.8 and pybind11. Existing SDK packages
and system libraries are unchanged.

```bash
cmake -S deploy/cpp -B deploy/cpp/build \
  -DPYTHON_EXECUTABLE="$HOME/miniconda3/envs/robomimic/bin/python" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/cpp/build -j2
```

Runtime files: deploy_g1_cpp.py, policy_core.py (shared policy logic), cpp_runtime.py,
wmrl_transport.cpython-38-aarch64-linux-gnu.so, policy.onnx, policy.json. The obsolete pure Python DDS entrypoint has been removed.

## Environment check

```bash
python deploy/check_environment.py --policy deploy/wmrl/policy.onnx
```

This imports the compiled native module and checks ONNX inference, but creates no
DDS participant or publisher. The board's existing robomimic environment is used;
no old Python-DDS environment YAML or shell launcher is required.
Only C++ native source and CMake configuration, the ARM64 extension, shared policy
logic, current entrypoint and relevant verification tools remain in this directory.
Historical JSON reports are retained as diagnostic evidence; their old script paths
may refer to removed implementations. The provided RoboMimic reference tree is unchanged.
