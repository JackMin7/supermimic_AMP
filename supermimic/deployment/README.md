# Standalone ONNX MuJoCo sim2sim

Export once in the training environment (`conda activate mjlab`):

```bash
python scripts/export_sim2sim.py \
  --checkpoint-file logs/rsl_rl/g1_supermimic_locomotion/2026-10-27_11-13-57_wmrl_v2/model_14000.pt \
  --output-dir deploy/wmrl
```

Output bundle (keep these three files together):

- `policy.onnx`: deterministic actor including observation normalization; float32 `[1,384]` input `observation`, `[1,29]` output `action`.
- `policy.json`: joint order, action scaling, default pose, Unitree motor parameters, physics options and export validation metadata.
- `scene.zip`: compiled-input MJCF and mesh assets, including ground and configured actuators/collisions.

Run the single standalone script:

```bash
python scripts/sim2sim.py Unitree-G1-AMP-Flat \
  --policy deploy/wmrl/policy.onnx \
  --viewer native --command 1.5 0 0 --duration 60
```

For headless evaluation, replace `--viewer native` with
`--viewer none --output /tmp/sim2sim.json`.
`--command VX VY WZ` is forward/lateral velocity in m/s and yaw rate in rad/s.
Native mode needs a graphical desktop and follows the robot at real time.
Headless runs as fast as possible. Only one robot is supported.

`sim2sim.py` can be copied outside the repository with the bundle. Runtime requires
only `mujoco`, `numpy`, and `onnxruntime`, with a MuJoCo version supporting
`MjSpec.from_zip` (use the version recorded in policy.json for reproducibility).
It imports no PyTorch, mjlab, project packages or training environment. No `.pt`
checkpoint is read at runtime. Export requires the training dependencies plus
`onnx` and `onnxruntime`, and checks ONNX against PyTorch on 100 seeded inputs.

The simulation runs 5 ms substeps and 50 Hz policy control. Four 96-dimensional
observation frames are ordered oldest to newest, with first-frame replication.
Each frame contains pelvis gyro, projected gravity, command, relative joint
positions, joint velocities and previous raw action. Unitree torque-speed and
friction calculations are applied every substep before implicit position actuation.

Initialization is nominal standing pose. No motion resets, automatic resets,
randomization or pushes are applied. Successful playback alone does not establish
parity with training: the tested checkpoint's 0.5 m/s command tracked poorly in
the previous PyTorch playback, while 1.5 m/s produced walking. Recovery validation
and matched-state mjlab comparisons remain separate work.

```bash
python -m unittest supermimic.tests.test_sim2sim -v
```

These tests export a temporary ONNX fixture, load the portable scene, and compare
observation history and high-speed motor control to training implementations.
Real model ONNX rollout results are in `supermimic/reports/sim2sim_onnx.json`.
