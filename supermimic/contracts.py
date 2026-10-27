"""Interfaces only. All tensors are batched; implementations come in P1-P4.

Internal quaternion convention is wxyz. Root velocities are world-frame link
velocities, not CoM velocities. Positions exclude the environment origin.
Joint/body ordering must be resolved by names and recorded in checkpoint metadata.
"""
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

# Keep inspection of the scaffold independent of torch/mjlab installation.
Tensor = Any


@dataclass
class PhysicsState:
    root_pos: Tensor             # [B, 3]
    root_quat_wxyz: Tensor       # [B, 4]
    root_lin_vel_w: Tensor       # [B, 3]
    root_ang_vel_w: Tensor       # [B, 3]
    joint_pos: Tensor            # [B, J]
    joint_vel: Tensor            # [B, J]


@dataclass
class TaskState:
    previous_action: Tensor
    observation_history: Tensor # [B, history, observation_dim], oldest first
    command: Tensor
    heading_target: Tensor
    command_time_left: Tensor
    recovery_enabled: Tensor
    recovery_counter: Tensor
    episode_step: Tensor
    extras: Mapping[str, Tensor] # Task-specific reference/contact/timer state


@dataclass
class Transition:
    state: PhysicsState
    next_state: PhysicsState    # MUST be captured before reset
    task_state: TaskState
    next_task_state: TaskState
    context: Tensor             # Actual applied randomization, schema in metadata
    action: Tensor              # Executed normalized action after clipping/noise
    reward: Tensor
    terminated: Tensor         # True terminal: no value bootstrap
    truncated: Tensor          # Timeout: bootstrap, but stop sequence sampling
    episode_id: Tensor
    dynamics_valid: Tensor     # False for unmodelled external state jumps
    labels: Mapping[str, Tensor] # Contact and other supervised targets


class SimulatorAdapter(Protocol):
    def reset(self, env_ids: Tensor | None = None) -> None: ...
    def read_state(self) -> PhysicsState: ...
    def read_task_state(self) -> TaskState: ...
    def read_context(self) -> Tensor: ...
    def step(self, action: Tensor) -> Transition:
        """Snapshot transition before resetting; reset ownership is in adapter."""
        ...


class DynamicsModel(Protocol):
    def step(self, state: PhysicsState, action: Tensor, context: Tensor
             ) -> tuple[PhysicsState, Mapping[str, Tensor]]: ...


class TaskHeads(Protocol):
    def observation(self, state: PhysicsState, task: TaskState,
                    context: Tensor) -> Tensor: ...
    def advance(self, state: PhysicsState, next_state: PhysicsState,
                action: Tensor, task: TaskState,
                auxiliary: Mapping[str, Tensor]) -> TaskState: ...
    def reward(self, state: PhysicsState, next_state: PhysicsState,
               action: Tensor, task: TaskState, next_task: TaskState,
               auxiliary: Mapping[str, Tensor]) -> Mapping[str, Tensor]: ...
    def continuation(self, state: PhysicsState, task: TaskState) -> Tensor:
        """Differentiable survival probability; exact hard rules used in evaluation."""
        ...
    def amp_features(self, state: PhysicsState, task: TaskState) -> Tensor: ...
