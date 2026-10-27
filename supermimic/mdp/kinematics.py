"""Differentiable FK from the compiled target model (free root + hinge joints)."""
import mujoco
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul


class RobotKinematics:
    def __init__(self, env):
        robot, model = env.scene['robot'], env.sim.mj_model
        self.device = env.device
        self.body_ids = [int(x) for x in robot.indexing.body_ids]
        local = {body_id: i for i, body_id in enumerate(self.body_ids)}
        self.root = int(robot.indexing.root_body_id)
        self.joint_ids = [int(x) for x in robot.indexing.joint_ids]
        joints = {joint_id: i for i, joint_id in enumerate(self.joint_ids)}
        self.nodes = []
        for bid in self.body_ids:
            if bid == self.root:
                self.nodes.append(None)
                continue
            parent = local[int(model.body_parentid[bid])]
            if parent >= local[bid]:
                raise ValueError('Body order must be topological')
            count, adr = int(model.body_jntnum[bid]), int(model.body_jntadr[bid])
            if count > 1 or (count and model.jnt_type[adr] != mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError('Only fixed or single-hinge child bodies supported')
            tensor = lambda x: torch.tensor(x.copy(), dtype=torch.float32, device=self.device)
            self.nodes.append((parent, tensor(model.body_pos[bid]), tensor(model.body_quat[bid]),
                               None if not count else (joints[adr], tensor(model.jnt_axis[adr]),
                                                       tensor(model.jnt_pos[adr]), float(model.qpos0[model.jnt_qposadr[adr]]))))

    def __call__(self, state):
        positions, rotations, velocities, angular = [], [], [], []
        batch = state.root_pos.shape[0]
        for node in self.nodes:
            if node is None:
                p, r, v, w = state.root_pos, state.root_quat_wxyz, state.root_lin_vel_w, state.root_ang_vel_w
            else:
                parent, offset, base_rotation, joint = node
                pp, pr, pv, pw = positions[parent], rotations[parent], velocities[parent], angular[parent]
                r0 = quat_mul(pr, base_rotation.expand(batch, -1))
                p0 = pp + quat_apply(pr, offset.expand(batch, -1))
                r, p, w = r0, p0, pw
                if joint is not None:
                    j, axis, pivot, reference = joint
                    angle = (state.joint_pos[:, j] - reference) * 0.5
                    delta = torch.cat((angle.cos()[:, None], angle.sin()[:, None] * axis), -1)
                    r = quat_mul(r0, delta)
                    pivot_w = p0 + quat_apply(r0, pivot.expand(batch, -1))
                    p = pivot_w - quat_apply(r, pivot.expand(batch, -1))
                    dw = quat_apply(r0, axis.expand(batch, -1)) * state.joint_vel[:, j:j+1]
                    w = pw + dw
                    v = pv + torch.cross(pw, p - pp, dim=-1) + torch.cross(dw, p - pivot_w, dim=-1)
                else:
                    v = pv + torch.cross(pw, p - pp, dim=-1)
            positions.append(p); rotations.append(r); velocities.append(v); angular.append(w)
        return tuple(torch.stack(xs, dim=1) for xs in (positions, rotations, velocities, angular))
