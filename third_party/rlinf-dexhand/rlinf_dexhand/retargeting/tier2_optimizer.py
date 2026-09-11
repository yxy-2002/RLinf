# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

# Tier 2 retargeting optimizer: Xin2025 Eq.6 (hand-only) loss via NLopt LD_SLSQP
# + torch autograd gradient.
#
# Loss = lambda_pos * L_fingertip_pos
#      + lambda_rot * L_fingertip_rot
#      + lambda_pinch * L_pinch
#      + L_joint + L_vel
#
#   L_fingertip_pos : huber(||v_r - v_h||),  v = wrist -> 5 fingertip vectors
#                     weight s_tilde(d) = sigmoid(d, eps1, -w)  (coordinates with pinch)
#   L_fingertip_rot : huber(||r_r - r_h||),  r = DIP -> 5 fingertip vectors
#   L_pinch         : huber(||gamma_r - l(d)*gamma_hat_h||), gamma = thumb -> 4 primary
#                     s(d) = sigmoid(d, eps1, +w); l(d) rescales [eps2,eps1] -> [0,eps1]
#   L_joint         : sum w_pos_j * (q_j - q_bar_j)^2
#   L_vel           : sum w_vel_j * (q_j - q_prev_j)^2
#
# Following Xin2025/dex-retargeting: the pinch switching (s, l) is computed in
# NumPy on the SCALED human vectors and the reference target is detached
# (requires_grad_=False), so within each SLSQP eval the loss is smooth Huber and
# the discrete-ish switching only re-targets between iterations.
#
# Robot model: pinocchio via robot_pinocchio.RobotPinocchio (no mimic; DOA==DOF).

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import nlopt
import numpy as np
import torch
from .robot_pinocchio import RobotPinocchio

Vec3 = np.ndarray
VecN = np.ndarray  # (N, 3)


@dataclass(slots=True, frozen=True)
class Tier2Config:
    """All tunable retargeting parameters (loaded from YAML)."""

    # Robot frame wiring
    wrist_link_name: str
    fingertip_link_names: List[str]   # 5, thumb-first
    dip_link_names: List[str]         # 5, robot DIP frames (Link_X_3)
    target_joint_names: List[str]     # 21
    # loss weights (lambda)
    lambda_fingertip_pos: float
    lambda_fingertip_rot: float
    lambda_pinch: float
    # pinch continuous sigmoid + rescaling (in SCALED = robot space, meters)
    epsilon1: float
    epsilon2: float
    sigmoid_w: float
    huber_delta: float
    scaling_factor: float            # human->robot scalar; resolved (auto calibrated upstream)
    # joint regularization
    joint_pos_ref: np.ndarray        # (21,) q_bar
    joint_pos_weights: np.ndarray    # (21,) w_pos
    joint_vel_weight: float          # w_vel (scalar)
    # solver
    opt_ftol_abs: float
    opt_maxtime: float
    opt_maxeval: int
    low_pass_alpha: float
    # forbid hyperextension (反关节): the wuji URDF lets flexion joints go
    # negative (hyperextend), and the optimizer prefers that solution in the
    # nullspace when pos+ori are both satisfiable. Clamp flexion joints
    # (joint1/3/4, NOT abduction joint2) lower bound to flexion_lower_bound.
    forbid_flexion_hyperextension: bool = True
    flexion_lower_bound: float = 0.0


def _sigmoid(x: np.ndarray, c: float, w: float) -> np.ndarray:
    """sigmoid(x, c, w) = 1 / (1 + exp(w*(x-c)))."""
    return 1.0 / (1.0 + np.exp(w * (x - c)))


def _rescale(d: np.ndarray, eps1: float, eps2: float) -> np.ndarray:
    """l(d): 0 if d<eps2; eps1/(eps1-eps2)*(d-eps2) if eps2<=d<=eps1; d if d>eps1."""
    out = np.zeros_like(d, dtype=np.float64)
    mid = (d >= eps2) & (d <= eps1)
    out[mid] = (eps1 / (eps1 - eps2)) * (d[mid] - eps2)
    high = d > eps1
    out[high] = d[high]
    return out


class Tier2Optimizer:
    """Run the Xin2025 Eq.6 (hand-only) retargeting via NLopt SLSQP + autograd."""

    def __init__(self, robot: RobotPinocchio, cfg: Tier2Config) -> None:
        self.robot = robot
        self.cfg = cfg
        self.n = len(cfg.target_joint_names)
        assert self.n == robot.dof, f"target_joint_names {self.n} != robot.dof {robot.dof}"

        # Pin pinocchio joint order to the config order.
        self.dof_joint_names = robot.dof_joint_names
        self.idx_cfg2pin = np.array(
            [self.dof_joint_names.index(n) for n in cfg.target_joint_names], dtype=int
        )
        # Box bounds (in cfg order, with epsilon margin).
        limits = robot.joint_limits[self.idx_cfg2pin]
        self._lower = limits[:, 0].copy()
        self._upper = limits[:, 1].copy()
        # Forbid hyperextension (反关节): the wuji URDF lets flexion joints go
        # negative, and the optimizer prefers the hyperextended solution in the
        # nullspace when pos+ori are both satisfiable. Clamp the FLEXION joints
        # (joint1/3/4 -- MCP/PIP/DIP) lower bound so they can't hyperextend;
        # leave abduction (joint2, symmetric +/-) at the URDF lower bound.
        if cfg.forbid_flexion_hyperextension:
            for i, name in enumerate(cfg.target_joint_names):
                if not name.endswith("joint2"):
                    self._lower[i] = max(self._lower[i], cfg.flexion_lower_bound)

        # Frames computed each eval: wrist + 5 tip + 5 DIP = 11 (wrist may overlap a tip? no)
        self.computed_frames: List[str] = list(
            dict.fromkeys([cfg.wrist_link_name, *cfg.fingertip_link_names, *cfg.dip_link_names])
        )
        self._wrist_idx = self.computed_frames.index(cfg.wrist_link_name)
        self._tip_idx = [self.computed_frames.index(n) for n in cfg.fingertip_link_names]
        self._dip_idx = [self.computed_frames.index(n) for n in cfg.dip_link_names]
        # pinch: thumb (finger 0) -> primary fingers (1..4)
        self._pinch_origin = self._tip_idx[0]
        self._pinch_task = self._tip_idx[1:]

        self.huber_loss = torch.nn.SmoothL1Loss(beta=cfg.huber_delta, reduction="none")

        # NLopt SLSQP in cfg order.
        self.opt = nlopt.opt(nlopt.LD_SLSQP, self.n)
        self.opt.set_lower_bounds((self._lower - 1e-3).tolist())
        self.opt.set_upper_bounds((self._upper + 1e-3).tolist())
        self.opt.set_ftol_abs(cfg.opt_ftol_abs)
        self.opt.set_maxtime(cfg.opt_maxtime)
        self.opt.set_maxeval(cfg.opt_maxeval)

        self._last_qpos = np.zeros(self.n, dtype=np.float64)
        self._lp_y: Optional[np.ndarray] = None  # low-pass state
        self._lp_alpha = cfg.low_pass_alpha

    # ---- public API ----

    def retarget(
        self, ref_values: Dict[str, np.ndarray], qpos_init: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Solve one retargeting step. ref_values keys: wrist_tip, thumb_primary,
        dip_tip, (all (N,3) SCALED human vectors); qpos_init optional warm start."""
        x0 = (
            (qpos_init if qpos_init is not None else self._last_qpos).astype(np.float64)
        )
        x0 = np.clip(x0, self._lower, self._upper)
        self._ref = ref_values  # stash for the objective closure

        self.opt.set_min_objective(self._objective)
        try:
            x_opt = self.opt.optimize(x0)
            qpos = np.clip(x_opt, self._lower, self._upper)
        except (ValueError, RuntimeError):
            qpos = x0
        # low-pass
        if self._lp_y is None:
            self._lp_y = qpos.copy()
        else:
            self._lp_y = self._lp_y + self._lp_alpha * (qpos - self._lp_y)
            qpos = self._lp_y.copy()
        self._last_qpos = qpos.copy()
        return qpos.astype(np.float32)

    def auto_calibrate_scaling(
        self, human_wrist_tip: np.ndarray, middle_idx: int = 2
    ) -> float:
        """scaling = robot wrist->middle_tip / human wrist->middle_tip (rest pose)."""
        q_rest = np.zeros(self.robot.dof, dtype=np.float64)
        robot_wrist = self.robot.get_frame_pose(self.cfg.wrist_link_name, q_rest)[:3, 3]
        robot_mid = self.robot.get_frame_pose(
            self.cfg.fingertip_link_names[middle_idx], q_rest
        )[:3, 3]
        robot_len = float(np.linalg.norm(robot_mid - robot_wrist))
        human_len = float(np.linalg.norm(human_wrist_tip))
        return robot_len / human_len if human_len > 1e-6 else 1.0

    def auto_calibrate_scaling_per_finger(
        self, human_wrist_tips: np.ndarray
    ) -> np.ndarray:
        """Per-finger scaling[f] = robot rest wrist->tip[f] / human wrist->tip[f].

        A single scalar (middle-finger ratio) can't match each finger's
        proportions (e.g. the human index is proportionally shorter than the
        robot index), leaving some fingers' targets too short -> spurious curl.
        Per-finger scaling makes each finger's target reach match the robot's.
        """
        q_rest = np.zeros(self.robot.dof, dtype=np.float64)
        robot_wrist = self.robot.get_frame_pose(self.cfg.wrist_link_name, q_rest)[:3, 3]
        s = np.ones(5, dtype=np.float64)
        for f, tip_name in enumerate(self.cfg.fingertip_link_names):
            robot_tip = self.robot.get_frame_pose(tip_name, q_rest)[:3, 3]
            robot_len = float(np.linalg.norm(robot_tip - robot_wrist))
            human_len = float(np.linalg.norm(human_wrist_tips[f]))
            s[f] = robot_len / human_len if human_len > 1e-6 else 1.0
        return s

    # ---- objective ----

    def _objective(self, x: np.ndarray, grad: np.ndarray) -> float:
        cfg = self.cfg
        # pinocchio qpos is in pinocchio order; x is in cfg order.
        qpos_pin = np.zeros(self.robot.dof, dtype=np.float64)
        qpos_pin[self.idx_cfg2pin] = x
        self.robot.compute_forward_kinematics(qpos_pin)

        # frame poses (11, 4, 4) -> positions (11, 3)
        poses = np.stack(
            [self.robot.get_frame_pose(f) for f in self.computed_frames], axis=0
        )
        # torch.tensor(...) copies into a writable leaf so autograd backward can
        # safely write grads (NLopt's x and pinocchio's oMf are read-only views;
        # as_tensor would share memory and backward would be undefined behavior).
        pos = poses[:, :3, 3]
        pos_t = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
        qpos_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)

        # ---- human (scaled) reference vectors (detached) ----
        wrist_pos = pos_t[self._wrist_idx]
        tip_pos = pos_t[self._tip_idx]            # (5,3)
        dip_pos = pos_t[self._dip_idx]            # (5,3)

        ref_wrist_tip = torch.as_tensor(self._ref["wrist_tip"], dtype=torch.float64)   # (5,3)
        ref_pinch = torch.as_tensor(self._ref["thumb_primary"], dtype=torch.float64)   # (4,3)
        ref_dip_tip = torch.as_tensor(self._ref["dip_tip"], dtype=torch.float64)       # (5,3)

        # ---- robot vectors ----
        robot_wrist_tip = tip_pos - wrist_pos                       # (5,3)
        robot_pinch = tip_pos[1:5] - tip_pos[0:1]                   # (4,3) thumb->primary
        robot_dip_tip = tip_pos - dip_pos                           # (5,3)

        # ---- pinch switching (numpy, detached) ----
        # distances d_i on SCALED human pinch vectors (= robot space)
        pinch_h = self._ref["thumb_primary"]
        d = np.linalg.norm(pinch_h, axis=1)                          # (4,)
        s = _sigmoid(d, cfg.epsilon1, cfg.sigmoid_w)                # (4,) high when pinching
        l = _rescale(d, cfg.epsilon1, cfg.epsilon2)                 # (4,)
        dir_h = pinch_h / (d[:, None] + 1e-9)                       # unit dir
        pinch_target = l[:, None] * dir_h                           # (4,3) rescaled target
        pinch_target_t = torch.as_tensor(pinch_target, dtype=torch.float64)

        # fingertip_pos switching weight s_tilde = sigmoid(d, eps1, -w); s_tilde + s = 1
        # (use per-finger pinch distance; thumb uses d_thumb = 1.0 -> s_tilde ~ 1)
        d_pos = np.ones(5, dtype=np.float64)
        d_pos[1:5] = d
        s_tilde = _sigmoid(d_pos, cfg.epsilon1, -cfg.sigmoid_w)    # (5,)

        # ---- losses (huber on L2 norm) ----
        def huber_on_vec_diff(robot_vec, ref_vec, weight_per_row):
            err = torch.norm(robot_vec - ref_vec, dim=-1)           # (N,)
            w = torch.as_tensor(weight_per_row, dtype=torch.float64)
            return (self.huber_loss(w * err, torch.zeros_like(err))).sum()

        L_pos = cfg.lambda_fingertip_pos * huber_on_vec_diff(
            robot_wrist_tip, ref_wrist_tip, s_tilde
        )
        L_rot = cfg.lambda_fingertip_rot * huber_on_vec_diff(
            robot_dip_tip, ref_dip_tip, np.ones(5)
        )
        L_pinch = cfg.lambda_pinch * huber_on_vec_diff(
            robot_pinch, pinch_target_t, s
        )

        # joint regularization (analytic, in cfg order)
        q_err = qpos_t - torch.as_tensor(cfg.joint_pos_ref, dtype=torch.float64)
        L_joint = (torch.as_tensor(cfg.joint_pos_weights, dtype=torch.float64) * q_err * q_err).sum()
        q_vel = qpos_t - torch.as_tensor(self._last_qpos, dtype=torch.float64)
        L_vel = (cfg.joint_vel_weight * q_vel * q_vel).sum()

        total = L_pos + L_rot + L_pinch + L_joint + L_vel

        if grad.size > 0:
            total.backward()
            # link-position gradient -> joint gradient via frame Jacobians (linear part)
            self.robot.compute_jacobians(qpos_pin)
            jac_list = []
            for f in self.computed_frames:
                jac = self.robot.get_frame_space_jacobian(f)  # (6, dof_pin)
                jac_list.append(jac[:3, :])
            jac_pin = np.stack(jac_list, axis=0)               # (11, 3, dof_pin)
            # slice to cfg joint order
            jac_cfg = jac_pin[:, :, self.idx_cfg2pin]          # (11, 3, n_cfg)
            grad_pos = pos_t.grad.cpu().numpy()[:, None, :]    # (11, 1, 3)
            link_grad = np.matmul(grad_pos, jac_cfg)           # (11, 1, n_cfg)
            link_grad = link_grad.mean(1).sum(0)               # (n_cfg,)
            # qpos-tensor grads (joint reg terms)
            q_grad = qpos_t.grad.cpu().numpy()
            grad[:] = link_grad + q_grad

        return float(total.cpu().detach().item())


def build_config(raw: Dict) -> Tuple[Tier2Config, str, bool]:
    """Build a Tier2Config from a parsed YAML dict (the 'retargeting:' block).

    Returns (config, urdf_path, scaling_auto). scaling_auto=True means the caller
    should auto-calibrate the scalar and scale human vectors before retarget().
    """
    names = raw["target_joint_names"]
    n = len(names)
    jpw_raw = raw.get("joint_pos_weights", {})
    jpw = np.zeros(n, dtype=np.float64)
    for jn, w in jpw_raw.items():
        if jn in names:
            jpw[names.index(jn)] = float(w)
    jpref = np.full(n, float(raw.get("joint_pos_ref", 0.0)), dtype=np.float64)

    sf = raw.get("scaling_factor", "auto")
    if isinstance(sf, str) and sf == "auto":
        scaling = 1.0  # placeholder; caller overrides via auto_calibrate_scaling_per_finger
        scaling_auto = True
    elif isinstance(sf, (list, tuple)) and len(sf) == 5:
        # manual PER-FINGER scaling [thumb, index, middle, ring, pinky] -- lets the
        # user tune a finger that auto-calibration under/over-scales (e.g. raise the
        # index entry if the robot index doesn't extend enough).
        scaling = np.array([float(x) for x in sf], dtype=np.float64)
        scaling_auto = False
    elif isinstance(sf, (list, tuple)):
        scaling = float(np.mean(sf))  # wrong-length list -> flat mean fallback
        scaling_auto = False
    else:
        scaling = float(sf)
        scaling_auto = False

    cfg = Tier2Config(
        wrist_link_name=raw["wrist_link_name"],
        fingertip_link_names=list(raw["fingertip_link_names"]),
        dip_link_names=list(raw["dip_link_names"]),
        target_joint_names=names,
        lambda_fingertip_pos=float(raw.get("lambda_fingertip_pos", 1.0)),
        lambda_fingertip_rot=float(raw.get("lambda_fingertip_rot", 10.0)),
        lambda_pinch=float(raw.get("lambda_pinch", 10.0)),
        epsilon1=float(raw.get("epsilon1", 0.1)),
        epsilon2=float(raw.get("epsilon2", 0.01)),
        sigmoid_w=float(raw.get("sigmoid_w", 10.0)),
        huber_delta=float(raw.get("huber_delta", 0.02)),
        scaling_factor=scaling,
        joint_pos_ref=jpref,
        joint_pos_weights=jpw,
        joint_vel_weight=float(raw.get("joint_vel_weight", 0.01)),
        opt_ftol_abs=float(raw.get("opt_ftol_abs", 1e-6)),
        opt_maxtime=float(raw.get("opt_maxtime", 0.02)),
        opt_maxeval=int(raw.get("opt_maxeval", 50)),
        low_pass_alpha=float(raw.get("low_pass_alpha", 0.3)),
        forbid_flexion_hyperextension=bool(raw.get("forbid_flexion_hyperextension", True)),
        flexion_lower_bound=float(raw.get("flexion_lower_bound", 0.0)),
    )
    return cfg, raw["urdf_path"], scaling_auto
