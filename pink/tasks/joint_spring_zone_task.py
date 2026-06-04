#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Joint spring-zone repulsion task."""

from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import TaskDefinitionError
from ..utils import get_joint_idx
from .joint_position_task import JointPositionTask
from .task import Task

EscapeMode = Literal["high", "low", "nearest"]
EscapePreference = Optional[Literal["high", "low", "+", "-"]]

# Tolerance to decide whether a spring-zone edge coincides with a URDF limit.
_LIMIT_COINCIDE_TOL = 1e-3


class JointSpringZoneTask(Task):
    r"""Repel 1-DoF joints out of a spring zone :math:`[q_{\mathrm{lo}}, q_{\mathrm{hi}}]`.

    While active, a linear error pushes the joint past the escape edge by
    ``exit_overshoot`` [rad] (e.g. zone upper edge 0 → settle near 0.02), which
    reduces chatter on the zone boundary.

    **Escape direction** (from URDF limits + optional preference):

    1. Zone touches lower limit only → escape ``high`` (target :math:`q_{\mathrm{hi}} + \varepsilon`).
    2. Zone touches upper limit only → escape ``low`` (target :math:`q_{\mathrm{lo}} - \varepsilon`).
    3. Else preference ``high``/``low`` if set.
    4. Else ``nearest`` (closer edge ± :math:`\varepsilon`).

    Errors follow :class:`PostureTask` / :func:`pin.difference` sign (:math:`\Delta q \approx -e`).
    """

    def __init__(
        self,
        model: pin.Model,
        joint_zones: Mapping[str, Tuple[float, float]],
        cost: Union[float, Mapping[str, float], Sequence[float]] = 1.0,
        lm_damping: float = 0.0,
        gain: float = 1.0,
        escape_preference: Optional[Mapping[str, EscapePreference]] = None,
        exit_overshoot: Union[float, Mapping[str, float]] = 0.0,
        limit_tol: float = _LIMIT_COINCIDE_TOL,
    ) -> None:
        r"""Create task.

        Args:
            model: Robot model (provides URDF position limits).
            joint_zones: Map joint name -> ``(q_zone_lo, q_zone_hi)`` [rad].
            cost: Scalar or per-joint costs.
            lm_damping: Levenberg-Marquardt damping (unitless).
            gain: Task gain :math:`\alpha \in [0, 1]`.
            escape_preference: Per-joint escape when zone is not limit-flush.
            exit_overshoot: Extra margin [rad] past the escape edge (scalar or
                per-joint). Active band extends through this margin so the joint
                can settle slightly outside the nominal zone edge.
            limit_tol: Tolerance [rad] for zone/limit coincidence.
        """
        if not joint_zones:
            raise TaskDefinitionError("joint_zones must not be empty")
        self.model = model
        self.joint_names: List[str] = []
        self.q_lo: Dict[str, float] = {}
        self.q_hi: Dict[str, float] = {}
        self._escape_mode: Dict[str, EscapeMode] = {}
        self._exit_overshoot: Dict[str, float] = {}
        self.idx_q: List[int] = []
        self.idx_v: List[int] = []
        prefs = dict(escape_preference or {})
        overshoots = self._resolve_overshoot_map(exit_overshoot, list(joint_zones.keys()))
        for name, bounds in joint_zones.items():
            if not model.existJointName(name):
                continue
            q_lo, q_hi = float(bounds[0]), float(bounds[1])
            if q_lo > q_hi:
                q_lo, q_hi = q_hi, q_lo
            joint_id = model.getJointId(name)
            joint = model.joints[joint_id]
            if joint.nq != 1:
                raise TaskDefinitionError(
                    f"Joint '{name}' has nq={joint.nq}; "
                    "JointSpringZoneTask supports 1-DoF joints only"
                )
            idx_q, idx_v = get_joint_idx(model, name)
            lim_lo = float(model.lowerPositionLimit[idx_q])
            lim_hi = float(model.upperPositionLimit[idx_q])
            self.joint_names.append(name)
            self.q_lo[name] = q_lo
            self.q_hi[name] = q_hi
            self._escape_mode[name] = self._resolve_escape_mode(
                q_lo, q_hi, lim_lo, lim_hi, prefs.get(name), limit_tol
            )
            self._exit_overshoot[name] = float(overshoots.get(name, 0.0))
            self.idx_q.append(idx_q)
            self.idx_v.append(idx_v)
        if not self.idx_v:
            raise TaskDefinitionError(
                f"No valid joints for JointSpringZoneTask: {list(joint_zones)}"
            )
        self._idx_q = np.asarray(self.idx_q, dtype=int)
        self._idx_v = np.asarray(self.idx_v, dtype=int)
        resolved_cost = JointPositionTask._resolve_cost_vector(cost, self.joint_names)
        super().__init__(cost=resolved_cost, gain=gain, lm_damping=lm_damping)

    @staticmethod
    def _resolve_overshoot_map(
        exit_overshoot: Union[float, Mapping[str, float]],
        joint_names: List[str],
    ) -> Dict[str, float]:
        if isinstance(exit_overshoot, (int, float)):
            return {name: max(0.0, float(exit_overshoot)) for name in joint_names}
        return {name: max(0.0, float(exit_overshoot.get(name, 0.0))) for name in joint_names}

    @staticmethod
    def _resolve_escape_mode(
        q_zone_lo: float,
        q_zone_hi: float,
        lim_lo: float,
        lim_hi: float,
        preference: EscapePreference,
        tol: float,
    ) -> EscapeMode:
        touch_lo = abs(q_zone_lo - lim_lo) <= tol
        touch_hi = abs(q_zone_hi - lim_hi) <= tol
        if touch_lo and not touch_hi:
            return "high"
        if touch_hi and not touch_lo:
            return "low"
        if preference in ("high", "+"):
            return "high"
        if preference in ("low", "-"):
            return "low"
        return "nearest"

    def _joint_angle(self, configuration: Configuration, row: int) -> float:
        return float(configuration.q[self._idx_q[row]])

    def _active_error(
        self,
        q: float,
        q_lo: float,
        q_hi: float,
        mode: EscapeMode,
        overshoot: float,
    ) -> Tuple[bool, float]:
        eps = max(0.0, float(overshoot))
        if mode == "high":
            q_tgt = q_hi + eps
            if q < q_lo or q > q_tgt:
                return False, 0.0
            return True, q - q_tgt
        if mode == "low":
            q_tgt = q_lo - eps
            if q > q_hi or q < q_tgt:
                return False, 0.0
            return True, q - q_tgt
        # nearest
        if q < q_lo or q > q_hi:
            return False, 0.0
        if (q - q_lo) <= (q_hi - q):
            q_tgt = q_lo - eps
            if q < q_tgt:
                return False, 0.0
            return True, q - q_tgt
        q_tgt = q_hi + eps
        if q > q_tgt:
            return False, 0.0
        return True, q - q_tgt

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Non-zero while pushing through the zone and exit-overshoot band."""
        err = np.zeros(len(self.joint_names), dtype=np.float64)
        for row, name in enumerate(self.joint_names):
            q = self._joint_angle(configuration, row)
            active, e = self._active_error(
                q,
                self.q_lo[name],
                self.q_hi[name],
                self._escape_mode[name],
                self._exit_overshoot[name],
            )
            if active:
                err[row] = e
        return err

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Unit tangent row when active."""
        nv = configuration.model.nv
        k = len(self.joint_names)
        jacobian = np.zeros((k, nv))
        for row, name in enumerate(self.joint_names):
            q = self._joint_angle(configuration, row)
            active, _ = self._active_error(
                q,
                self.q_lo[name],
                self.q_hi[name],
                self._escape_mode[name],
                self._exit_overshoot[name],
            )
            if active:
                jacobian[row, self._idx_v[row]] = 1.0
        return jacobian

    def escape_mode(self, joint_name: str) -> EscapeMode:
        """Return the resolved escape mode for a joint in this task."""
        return self._escape_mode[joint_name]

    def __repr__(self) -> str:
        """Human-readable representation of the task."""
        zones = {
            n: (self.q_lo[n], self.q_hi[n], self._escape_mode[n], self._exit_overshoot[n])
            for n in self.joint_names
        }
        return (
            "JointSpringZoneTask("
            f"joints={self.joint_names!r}, "
            f"zones={zones!r}, "
            f"cost={self.cost}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )
