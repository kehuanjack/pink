#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Joint position preference task."""

from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import TaskDefinitionError
from ..utils import get_joint_idx
from .task import Task


class JointPositionTask(Task):
    r"""Regulate one or more 1-DoF joints toward preferred angles.

    For each joint name :math:`i` in ``joint_targets``, the task drives
    :math:`q_i \rightarrow q^\*_i` with error
    :math:`e_i = q^\*_i \ominus q_i` (Pinocchio configuration difference).

    This is a generic secondary IK task for redundancy resolution (e.g. elbow
    joint preference at zero, nominal posture on a subset of joints).
    """

    def __init__(
        self,
        model: pin.Model,
        joint_targets: Mapping[str, float],
        cost: Union[float, Mapping[str, float], Sequence[float]] = 1.0,
        lm_damping: float = 0.0,
        gain: float = 1.0,
    ) -> None:
        r"""Create task.

        Args:
            model: Robot model.
            joint_targets: Map from joint name to preferred position [rad].
            cost: Scalar cost for all joints, or per-joint costs keyed by joint
                name (same keys as ``joint_targets``), or a sequence in
                ``joint_targets`` iteration order.
            lm_damping: Levenberg-Marquardt damping (unitless).
            gain: Task gain :math:`\alpha \in [0, 1]`.
        """
        if not joint_targets:
            raise TaskDefinitionError("joint_targets must not be empty")
        self.model = model
        self.joint_names: List[str] = []
        self.joint_targets: Dict[str, float] = {}
        self.idx_q: List[int] = []
        self.idx_v: List[int] = []
        self._target_q = pin.neutral(model).copy()
        for name, q_pref in joint_targets.items():
            if not model.existJointName(name):
                continue
            joint_id = model.getJointId(name)
            joint = model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise TaskDefinitionError(
                    f"Joint '{name}' has nq={joint.nq}, nv={joint.nv}; "
                    "JointPositionTask supports 1-DoF joints only"
                )
            idx_q, idx_v = get_joint_idx(model, name)
            self.joint_names.append(name)
            self.joint_targets[name] = float(q_pref)
            self.idx_q.append(idx_q)
            self.idx_v.append(idx_v)
            self._target_q[idx_q] = float(q_pref)
        if not self.idx_v:
            raise TaskDefinitionError(
                f"No valid joints for JointPositionTask: {list(joint_targets)}"
            )
        self._idx_v = np.asarray(self.idx_v, dtype=int)
        self._idx_q = np.asarray(self.idx_q, dtype=int)
        self._jacobian = np.zeros((len(self.idx_v), model.nv))
        self._jacobian[np.arange(len(self.idx_v)), self._idx_v] = 1.0
        self._jacobian.setflags(write=False)
        resolved_cost = JointPositionTask._resolve_cost_vector(cost, self.joint_names)
        super().__init__(cost=resolved_cost, gain=gain, lm_damping=lm_damping)

    @staticmethod
    def _resolve_cost_vector(
        cost: Union[float, Mapping[str, float], Sequence[float]],
        joint_names: List[str],
    ) -> Union[float, np.ndarray]:
        if isinstance(cost, (int, float)):
            return float(cost)
        if isinstance(cost, Mapping):
            try:
                return np.array(
                    [float(cost[name]) for name in joint_names], dtype=np.float64
                )
            except KeyError as exc:
                raise TaskDefinitionError(
                    f"Missing cost for joint {exc.args[0]!r}"
                ) from exc
        cost_arr = np.asarray(cost, dtype=np.float64).reshape(-1)
        if cost_arr.shape[0] != len(joint_names):
            raise TaskDefinitionError(
                f"cost length {cost_arr.shape[0]} != number of joints {len(joint_names)}"
            )
        return cost_arr

    def set_joint_targets(self, joint_targets: Mapping[str, float]) -> None:
        """Update preferred positions for joints already registered in this task."""
        for name, q_pref in joint_targets.items():
            if name not in self.joint_targets:
                raise TaskDefinitionError(
                    f"Joint '{name}' is not in this JointPositionTask "
                    f"({self.joint_names})"
                )
            idx_q, _ = get_joint_idx(self.model, name)
            self.joint_targets[name] = float(q_pref)
            self._target_q[idx_q] = float(q_pref)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute scalar joint position errors :math:`q - q^\*`."""
        return configuration.q[self._idx_q] - self._target_q[self._idx_q]

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Return the cached selector Jacobian."""
        return self._jacobian

    def __repr__(self) -> str:
        """Human-readable representation of the task."""
        return (
            "JointPositionTask("
            f"joints={self.joint_names!r}, "
            f"targets={self.joint_targets!r}, "
            f"cost={self.cost}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )


class JointLimitCenteringTask(JointPositionTask):
    r"""Legacy helper: drive joints toward limit mid-range or fixed overrides.

    Prefer :class:`JointPositionTask` with an explicit ``joint_targets`` map.
    """

    def __init__(
        self,
        model: pin.Model,
        joint_names: Sequence[str],
        cost: float = 1.0,
        lm_damping: float = 0.0,
        gain: float = 1.0,
        preferred_positions: Optional[Mapping[str, float]] = None,
    ) -> None:
        overrides = dict(preferred_positions or {})
        joint_targets: Dict[str, float] = {}
        for name in joint_names:
            if not model.existJointName(name):
                continue
            if name in overrides:
                joint_targets[name] = float(overrides[name])
                continue
            idx_q, _ = get_joint_idx(model, name)
            lo = float(model.lowerPositionLimit[idx_q])
            hi = float(model.upperPositionLimit[idx_q])
            if lo >= hi:
                continue
            joint_targets[name] = 0.5 * (lo + hi)
        super().__init__(
            model,
            joint_targets,
            cost=cost,
            lm_damping=lm_damping,
            gain=gain,
        )
