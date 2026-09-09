#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Gaze task implementation."""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import TargetNotSet, TaskDefinitionError
from ..utils import get_joint_idx
from .task import Task


def _skew(vector: np.ndarray) -> np.ndarray:
    """Return the cross-product matrix of a 3D vector."""
    x, y, z = vector
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


class GazeTask(Task):
    r"""Point a frame-fixed optical axis toward a world-space target.

    The task error is ``direction - axis`` (both unit vectors). Unlike the raw
    cross product, this is zero only when the optical axis is aligned with the
    target direction (dot product +1), not at the anti-parallel null (dot -1).
    Its Jacobian can be restricted to selected 1-DoF joints, which is useful for
    a pan-tilt head: other robot joints then cannot compensate for gaze error.
    """

    frame: str
    target_world: Optional[np.ndarray]

    def __init__(
        self,
        frame: str,
        model: pin.Model,
        joint_names: Sequence[str],
        cost: Union[float, Sequence[float], np.ndarray] = 1.0,
        optical_axis: Sequence[float] = (1.0, 0.0, 0.0),
        offset: Sequence[float] = (0.0, 0.0, 0.0),
        lm_damping: float = 0.0,
        gain: float = 1.0,
        min_target_distance: float = 1e-4,
    ) -> None:
        r"""Create a gaze task.

        Args:
            frame: Frame carrying the optical axis.
            model: Robot model.
            joint_names: One-DoF joints allowed to reduce gaze error.
            cost: Scalar or 3D cost for the gaze error, in
                :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
            optical_axis: Optical axis expressed in ``frame``.
            offset: Optical origin offset expressed in ``frame`` [m].
            lm_damping: Levenberg-Marquardt damping.
            gain: Task gain :math:`\alpha \in [0, 1]`.
            min_target_distance: Disable the task below this distance [m].
        """
        if not model.existFrame(frame):
            raise TaskDefinitionError(f"Frame '{frame}' does not exist")
        if not joint_names:
            raise TaskDefinitionError("joint_names must not be empty")
        self.frame = str(frame)
        self.model = model
        self.target_world = None
        self.offset = np.asarray(offset, dtype=float).reshape(3).copy()
        axis = np.asarray(optical_axis, dtype=float).reshape(3).copy()
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            raise TaskDefinitionError("optical_axis must be non-zero")
        self.optical_axis = axis / axis_norm
        self.min_target_distance = float(min_target_distance)
        if self.min_target_distance < 0.0:
            raise TaskDefinitionError("min_target_distance must be non-negative")

        idx_v = []
        resolved_names = []
        for name in joint_names:
            if not model.existJointName(name):
                raise TaskDefinitionError(f"Joint '{name}' does not exist")
            joint_id = model.getJointId(name)
            joint = model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise TaskDefinitionError(
                    f"Joint '{name}' has nq={joint.nq}, nv={joint.nv}; "
                    "GazeTask supports 1-DoF joints only"
                )
            _, velocity_index = get_joint_idx(model, name)
            idx_v.append(int(velocity_index))
            resolved_names.append(str(name))
        self.joint_names = tuple(resolved_names)
        self.idx_v = tuple(idx_v)

        if isinstance(cost, (int, float)):
            resolved_cost: Union[float, np.ndarray] = float(cost)
        else:
            resolved_cost = np.asarray(cost, dtype=float).reshape(3).copy()
        if np.any(np.asarray(resolved_cost) < 0.0):
            raise TaskDefinitionError("cost must be non-negative")
        super().__init__(
            cost=resolved_cost,
            gain=gain,
            lm_damping=lm_damping,
        )

    def set_target(self, target_world: Sequence[float]) -> None:
        """Set the target point in world coordinates."""
        target = np.asarray(target_world, dtype=float).reshape(3)
        if not np.all(np.isfinite(target)):
            raise TaskDefinitionError("target_world must be finite")
        self.target_world = target.copy()

    def _geometry(
        self, configuration: Configuration
    ) -> tuple[pin.SE3, np.ndarray, np.ndarray, float]:
        if self.target_world is None:
            raise TargetNotSet(f"no target set for gaze frame '{self.frame}'")
        transform = configuration.get_transform_frame_to_world(self.frame)
        origin = transform.translation + transform.rotation @ self.offset
        axis = transform.rotation @ self.optical_axis
        delta = self.target_world - origin
        distance = float(np.linalg.norm(delta))
        if distance <= self.min_target_distance:
            direction = np.zeros(3)
        else:
            direction = delta / distance
        return transform, axis, direction, distance

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute optical-axis alignment error in world coordinates."""
        _, axis, direction, distance = self._geometry(configuration)
        if distance <= self.min_target_distance:
            return np.zeros(3)
        return direction - axis

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the alignment-error Jacobian, masked to selected joints."""
        _, jacobian = self._error_and_jacobian(configuration)
        return jacobian

    def _error_and_jacobian(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray]:
        transform, axis, direction, distance = self._geometry(configuration)
        jacobian = np.zeros((3, configuration.model.nv))
        if distance <= self.min_target_distance:
            return np.zeros(3), jacobian

        frame_jacobian_local = configuration.get_frame_jacobian(self.frame)
        rotation = transform.rotation
        origin_jacobian = rotation @ frame_jacobian_local[:3]
        angular_jacobian = rotation @ frame_jacobian_local[3:]
        offset_world = rotation @ self.offset
        origin_jacobian -= _skew(offset_world) @ angular_jacobian

        direction_projection = np.eye(3) - np.outer(direction, direction)
        axis_jacobian = -_skew(axis) @ angular_jacobian
        direction_jacobian = -direction_projection @ origin_jacobian / distance
        full_jacobian = direction_jacobian - axis_jacobian
        jacobian[:, self.idx_v] = full_jacobian[:, self.idx_v]
        return direction - axis, jacobian

    def compute_qp_objective(
        self, configuration: Configuration
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the QP objective with damping restricted to gaze joints."""
        error, jacobian = self._error_and_jacobian(configuration)
        minus_gain_error = -self.gain * error
        weight = (
            np.eye(3)
            if self.cost is None
            else np.diag(
                [self.cost] * 3 if isinstance(self.cost, float) else self.cost
            )
        )
        weighted_jacobian = weight @ jacobian
        weighted_error = weight @ minus_gain_error
        hessian = weighted_jacobian.T @ weighted_jacobian
        mu = self.lm_damping * weighted_error @ weighted_error
        hessian[self.idx_v, self.idx_v] += mu
        linear = -weighted_error.T @ weighted_jacobian
        return hessian, linear

    def __repr__(self) -> str:
        """Human-readable representation of the task."""
        return (
            "GazeTask("
            f"frame={self.frame!r}, "
            f"joints={self.joint_names!r}, "
            f"cost={self.cost}, "
            f"lm_damping={self.lm_damping}, "
            f"gain={self.gain})"
        )
