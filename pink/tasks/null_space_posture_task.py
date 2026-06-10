#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Null-space posture task implementation."""

from typing import Optional, Sequence

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import TargetNotSet
from ..utils import get_root_joint_dim
from .task import Task


class NullSpacePostureTask(Task):
    r"""Regulate joint angles within the null space of protected frames.

    This task drives the configuration toward a target posture
    :math:`q^*`, like :class:`PostureTask`, but both its error and its
    Jacobian are projected onto the null space of the stacked Jacobian of
    a set of *protected frames* (typically end effectors):

    .. math::

        N = I_{n_v} - J_{ee}^{+} J_{ee}, \qquad
        e(q) = N \, (q \ominus q^*), \qquad
        J(q) = N

    Since :math:`N` is orthogonal to the row space of :math:`J_{ee}`, the
    posture objective cannot pull joints in directions that move the
    protected frames (to first order). Combined with
    :class:`FrameTask` objectives on the same frames, this removes the
    structural conflict between Cartesian tracking and joint-space posture
    tracking: posture convergence happens only through redundant degrees
    of freedom.

    When an integration timestep is provided via
    :func:`set_integration_timestep`, the projected posture step is
    additionally scaled *uniformly* so that no joint exceeds its velocity
    limit. This keeps the displacement direction inside the null space:
    without it, per-joint velocity clipping by box inequalities in the IK
    problem would distort the direction of the solution and leak motion
    into the protected frames ("fast joints wait for slow joints").

    Attributes:
        target_q: Target vector in the configuration space. If the model
            has a floating base, the vector includes floating-base
            coordinates (they have no effect on this task).
        frames: Names of the protected frames whose stacked Jacobian
            defines the null-space projector.
        projector_damping: Damping :math:`\lambda` of the damped
            pseudo-inverse :math:`J^{+} = J^T (J J^T + \lambda^2 I)^{-1}`
            used to build the projector. Increase near singularities.
        velocity_limit_scale: Fraction of model velocity limits available
            to this task when uniform scaling is active (leave headroom
            for concurrent tasks such as frame tracking).
    """

    target_q: Optional[np.ndarray]

    def __init__(
        self,
        frames: Sequence[str],
        cost: float,
        projector_damping: float = 1e-6,
        velocity_limit_scale: float = 0.8,
        lm_damping: float = 0.0,
        gain: float = 1.0,
    ) -> None:
        r"""Create task.

        Args:
            frames: Names of the protected frames (typically end-effector
                links). Their stacked Jacobian defines the null space in
                which the posture error is regulated. An empty sequence
                makes this task behave like a regular posture task over the
                full tangent space.
            cost: value used to cast joint angle differences to a
                homogeneous cost, in :math:`[\mathrm{cost}] /
                [\mathrm{rad}]`.
            projector_damping: Damping of the damped pseudo-inverse used to
                build the null-space projector.
            velocity_limit_scale: Fraction of model velocity limits
                available to this task when uniform scaling is active (see
                :func:`set_integration_timestep`).
            lm_damping: Unitless scale of the Levenberg-Marquardt (only
                when the error is large) regularization term, which helps
                when targets are unfeasible.
            gain: Task gain :math:`\alpha \in [0, 1]` for additional
                low-pass filtering. Defaults to 1.0 (no filtering) for
                dead-beat control.
        """
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.target_q = None
        self.projector_damping = float(projector_damping)
        self.velocity_limit_scale = float(velocity_limit_scale)
        self._dt: Optional[float] = None
        self._frames = list(frames)
        self._cached_q: Optional[np.ndarray] = None
        self._cached_projector: Optional[np.ndarray] = None

    def set_integration_timestep(self, dt: Optional[float]) -> None:
        """Enable uniform velocity-limit scaling of the posture step.

        When a timestep is set, the projected posture step requested by
        this task is scaled uniformly (same factor on all joints) so that
        no joint is asked to move faster than ``velocity_limit_scale``
        times its model velocity limit over ``dt``. This prevents the
        per-joint velocity box constraints of the IK problem from clipping
        the solution joint by joint, which would distort its direction out
        of the null space.

        Args:
            dt: Integration timestep in [s], or ``None`` to disable
                scaling.
        """
        self._dt = None if dt is None else float(dt)

    @property
    def frames(self) -> Sequence[str]:
        """Names of the protected frames."""
        return list(self._frames)

    def set_frames(self, frames: Sequence[str]) -> None:
        """Set protected frames and invalidate the cached projector.

        Args:
            frames: Names of the protected frames.
        """
        self._frames = list(frames)
        self._cached_q = None
        self._cached_projector = None

    def set_target(self, target_q: np.ndarray) -> None:
        """Set target posture.

        Args:
            target_q: Target vector in the configuration space.
        """
        self.target_q = target_q.copy()

    def set_target_from_configuration(
        self, configuration: Configuration
    ) -> None:
        """Set target posture from a robot configuration.

        Args:
            configuration: Robot configuration.
        """
        self.set_target(configuration.q)

    def _compute_projector(self, configuration: Configuration) -> np.ndarray:
        r"""Compute (or fetch from cache) the null-space projector.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Projector :math:`N = I - J^{+} J \in \mathbb{R}^{n_v \times
            n_v}` onto the null space of the stacked protected-frame
            Jacobian.
        """
        q = configuration.q
        if self._cached_projector is not None and np.array_equal(
            q, self._cached_q
        ):
            return self._cached_projector
        nv = configuration.model.nv
        if not self._frames:
            projector = np.eye(nv)
        else:
            jacobian = np.vstack(
                [
                    configuration.get_frame_jacobian(frame)
                    for frame in self._frames
                ]
            )
            k = jacobian.shape[0]
            gram = jacobian @ jacobian.T + (
                self.projector_damping**2
            ) * np.eye(k)
            # J^+ = J^T (J J^T + lambda^2 I)^{-1}, solved without inverting.
            pinv = np.linalg.solve(gram, jacobian).T
            projector = np.eye(nv) - pinv @ jacobian
        self._cached_q = q.copy()
        self._cached_projector = projector
        return projector

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute null-space posture task error.

        The error is the posture difference projected onto the null space
        of the protected frames:

        .. math::

            e(q) = N \, (q \ominus q^*)

        so that the task dynamics :math:`J(q) \Delta q = -\alpha e(q)`
        drive the configuration toward :math:`q^*` only along directions
        that do not move the protected frames.

        If an integration timestep is set (see
        :func:`set_integration_timestep`), the error is scaled uniformly
        so the requested step stays within joint velocity limits.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Null-space posture task error :math:`e(q)`.
        """
        if self.target_q is None:
            raise TargetNotSet("no posture target")
        diff = pin.difference(
            configuration.model,
            self.target_q,
            configuration.q,
        )
        _, root_nv = get_root_joint_dim(configuration.model)
        if root_nv > 0:
            diff[:root_nv] = 0.0
        error = self._compute_projector(configuration) @ diff
        if self._dt is not None:
            # Requested displacement is Delta_q = -gain * error. Scale it
            # uniformly so no joint exceeds its velocity limit: this keeps
            # the step direction inside the null space instead of letting
            # per-joint box constraints clip it joint by joint.
            v_max = np.asarray(
                configuration.model.velocityLimit, dtype=float
            )
            step = self.gain * np.abs(error)
            budget = self.velocity_limit_scale * v_max * self._dt
            valid = np.isfinite(budget) & (budget > 0.0)
            if np.any(valid):
                ratio = float(np.max(step[valid] / budget[valid]))
                if ratio > 1.0:
                    error = error / ratio
        return error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the null-space posture task Jacobian.

        The task Jacobian is the null-space projector :math:`N \in
        \mathbb{R}^{n_v \times n_v}` itself, so the QP contribution of this
        task is :math:`\| N \Delta q + \alpha N (q \ominus q^*) \|^2_W`:
        only the null-space component of the displacement is compared
        against the (projected) posture error.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task Jacobian :math:`J(q) = N`.
        """
        return self._compute_projector(configuration)

    def __repr__(self):
        """Human-readable representation of the task."""
        return (
            "NullSpacePostureTask("
            f"frames={self._frames}, "
            f"cost={self.cost}, "
            f"projector_damping={self.projector_damping}, "
            f"velocity_limit_scale={self.velocity_limit_scale}, "
            f"dt={self._dt}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )
