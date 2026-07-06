#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Null-space posture task implementation."""

from typing import Optional, Sequence, Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import TargetNotSet
from ..utils import get_root_joint_dim
from .posture_task import PostureTask


def masked_null_space_projector_block(
    configuration: Configuration,
    frames: Sequence[str],
    v_indices: np.ndarray,
    *,
    damping: float,
) -> np.ndarray:
    """Return ``P_m = I - J^+ J`` on the velocity subset *v_indices* only."""
    v_idx = np.asarray(v_indices, dtype=int)
    nv_m = v_idx.size
    if not frames or nv_m == 0:
        return np.eye(nv_m, dtype=float)

    jacobian = np.vstack(
        [configuration.get_frame_jacobian(frame) for frame in frames]
    )
    jacobian_m = jacobian[:, v_idx]
    k = jacobian_m.shape[0]
    gram = jacobian_m @ jacobian_m.T + (float(damping) ** 2) * np.eye(k)
    pinv = np.linalg.solve(gram, jacobian_m).T
    return np.eye(nv_m, dtype=float) - pinv @ jacobian_m


def masked_null_space_projector(
    configuration: Configuration,
    frames: Sequence[str],
    v_indices: np.ndarray,
    *,
    damping: float,
) -> np.ndarray:
    """Embed the chain-local projector into an ``nv x nv`` matrix (legacy helper)."""
    nv = configuration.model.nv
    v_idx = np.asarray(v_indices, dtype=int)
    projector = np.eye(nv, dtype=float)
    if not frames or v_idx.size == 0:
        return projector
    projector[np.ix_(v_idx, v_idx)] = masked_null_space_projector_block(
        configuration, frames, v_idx, damping=damping
    )
    return projector


class NullSpacePostureTask(PostureTask):
    r"""Regulate joint angles within the null space of protected frames.

    Like :class:`PostureTask`, this task drives the configuration toward a
    target posture :math:`q^*`. Both error and Jacobian are projected onto
    the null space of a stacked end-effector Jacobian:

    .. math::

        N = I_{n_v} - J_{ee}^{+} J_{ee}, \qquad
        e(q) = N \, (q^* \ominus q), \qquad
        J(q) = N

    When ``v_indices`` is set, the projector and task error apply **only** on that
    joint-chain velocity subset (dual-arm: each arm's NS task does not pull other
    chains). Otherwise the projector applies to the full tangent space.

    Optional uniform velocity-limit scaling (see
    :func:`set_integration_timestep`) keeps the requested step direction inside
    the null space instead of letting per-joint box constraints clip it.
    """

    def __init__(
        self,
        frames: Union[str, Sequence[str]],
        cost: float,
        projector_damping: float = 1e-6,
        velocity_limit_scale: float = 0.8,
        lm_damping: float = 0.0,
        gain: float = 1.0,
    ) -> None:
        super().__init__(cost=cost, lm_damping=lm_damping, gain=gain)
        if isinstance(frames, str):
            frame_list = [frames]
        else:
            frame_list = list(frames)
        self.projector_damping = float(projector_damping)
        self.velocity_limit_scale = float(velocity_limit_scale)
        self._dt: Optional[float] = None
        self._frames = frame_list
        self._v_indices: Optional[np.ndarray] = None
        self._cached_q: Optional[np.ndarray] = None
        self._cached_projector: Optional[np.ndarray] = None
        self._cached_masked_block: Optional[np.ndarray] = None

    def set_integration_timestep(self, dt: Optional[float]) -> None:
        """Enable uniform velocity-limit scaling over ``dt`` [s] (or disable)."""
        self._dt = None if dt is None else float(dt)

    def _invalidate_projector_cache(self) -> None:
        self._cached_q = None
        self._cached_projector = None
        self._cached_masked_block = None

    @property
    def frames(self) -> Sequence[str]:
        """Names of the protected frames."""
        return list(self._frames)

    def set_frames(self, frames: Sequence[str]) -> None:
        """Set protected frames and invalidate the cached projector."""
        self._frames = list(frames)
        self._invalidate_projector_cache()

    def set_v_index_mask(self, indices: Sequence[int]) -> None:
        """Limit the null-space projector to selected velocity DOFs."""
        self._v_indices = np.asarray(indices, dtype=int)
        self._invalidate_projector_cache()

    def _masked_projector_block(
        self, configuration: Configuration
    ) -> Optional[np.ndarray]:
        if self._v_indices is None or self._v_indices.size == 0:
            return None
        q = configuration.q
        if (
            self._cached_masked_block is not None
            and self._cached_q is not None
            and np.allclose(q, self._cached_q, atol=1e-12, rtol=0)
        ):
            return self._cached_masked_block
        block = masked_null_space_projector_block(
            configuration,
            self._frames,
            self._v_indices,
            damping=self.projector_damping,
        )
        self._cached_q = q.copy()
        self._cached_masked_block = block
        self._cached_projector = None
        return block

    def _compute_projector(self, configuration: Configuration) -> np.ndarray:
        q = configuration.q
        if self._cached_projector is not None and np.allclose(q, self._cached_q, atol=1e-12, rtol=0):
            return self._cached_projector

        nv = configuration.model.nv
        if not self._frames:
            projector = np.eye(nv, dtype=float)
        elif self._v_indices is not None and self._v_indices.size > 0:
            projector = masked_null_space_projector(
                configuration,
                self._frames,
                self._v_indices,
                damping=self.projector_damping,
            )
        else:
            jacobian = np.vstack(
                [configuration.get_frame_jacobian(frame) for frame in self._frames]
            )
            k = jacobian.shape[0]
            gram = jacobian @ jacobian.T + (self.projector_damping**2) * np.eye(k)
            pinv = np.linalg.solve(gram, jacobian).T
            projector = np.eye(nv, dtype=float) - pinv @ jacobian

        self._cached_q = q.copy()
        self._cached_projector = projector
        self._cached_masked_block = None
        return projector

    def _scale_error(self, configuration: Configuration, error: np.ndarray) -> np.ndarray:
        if self._dt is None:
            return error
        v_max = np.asarray(configuration.model.velocityLimit, dtype=float)
        step = self.gain * np.abs(error)
        budget = self.velocity_limit_scale * v_max * self._dt
        valid = np.isfinite(budget) & (budget > 0.0)
        if not np.any(valid):
            return error
        ratio = float(np.max(step[valid] / budget[valid]))
        if ratio > 1.0:
            return error / ratio
        return error

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.target_q is None:
            raise TargetNotSet("no posture target")
        _, root_nv = get_root_joint_dim(configuration.model)
        nv = configuration.model.nv
        diff = pin.difference(
            configuration.model,
            self.target_q,
            configuration.q,
        )
        block = self._masked_projector_block(configuration)
        if block is not None:
            v_idx = self._v_indices
            error = np.zeros(nv, dtype=float)
            error[v_idx] = block @ diff[v_idx]
        else:
            error = self._compute_projector(configuration) @ diff
        error = self._scale_error(configuration, error)
        return error[root_nv:]

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        _, root_nv = get_root_joint_dim(configuration.model)
        nv = configuration.model.nv
        block = self._masked_projector_block(configuration)
        if block is not None:
            v_idx = self._v_indices
            jacobian = np.zeros((nv - root_nv, nv), dtype=float)
            for local_i, vi in enumerate(v_idx):
                if vi < root_nv:
                    continue
                jacobian[vi - root_nv, v_idx] = block[local_i, :]
            return jacobian
        return self._compute_projector(configuration)[root_nv:, :]

    def __repr__(self) -> str:
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
