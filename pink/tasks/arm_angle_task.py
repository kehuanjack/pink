#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

r"""Arm-angle (redundancy) task for 7-DOF (or similar) serial arms.

This implements the **shoulder–elbow–wrist (SEW)** redundancy angle in the usual
geometric sense: for fixed shoulder :math:`S`, wrist :math:`W`, and link
lengths :math:`\ell_{SE},\ell_{EW}`, the elbow lies on a circle in the plane
orthogonal to :math:`\hat{s}_w = \overrightarrow{SW}/\|\overrightarrow{SW}\|`.
The scalar angle measures the in-plane direction of
:math:`\overrightarrow{CE}` (chord from circle center :math:`C` to elbow
:math:`E`) in an orthonormal basis :math:`(e_1,e_2)` of that plane.

**Reference direction (gauge)**

The conventional SEW angle (e.g. Hollerbach, 1985; Kreutz-Delgado et al.) takes
a **fixed unit reference vector** :math:`e_r` (often world :math:`+\mathbf{z}`)
and builds a reference normal
:math:`e_y \propto \hat{s}_w \times e_r`,
:math:`e_x = e_y \times \hat{s}_w` in the circle plane. Then :math:`\psi` is
the angle of the SEW plane about the shoulder–wrist line w.r.t. the plane
through :math:`SW` and :math:`e_r`.

By default this task uses :math:`e_r = (0,0,1)` in the world frame (“up”),
which matches that **conventional** convention when
`auxiliary_vector_world` is left as ``None``.

**Singularities**

When :math:`\hat{s}_w` is **parallel** to :math:`e_r`, the reference plane is
undefined (**algorithmic singularity** of conventional SEW). Elias & Wen
(2024, arXiv:2307.13122) discuss **generalized / stereographic** choices of
reference function :math:`f_x(\mathbf{p}_{SW})` to shrink the singular set; this
module does not implement stereographic SEW—use a custom reference or switch
`auxiliary_vector_world` near singularities if needed.

**Modelling**

:math:`\ell_{SE},\ell_{EW}` must be the kinematic distances consistent with
the chosen shoulder / elbow / wrist **points** (frame origins). If URDF frame
origins do not coincide with analytic S/E/W points, either calibrate lengths or
accept that measured :math:`E` is projected radially onto the circle (see
`_arm_angle_from_points`)—a consistent scalar, but not identical to every
paper’s closed-form IK definition for a specific DH chain.

References:

- A. J. Elias and J. T. Wen, “Redundancy parameterization and inverse
  kinematics of 7-DOF revolute manipulators,” arXiv:2307.13122, 2024.
- J. M. Hollerbach, “Optimum kinematic design for a seven degree of freedom
  manipulator,” 1985.
"""

import math
from typing import Optional, Tuple

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import TargetNotSet
from .task import Task


def _wrap_to_pi_scalar(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _sw_plane_basis(
    S: np.ndarray,
    W: np.ndarray,
    aux_world: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    r"""Unit shoulder-to-wrist axis and orthonormal (e1, e2) spanning the elbow circle plane.

    When ``aux_world`` is None, use **conventional SEW** reference
    :math:`e_r = (0,0,1)` and set :math:`e_1 \propto \hat{s}_w \times e_r`,
    :math:`e_2 = \hat{s}_w \times e_1`, matching :math:`e_y,e_x` in Elias &
    Wen (arXiv:2307.13122, Table~1, conventional SEW).

    If ``aux_world`` is set, it is used as a custom :math:`e_r` (not necessarily
    unit—only the direction of the cross products matters until normalization).
    """
    sw = W - S
    L = float(np.linalg.norm(sw))
    if L < 1e-10:
        return None, None, None
    u_sw = sw / L

    if aux_world is None:
        # Conventional SEW: fixed world "up" (Hollerbach 1985; Elias & Wen §3).
        aux = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        aux = np.array(aux_world, dtype=float).reshape(3)
    t = np.cross(u_sw, aux)
    if np.linalg.norm(t) < 1e-10:
        t = np.cross(u_sw, np.array([1.0, 0.0, 0.0], dtype=float))
    if np.linalg.norm(t) < 1e-10:
        t = np.cross(u_sw, np.array([0.0, 1.0, 0.0], dtype=float))
    if np.linalg.norm(t) < 1e-10:
        return u_sw, None, None
    e1 = t / np.linalg.norm(t)
    e2 = np.cross(u_sw, e1)
    e2 = e2 / max(np.linalg.norm(e2), 1e-12)
    return u_sw, e1, e2


def _circle_center_radius(
    S: np.ndarray, W: np.ndarray, l_se: float, l_ew: float
) -> Tuple[Optional[np.ndarray], float, float]:
    """Center on segment S–W and radius of elbow circle (lengths l_se, l_ew)."""
    sw = W - S
    L = float(np.linalg.norm(sw))
    if L < 1e-10:
        return None, 0.0, 0.0
    if l_se <= 0.0 or l_ew <= 0.0:
        return None, 0.0, 0.0
    if L > l_se + l_ew + 1e-9 or L < abs(l_se - l_ew) - 1e-9:
        return None, 0.0, L
    x = (l_se * l_se - l_ew * l_ew + L * L) / (2.0 * L)
    r2 = l_se * l_se - x * x
    if r2 < -1e-8:
        return None, 0.0, L
    r = float(math.sqrt(max(0.0, r2)))
    u_sw = sw / L
    C = S + x * u_sw
    return C, r, L


def _arm_angle_from_points(
    S: np.ndarray,
    E: np.ndarray,
    W: np.ndarray,
    l_se: float,
    l_ew: float,
    aux_world: Optional[np.ndarray] = None,
) -> Tuple[bool, float]:
    """Scalar arm angle theta = atan2((E-C)·e2, (E-C)·e1) on the SEW elbow circle."""
    C, r, _ = _circle_center_radius(S, W, l_se, l_ew)
    if C is None or r < 1e-8:
        return False, 0.0

    _, e1, e2 = _sw_plane_basis(S, W, aux_world)
    if e1 is None or e2 is None:
        return False, 0.0

    v = E - C
    radial = float(np.linalg.norm(v))
    if radial < 1e-10:
        return False, 0.0
    # Numerical drift: re-scale to circle for angle extraction
    v = v * (r / radial)
    c1 = float(np.dot(v, e1))
    c2 = float(np.dot(v, e2))
    if abs(c1) < 1e-12 and abs(c2) < 1e-12:
        return False, 0.0
    return True, float(math.atan2(c2, c1))


class ArmAngleTask(Task):
    r"""Regulate the arm-angle redundancy of an anthropomorphic arm.

    The arm angle :math:`\theta` parametrizes the elbow location on the circle
    consistent with shoulder :math:`S`, wrist :math:`W`, upper-arm length
    :math:`\ell_{SE}` and forearm length :math:`\ell_{EW}` (triangle SEW). A
    plane orthogonal to :math:`\overrightarrow{SW}` and a reference plane
    (fixed :math:`e_r`, default world :math:`+\mathbf{z}`, per conventional SEW)
    define :math:`\theta = \mathrm{atan2}((E-C)\cdot e_2, (E-C)\cdot e_1)` where
    :math:`E` is the elbow origin and :math:`C` is the circle center.

    **Frames**

    Select URDF frames whose origins approximate shoulder, elbow, and wrist
    centers (for wrist, prefer a frame placed at the kinematic wrist center if
    available).

    **Jacobian**

    The task Jacobian is assembled via the geometric chain rule
    :math:`J_\theta = (\partial\theta/\partial S)\,J_S
    + (\partial\theta/\partial E)\,J_E
    + (\partial\theta/\partial W)\,J_W`, where :math:`J_S,J_E,J_W` are
    translational frame Jacobians from :class:`Configuration` and
    :math:`\partial\theta/\partial S` etc. are obtained by symmetric finite
    differences on the three SEW points only (nine evaluations of
    :func:`_arm_angle_from_points`).

    Attributes:
        shoulder_frame: Name of the shoulder link/joint frame in the model.
        elbow_frame: Name of the elbow link/joint frame.
        wrist_frame: Name of the wrist / wrist-center frame.
        upper_arm_length: Nominal upper-arm length :math:`\ell_{SE}` [m].
        forearm_length: Nominal forearm length :math:`\ell_{EW}` [m].
        target_theta: Target arm angle [rad].
    """

    target_theta: Optional[float]

    def __init__(
        self,
        shoulder_frame: str,
        elbow_frame: str,
        wrist_frame: str,
        upper_arm_length: float,
        forearm_length: float,
        cost: float = 1.0,
        lm_damping: float = 0.0,
        gain: float = 1.0,
        finite_difference_step: float = 1e-5,
        auxiliary_vector_world: Optional[np.ndarray] = None,
    ) -> None:
        r"""Initialize task.

        Args:
            shoulder_frame: Shoulder frame name.
            elbow_frame: Elbow frame name.
            wrist_frame: Wrist frame name.
            upper_arm_length: Upper-arm length ℓ_SE [m].
            forearm_length: Forearm length ℓ_EW [m].
            cost: Cost weight in [cost] / [rad].
            lm_damping: Levenberg-Marquardt damping (see :class:`Task`).
            gain: Task gain α.
            finite_difference_step: Step size for Jacobian finite differencing
                in tangent space.
            auxiliary_vector_world: Optional reference vector e_r in the world
                frame; builds e1 ∝ ŝ_w × e_r. If ``None``, uses (0, 0, 1).
        """
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.shoulder_frame = shoulder_frame
        self.elbow_frame = elbow_frame
        self.wrist_frame = wrist_frame
        self.upper_arm_length = float(upper_arm_length)
        self.forearm_length = float(forearm_length)
        self.finite_difference_step = float(finite_difference_step)
        self.auxiliary_vector_world = (
            None
            if auxiliary_vector_world is None
            else np.array(auxiliary_vector_world, dtype=float).reshape(3).copy()
        )
        self.target_theta = None

        self._shoulder_fid = -1
        self._elbow_fid = -1
        self._wrist_fid = -1

    def _resolve_frame_ids(self, model: pin.Model) -> None:
        if self._shoulder_fid >= 0:
            return
        self._shoulder_fid = model.getFrameId(self.shoulder_frame)
        self._elbow_fid = model.getFrameId(self.elbow_frame)
        self._wrist_fid = model.getFrameId(self.wrist_frame)

    def set_target(self, theta: float) -> None:
        """Set target arm angle [rad]."""
        self.target_theta = float(theta)

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set target to the current arm angle at this configuration."""
        ok, th = self._theta_from_configuration(configuration)
        if not ok:
            raise ValueError(
                "arm angle is ill-defined at this configuration "
                "(triangle / circle degeneracy)"
            )
        self.target_theta = th

    def _theta_from_model(
        self, model: pin.Model, data: pin.Data, q: np.ndarray
    ) -> Tuple[bool, float]:
        self._resolve_frame_ids(model)
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        S = np.array(data.oMf[self._shoulder_fid].translation, dtype=float)
        E = np.array(data.oMf[self._elbow_fid].translation, dtype=float)
        W = np.array(data.oMf[self._wrist_fid].translation, dtype=float)
        return _arm_angle_from_points(
            S,
            E,
            W,
            self.upper_arm_length,
            self.forearm_length,
            self.auxiliary_vector_world,
        )

    def _theta_from_configuration(
        self, configuration: Configuration
    ) -> Tuple[bool, float]:
        return self._theta_from_model(
            configuration.model, configuration.data, configuration.q
        )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.target_theta is None:
            raise TargetNotSet("no arm-angle target; call set_target or set_target_from_configuration")
        ok, th = self._theta_from_configuration(configuration)
        if not ok:
            return np.zeros(1)
        e = _wrap_to_pi_scalar(th - self.target_theta)
        return np.array([e], dtype=float)

    def _frame_positions_and_jacobians(
        self, configuration: Configuration
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Shoulder, elbow, wrist positions and translational frame Jacobians.

        Args:
            configuration: Robot configuration :math:`q` with forward kinematics
                already computed.

        Returns:
            Tuple ``(S, E, W, J_S, J_E, J_W)`` where positions are in the world
            frame. Each :math:`J_\*` is the translational Jacobian
            :math:`\partial p / \partial q` in the world frame. Because
            :meth:`Configuration.get_frame_jacobian` uses
            :data:`pin.ReferenceFrame.LOCAL`, the first three rows are rotated
            into the world frame (same convention as
            :class:`~pink.barriers.position_barrier.PositionBarrier`).
        """
        TS = configuration.get_transform_frame_to_world(self.shoulder_frame)
        TE = configuration.get_transform_frame_to_world(self.elbow_frame)
        TW = configuration.get_transform_frame_to_world(self.wrist_frame)
        S = np.array(TS.translation, dtype=float)
        E = np.array(TE.translation, dtype=float)
        W = np.array(TW.translation, dtype=float)
        JS = configuration.get_frame_jacobian(self.shoulder_frame)[:3]
        JE = configuration.get_frame_jacobian(self.elbow_frame)[:3]
        JW = configuration.get_frame_jacobian(self.wrist_frame)[:3]
        JS = TS.rotation @ JS
        JE = TE.rotation @ JE
        JW = TW.rotation @ JW
        return S, E, W, JS, JE, JW

    def _theta_gradient_points(
        self,
        S: np.ndarray,
        E: np.ndarray,
        W: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Gradients of arm angle w.r.t. shoulder, elbow, and wrist positions.

        Symmetric finite differences perturb each world-frame coordinate of
        :math:`S`, :math:`E`, and :math:`W` (eighteen calls to
        :func:`_arm_angle_from_points`).

        Args:
            S: Shoulder position in the world frame.
            E: Elbow position in the world frame.
            W: Wrist position in the world frame.

        Returns:
            Tuple ``(g_S, g_E, g_W)`` with each gradient shaped ``(3,)``.
        """
        h = self.finite_difference_step
        l_se = self.upper_arm_length
        l_ew = self.forearm_length
        aux = self.auxiliary_vector_world

        gS = np.zeros(3, dtype=float)
        gE = np.zeros(3, dtype=float)
        gW = np.zeros(3, dtype=float)
        if h <= 0.0:
            return gS, gE, gW

        for axis in range(3):
            S_p = S.copy()
            S_m = S.copy()
            S_p[axis] += h
            S_m[axis] -= h
            ok_p, th_p = _arm_angle_from_points(S_p, E, W, l_se, l_ew, aux)
            ok_m, th_m = _arm_angle_from_points(S_m, E, W, l_se, l_ew, aux)
            if ok_p and ok_m:
                gS[axis] = _wrap_to_pi_scalar(th_p - th_m) / (2.0 * h)

            E_p = E.copy()
            E_m = E.copy()
            E_p[axis] += h
            E_m[axis] -= h
            ok_p, th_p = _arm_angle_from_points(S, E_p, W, l_se, l_ew, aux)
            ok_m, th_m = _arm_angle_from_points(S, E_m, W, l_se, l_ew, aux)
            if ok_p and ok_m:
                gE[axis] = _wrap_to_pi_scalar(th_p - th_m) / (2.0 * h)

            W_p = W.copy()
            W_m = W.copy()
            W_p[axis] += h
            W_m[axis] -= h
            ok_p, th_p = _arm_angle_from_points(S, E, W_p, l_se, l_ew, aux)
            ok_m, th_m = _arm_angle_from_points(S, E, W_m, l_se, l_ew, aux)
            if ok_p and ok_m:
                gW[axis] = _wrap_to_pi_scalar(th_p - th_m) / (2.0 * h)

        return gS, gE, gW

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        nv = configuration.model.nv
        J = np.zeros((1, nv))

        if self.finite_difference_step <= 0.0:
            return J

        S, E, W, JS, JE, JW = self._frame_positions_and_jacobians(configuration)
        ok0, _ = _arm_angle_from_points(
            S,
            E,
            W,
            self.upper_arm_length,
            self.forearm_length,
            self.auxiliary_vector_world,
        )
        if not ok0:
            return J

        gS, gE, gW = self._theta_gradient_points(S, E, W)
        J[0, :] = gS @ JS + gE @ JE + gW @ JW
        return J

    def __repr__(self) -> str:
        return (
            "ArmAngleTask("
            f"shoulder_frame={self.shoulder_frame!r}, "
            f"elbow_frame={self.elbow_frame!r}, "
            f"wrist_frame={self.wrist_frame!r}, "
            f"upper_arm_length={self.upper_arm_length}, "
            f"forearm_length={self.forearm_length}, "
            f"cost={self.cost}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )
