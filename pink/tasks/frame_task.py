#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Frame task implementation."""

from typing import Literal, Optional, Sequence, Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import TargetNotSet, TaskDefinitionError
from .task import Task

CostFrame = Literal["local", "world"]


class FrameTask(Task):
    r"""Regulate the pose of a robot frame in the world frame.

    Attributes:
        frame: Frame name, typically the name of a link or joint from the robot
            description.
        cost_frame: Frame in which anisotropic ``position_cost`` /
            ``orientation_cost`` axes are interpreted. ``"local"`` (default)
            uses body axes of the task frame; ``"world"`` uses axes parallel to
            the inertial frame (Pinocchio ``LOCAL_WORLD_ALIGNED``).
        transform_target_to_world: Target pose of the frame.

    Costs are designed so that errors with varying SI units, here position and
    orientation displacements, can be cast to homogeneous values. For example,
    if task "foo" has a position cost of 1.0 and task "bar" a position cost of
    0.1, then a 1 [cm] error in task "foo" costs as much as a 10 [cm] error in
    task "bar".

    Dimensionally, the 6D cost vector is a (normalized) force screw and our
    objective function is a (normalized) energy.

    See also:
        The :class:`RelativeFrameTask` also regulates the pose of a robot
        frame, but its target can be expressed in an arbitrary frame (inertial
        or mobile) and not necessarily in the (inertial) world frame.
    """

    frame: str
    cost_frame: CostFrame
    transform_target_to_world: Optional[pin.SE3]

    def __init__(
        self,
        frame: str,
        position_cost: Union[float, Sequence[float]],
        orientation_cost: Union[float, Sequence[float]],
        lm_damping: float = 0.0,
        gain: float = 1.0,
        cost_frame: CostFrame = "local",
    ) -> None:
        r"""Define a new frame task.

        Args:
            frame: Frame name, typically the name of a link or joint from the
                robot description.
            position_cost: Contribution of position errors to the normalized
                cost, in :math:`[\mathrm{cost}] / [\mathrm{m}]`. If this is a
                vector, the cost is anisotropic and each coordinate corresponds
                to an axis of ``cost_frame``.
            orientation_cost: Contribution of orientation errors to the
                normalized cost, in :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
                If this is a vector, the cost is anisotropic and each
                coordinate corresponds to an axis of ``cost_frame``.
            lm_damping: Levenberg-Marquardt damping (see class attributes). The
                default value is conservatively low.
            gain: Task gain :math:`\alpha \in [0, 1]` for additional low-pass
                filtering. Defaults to 1.0 (no filtering) for dead-beat
                control.
            cost_frame: ``"local"`` to weight body-frame twist coordinates
                (Pink default), or ``"world"`` to weight world-aligned
                coordinates so that e.g. ``position_cost=[0, 0, 1]`` always
                means world :math:`z`.
        """
        if cost_frame not in ("local", "world"):
            raise TaskDefinitionError(
                f"cost_frame must be 'local' or 'world', got {cost_frame!r}"
            )
        super().__init__(
            cost=np.ones(6),  # updated below
            gain=gain,
            lm_damping=lm_damping,
        )
        self.frame = frame
        self.cost_frame = cost_frame
        self.lm_damping = lm_damping
        self.transform_target_to_world = None
        #
        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(
        self, position_cost: Union[float, Sequence[float], np.ndarray]
    ) -> None:
        r"""Set a new cost for all 3D position coordinates.

        Args:
            position_cost: Contribution of position errors to the normalized
                cost, in :math:`[\mathrm{cost}] / [\mathrm{m}]`. If this is a
                vector, the cost is anisotropic and each coordinate corresponds
                to an axis of :attr:`cost_frame`.
        """
        if isinstance(position_cost, float):
            assert position_cost >= 0.0
        else:  # not isinstance(position_cost, float)
            assert all(cost >= 0.0 for cost in position_cost)
        if isinstance(self.cost, np.ndarray):
            self.cost[0:3] = position_cost
        else:  # self.cost is not a vector
            raise TaskDefinitionError(
                "Frame task cost should be a vector, "
                f"currently cost={self.cost}"
            )

    def set_orientation_cost(
        self, orientation_cost: Union[float, Sequence[float], np.ndarray]
    ) -> None:
        r"""Set a new cost for all 3D orientation coordinates.

        Args:
            orientation_cost: Contribution of orientation errors to the
                normalized cost, in :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
                If this is a vector, the cost is anisotropic and each
                coordinate corresponds to an axis of :attr:`cost_frame`.
        """
        if isinstance(orientation_cost, float):
            assert orientation_cost >= 0.0
        else:  # not isinstance(orientation_cost, float)
            assert all(cost >= 0.0 for cost in orientation_cost)
        if isinstance(self.cost, np.ndarray):
            self.cost[3:6] = orientation_cost
        else:  # self.cost is not a vector
            raise TaskDefinitionError(
                "Frame task cost should be a vector, "
                f"currently cost={self.cost}"
            )

    @staticmethod
    def _align_twist(twist: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Map a body twist to world-aligned coordinates via ``diag(R, R)``."""
        return np.hstack((rotation @ twist[:3], rotation @ twist[3:]))

    def set_target(self, transform_target_to_world: pin.SE3) -> None:
        """Set task target pose in the world frame.

        Args:
            transform_target_to_world: Transform from the task target frame to
                the world frame.
        """
        self.transform_target_to_world = transform_target_to_world.copy()

    def set_target_from_configuration(
        self, configuration: Configuration
    ) -> None:
        """Set task target pose from a robot configuration.

        Args:
            configuration: Robot configuration.
        """
        self.set_target(configuration.get_transform_frame_to_world(self.frame))

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute frame task error.

        By default (``cost_frame="local"``) this error is a twist
        :math:`e(q) \in se(3)` expressed in the local frame (i.e. it is a
        *body* twist). With ``cost_frame="world"`` the same body twist is
        mapped to world-aligned coordinates
        :math:`e_{\mathrm{lwa}} = \mathrm{diag}(R_{0b}, R_{0b})\, e_b` so that
        anisotropic costs apply along inertial axes.

        We map the twist to :math:`\mathbb{R}^6` using Pinocchio's convention
        where linear coordinates are followed by angular coordinates.

        The body error is the right-minus difference between the target pose
        :math:`T_{0t}` and current frame pose :math:`T_{0b}`:

        .. math::

            e(q) := {}_b \xi_{0b} = -(T_{t0} \boxminus T_{b0})
            = -\log(T_{t0} \cdot T_{0b}) = -\log(T_{tb}) = \log(T_{bt})

        where :math:`b` denotes our frame, :math:`t` the target frame and
        :math:`0` the inertial frame.

        See :func:`pink.tasks.task.Task.compute_error` for more context, and
        [Sola2018]_ for details on the right-minus operator.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Frame task error :math:`e(q)`.
        """
        if self.transform_target_to_world is None:
            raise TargetNotSet(f"no target set for frame '{self.frame}'")
        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.frame
        )
        transform_target_to_frame = transform_frame_to_world.actInv(
            self.transform_target_to_world
        )
        # Reminder: log(transform_bar_to_foo) is always a twist in frame "foo".
        # Here the error is such that the new frame after integration of the
        # stationary body twist coincides with the target:
        #
        #     transform_new_to_world = transform_frame_to_world * exp(error)
        #         = transform_frame_to_world * transform_target_to_frame
        #         = transform_target_to_frame
        #
        error_in_frame: np.ndarray = pin.log(transform_target_to_frame).vector
        if self.cost_frame == "world":
            return self._align_twist(
                error_in_frame, transform_frame_to_world.rotation
            )
        return error_in_frame

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the frame task Jacobian.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{6 \times n_v}` is the
        derivative of the task error :math:`e(q) \in \mathbb{R}^6` with respect
        to the configuration :math:`q`. The formula for the local frame task
        is:

        .. math::

            J(q) = -\text{Jlog}_6(T_{tb}) {}_b J_{0b}(q)

        With ``cost_frame="world"``, rows are left-multiplied by
        :math:`\mathrm{diag}(R_{0b}, R_{0b})` to match the world-aligned
        error.

        The derivation of the formula for this Jacobian is detailed in
        [Caron2023]_. See also
        :func:`pink.tasks.task.Task.compute_jacobian` for more context on task
        Jacobians.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix :math:`J`, in the same frame as
            :func:`compute_error`.
        """
        if self.transform_target_to_world is None:
            raise TargetNotSet(f"no target set for frame '{self.frame}'")
        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.frame
        )
        transform_frame_to_target = self.transform_target_to_world.actInv(
            transform_frame_to_world
        )
        jacobian_in_frame = configuration.get_frame_jacobian(self.frame)
        J = -pin.Jlog6(transform_frame_to_target) @ jacobian_in_frame
        if self.cost_frame == "world":
            R = transform_frame_to_world.rotation
            J = np.vstack((R @ J[:3], R @ J[3:]))
        return J

    @property
    def position_cost(self) -> Union[float, np.ndarray]:
        """Get the position cost."""
        if isinstance(self.cost, np.ndarray):
            return self.cost[0:3]
        if isinstance(self.cost, float):
            return self.cost
        raise TaskDefinitionError(
            "Frame task cost should be a vector or a scalar, "
            f"currently cost={self.cost}"
        )

    @property
    def orientation_cost(self) -> Union[float, np.ndarray]:
        """Get the orientation cost."""
        if isinstance(self.cost, np.ndarray):
            return self.cost[3:6]
        if isinstance(self.cost, float):
            return self.cost
        raise TaskDefinitionError(
            "Frame task cost should be a vector or a scalar, "
            f"currently cost={self.cost}"
        )

    def __repr__(self):
        """Human-readable representation of the task."""
        return (
            "FrameTask("
            f"frame={self.frame}, "
            f"cost_frame={self.cost_frame}, "
            f"position_cost={self.position_cost}, "
            f"orientation_cost={self.orientation_cost}, "
            f"lm_damping={self.lm_damping}, "
            f"gain={self.gain})"
        )
