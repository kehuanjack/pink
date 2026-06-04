#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ArmAngleTask.

Uses the same 7-DOF manipulator as :doc:`../examples/arm_panda` ---
``panda_description`` from the ``robot_descriptions`` package (Pinocchio
loader), which is a dependency of the upstream Pink development environment
(:file:`pyproject.toml` / Pixi workspace).
"""

import unittest

import numpy as np
import pinocchio as pin

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ImportError:  # pragma: no cover
    load_robot_description = None

from pink import Configuration
from pink.exceptions import TargetNotSet
from pink.tasks import ArmAngleTask
from pink.utils import custom_configuration_vector

_SKIP = load_robot_description is None

# Canonical frames on Franka Emika Panda (example-robot-data / panda_description.)
_SHOULDER = "panda_link2"
_ELBOW = "panda_link4"
_WRIST = "panda_link6"


class TestArmAngleTask(unittest.TestCase):
    """Test ArmAngleTask on the library's example 7-DOF Panda model."""

    @staticmethod
    def _nominal_lengths_from_frames(
        model: pin.Model, data: pin.Data, q: np.ndarray
    ) -> tuple[float, float]:
        """Match analytic arm-angle geometry to Pinocchio frame origins."""
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        ids = (
            model.getFrameId(_SHOULDER),
            model.getFrameId(_ELBOW),
            model.getFrameId(_WRIST),
        )
        S = np.array(data.oMf[ids[0]].translation, dtype=float)
        E = np.array(data.oMf[ids[1]].translation, dtype=float)
        W = np.array(data.oMf[ids[2]].translation, dtype=float)
        l_se = float(np.linalg.norm(E - S))
        l_ew = float(np.linalg.norm(W - E))
        return l_se, l_ew

    def _load_panda_bent(self) -> tuple[pin.Model, pin.Data, np.ndarray]:
        assert load_robot_description is not None
        robot = load_robot_description("panda_description", root_joint=None)
        # Bent pose so the SEW elbow circle has non-zero radius.
        q = custom_configuration_vector(
            robot,
            panda_joint1=0.1,
            panda_joint2=-0.8,
            panda_joint3=0.2,
            panda_joint4=-2.0,
            panda_joint5=0.3,
            panda_joint6=1.8,
            panda_joint7=0.0,
        )
        return robot.model, robot.data, q

    @unittest.skipIf(
        _SKIP,
        "robot_descriptions not installed (install per pink/pyproject.toml dev deps)",
    )
    def test_zero_error_at_recorded_target(self):
        model, data, q = self._load_panda_bent()
        configuration = Configuration(model, data, q)
        l_se, l_ew = self._nominal_lengths_from_frames(model, data, q)
        task = ArmAngleTask(
            shoulder_frame=_SHOULDER,
            elbow_frame=_ELBOW,
            wrist_frame=_WRIST,
            upper_arm_length=l_se,
            forearm_length=l_ew,
            cost=1.0,
            finite_difference_step=1e-4,
        )
        task.set_target_from_configuration(configuration)
        e = task.compute_error(configuration)
        self.assertAlmostEqual(float(e[0]), 0.0, places=5)

    @unittest.skipIf(_SKIP, "robot_descriptions not installed")
    def test_jacobian_shape(self):
        model, data, q = self._load_panda_bent()
        configuration = Configuration(model, data, q)
        l_se, l_ew = self._nominal_lengths_from_frames(model, data, q)
        task = ArmAngleTask(
            shoulder_frame=_SHOULDER,
            elbow_frame=_ELBOW,
            wrist_frame=_WRIST,
            upper_arm_length=l_se,
            forearm_length=l_ew,
            finite_difference_step=1e-4,
        )
        task.set_target_from_configuration(configuration)
        J = task.compute_jacobian(configuration)
        self.assertEqual(J.shape, (1, model.nv))
        self.assertTrue(np.all(np.isfinite(J)))

    def test_target_not_set_raises(self):
        model = pin.Model()
        model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "j1")
        data = model.createData()
        q = pin.neutral(model)
        configuration = Configuration(model, data, q)
        task = ArmAngleTask(
            shoulder_frame="universe",
            elbow_frame="universe",
            wrist_frame="universe",
            upper_arm_length=0.3,
            forearm_length=0.3,
        )
        with self.assertRaises(TargetNotSet):
            task.compute_error(configuration)
