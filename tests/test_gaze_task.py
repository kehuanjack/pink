#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gaze task."""

import unittest

import numpy as np
import pinocchio as pin

from pink import Configuration
from pink.exceptions import TargetNotSet
from pink.tasks import GazeTask


class TestGazeTask(unittest.TestCase):
    """Check gaze geometry, Jacobian, masking and target handling."""

    def setUp(self):
        model = pin.Model()
        pitch_id = model.addJoint(
            0,
            pin.JointModelRY(),
            pin.SE3.Identity(),
            "pitch",
        )
        model.appendBodyToJoint(
            pitch_id,
            pin.Inertia.Random(),
            pin.SE3.Identity(),
        )
        yaw_id = model.addJoint(
            pitch_id,
            pin.JointModelRZ(),
            pin.SE3(np.eye(3), np.array([0.2, 0.0, 0.0])),
            "yaw",
        )
        model.appendBodyToJoint(
            yaw_id,
            pin.Inertia.Random(),
            pin.SE3.Identity(),
        )
        model.addFrame(
            pin.Frame(
                "camera",
                yaw_id,
                0,
                pin.SE3(np.eye(3), np.array([0.1, 0.02, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )
        self.model = model
        self.q = np.array([0.2, -0.3])
        self.configuration = Configuration(model, model.createData(), self.q)
        self.task = GazeTask(
            "camera",
            model,
            ["pitch", "yaw"],
            offset=[0.01, 0.0, 0.0],
        )

    def test_target_not_set(self):
        with self.assertRaises(TargetNotSet):
            self.task.compute_error(self.configuration)
        with self.assertRaises(TargetNotSet):
            self.task.compute_jacobian(self.configuration)

    def test_zero_error_on_optical_axis(self):
        transform = self.configuration.get_transform_frame_to_world("camera")
        origin = transform.translation + transform.rotation @ self.task.offset
        axis = transform.rotation @ self.task.optical_axis
        self.task.set_target(origin + axis)
        self.assertLess(
            np.linalg.norm(self.task.compute_error(self.configuration)),
            1e-12,
        )

    def test_analytic_jacobian_matches_finite_difference(self):
        self.task.set_target([1.0, 0.4, 0.6])
        jacobian = self.task.compute_jacobian(self.configuration)
        jacobian_fd = np.empty_like(jacobian)
        epsilon = 1e-7
        for idx_v in range(self.model.nv):
            displacement = np.zeros(self.model.nv)
            displacement[idx_v] = epsilon
            q_plus = pin.integrate(self.model, self.q, displacement)
            q_minus = pin.integrate(self.model, self.q, -displacement)
            config_plus = Configuration(self.model, self.model.createData(), q_plus)
            config_minus = Configuration(self.model, self.model.createData(), q_minus)
            jacobian_fd[:, idx_v] = (
                self.task.compute_error(config_plus)
                - self.task.compute_error(config_minus)
            ) / (2.0 * epsilon)
        self.assertTrue(np.allclose(jacobian, jacobian_fd, atol=1e-6))

    def test_jacobian_masks_unselected_joint(self):
        task = GazeTask("camera", self.model, ["yaw"])
        task.set_target([1.0, 0.4, 0.6])
        jacobian = task.compute_jacobian(self.configuration)
        self.assertTrue(np.allclose(jacobian[:, 0], 0.0))
        self.assertGreater(np.linalg.norm(jacobian[:, 1]), 1e-6)

    def test_lm_damping_is_restricted_to_selected_joints(self):
        task = GazeTask(
            "camera",
            self.model,
            ["yaw"],
            lm_damping=1.0,
        )
        task.set_target([1.0, 0.4, 0.6])
        hessian, linear = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(hessian[0], 0.0))
        self.assertTrue(np.allclose(hessian[:, 0], 0.0))
        self.assertAlmostEqual(linear[0], 0.0)

    def test_close_target_disables_error_and_jacobian(self):
        task = GazeTask(
            "camera",
            self.model,
            ["pitch", "yaw"],
            offset=[0.01, 0.0, 0.0],
            min_target_distance=0.1,
        )
        transform = self.configuration.get_transform_frame_to_world("camera")
        origin = transform.translation + transform.rotation @ task.offset
        task.set_target(origin)
        self.assertTrue(np.allclose(task.compute_error(self.configuration), 0.0))
        self.assertTrue(np.allclose(task.compute_jacobian(self.configuration), 0.0))

    def test_rejects_anti_parallel_null_of_cross_product(self):
        """Cross product is zero at 180 deg; direction-axis error must stay non-zero."""
        transform = self.configuration.get_transform_frame_to_world("camera")
        origin = transform.translation + transform.rotation @ self.task.offset
        axis = transform.rotation @ self.task.optical_axis
        self.task.set_target(origin - axis)
        error = self.task.compute_error(self.configuration)
        self.assertGreater(np.linalg.norm(error), 1.0)
        self.assertAlmostEqual(float(np.dot(axis, -axis)), -1.0)


if __name__ == "__main__":
    unittest.main()
