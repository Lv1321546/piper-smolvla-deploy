"""Regression for the 8.4ms live callback burst that stopped XYZ v2."""
from pathlib import Path
import unittest

import numpy as np

from servo_bridge.candidate_cadence import CandidateCadence
from servo_bridge.core import Feedback, NAMES, load_config
from servo_bridge.execution import VelocityTarget
from servo_bridge.sdk_worker import Worker


class Writer:
    def joint(self, raw):
        pass


class CandidateCadenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[1]/'config.cartesian.example.json')
        self.gate = CandidateCadence(self.cfg['publish_period_s'])

    def test_actual_burst_skips_integration_without_relaxing_executor(self):
        executor = VelocityTarget(self.cfg)
        q = np.radians(self.cfg['sim_initial_joints_deg'])
        sent = []
        for now in (1., 1.020, 1.02842, 1.048, 1.068):
            if self.gate.should_defer(now):
                self.assertAlmostEqual(now, 1.02842)
                self.assertEqual(executor.last_time, 1.020)
                continue
            _, q, _ = executor.step(NAMES, [.03, 0, 0, 0, 0, 0],
                                    feedback=Feedback(q.copy(), now, ready=True), now=now,
                                    stamp_age_s=0., horizon_s=.02)
            self.gate.commit(now)
            sent.append(now)
        self.assertEqual(sent, [1., 1.020, 1.048, 1.068])

    def test_bursts_and_receipt_jitter_pass_independent_worker_guards(self):
        executor = VelocityTarget(self.cfg)
        q = np.radians(self.cfg['sim_initial_joints_deg'])
        receipt = [0.]
        worker = Worker(self.cfg, Writer(), clock=lambda: receipt[0])
        transmitted = 0
        for index, now in enumerate(1.+np.arange(200)*.02):
            # A late callback followed by an early callback reproduces the
            # original failure; other callbacks remain near the nominal rate.
            if index % 10 == 0:
                now += .012
            if self.gate.should_defer(float(now)):
                continue
            _, target, velocity = executor.step(NAMES, [.04, 0, 0, 0, 0, 0],
                feedback=Feedback(q.copy(), now, ready=True), now=now, stamp_age_s=0., horizon_s=.02)
            receipt[0] = now + (.003 if index % 2 else 0.)
            worker.packet(dict(version=1, monotonic=float(now), servo_status=0,
                status_monotonic=float(now), deadman=True, joint_names=list(NAMES),
                positions=target.tolist(), velocities=velocity.tolist(), horizon_s=.02),
                Feedback(q.copy(), receipt[0], ready=True))
            q = target.copy()
            self.gate.commit(float(now))
            transmitted += 1
        self.assertGreater(transmitted, 150)

    def test_true_timeout_is_not_hidden(self):
        executor = VelocityTarget(self.cfg)
        q = np.radians(self.cfg['sim_initial_joints_deg'])
        executor.step(NAMES, [0.]*6, feedback=Feedback(q, 1., ready=True), now=1., stamp_age_s=0., horizon_s=.02)
        self.gate.commit(1.)
        self.assertFalse(self.gate.should_defer(1.2))
        with self.assertRaisesRegex(ValueError, 'cadence'):
            executor.step(NAMES, [0.]*6, feedback=Feedback(q, 1.2, ready=True), now=1.2, stamp_age_s=0., horizon_s=.02)

    def test_bad_clock_is_rejected(self):
        self.gate.commit(1.)
        for now in (.9, float('nan'), float('inf')):
            with self.subTest(now=now), self.assertRaises(ValueError):
                self.gate.should_defer(now)


if __name__ == '__main__':
    unittest.main()
