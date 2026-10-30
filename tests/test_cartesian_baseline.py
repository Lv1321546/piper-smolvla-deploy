"""Historical controller comparison and per-cycle diagnostics; synthetic only."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from e14.gamepad_control import GamepadControl
from tests import test_gamepad as fixtures


class BaselineTests(unittest.TestCase):
    setUp = fixtures.GamepadTests.setUp
    tearDown = fixtures.GamepadTests.tearDown
    state = fixtures.GamepadTests.state
    advance = fixtures.GamepadTests.advance
    activate = fixtures.GamepadTests.activate
    make_session = fixtures.GamepadTests.make_session

    def test_measured_reference_matches_historical_controller_on_identical_feedback(self):
        data=json.loads((Path(__file__).parent/'fixtures/cartesian_3e30c52_low.json').read_text())
        for case in data['cases']:
            with self.subTest(initial=case['initial']):
                self.backend.state=case['initial'].copy();self.backend.refresh()
                self.core=GamepadControl(self.robot,{**data['settings'],'cartesian_reference':'measured'},
                                         execute=False,clock=self.clock)
                self.activate()
                for frame in case['frames']:
                    self.backend.state=frame['actual'].copy();self.backend.refresh()
                    event=self.advance(lb=True,**frame['axes'])
                    if frame['expected'] is not None:
                        self.assertIsNotNone(event)
                        np.testing.assert_allclose(event['target'],frame['expected'],atol=1e-9,rtol=0)
                self.assertFalse(self.backend.commands)

    def test_measured_reference_does_not_integrate_xyz_ahead_of_stationary_feedback(self):
        self.backend.joint=lambda raw:self.backend.commands.append(('joint',list(raw)))
        self.core=GamepadControl(self.robot,{'linear_speed_m_s':.01,'cartesian_reference':'measured'},
                                 execute=True,clock=self.clock)
        self.activate()
        for _ in range(30):
            self.advance(lb=True,left_y=-1.)
            np.testing.assert_allclose(self.core.diagnostics['reference_tcp_m'],
                                       self.core.diagnostics['measured_tcp_m'],atol=1e-12)
        targets=[command[1] for command in self.backend.commands]
        self.assertGreater(len(targets),10)
        np.testing.assert_allclose(targets[-1],targets[-10],atol=0)
        self.assertFalse(self.core.faulted)

    def test_trace_records_input_targets_feedback_and_reference_without_recording(self):
        self.robot.config['gamepad']={'cartesian_reference':'measured','linear_speed_m_s':.01}
        session,dashboard,pad,_=self.make_session()
        session.core.execute=True
        stream=io.StringIO();session.trace_stream=stream
        pad.current=self.state();session.poll()
        self.clock.sleep(.05);pad.current=self.state(lb=True,left_y=-.5);session.poll()
        cycles=[json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(cycles),2)
        cycle=cycles[-1]
        self.assertEqual(cycle['input']['axes']['left_y'],-.5)
        self.assertEqual(cycle['planner']['cartesian_reference'],'measured')
        self.assertEqual(cycle['target_after_deg'],self.robot.last_joint_target)
        self.assertEqual(cycle['measured_after_deg'],self.backend.state[:6])
        self.assertEqual(cycle['feedback_after']['enabled'],[True]*6)
        self.assertIsNotNone(cycle['proposal'])
        self.assertFalse(cycle['faulted'])
        self.assertEqual(dashboard.state['teleop_cartesian_reference'],'measured')
        self.assertIn('measured-baseline',dashboard.state['teleop_revision'])
        self.assertIsNone(session.event_stream)

    def test_measured_reference_keeps_sender_limits_under_motor_lag_and_reversals(self):
        self.backend.state=[-6.282,16.686,-15.465,3.904,27.699,1.874,.0004]
        self.backend.refresh()
        self.core=GamepadControl(self.robot,{'linear_speed_m_s':.03,'angular_speed_rad_s':.15,
                                            'cartesian_reference':'measured'},execute=True,clock=self.clock)
        desired=np.array(self.backend.state[:6]);last=[self.clock()]
        refresh_original=self.backend.refresh;sent=[]
        def refresh():
            if self.clock()>last[0]:
                q=np.array(self.backend.state[:6])
                self.backend.state[:6]=(q+np.clip((desired-q)*.2,-.5,.5)).tolist()
                last[0]=self.clock()
            refresh_original()
        def joint(raw):
            target=np.array(raw)/1000
            self.assertTrue(np.all(np.abs(target-self.backend.state[:6])<=1.+1e-9))
            if sent:self.assertTrue(np.all(np.abs(target-sent[-1])<=1.+1e-9))
            sent.append(target.copy());desired[:]=target
        self.backend.refresh=refresh;self.backend.joint=joint;self.activate()
        random=np.random.default_rng(19)
        for tick in range(180):
            if tick%5==0:axes=random.choice([-1.,0.,1.],size=6)
            self.clock.sleep(.05)
            state=self.state(lb=True,left_x=axes[0],left_y=axes[1],right_y=axes[2],right_x=axes[3])
            state.hat=tuple(axes[4:].astype(int));self.core.tick(state)
        self.assertGreater(len(sent),50)
        self.assertFalse(self.core.faulted)
        self.advance(lb=False)

    def test_trace_records_original_error_and_locked_state(self):
        session,_,pad,_=self.make_session()
        stream=io.StringIO();session.trace_stream=stream
        pad.current=self.state()
        with patch.object(session.core,'tick',side_effect=RuntimeError('diagnostic rejection')):
            session.poll()
        cycle=json.loads(stream.getvalue())
        self.assertEqual(cycle['error'],'diagnostic rejection')
        self.assertTrue(cycle['faulted'])
        self.assertFalse(self.backend.commands)

    def test_invalid_nan_input_is_loggable_without_masking_controller_fault(self):
        session,_,pad,_=self.make_session()
        stream=io.StringIO();session.trace_stream=stream
        pad.current=self.state(lb=True,left_y=float('nan'));session.poll()
        cycle=json.loads(stream.getvalue())
        self.assertEqual(cycle['input']['axes']['left_y'],'nan')
        self.assertTrue(cycle['faulted'])
        self.assertIn('无效手柄输入',cycle['message'])

    def test_unknown_reference_mode_is_rejected_before_control(self):
        for value in ('unknown',None,{},True):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'cartesian_reference'):
                GamepadControl(self.robot,{'cartesian_reference':value})

    def test_trace_includes_device_error_stop_in_the_same_cycle(self):
        session,_,pad,_=self.make_session()
        stream=io.StringIO();session.trace_stream=stream
        pad.current=self.state()
        with patch.object(session.cameras,'read',side_effect=RuntimeError('camera unavailable')):
            session.poll()
        cycle=json.loads(stream.getvalue())
        self.assertTrue(cycle['faulted'])
        self.assertIn('camera unavailable',cycle['device_errors'][0])
        self.assertIn('camera unavailable',cycle['message'])
