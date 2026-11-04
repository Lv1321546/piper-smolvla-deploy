"""Offline motion-shape and timing contracts; no SDK or hardware."""
import unittest
from pathlib import Path
import numpy as np
from e14.replay_smoothing import (fixed_ticks,pchip_coefficients,evaluate,
                                  curve_metrics,prepare_trajectory,filtered_joints)
from servo_bridge.core import load_config


class SmoothingTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config(Path(__file__).resolve().parents[1]/'servo_bridge/config.joint.example.json')

    def test_fixed_cadence_merges_tiny_terminal_interval(self):
        ticks=fixed_ticks(.100001,.02)
        np.testing.assert_allclose(ticks,[0,.02,.04,.06,.08,.100001])
        self.assertGreaterEqual(np.min(np.diff(ticks)),.01)
        self.assertEqual(ticks[-1],.100001)

    def test_shape_preserving_turns_and_continuous_velocity(self):
        t=np.array([0.,.1,.23,.4,.6])
        q=np.array([0.,.2,.3,.3,.1])[:,None]
        coef=pchip_coefficients(t,q)
        a,b,c,d=coef
        for i,h in enumerate(np.diff(t)):
            points=np.linspace(t[i],t[i+1],101)
            curve=evaluate(t,coef,points)[:,0]
            self.assertGreaterEqual(curve.min(),min(q[i,0],q[i+1,0])-1e-12)
            self.assertLessEqual(curve.max(),max(q[i,0],q[i+1,0])+1e-12)
            if i<len(t)-2:
                np.testing.assert_allclose(3*a[i]*h*h+2*b[i]*h+c[i],c[i+1],atol=1e-12)
        np.testing.assert_allclose(evaluate(t,coef,t),q)

    def test_analytic_extrema_cover_dense_velocity_and_acceleration(self):
        t=np.array([0.,.13,.3,.5]);q=np.array([0.,.1,.14,0.])[:,None]
        coef=pchip_coefficients(t,q);v,a,error=curve_metrics(t,coef,q)
        aa,bb,cc,dd=coef
        for i,h in enumerate(np.diff(t)):
            x=np.linspace(0,h,1001)[:,None]
            self.assertLessEqual(abs(3*aa[i]*x*x+2*bb[i]*x+cc[i]).max(),v+1e-10)
            self.assertLessEqual(abs(6*aa[i]*x+2*bb[i]).max(),a+1e-10)
            original=q[i]+x*(q[i+1]-q[i])/h
            curve=((aa[i]*x+bb[i])*x+cc[i])*x+dd[i]
            self.assertLessEqual(abs(curve-original).max(),error+1e-10)

    def test_joint_smoothing_preserves_duration_gripper_and_raw_arrays(self):
        t=np.arange(31)*.05
        q=np.tile([0.,90.,-40.,0.,20.,0.,.02],(len(t),1))
        q[:,0]=2*t+.01*np.sin(np.arange(len(t))*2)
        q[10:20,6]=.024
        original=q.copy();original_t=t.copy()
        ticks,values,report=prepare_trajectory(t,q,.02,self.cfg)
        np.testing.assert_array_equal(q,original);np.testing.assert_array_equal(t,original_t)
        np.testing.assert_allclose(values[[0,-1]],q[[0,-1]])
        np.testing.assert_allclose(values[:,6],np.interp(ticks,t,q[:,6]))
        self.assertEqual(ticks[-1],t[-1])
        self.assertLessEqual(report['joint_max_deviation_deg'],.5)
        self.assertLessEqual(report['peak_joint_acceleration_deg_s2'],np.degrees(self.cfg['maximum_joint_acceleration_rad_s2']))
        self.assertEqual(len(ticks),76)

    def test_unsafe_sharp_trajectory_is_rejected_without_time_stretch(self):
        t=np.array([0.,.05,.1]);q=np.tile([0.,90.,-40.,0.,20.,0.,.02],(3,1));q[1,0]=2.
        with self.assertRaises(ValueError):prepare_trajectory(t,q,.02,self.cfg)

    def test_endpoint_chatter_stays_filtered_while_exact_endpoint_is_kept(self):
        # Small measured wobble in the final 0.3 s used to be restored by the
        # endpoint fade, causing every smoothing candidate to fail acceleration.
        t=np.arange(41)*.05
        q=np.tile([0.,90.,-40.,0.,51.132,0.,.02],(len(t),1))
        q[:,2]=-40+5*t
        q[-7:,2]=[-31.2,-30.7,-30.2,-29.9,-29.8,-29.9,-29.9]
        q[-3,4]-=.121
        original=q.copy()
        ticks,values,report=prepare_trajectory(t,q,.02,self.cfg)
        np.testing.assert_array_equal(q,original)
        np.testing.assert_array_equal(values[[0,-1]],q[[0,-1]])
        np.testing.assert_array_equal(values[:,6],np.interp(ticks,t,q[:,6]))
        self.assertEqual(ticks[-1],t[-1])
        self.assertLessEqual(report['joint_max_deviation_deg'],.5)
        self.assertLessEqual(report['peak_joint_acceleration_deg_s2'],
                             np.degrees(self.cfg['maximum_joint_acceleration_rad_s2']))

    def test_short_recording_endpoint_corrections_do_not_overlap(self):
        for duration in (.05,.1,.2):
            t=np.linspace(0,duration,5)
            q=np.column_stack([t*2,t**2])
            result=filtered_joints(t,q,.05)
            np.testing.assert_array_equal(result[[0,-1]],q[[0,-1]])
            self.assertTrue(np.isfinite(result).all())

    def test_linear_comparison_keeps_interpolation_on_uniform_schedule(self):
        t=np.array([0.,.049,.1]);q=np.tile([0.,90.,-40.,0.,20.,0.,.02],(3,1));q[:,0]=t*2
        ticks,v,report=prepare_trajectory(t,q,.02,self.cfg,smoothing='linear')
        np.testing.assert_allclose(ticks,np.arange(6)*.02)
        np.testing.assert_allclose(v[:,0],ticks*2)
        self.assertEqual(report['smoothing'],'linear')


if __name__=='__main__':unittest.main()
