import unittest,numpy as np
from servo_bridge.core import Feedback,NAMES,load_config
from servo_bridge.execution import VelocityTarget

class ExecutionTests(unittest.TestCase):
 def setUp(self):
  self.cfg=load_config();self.e=VelocityTarget(self.cfg);self.q=np.radians([0,80,-60,0,40,0])
 def test_deadband_plant_makes_progress(self):
  q=self.q.copy();initial=q.copy()
  for k in range(100):
   t=1+k*.05;f=Feedback(q.copy(),t,ready=True)
   _,target,v=self.e.step(NAMES,[.02,0,0,0,0,0],feedback=f,now=t,stamp_age_s=0,horizon_s=.05)
   error=target-q
   q+=np.where(abs(error)>np.radians(.15),error*.5,0)
  self.assertGreater(np.degrees(q[0]-initial[0]),5)
 def test_stuck_arm_stops_at_existing_lead_bound(self):
  for k in range(100):
   t=1+k*.05
   try:self.e.step(NAMES,[.02,0,0,0,0,0],feedback=Feedback(self.q.copy(),t,ready=True),now=t,stamp_age_s=0,horizon_s=.05)
   except ValueError as e:
    self.assertIn('tracking',str(e));break
  else:self.fail('Stuck plant did not stop')
  self.assertLessEqual(np.degrees(self.e.position[0]),1)
 def test_reversal_and_reset(self):
  q=self.q.copy();lastv=np.zeros(6)
  for k in range(60):
   t=1+k*.05
   _,q,v=self.e.step(NAMES,[.05 if k<30 else -.05,0,0,0,0,0],feedback=Feedback(q,t,ready=True),now=t,stamp_age_s=0,horizon_s=.05)
   self.assertLessEqual(np.max(abs(v-lastv)),.02500001);lastv=v
  self.e.reset();self.assertIsNone(self.e.position)
 def test_alternating_velocity_ripple_is_attenuated(self):
  q=self.q.copy();out=[]
  for k in range(60):
   t=1+k*.05;demand=.02 + (.01 if k%2 else -.01)
   _,q,v=self.e.step(NAMES,[demand,0,0,0,0,0],feedback=Feedback(q,t,ready=True),now=t,stamp_age_s=0,horizon_s=.05)
   out.append(v[0])
  self.assertLess(np.ptp(out[-20:]),.005)
  self.assertAlmostEqual(np.mean(out[-20:]),.02,places=4)

 def test_stale_and_bad_order(self):
  with self.assertRaises(ValueError):self.e.step(NAMES,[.02]*6,feedback=Feedback(self.q,0,ready=True),now=1,stamp_age_s=0,horizon_s=.05)
  self.assertIsNone(self.e.position)
  with self.assertRaises(ValueError):self.e.step(['joint1']*6,[.02]*6,feedback=Feedback(self.q,1,ready=True),now=1,stamp_age_s=0,horizon_s=.05)

if __name__=='__main__':unittest.main()
