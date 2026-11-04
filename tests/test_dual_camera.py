"""Dual RGB paths, independent streams and actual LeRobot export; synthetic data."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from PIL import Image

from e14.cameras import CameraReader
from e14.collect_data import record_episode
from e14.export_dataset import load_raw_episode,build_samples,export_dataset
from e14.teaching_ui import Dashboard
from tests.test_data import Clock,Robot

ROOT=Path(__file__).resolve().parents[1]


class DualCameraTests(unittest.TestCase):
    def config(self):
        config=json.loads((ROOT/'config.joint.dual.example.json').read_text())
        for name,camera in config['cameras'].items():camera.update(width=16,height=12,device=f'/dev/mock-{name}')
        return config

    def record(self,folder):
        clock=Clock();config=self.config()
        with CameraReader(config['cameras'],mock=True,clock=clock) as cameras:
            return record_episode(Robot(clock),cameras,folder,'pick block',config,.2,
                                  mock=True,outcome='success',clock=clock,sleep=clock.sleep)

    def test_views_saved_separately_and_both_used_as_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'episode_000000';self.record(p)
            episode=load_raw_episode(p)
            row=episode['rows'][0]
            self.assertEqual(set(row['images']),{'front','astra'})
            self.assertIn('Astra',episode['manifest']['cameras']['astra']['label'])
            self.assertEqual('/dev/mock-astra',episode['manifest']['cameras']['astra']['device'])
            for name in ['front','astra']:
                self.assertEqual(row['images'][name],f'images/{name}/000000.jpg')
                self.assertTrue((p/row['images'][name]).is_file())
                with Image.open(p/row['images'][name]) as image:
                    self.assertEqual(image.format,'JPEG')
            self.assertEqual(episode['manifest']['cameras']['front']['jpeg_quality'],90)
            self.assertEqual(episode['manifest']['cameras']['astra']['jpeg_quality'],80)
            sample=next(iter(build_samples(episode)))
            self.assertIn('observation.images.front',sample)
            self.assertIn('observation.images.astra',sample)
            np.testing.assert_allclose(sample['action'],episode['rows'][1]['state'])
            self.assertFalse(np.array_equal(sample['observation.images.front'],sample['observation.images.astra']))

    def test_stale_second_view_fails_recording_instead_of_silent_omission(self):
        with tempfile.TemporaryDirectory() as temp:
            clock=Clock();config=self.config();p=Path(temp)/'episode_000000'
            with CameraReader(config['cameras'],mock=True,clock=clock) as cameras:
                read=cameras.read;first_time=clock()
                def frozen():
                    sample=read();sample['image_timestamps']['astra']=first_time
                    return sample
                cameras.read=frozen
                with self.assertRaisesRegex(RuntimeError,'Repeated camera frame'):
                    record_episode(Robot(clock),cameras,p,'pick block',config,.2,
                                   mock=True,clock=clock,sleep=clock.sleep)
            self.assertEqual(json.loads((p/'manifest.json').read_text())['status'],'failed')

    def test_actual_lerobot_export_contains_both_views(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'raw'
            self.record(raw/'episode_000000');self.record(raw/'episode_000001')
            out=root/'dataset'
            report=export_dataset(raw,out,'local/piper_dual_test')
            self.assertTrue(report['original_encoded_image_bytes_preserved'])
            info=json.loads((out/'meta/info.json').read_text())
            for name in ['front','astra']:
                self.assertIn(f'observation.images.{name}',info['features'])
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset=LeRobotDataset('local/piper_dual_test',root=out)
            sample=dataset[0]
            self.assertEqual(tuple(sample['observation.images.front'].shape),(3,12,16))
            self.assertEqual(tuple(sample['observation.images.astra'].shape),(3,12,16))
            import pyarrow.parquet as pq
            table=pq.read_table(next((out/'data').rglob('*.parquet')))
            payload=table.column('observation.images.front')[0].as_py()['bytes']
            self.assertEqual(payload,(raw/'episode_000000/images/front/000000.jpg').read_bytes())

    def test_streams_deliver_distinct_cameras_and_unknown_names_rejected(self):
        front=np.full((12,16,3),30,np.uint8);astra=np.full((12,16,3),200,np.uint8)
        with Dashboard(0) as dashboard:
            dashboard.update({'status':'preparing'},dict(front=front,astra=astra))
            with urlopen(dashboard.url+'/state') as response:
                state=json.load(response)
                self.assertEqual(state['camera_names'],['front','astra'])
                self.assertEqual(state['session_id'],dashboard.session_id)
                self.assertNotIn('token',state)
            with urlopen(dashboard.url) as response:
                html=response.read().decode()
                self.assertNotIn('__SESSION_ID__',html)
                self.assertIn(dashboard.session_id,html)
                self.assertIn('physicalKeys',html)
                self.assertIn('location.reload()',html)
            for name,value in [('front',30),('astra',200)]:
                with urlopen(dashboard.url+'/stream/'+name,timeout=2) as response:
                    response.readline();response.readline()
                    size=int(response.readline().split(b':')[1]);response.readline()
                    image=np.asarray(Image.open(io.BytesIO(response.read(size))))
                    self.assertAlmostEqual(float(image.mean()),value,delta=2)
            with self.assertRaises(HTTPError) as error:urlopen(dashboard.url+'/stream/missing')
            self.assertEqual(error.exception.code,404)


if __name__=='__main__':unittest.main()
