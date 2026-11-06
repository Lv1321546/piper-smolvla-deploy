"""GPU choice uses both process ownership and memory, never utilization alone."""
import unittest
from server.training_queue import idle_gpus

class GPUQueueTests(unittest.TestCase):
    def test_busy_process_blocks_even_zero_utilization(self):
        self.assertEqual(idle_gpus('GPU-A,L40,10,0\n','GPU-A\n'),[])
    def test_allocated_memory_blocks_even_if_process_query_is_empty(self):
        self.assertEqual(idle_gpus('GPU-A,L40,2048,0\n',''),[])
    def test_only_baseline_and_zero_utilization_are_eligible(self):
        rows='GPU-A,L40,10,0\nGPU-B,L40,10,5\nGPU-C,RTX A6000,5,0\n'
        self.assertEqual([g['uuid'] for g in idle_gpus(rows,'GPU-C\n')],['GPU-A'])
    def test_unknown_query_fields_do_not_allow_launch(self):
        with self.assertRaises(ValueError):idle_gpus('GPU-A,L40,N/A,0\n','')

if __name__=='__main__':unittest.main()
