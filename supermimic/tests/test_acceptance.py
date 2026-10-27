import unittest
import json
from supermimic.evaluation.acceptance import compare, json_safe_metrics

class ComparabilityTest(unittest.TestCase):
    def setUp(self):
        self.base=dict(mean_task_return=100.,xy_rmse=1.,yaw_rmse=2.,velocity_survival_rate=1.,recovery_success_rate=1.)
    def test_boundary_and_each_regression(self):
        candidate={**self.base,'mean_task_return':98.,'xy_rmse':1.05,'yaw_rmse':2.1}
        self.assertTrue(compare(self.base,candidate)['passed'])
        for key,value in [('mean_task_return',97.99),('xy_rmse',1.051),('yaw_rmse',2.101),('velocity_survival_rate',.99),('recovery_success_rate',.99)]:
            self.assertFalse(compare(self.base,{**candidate,key:value})['passed'],key)
    def test_failed_episode_serialization(self):
        original={'episodes':[{'reward':float('-inf'),'success':False}]}
        clean,bad=json_safe_metrics(original)
        self.assertEqual(clean,{'episodes':[{'reward':None,'success':False}]})
        self.assertEqual(bad,[{'path':'/episodes/0/reward','value':'-inf'}])
        json.dumps(clean,allow_nan=False)
        self.assertEqual(original['episodes'][0]['reward'],float('-inf'))

    def test_negative_return_and_nonfinite(self):
        base={**self.base,'mean_task_return':-100.}
        self.assertTrue(compare(base,{**base,'mean_task_return':-102.})['passed'])
        self.assertFalse(compare(base,{**base,'mean_task_return':-102.1})['passed'])
        for value in (float('nan'),float('inf'),float('-inf')):
            self.assertFalse(compare(self.base,{**self.base,'mean_task_return':value})['passed'])

if __name__=='__main__': unittest.main()
