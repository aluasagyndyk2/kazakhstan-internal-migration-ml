import unittest
import pandas as pd
from run_analysis import temporal_splits, joint_shuffle

class EvaluationTests(unittest.TestCase):
    def test_each_origin_excludes_its_own_and_future_years(self):
        d=pd.DataFrame({'year':[2018,2019,2020,2024,2025]})
        splits=list(temporal_splits(d,[2019,2020,2024,2025]))
        for year,train,valid in splits:
            self.assertTrue((train.year<year).all())
            self.assertTrue((valid.year==year).all())
        self.assertIn(2024,splits[-1][1].year.tolist())

    def test_group_shuffle_keeps_within_group_relationship(self):
        import numpy as np
        d=pd.DataFrame({'a':[1,2,3,4],'b':[10,20,30,40],'c':[9,8,7,6]})
        q=joint_shuffle(d,['a','b'],np.random.default_rng(42))
        self.assertTrue((q.b==q.a*10).all())
        self.assertTrue((q.c==d.c).all())
        self.assertFalse(q.a.equals(d.a))

if __name__=='__main__':unittest.main()
