"""Source-value and temporal integrity tests for the extended migration study."""
import unittest
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from build_dataset import build_dataset, FEATURES, CORE_FEATURES, read_new_sources

ROOT=Path(__file__).resolve().parent

class ExtendedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=build_dataset()
        cls.values,_=read_new_sources()

    def test_exact_source_values_and_units(self):
        v=self.values
        self.assertEqual(v[(110000000,2014,'schools')],613)
        self.assertEqual(v[(110000000,2014,'students')],10289)
        self.assertEqual(v[(110000000,2014,'housing_sqm')],268000)
        self.assertTrue(np.isnan(v.get((100000000,2014,'schools'),np.nan)))

    def test_matching_and_no_future_leakage(self):
        d=self.df
        self.assertEqual(len(FEATURES),10)
        self.assertEqual(len(CORE_FEATURES),5)
        self.assertEqual(len(d),164, 'Official spelling variants must not drop Zhetisu')
        self.assertEqual(len(d.loc[d.kato==330000000]),3)
        self.assertFalse(d.duplicated(['kato','year']).any())
        self.assertTrue((d.feature_year==d.year-1).all())
        self.assertFalse(d[FEATURES].isna().any().any())
        self.assertTrue(np.isfinite(d[FEATURES].to_numpy()).all())
        self.assertTrue((d.target==(d.migration_net>0).astype(int)).all())
        self.assertTrue((d.loc[d.kato.isin([100000000,190000000,330000000,350000000,620000000,630000000]),'feature_year']>=2022).all())
        self.assertTrue((d.loc[d.kato.isin([610000000,790000000]),'feature_year']>=2018).all())

    def test_lagged_density_is_from_the_same_region_year(self):
        d=self.df
        r=d.loc[(d.kato==110000000)&(d.year==2015)].iloc[0]
        panel=pd.read_csv(ROOT/'data/processed/regional_panel.csv')
        pop=panel.loc[(panel.kato==110000000)&(panel.year==2014),'population'].iloc[0]
        self.assertAlmostEqual(r.schools_per_10000_lag1,613/pop*10000)
        self.assertAlmostEqual(r.housing_sqm_per_person_lag1,268000/pop)

    def test_all_archived_originals_unchanged(self):
        manifest=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(manifest),12)
        for rec in manifest:
            self.assertEqual(hashlib.sha256((ROOT/rec['file']).read_bytes()).hexdigest(),rec['sha256'])

    def test_dictionary_discloses_previously_seen_slice(self):
        dictionary=pd.read_csv(ROOT/'data/processed/data_dictionary.csv')
        row=dictionary.loc[dictionary.variable=='split'].iloc[0]
        self.assertIn('previously',row.definition.lower())
        self.assertNotIn('held-out',row.definition.lower())

if __name__=='__main__': unittest.main()
