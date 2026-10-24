"""Independent source-control and chronological-integrity checks."""
import unittest
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent

class DataIntegrity(unittest.TestCase):
    def test_known_official_migration_controls(self):
        df=pd.read_csv(ROOT/'data'/'processed'/'regional_panel.csv')
        rows=df.set_index(['region','year'])
        self.assertEqual(rows.loc[('Astana',2024),'migration_net'],73238)
        self.assertEqual(rows.loc[('Almaty city',2024),'migration_net'],36061)
        self.assertEqual(rows.loc[('Shymkent',2024),'migration_net'],9361)
        self.assertEqual(rows.loc[('Almaty region',2024),'migration_net'],2639)
        for year,group in df.groupby('year'):
            self.assertEqual(group.migration_net.sum(),0)
            self.assertTrue((group.migration_in-group.migration_out==group.migration_net).all())

    def test_territory_and_lag_integrity(self):
        df=pd.read_csv(ROOT/'data'/'processed'/'model_dataset.csv')
        self.assertFalse(df.duplicated(['kato','year']).any())
        self.assertTrue((df.feature_year==df.year-1).all())
        self.assertTrue((df.target==(df.migration_net>0).astype(int)).all())
        self.assertTrue((df.migration_net!=0).all())
        affected=['Abay','Zhetisu','Ulytau','Almaty region','Karaganda','East Kazakhstan']
        self.assertTrue((df.loc[df.region.isin(affected),'feature_year']>=2022).all())
        self.assertTrue((df.loc[df.region.isin(['Turkistan','Shymkent']),'feature_year']>=2018).all())
        for name in ['income_ratio_lag1','wage_ratio_lag1','unemployment_lag1','urban_share_lag1','log_population_lag1']:
            self.assertFalse(df[name].isna().any())
        self.assertTrue(df.loc[df.split=='test','year'].isin([2024,2025]).all())
        self.assertTrue((df.loc[df.split!='test','year']<=2023).all())

if __name__=='__main__': unittest.main()
