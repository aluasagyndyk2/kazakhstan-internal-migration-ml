"""Independently audit all saved predictions, chronology and SHAP sums."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
def manual_metrics(g):
    y=g.target.to_numpy();p=g.prediction.to_numpy()
    tn=int(((y==0)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum());tp=int(((y==1)&(p==1)).sum())
    safe=lambda a,b:a/b if b else 0.0
    return dict(n=len(g),positive=int(y.sum()),tn=tn,fp=fp,fn=fn,tp=tp,
        accuracy=safe(tn+tp,len(g)),macro_f1=(safe(2*tp,2*tp+fp+fn)+safe(2*tn,2*tn+fp+fn))/2,
        balanced_accuracy=(safe(tp,tp+fn)+safe(tn,tn+fp))/2,
        precision_positive=safe(tp,tp+fp),recall_positive=safe(tp,tp+fn))

def main():
    r=ROOT/'results';d=pd.read_csv(ROOT/'data/processed/model_dataset.csv')
    p=pd.read_csv(r/'rolling_predictions.csv');folds=pd.read_csv(r/'rolling_by_year.csv')
    assert not p.duplicated(['feature_set','model','year','kato']).any()
    assert len(p)==1392 and len(folds)==84
    source=d.set_index(['kato','year']).target
    for a in p.itertuples():assert source.loc[(a.kato,a.year)]==a.target
    for row in folds.itertuples():
        g=p.loc[(p.feature_set==row.feature_set)&(p.model==row.model)&(p.year==row.year)]
        for key,val in manual_metrics(g).items():assert np.isclose(val,getattr(row,key)),(row.feature_set,row.model,row.year,key)
        assert row.train_max_year<row.year
        assert row.train_n==len(d.loc[d.year<row.year])
        assert row.n==len(d.loc[d.year==row.year])
    final=pd.read_csv(r/'final_slice_metrics.csv')
    for row in final.itertuples():
        g=p.loc[(p.feature_set==row.feature_set)&(p.model==row.model)&(p.year>=2024)]
        for key,val in manual_metrics(g).items():assert np.isclose(val,getattr(row,key))
    for (m,y),g in p.groupby(['model','year']):
        a=g.loc[g.feature_set=='Core (5)'].sort_values('kato')
        b=g.loc[g.feature_set=='Extended (10)'].sort_values('kato')
        assert a.kato.tolist()==b.kato.tolist()
        if m in ['Majority dummy','Previous-year sign']:assert a.prediction.tolist()==b.prediction.tolist()
    select=json.loads((r/'model_selection.json').read_text())
    for feature_set,name in select['selected'].items():
        avg=folds.loc[(folds.feature_set==feature_set)&(folds.year<=2023)&(~folds.model.isin(['Majority dummy','Previous-year sign']))].groupby('model').macro_f1.mean()
        assert avg.idxmax()==name
    shap_values=pd.read_csv(r/'shap_values_2023.csv');cfg=json.loads((r/'experiment_config.json').read_text())
    cols=cfg['feature_sets']['Extended (10)']
    assert np.allclose(shap_values[cols].sum(axis=1)+shap_values.base_probability,shap_values.predicted_probability,atol=1e-8)
    back=pd.read_csv(r/'shap_background.csv');assert back.year.max()<2023 and len(back)==32
    q=p.loc[(p.feature_set=='Extended (10)')&(p.model==select['selected']['Extended (10)'])&(p.year==2023)].set_index('region')
    assert np.allclose(shap_values.predicted_probability,shap_values.region.map(q.probability))
    repeats=pd.read_csv(r/'importance_repeats.csv')
    assert repeats.groupby(['kind','name','year']).size().eq(50).all()
    assert set(repeats.year)==set(range(2019,2024))
    text='PASS: 84 yearly metric rows and 12 final-slice rows reproduced independently from 1392 predictions.\nPASS: identical samples, labels, earlier-year training, and pre-2024 family selection.\nPASS: SHAP values reconstruct saved probabilities; all 32 background rows precede 2023.\nPASS: feature/group importance has 50 repetitions in every selection year.\n'
    (r/'verification_log.txt').write_text(text,encoding='utf-8');print(text)

if __name__=='__main__':main()
