"""Recompute saved scores independently and verify selection/test separation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,average_precision_score,confusion_matrix

ROOT=Path(__file__).resolve().parent
def main():
    predictions=pd.read_csv(ROOT/'results'/'test_predictions.csv')
    metrics=pd.read_csv(ROOT/'results'/'test_metrics.csv').set_index('model')
    for name,group in predictions.groupby('model'):
        assert len(group)==40 and not group.duplicated(['kato','year']).any()
        assert set(group.year)=={2024,2025}
        y=group.target; pred=group.prediction; probability=group.probability
        checks={'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),
                'precision_positive':precision_score(y,pred,zero_division=0),'recall_positive':recall_score(y,pred,zero_division=0),
                'roc_auc':roc_auc_score(y,probability),'average_precision':average_precision_score(y,probability)}
        for metric,value in checks.items(): assert np.isclose(value,metrics.loc[name,metric]),(name,metric)
        assert np.array_equal(confusion_matrix(y,pred,labels=[0,1]).ravel(),metrics.loc[name,['tn','fp','fn','tp']].to_numpy(dtype=int))
    cv=pd.read_csv(ROOT/'results'/'validation_by_fold.csv')
    assert cv.validation_year.max()==2023
    learned=cv.loc[cv.model.isin(['Logistic Regression','Decision Tree','Random Forest','Gradient Boosting'])]
    selected=learned.groupby('model').macro_f1.mean().idxmax()
    saved=json.loads((ROOT/'results'/'model_selection.json').read_text())
    assert selected==saved['selected_model'] and saved['selection_uses_test'] is False
    source=pd.read_csv(ROOT/'data'/'processed'/'model_dataset.csv').set_index(['kato','year'])
    for row in predictions.itertuples(): assert row.target==source.loc[(row.kato,row.year),'target']
    assert (cv.groupby('validation_year').train_n.nunique()==1).all()
    print('PASS: all 6 comparator metric rows independently reproduced from 240 saved predictions.')
    print('PASS: validation selection reproduces '+selected+' without test scores.')
    print('PASS: held-out labels match the frozen-source modelling dataset.')
if __name__=='__main__': main()
