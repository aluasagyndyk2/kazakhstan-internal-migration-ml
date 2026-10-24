"""Reproducible chronological benchmark. Never tune on the test set.

python build_dataset.py
python run_analysis.py
"""
from pathlib import Path
import json, platform, sys, datetime
import numpy as np
import pandas as pd
import sklearn, scipy, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,precision_score,
                            recall_score,roc_auc_score,average_precision_score,confusion_matrix)
from build_dataset import FEATURES

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'; FIGURES=RESULTS/'figures'

def models():
    choices={
        'Majority dummy':DummyClassifier(strategy='most_frequent'),
        'Logistic Regression':LogisticRegression(C=1.0,max_iter=2000,random_state=42),
        'Decision Tree':DecisionTreeClassifier(max_depth=3,min_samples_leaf=5,random_state=42),
        'Random Forest':RandomForestClassifier(n_estimators=300,max_depth=4,min_samples_leaf=3,max_features='sqrt',random_state=42,n_jobs=1),
        'Gradient Boosting':GradientBoostingClassifier(n_estimators=100,learning_rate=0.05,max_depth=2,min_samples_leaf=5,random_state=42)
    }
    result={}
    for name,model in choices.items():
        steps=[('impute',SimpleImputer(strategy='median'))]
        if name=='Logistic Regression': steps.append(('scale',StandardScaler()))
        result[name]=Pipeline(steps+[('model',model)])
    return result

def fit_model(model,name,X,y):
    # Same inverse-class-frequency sample weighting for all four learned models.
    args={} if name=='Majority dummy' else {'model__sample_weight':compute_sample_weight('balanced',y)}
    return model.fit(X,y,**args)

def metrics(y,pred,prob):
    cm=confusion_matrix(y,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    return dict(n=len(y),positive=int(np.sum(y)),accuracy=accuracy_score(y,pred),
                balanced_accuracy=balanced_accuracy_score(y,pred),macro_f1=f1_score(y,pred,average='macro',zero_division=0),
                precision_positive=precision_score(y,pred,zero_division=0),recall_positive=recall_score(y,pred,zero_division=0),
                roc_auc=roc_auc_score(y,prob),average_precision=average_precision_score(y,prob),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def main():
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    df=pd.read_csv(ROOT/'data'/'processed'/'model_dataset.csv')
    development=df.loc[df.year<=2023].copy(); test=df.loc[df.year>=2024].copy()
    assert development.year.max()<test.year.min()
    candidates=models(); fold_years=list(range(2019,2024))
    config=dict(seed=42,primary_selection_metric='mean rolling-validation macro_f1',
                fold_years=fold_years,initial_training_years=[2015,2016,2017,2018],
                final_development_years=list(range(2015,2024)),test_years=[2024,2025],
                decision_threshold=0.5,features=FEATURES,class_weighting='inverse class frequency from each training fold only',
                model_parameters={name:model.get_params()['model'].get_params() for name,model in candidates.items()},
                tuning='No hyperparameter search; fixed modest-capacity configurations, selection by validation only',
                interpretation='50 within-year validation permutations per feature per fold, selected model family only')
    # Config saved before any model is trained or held-out test is scored.
    (RESULTS/'experiment_config.json').write_text(json.dumps(config,indent=2,default=str),encoding='utf-8')
    fold_metrics=[]; val_predictions=[]; fitted_folds={}
    for year in fold_years:
        train=development.loc[development.year<year]; valid=development.loc[development.year==year]
        assert train.year.max()<year and len(valid)>0
        for name,template in candidates.items():
            fitted=fit_model(clone(template),name,train[FEATURES],train.target)
            pred=fitted.predict(valid[FEATURES]); prob=fitted.predict_proba(valid[FEATURES])[:,1]
            fold_metrics.append(dict(model=name,validation_year=year,train_n=len(train),**metrics(valid.target,pred,prob)))
            fitted_folds[(name,year)]=fitted
            for (_,row),yp,pp in zip(valid.iterrows(),pred,prob):
                val_predictions.append(dict(model=name,year=year,kato=row.kato,region=row.region,target=int(row.target),prediction=int(yp),probability=float(pp)))
        pred=valid.persistence_prediction.to_numpy()
        fold_metrics.append(dict(model='Previous-year sign',validation_year=year,train_n=len(train),**metrics(valid.target,pred,pred)))
    cv=pd.DataFrame(fold_metrics); cv.to_csv(RESULTS/'validation_by_fold.csv',index=False)
    pd.DataFrame(val_predictions).to_csv(RESULTS/'validation_predictions.csv',index=False)
    summary=cv.groupby('model',sort=False).agg(validation_macro_f1=('macro_f1','mean'),validation_macro_f1_sd=('macro_f1','std'),
                    validation_balanced_accuracy=('balanced_accuracy','mean'),validation_accuracy=('accuracy','mean')).reset_index()
    summary.to_csv(RESULTS/'validation_summary.csv',index=False)
    learned=[x for x in candidates if x!='Majority dummy']
    selected=summary.loc[summary.model.isin(learned)].sort_values('validation_macro_f1',ascending=False,kind='stable').iloc[0].model
    (RESULTS/'model_selection.json').write_text(json.dumps(dict(selected_model=selected,rule=config['primary_selection_metric'],selection_uses_test=False),indent=2),encoding='utf-8')

    # Interpret the selected family only using out-of-time validation data.
    permutations=[]; rng=np.random.default_rng(42)
    for year in fold_years:
        valid=development.loc[development.year==year]
        fitted=fitted_folds[(selected,year)]; baseline=f1_score(valid.target,fitted.predict(valid[FEATURES]),average='macro',zero_division=0)
        for feature in FEATURES:
            for repeat in range(50):
                shuffled=valid[FEATURES].copy(); shuffled[feature]=rng.permutation(shuffled[feature].to_numpy())
                score=f1_score(valid.target,fitted.predict(shuffled),average='macro',zero_division=0)
                permutations.append(dict(feature=feature,validation_year=year,repeat=repeat,drop_macro_f1=baseline-score))
    importance_long=pd.DataFrame(permutations)
    importance_long.to_csv(RESULTS/'importance_repeats.csv',index=False)
    importance=importance_long.groupby('feature').agg(mean_drop_macro_f1=('drop_macro_f1','mean'),sd_permutations=('drop_macro_f1','std')).sort_values('mean_drop_macro_f1',ascending=False)
    importance.to_csv(RESULTS/'feature_importance.csv')

    # Freeze selection first, then evaluate all predeclared comparators on the same test.
    test_rows=[]; test_predictions=[]; year_rows=[]; selected_estimator=None
    for name,template in candidates.items():
        fitted=fit_model(clone(template),name,development[FEATURES],development.target)
        pred=fitted.predict(test[FEATURES]); prob=fitted.predict_proba(test[FEATURES])[:,1]
        test_rows.append(dict(model=name,selected_before_test=name==selected,**metrics(test.target,pred,prob)))
        for (_,row),yp,pp in zip(test.iterrows(),pred,prob):
            test_predictions.append(dict(model=name,kato=row.kato,region=row.region,year=int(row.year),target=int(row.target),prediction=int(yp),probability=float(pp),correct=int(yp)==int(row.target)))
        for year in [2024,2025]:
            mask=(test.year==year).to_numpy()
            year_rows.append(dict(model=name,year=year,**metrics(test.target.to_numpy()[mask],pred[mask],prob[mask])))
        if name==selected: selected_estimator=fitted
        if name=='Decision Tree':
            (RESULTS/'decision_tree_rules.txt').write_text(export_text(fitted.named_steps['model'],feature_names=FEATURES),encoding='utf-8')
        if name=='Logistic Regression':
            pd.DataFrame(dict(feature=FEATURES,standardised_log_odds_coefficient=fitted.named_steps['model'].coef_[0])).to_csv(RESULTS/'logistic_coefficients.csv',index=False)
    pred=test.persistence_prediction.to_numpy()
    test_rows.append(dict(model='Previous-year sign',selected_before_test=False,**metrics(test.target,pred,pred)))
    for (_,row),yp in zip(test.iterrows(),pred):
        test_predictions.append(dict(model='Previous-year sign',kato=row.kato,region=row.region,year=int(row.year),target=int(row.target),prediction=int(yp),probability=float(yp),correct=int(yp)==int(row.target)))
    for year in [2024,2025]:
        part=test.loc[test.year==year]; yp=part.persistence_prediction.to_numpy()
        year_rows.append(dict(model='Previous-year sign',year=year,**metrics(part.target,yp,yp)))
    final=pd.DataFrame(test_rows); final.to_csv(RESULTS/'test_metrics.csv',index=False)
    predictions=pd.DataFrame(test_predictions); predictions.to_csv(RESULTS/'test_predictions.csv',index=False)
    predictions.loc[(predictions.model==selected)&(~predictions.correct)].to_csv(RESULTS/'selected_model_errors.csv',index=False)
    pd.DataFrame(year_rows).to_csv(RESULTS/'test_metrics_by_year.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    palette={'0':'#b7c3d0','1':'#245b83'}
    counts=df.groupby(['year','target']).size().unstack(fill_value=0)
    ax=counts.rename(columns={0:'Negative net migration',1:'Positive net migration'}).plot.bar(stacked=True,color=[palette['0'],palette['1']],figsize=(8.4,3.3),width=.7)
    ax.set(xlabel='Outcome year',ylabel='Region-year observations',title='Observed class balance after territorial exclusions')
    ax.legend(loc='upper left',frameon=False,fontsize=8); ax.set_ylim(0,25)
    plt.tight_layout(); plt.savefig(FIGURES/'class_balance.png'); plt.close()

    fig,ax=plt.subplots(figsize=(8.4,3.8))
    short={'Majority dummy':'Majority','Logistic Regression':'Logistic','Decision Tree':'Tree','Random Forest':'Forest','Gradient Boosting':'Boosting','Previous-year sign':'Previous sign'}
    names=final.model.tolist(); x=np.arange(len(names)); scores=summary.set_index('model')
    ax.bar(x-.18,[scores.loc[n,'validation_macro_f1'] for n in names],width=.35,label='Rolling validation mean',color='#92a8bb')
    ax.bar(x+.18,final.macro_f1,width=.35,label='Held-out 2024-2025',color='#245b83')
    ax.set_xticks(x,[short[n] for n in names]); ax.set_ylim(0,1.1); ax.set(ylabel='Macro F1',title='Predeclared model comparison')
    ax.legend(frameon=False,fontsize=8)
    for xx,yy in zip(x+.18,final.macro_f1): ax.text(xx,yy+.025,f'{yy:.3f}',ha='center',fontsize=8)
    plt.tight_layout(); plt.savefig(FIGURES/'model_comparison.png'); plt.close()

    picked=final.loc[final.model==selected].iloc[0]
    cm=np.array([[picked.tn,picked.fp],[picked.fn,picked.tp]],dtype=int)
    fig,ax=plt.subplots(figsize=(4.7,3.5)); ax.imshow(cm,cmap='Blues',vmin=0)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],xlabel='Predicted class',ylabel='Actual class',title=selected+' | test confusion matrix')
    for (i,j),value in np.ndenumerate(cm): ax.text(j,i,str(value),ha='center',va='center',fontsize=20,color='white' if value>cm.max()/2 else '#142537')
    plt.tight_layout(); plt.savefig(FIGURES/'confusion_matrix.png'); plt.close()

    display={'income_ratio_lag1':'Relative income (previous year)','wage_ratio_lag1':'Relative wage (previous year)','unemployment_lag1':'Unemployment (previous year)','urban_share_lag1':'Urban population share (previous year)','log_population_lag1':'Log population (previous year)'}
    fig,ax=plt.subplots(figsize=(8.4,3.4)); values=importance.iloc[::-1]
    ax.barh([display[f] for f in values.index],values.mean_drop_macro_f1,color='#245b83')
    ax.axvline(0,color='#667',lw=.8); ax.set(xlabel='Mean loss of validation macro F1 after permutation',title=selected+' | validation feature importance')
    plt.tight_layout(); plt.savefig(FIGURES/'feature_importance.png'); plt.close()

    environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,scikit_learn=sklearn.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,run_date='2026-10-24')
    (RESULTS/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    (ROOT/'requirements.txt').write_text('\n'.join([f'numpy=={np.__version__}',f'pandas=={pd.__version__}',f'scikit-learn=={sklearn.__version__}',f'scipy=={scipy.__version__}',f'matplotlib=={matplotlib.__version__}',f'openpyxl=={__import__("openpyxl").__version__}'])+'\n',encoding='utf-8')
    log=['Actual run completed on 2026-10-24',f'Raw panel: 240 rows; model dataset: {len(df)} rows',f'Development: {len(development)}; held-out test: {len(test)}',
         'Features: '+', '.join(FEATURES),'Validation: expanding windows, outcomes 2019 through 2023',
         'Selected before test: '+selected,'\nVALIDATION\n'+summary.to_string(index=False),'\nTEST\n'+final.to_string(index=False),
         '\nSELECTED MODEL ERRORS\n'+predictions.loc[(predictions.model==selected)&(~predictions.correct)].to_string(index=False),
         '\nVALIDATION PERMUTATION IMPORTANCE\n'+importance.to_string(),'\nENVIRONMENT\n'+json.dumps(environment,indent=2)]
    (RESULTS/'run_log.txt').write_text('\n'.join(log),encoding='utf-8')
    print('\n'.join(log))

if __name__=='__main__': main()
