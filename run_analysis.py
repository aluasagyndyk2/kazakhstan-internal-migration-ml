"""Version 2: matched feature-set comparison with expanding-year evaluation.

2024-2025 were already inspected in Version 1. They are a final descriptive
slice, never described as a fresh untouched test.
"""
from pathlib import Path
from datetime import datetime, timezone
import json,sys,platform,importlib.metadata,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score
from model_utils_v1 import models,fit_model,metrics
from build_dataset import CORE_FEATURES,FEATURES,GROUPS

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results';FIGURES=RESULTS/'figures'
SETS={'Core (5)':CORE_FEATURES,'Extended (10)':FEATURES}
LABELS={'income_ratio_lag1':'Relative income','wage_ratio_lag1':'Relative wage',
 'unemployment_lag1':'Unemployment','urban_share_lag1':'Urban population share',
 'log_population_lag1':'Population size (log)','schools_per_10000_lag1':'Schools per 10,000 people',
 'students_per_10000_lag1':'Students per 10,000 people','doctors_per_10000_lag1':'Doctors per 10,000 people',
 'beds_per_10000_lag1':'Hospital beds per 10,000 people','housing_sqm_per_person_lag1':'New housing m2 per person'}

def temporal_splits(df,years):
    for year in years:
        train=df.loc[df.year<year];valid=df.loc[df.year==year]
        if not len(train) or not len(valid):raise ValueError(f'Empty chronological split {year}')
        assert train.year.max()<valid.year.min()
        yield year,train,valid

def joint_shuffle(X,columns,rng):
    result=X.copy();order=rng.permutation(len(X))
    result.loc[:,columns]=X[columns].to_numpy()[order]
    return result

def main():
    RESULTS.mkdir(exist_ok=True);FIGURES.mkdir(exist_ok=True)
    df=pd.read_csv(ROOT/'data/processed/model_dataset.csv')
    candidates=models();years=list(range(2019,2026));selection_years=list(range(2019,2024))
    config=dict(version=2,seed=42,feature_sets=SETS,groups=GROUPS,rolling_years=years,
        selection_years=selection_years,final_slice=[2024,2025],decision_threshold=0.5,
        evaluation='Train on all earlier eligible outcome years; evaluate the next year. Both feature sets use identical rows.',
        selection='Learned model family chosen by mean macro F1 on 2019-2023 only; fixed hyperparameters; no search.',
        known_test_caution='2024-2025 already inspected in V1; this is an exploratory extension, not a newly untouched test.',
        model_parameters={n:m.named_steps['model'].get_params() for n,m in candidates.items()},
        permutation_repeats=50,shap_year=2023,shap_background_size=32,shap_max_evals=210,
        dataset_sha256=hashlib.sha256((ROOT/'data/processed/model_dataset.csv').read_bytes()).hexdigest(),
        protocol_locked_utc=datetime.now(timezone.utc).isoformat())
    (RESULTS/'experiment_config.json').write_text(json.dumps(config,indent=2,default=str),encoding='utf-8')
    print('Frozen V2 protocol saved. Starting identical-row comparisons.',flush=True)
    folds=[];predictions=[];fits={}
    for year,train,valid in temporal_splits(df,years):
        for set_name,features in SETS.items():
            for name,template in candidates.items():
                fitted=fit_model(clone(template),name,train[features],train.target)
                pred=fitted.predict(valid[features]);prob=fitted.predict_proba(valid[features])[:,1]
                fits[(set_name,name,year)]=fitted
                folds.append(dict(feature_set=set_name,model=name,year=year,train_n=len(train),train_max_year=int(train.year.max()),**metrics(valid.target,pred,prob)))
                for r,yp,pp in zip(valid.itertuples(),pred,prob):
                    predictions.append(dict(feature_set=set_name,model=name,year=year,kato=int(r.kato),region=r.region,
                        target=int(r.target),prediction=int(yp),probability=float(pp),correct=int(r.target)==int(yp)))
            pred=valid.persistence_prediction.to_numpy()
            folds.append(dict(feature_set=set_name,model='Previous-year sign',year=year,train_n=len(train),train_max_year=int(train.year.max()),**metrics(valid.target,pred,pred)))
            for r,yp in zip(valid.itertuples(),pred):
                predictions.append(dict(feature_set=set_name,model='Previous-year sign',year=year,kato=int(r.kato),region=r.region,
                    target=int(r.target),prediction=int(yp),probability=float(yp),correct=int(r.target)==int(yp)))
        print(f'Rolling year {year} complete: train {len(train)}, evaluate {len(valid)}',flush=True)
    cv=pd.DataFrame(folds);preds=pd.DataFrame(predictions)
    cv.to_csv(RESULTS/'rolling_by_year.csv',index=False);preds.to_csv(RESULTS/'rolling_predictions.csv',index=False)
    selection=cv.loc[cv.year.isin(selection_years)].groupby(['feature_set','model'],sort=False).agg(
        selection_macro_f1=('macro_f1','mean'),selection_accuracy=('accuracy','mean')).reset_index()
    selection.to_csv(RESULTS/'selection_summary.csv',index=False)
    learned=selection.loc[~selection.model.isin(['Majority dummy','Previous-year sign'])]
    selected={s:learned.loc[learned.feature_set==s].sort_values('selection_macro_f1',ascending=False,kind='stable').iloc[0].model for s in SETS}
    (RESULTS/'model_selection.json').write_text(json.dumps(dict(selected=selected,years=selection_years,uses_final_slice=False),indent=2),encoding='utf-8')
    rolling=cv.groupby(['feature_set','model'],sort=False).agg(mean_macro_f1=('macro_f1','mean'),sd_macro_f1=('macro_f1','std'),mean_accuracy=('accuracy','mean')).reset_index()
    rolling.to_csv(RESULTS/'rolling_summary.csv',index=False)
    slices=[]
    for (s,n),g in preds.loc[preds.year>=2024].groupby(['feature_set','model'],sort=False):
        slices.append(dict(feature_set=s,model=n,selected_on_2019_2023=n==selected[s],
            **metrics(g.target,g.prediction,g.probability)))
    final=pd.DataFrame(slices);final.to_csv(RESULTS/'final_slice_metrics.csv',index=False)
    errors=preds.loc[(preds.model==preds.feature_set.map(selected))&(~preds.correct)].copy()
    errors.to_csv(RESULTS/'selected_model_errors.csv',index=False)

    # Validation interpretation only. One joint row permutation per feature group.
    selected_name=selected['Extended (10)'];rng=np.random.default_rng(42);permutations=[]
    for year in selection_years:
        valid=df.loc[df.year==year];fitted=fits[('Extended (10)',selected_name,year)]
        baseline=f1_score(valid.target,fitted.predict(valid[FEATURES]),average='macro',zero_division=0)
        units=[('feature',f,[f]) for f in FEATURES]+[('group',g,cols) for g,cols in GROUPS.items()]
        for kind,label,cols in units:
            for repeat in range(50):
                shuffled=joint_shuffle(valid[FEATURES],cols,rng)
                score=f1_score(valid.target,fitted.predict(shuffled),average='macro',zero_division=0)
                permutations.append(dict(kind=kind,name=label,year=year,repeat=repeat,drop_macro_f1=baseline-score))
        print(f'Validation importance {year} complete.',flush=True)
    repeated=pd.DataFrame(permutations);repeated.to_csv(RESULTS/'importance_repeats.csv',index=False)
    importance=repeated.groupby(['kind','name']).agg(mean_drop_macro_f1=('drop_macro_f1','mean'),sd=('drop_macro_f1','std')).reset_index()
    importance.to_csv(RESULTS/'permutation_importance.csv',index=False)

    # Real SHAP library, model-agnostic explanation in positive-class probability units.
    import shap
    train=df.loc[df.year<2023];valid=df.loc[df.year==2023]
    background=train[FEATURES].sample(n=min(32,len(train)),random_state=42)
    fitted=fits[('Extended (10)',selected_name,2023)]
    def predict_positive(x):return fitted.predict_proba(pd.DataFrame(x,columns=FEATURES))[:,1]
    explainer=shap.PermutationExplainer(predict_positive,background,feature_names=FEATURES,seed=42)
    explanation=explainer(valid[FEATURES],max_evals=210,silent=True)
    values=np.asarray(explanation.values);base=np.asarray(explanation.base_values)
    actual=predict_positive(valid[FEATURES]);reconstructed=base+values.sum(axis=1)
    max_error=float(np.max(np.abs(actual-reconstructed)));assert max_error<1e-8,max_error
    pd.DataFrame(values,columns=FEATURES).assign(region=valid.region.to_numpy(),year=2023,
        base_probability=base,predicted_probability=actual).to_csv(RESULTS/'shap_values_2023.csv',index=False)
    pd.DataFrame(dict(feature=FEATURES,mean_abs_shap=np.abs(values).mean(axis=0))).sort_values('mean_abs_shap',ascending=False).to_csv(RESULTS/'shap_summary.csv',index=False)
    pd.concat([train.loc[background.index,['kato','region','year']].reset_index(drop=True),background.reset_index(drop=True)],axis=1).to_csv(RESULTS/'shap_background.csv',index=False)
    (RESULTS/'shap_audit.json').write_text(json.dumps(dict(explainer='shap.PermutationExplainer',version=shap.__version__,
        model=selected_name,validation_year=2023,train_max_year=int(train.year.max()),background_n=len(background),
        explained_n=len(valid),max_evals=210,seed=42,max_additivity_error=max_error,units='positive-class probability; descriptive, not causal'),indent=2),encoding='utf-8')
    print('SHAP complete; maximum additivity error:',max_error,flush=True)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':160,'axes.spines.top':False,'axes.spines.right':False})
    order=['Majority dummy','Logistic Regression','Decision Tree','Random Forest','Gradient Boosting','Previous-year sign']
    short=['Majority','Logistic','Tree','Forest','Boosting','Previous sign']
    for table,value,title,filename in [(rolling,'mean_macro_f1','Mean yearly macro F1 | rolling origins 2019-2025','model_comparison.png'),
         (final,'macro_f1','Macro F1 | previously inspected 2024-2025 slice','final_slice_comparison.png')]:
        fig,ax=plt.subplots(figsize=(8.5,3.5));x=np.arange(6)
        for offset,(s,color) in zip([-.18,.18],[('Core (5)','#94a7b5'),('Extended (10)','#245b83')]):
            scores=table.loc[table.feature_set==s].set_index('model').loc[order,value]
            ax.bar(x+offset,scores,width=.35,color=color,label=s)
        ax.set_xticks(x,short);ax.set_ylim(0,1.05);ax.set(ylabel='Macro F1',title=title);ax.legend(frameon=False)
        fig.tight_layout();fig.savefig(FIGURES/filename);plt.close(fig)
    for kind,filename,title in [('feature','feature_importance.png','Validation feature permutation importance'),('group','group_importance.png','Validation grouped permutation importance')]:
        tab=importance.loc[importance.kind==kind].sort_values('mean_drop_macro_f1')
        fig,ax=plt.subplots(figsize=(8.5,4.4 if kind=='feature' else 3))
        ax.barh([LABELS.get(n,n) for n in tab.name],tab.mean_drop_macro_f1,color='#245b83');ax.axvline(0,color='gray',lw=.7)
        ax.set(xlabel='Mean decrease in macro F1 after shuffling',title=title+' | '+selected_name)
        fig.tight_layout();fig.savefig(FIGURES/filename);plt.close(fig)
    ss=pd.read_csv(RESULTS/'shap_summary.csv').sort_values('mean_abs_shap')
    fig,ax=plt.subplots(figsize=(8.5,4.4));ax.barh([LABELS[x] for x in ss.feature],ss.mean_abs_shap,color='#245b83')
    ax.set(xlabel='Mean absolute SHAP value (probability units)',title='SHAP | 2023 out-of-time predictions | '+selected_name)
    fig.tight_layout();fig.savefig(FIGURES/'shap_importance.png');plt.close(fig)
    row=final.loc[(final.feature_set=='Extended (10)')&(final.model==selected_name)].iloc[0]
    cm=np.array([[row.tn,row.fp],[row.fn,row.tp]],int)
    fig,ax=plt.subplots(figsize=(4.7,3.5));ax.imshow(cm,cmap='Blues')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],
        xlabel='Predicted class',ylabel='Actual class',title=selected_name+' | 2024-2025 slice')
    for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',fontsize=20,color='white' if v>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(FIGURES/'confusion_matrix.png');plt.close(fig)

    packages=['numpy','pandas','scikit-learn','scipy','matplotlib','openpyxl','xlrd','shap','nbformat','nbclient','ipykernel']
    versions={p:importlib.metadata.version(p) for p in packages}
    (ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n',encoding='utf-8')
    env=dict(python=sys.version,platform=platform.platform(),packages=versions,completed_utc=datetime.now(timezone.utc).isoformat())
    (RESULTS/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    log=['VERSION 2: actual execution',f'Matched sample: {len(df)}; core features 5; extended features 10.',
         'Origins 2019-2025; each model trains only on earlier years. 2024-2025 were already seen in Version 1.',
         'Selected families from 2019-2023: '+json.dumps(selected),'\nROLLING SUMMARY\n'+rolling.to_string(index=False),
         '\nPREVIOUSLY INSPECTED FINAL SLICE\n'+final.to_string(index=False),
         '\nGROUP IMPORTANCE\n'+importance.loc[importance.kind=='group'].to_string(index=False),
         '\nSHAP\n'+json.dumps(json.loads((RESULTS/'shap_audit.json').read_text()),indent=2),
         '\nFINAL SLICE ERRORS\n'+errors.loc[errors.year>=2024].to_string(index=False),'\nENVIRONMENT\n'+json.dumps(env,indent=2)]
    (RESULTS/'run_log.txt').write_text('\n'.join(log),encoding='utf-8');print('\n'.join(log))
    return final

if __name__=='__main__':main()
