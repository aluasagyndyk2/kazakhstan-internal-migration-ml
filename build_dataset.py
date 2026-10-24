"""Build the matched core/extended panel from 12 frozen official workbooks."""
from pathlib import Path
import re,json
import io,contextlib,warnings
import numpy as np
import pandas as pd
from build_core_dataset import build_dataset as build_core, REGIONS, FEATURES as CORE_FEATURES

ROOT=Path(__file__).resolve().parent
NEW_BASES=['schools_per_10000','students_per_10000','doctors_per_10000','beds_per_10000','housing_sqm_per_person']
FEATURES=CORE_FEATURES+[x+'_lag1' for x in NEW_BASES]
GROUPS={'Core socioeconomic':CORE_FEATURES,'Education':FEATURES[5:7],'Healthcare':FEATURES[7:9],'Housing':FEATURES[9:]}

def norm(v):
    return re.sub(r'[^а-яa-zәғқңөұүһіё]','',str(v).lower().replace('город','').replace('г.','').replace('область','')).replace('ұлытау','улытау')

def read_new_sources():
    """Preserve actual source worksheet/cell addresses for every extracted value."""
    names={norm(ru):k for k,(_,ru) in REGIONS.items()}
    names[norm('Алматы')]=750000000
    # Same current unit, three official Russian/Kazakh spellings across sources.
    names[norm('Жетису')]=names[norm('Жетысу')]=330000000
    values={};lineage=[]
    specs=[('schools','xlsx','Число общеобраз. школ',3,2,6,25,1,'schools'),
           ('students','xls','Численность студентов',2,2,4,23,1,'students'),
           ('doctors','xlsx','Показатель',5,2,7,26,1,'doctors'),
           ('beds','xlsx','Показатель',5,2,7,26,1,'beds'),
           ('housing','xlsx','показатель',3,1,5,25,1000,'housing_sqm')]
    for source,ext,sheet,header,label,start,end,mult,var in specs:
        with warnings.catch_warnings():
            # XLS reading is pinned to xlrd; pandas 3.1 announces a future
            # engine removal, which is irrelevant to this frozen environment.
            warnings.filterwarnings('ignore',message='The xlrd engine is deprecated.*')
            data=pd.read_excel(ROOT/'data/raw'/f'{source}.{ext}',sheet_name=sheet,header=None)
        columns={}
        for c,val in data.iloc[header-1].items():
            m=re.match(r'^(20\d{2})',str(val))
            if m and 2014<=int(m.group(1))<=2025:columns[int(m.group(1))]=c
        for r in range(start-1,min(end,len(data))):
            key=names.get(norm(data.iat[r,label-1]))
            if key is None:continue
            for year,c in columns.items():
                raw=data.iat[r,c]
                if isinstance(raw,(int,float,np.number)):v=float(raw)*mult
                elif str(raw).strip()=='-':v=0.0 # Official legend: phenomenon absent.
                else:v=np.nan
                assert (key,year,var) not in values,(source,key,year)
                values[(key,year,var)]=v
                # Excel letters without a workbook rewrite.
                n=c+1;letters=''
                while n:n,rem=divmod(n-1,26);letters=chr(65+rem)+letters
                lineage.append(dict(source_id=source,sheet=sheet,cell=f'{letters}{r+1}',kato=key,year=year,
                    variable=var,raw_value=raw,value=v,multiplier=mult,
                    year_rule='academic-year start' if source in ['schools','students'] else 'calendar year'))
    return values,pd.DataFrame(lineage)

def build_dataset():
    # The reused extractor writes V1 terminology internally; expose only V2's
    # corrected summary and split labels in the student notebook.
    with contextlib.redirect_stdout(io.StringIO()):
        core=build_core()
    out=ROOT/'data/processed'
    core['split']=np.where(core.year>=2024,'previously_seen_final_slice','development')
    core.to_csv(out/'core_eligible_dataset.csv',index=False,encoding='utf-8-sig')
    values,lineage=read_new_sources()
    panel=pd.read_csv(out/'regional_panel.csv')
    for v in ['schools','students','doctors','beds','housing_sqm']:
        panel[v]=[values.get((int(r.kato),int(r.year),v),np.nan) for r in panel.itertuples()]
    for name,raw in [('schools_per_10000','schools'),('students_per_10000','students'),('doctors_per_10000','doctors'),('beds_per_10000','beds')]:
        panel[name]=panel[raw]/panel.population*10000
    panel['housing_sqm_per_person']=panel.housing_sqm/panel.population
    panel.to_csv(out/'regional_panel.csv',index=False,encoding='utf-8-sig')
    indexed=panel.set_index(['kato','year'])
    for name in NEW_BASES:
        core[name+'_lag1']=[indexed.loc[(int(r.kato),int(r.feature_year)),name] for r in core.itertuples()]
    missing=core[FEATURES].isna().any(axis=1)
    excluded=core.loc[missing,['kato','region','year']].copy()
    excluded['reason']='missing extended lagged predictor; omitted from BOTH feature-set comparisons'
    excluded.to_csv(out/'extended_excluded_rows.csv',index=False)
    matched=core.loc[~missing].copy()
    matched['split']=np.where(matched.year>=2024,'previously_seen_final_slice','development')
    matched.to_csv(out/'model_dataset.csv',index=False,encoding='utf-8-sig')
    old=pd.read_csv(out/'cell_lineage.csv')
    pd.concat([old,lineage],ignore_index=True).to_csv(out/'cell_lineage.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'variable':f,'group':g,'lag_years':1,
        'definition':f.replace('_lag1','').replace('_',' ')} for g,cols in GROUPS.items() for f in cols]).to_csv(out/'feature_dictionary.csv',index=False)
    with (out/'data_dictionary.csv').open('a',encoding='utf-8') as stream:
        for base in NEW_BASES:
            unit='square metres per person' if base.startswith('housing') else 'count per 10000 average annual population'
            stream.write(f'{base}_lag1,Previous-year {base.replace("_"," ")},{unit},Frozen BNS source; academic start year for education\n')
    dictionary=pd.read_csv(out/'data_dictionary.csv')
    dictionary.loc[dictionary.variable=='split','definition']='Development or previously inspected final slice'
    dictionary.loc[dictionary.variable=='split','source_or_rule']='Development <=2023; 2024-2025 seen in V1; V2 uses expanding origins'
    dictionary.to_csv(out/'data_dictionary.csv',index=False,encoding='utf-8-sig')
    counts=matched.groupby('year').agg(rows=('target','size'),positive=('target','sum'))
    counts['negative']=counts.rows-counts.positive;counts.to_csv(out/'class_counts_by_year.csv')
    summary=dict(version=2,raw_rows=240,core_eligible_rows=len(core),matched_rows=len(matched),
        missing_extended_rows=int(missing.sum()),positive=int(matched.target.sum()),negative=int((matched.target==0).sum()),
        development_rows=int((matched.year<=2023).sum()),previously_seen_final_slice_rows=int((matched.year>=2024).sum()),
        features=FEATURES,feature_groups=GROUPS,feature_missing=int(matched[FEATURES].isna().sum().sum()),
        school_year_rule='2014/15 uses start year 2014; all predictors lag outcome by one year',
        housing_units='Actual data sheet is thousand m2; multiply by 1000 before dividing by population',
        temporal_caution='2024-2025 were inspected in V1. V2 uses rolling origin; they are not an untouched test.')
    (out/'dataset_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('VERSION 2 MATCHED SAMPLE\n'+json.dumps(summary,indent=2))
    return matched

if __name__=='__main__':build_dataset()
