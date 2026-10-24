"""One-shot evaluation of heldout-v1. Self-test never reads official test data."""
import argparse
import io
import json
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import improvement_experiments as e
import check_frozen_candidate
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

a = e.a
ROOT = a.ROOT
FROZEN = ROOT/'results/frozen_candidate_v1'
OUT = ROOT/'results/official_holdout_v1'
PLAN = ROOT/'report/heldout_test_plan.json'
SELFTEST = ROOT/'results/heldout_runner_selftest.json'
MODELS = ['count_standard', 'binary_standard']

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, obj):
    a.pilot.save_json(path, obj)

def operating(y, p, t):
    positive = p >= t
    n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
    fp, fn = int(np.sum((y == 0) & positive)), int(np.sum((y == 1) & ~positive))
    tp, tn = n1-fn, n0-fp
    recall, fpr = tp/n1 if n1 else None, fp/n0 if n0 else None
    return dict(n=len(y), n_positive=n1, n_negative=n0, positive_fraction=n1/len(y) if len(y) else None,
                tp=tp, tn=tn, fp=fp, fn=fn, recall=recall, fnr=fn/n1 if n1 else None, fpr=fpr,
                precision=tp/(tp+fp) if tp+fp else 0.,
                balanced_accuracy=(recall+1-fpr)/2 if n0 and n1 else None,
                accuracy=(tp+tn)/len(y) if len(y) else None, threshold=t)

def probability(y, p):
    return dict(auroc=float(roc_auc_score(y,p)), average_precision=float(average_precision_score(y,p)),
                brier=float(brier_score_loss(y,p)), log_loss=float(log_loss(y,p,labels=[0,1])))

def metric_vector(y, probs, policies):
    out = {}
    for name, p in probs.items():
        out.update({f'{name}/{k}':v for k,v in probability(y,p).items()})
        for policy,t in policies[name].items():
            m = operating(y,p,t)
            out.update({f'{name}/{policy}/{k}':m[k] for k in ['fnr','fpr','recall','precision']})
    out['paired_auroc_difference'] = out['count_standard/auroc']-out['binary_standard/auroc']
    return out

def bootstrap(y, probs, policies, repetitions, seed, progress=False):
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y==v) for v in [0,1]]
    assert all(len(c) for c in classes)
    rows = []
    for i in range(repetitions):
        indices = np.concatenate([rng.choice(c,size=len(c),replace=True) for c in classes])
        rows.append(metric_vector(y[indices],{k:p[indices] for k,p in probs.items()},policies))
        if progress and (i+1)%100==0:
            print(f'Bootstrap {i+1}/{repetitions}',flush=True)
    return pd.DataFrame(rows)

def predict(sample, d, manifest):
    sc = read(FROZEN/'scaler.json')
    assert sc['features'] == a.FEATURES
    scaled = (d-np.array(sc['mean']))/np.array(sc['scale'])
    counts = e.count_features(sample)
    binary = (counts>0).astype(np.float32)
    probs, individual = {}, {}
    for condition in MODELS:
        x = e.inputs(binary,counts,scaled,condition)
        arrays = []
        for seed in manifest['seeds']:
            cp = torch.load(FROZEN/condition/f'seed{seed}/model.pt',map_location='cpu',weights_only=True)
            assert cp['condition']==condition and cp['seed']==seed
            model = e.network(condition)
            model.load_state_dict(cp['state_dict'])
            p = a.predict(model,x).astype(np.float64)
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            arrays.append(p)
            individual[f'{condition}_seed{seed}'] = p
        probs[condition] = np.mean(arrays,axis=0)
    return probs, individual

def selftest():
    check_frozen_candidate.main()
    y = np.array([0,0,1,1])
    p = np.array([.1,.5,.5,.9])
    m = operating(y,p,.5)
    assert (m['tp'],m['tn'],m['fp'],m['fn'])==(2,1,1,0)
    assert m['recall']==1 and m['fpr']==.5 and m['precision']==2/3
    assert probability(y,p)['auroc']==.875
    assert operating(np.zeros(3),np.zeros(3),.5)['fnr'] is None
    policy = {name:{'recall90':.5,'balanced':.7} for name in MODELS}
    same = bootstrap(y,{name:p for name in MODELS},policy,20,20261004)
    assert np.array_equal(same.paired_auroc_difference,np.zeros(20))
    assert same.equals(bootstrap(y,{name:p for name in MODELS},policy,20,20261004))
    sample = pd.read_csv(a.DATA/'pilot_sample.csv')
    va = np.load(FROZEN/'partition_indices.npz')['validation']
    manifest = read(FROZEN/'frozen_manifest.json')
    subset = sample.iloc[va].reset_index(drop=True)
    probs, singles = predict(subset,a.descriptors(subset),manifest)
    maximum = 0.
    for name in MODELS:
        for seed in manifest['seeds']:
            stored = np.load(FROZEN/name/f'seed{seed}/validation_probability.npy')
            fresh = singles[f'{name}_seed{seed}']
            np.testing.assert_allclose(fresh,stored,rtol=1e-6,atol=1e-7)
            maximum = max(maximum,float(np.max(abs(fresh-stored))))
        saved = pd.read_csv(FROZEN/name/'validation_predictions.csv',float_precision='round_trip')
        np.testing.assert_array_equal(saved.source_id.to_numpy(),subset.source_id.to_numpy())
        np.testing.assert_allclose(probs[name],saved.probability,rtol=1e-6,atol=1e-7)
        for policy,t in manifest['thresholds'][name].items():
            assert np.array_equal(probs[name]>=t,saved.probability.to_numpy()>=t)
    save(SELFTEST,dict(status='PASS',utc=now(),runner_sha256=a.sha(Path(__file__)),
                     maximum_validation_probability_difference=maximum,
                     checks=['synthetic tied threshold','known AUROC','absent class','paired bootstrap identity and reproducibility',
                             'six frozen checkpoints reproduce all validation probabilities and fixed decisions'],
                     official_test_opened=False))
    print('PASS: synthetic statistics and all six validation predictions verified; official test unopened.',flush=True)

def execute():
    check_frozen_candidate.main()
    verification = read(SELFTEST)
    assert verification['status']=='PASS' and verification['runner_sha256']==a.sha(Path(__file__))
    plan, manifest = read(PLAN), read(FROZEN/'frozen_manifest.json')
    assert plan['policies']==manifest['thresholds']
    archive = ROOT.parents[1]/plan['archive_path_relative_to_workspace']
    assert a.sha(archive)==plan['archive_sha256']
    sample = pd.read_csv(a.DATA/'pilot_sample.csv')
    train = np.load(FROZEN/'partition_indices.npz')['train']
    development_d = a.descriptors(sample)
    cuts = {f:float(np.quantile(development_d[train,a.FEATURES.index(f)],.75)) for f in plan['structure_subgroups']['features']}
    # Exclusive directory and marker are created before opening official test labels.
    OUT.mkdir(exist_ok=False)
    marker = dict(started_utc=now(),authorization='User: 执行保留测试',runner_sha256=a.sha(Path(__file__)),
                  plan_sha256=a.sha(PLAN),manifest_sha256=a.sha(FROZEN/'frozen_manifest.json'),
                  archive_sha256=a.sha(archive),prior_exposure=plan['prior_exposure'],
                  preflight_sha256=a.sha(SELFTEST),structure_q75=cuts,official_test_labels_loaded=False)
    with (OUT/'run_started.json').open('x',encoding='utf-8') as handle:
        json.dump(marker,handle,indent=2,ensure_ascii=False,allow_nan=False)
    start = time.perf_counter()
    stage = 'opening_labels'
    try:
        with zipfile.ZipFile(archive) as z:
            test = pd.read_csv(io.BytesIO(z.read(plan['test_member'])),index_col=0)
        stage = 'data_audit_labels_opened'
        assert list(test.columns)==['smi','activity','dataset'],test.columns.tolist()
        assert len(test)==plan['expected_raw_rows']
        assert test.index.is_unique and test.index.notna().all()
        assert test.activity.isin([0,1]).all()
        test.index.name='source_id'
        test=test.reset_index()
        canonical,scaffolds=[],[]
        for smi in test.smi:
            assert isinstance(smi,str) and smi.strip()
            mol=Chem.MolFromSmiles(smi)
            assert mol is not None and mol.GetNumAtoms()>0
            cs=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
            canonical.append(cs)
            scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(cs),includeChirality=False) or '__ACYCLIC__')
        test['canonical_smiles'],test['scaffold']=canonical,scaffolds
        assert not test.canonical_smiles.duplicated().any()
        assert not set(canonical)&set(sample.canonical_smiles)
        assert not set(test.source_id)&set(sample.source_id)
        y=test.activity.to_numpy(dtype=np.int64)
        d=a.descriptors(test)
        for i,f in enumerate(a.FEATURES):test[f]=d[:,i]
        test['unseen_development_scaffold']=~test.scaffold.isin(set(sample.scaffold))
        save(OUT/'data_audit.json',dict(n=len(y),n_positive=int(y.sum()),n_negative=int((y==0).sum()),
            positive_fraction=float(y.mean()),invalid_smiles=0,duplicate_canonical=0,conflicting_labels=0,
            development_canonical_overlap=0,development_source_id_overlap=0,discarded_rows=0,
            n_unseen_scaffold=int(test.unseen_development_scaffold.sum()),structure_q75=cuts))
        print(f'Data checks passed: {len(test)} rows; no records dropped. Frozen inference starts.',flush=True)
        stage='frozen_inference'
        inference_start=time.perf_counter()
        probs,singles=predict(test,d,manifest)
        inference_seconds=time.perf_counter()-inference_start
        for key,p in singles.items():test[key]=p
        for name,p in probs.items():
            test[name]=p
            for policy,t in plan['policies'][name].items():test[f'{name}_{policy}']=(p>=t).astype(int)
        test.to_csv(OUT/'predictions.csv',index=False)
        stage='test_scored'
        rows=[]
        for name,p in probs.items():
            for policy,t in plan['policies'][name].items():
                rows.append(dict(model=name,policy=policy,**probability(y,p),**operating(y,p,t)))
        pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
        reliability=[]
        for name,p in probs.items():
            bins=np.minimum((p*10).astype(int),9)
            for b in range(10):
                select=bins==b
                reliability.append(dict(model=name,bin=b,lower=b/10,upper=(b+1)/10,n=int(select.sum()),
                    mean_probability=float(p[select].mean()) if select.any() else None,
                    observed_positive_fraction=float(y[select].mean()) if select.any() else None))
        pd.DataFrame(reliability).to_csv(OUT/'reliability.csv',index=False)
        subgroups=[]
        for feature,cut in cuts.items():
            high=test[feature].to_numpy()>cut
            for group,mask in [('high',high),('rest',~high)]:
                for name,p in probs.items():
                    for policy,t in plan['policies'][name].items():
                        m=operating(y[mask],p[mask],t)
                        subgroups.append(dict(feature=feature,cutpoint=cut,group=group,model=name,policy=policy,
                            sparse_positive=m['n_positive']<30,sparse_negative=m['n_negative']<30,**m))
        pd.DataFrame(subgroups).to_csv(OUT/'structure_subgroups.csv',index=False)
        unseen=[]
        mask=test.unseen_development_scaffold.to_numpy()
        for name,p in probs.items():
            for policy,t in plan['policies'][name].items():
                m=operating(y[mask],p[mask],t)
                enough=min(m['n_positive'],m['n_negative'])>=100
                unseen.append(dict(model=name,policy=policy,**m,
                    auroc=float(roc_auc_score(y[mask],p[mask])) if enough else None,
                    auroc_omission_reason=None if enough else 'Fewer than 100 in a class'))
        save(OUT/'unseen_scaffold.json',unseen)
        costs=[dict(model=r['model'],policy=r['policy'],relative_fn_cost=lam,
                    loss=(r['fp']+lam*r['fn'])/len(y)) for r in rows for lam in plan['cost_sensitivity']['lambda']]
        pd.DataFrame(costs).to_csv(OUT/'cost_sensitivity.csv',index=False)
        stage='bootstrap'
        boot=bootstrap(y,probs,plan['policies'],plan['uncertainty']['resamples'],plan['uncertainty']['seed'],True)
        boot.to_csv(OUT/'bootstrap_replicates.csv',index=False)
        point=metric_vector(y,probs,plan['policies'])
        intervals=[dict(metric=k,estimate=point[k],lower=float(boot[k].quantile(.025)),upper=float(boot[k].quantile(.975))) for k in boot]
        pd.DataFrame(intervals).to_csv(OUT/'confidence_intervals.csv',index=False)
        primary=next(v for v in intervals if v['metric']=='paired_auroc_difference')
        # Verification from serialized predictions, without another fit or inference.
        loaded=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip')
        assert np.array_equal(loaded.activity,y)
        for r in rows:
            p=loaded[r['model']].to_numpy()
            assert np.array_equal(p,probs[r['model']])
            assert operating(y,p,r['threshold'])=={k:r[k] for k in operating(y,p,r['threshold'])}
            assert r['tp']+r['tn']+r['fp']+r['fn']==len(y)
        check_frozen_candidate.main()
        save(OUT/'execution.json',dict(status='completed',started_utc=marker['started_utc'],completed_utc=now(),
            official_test_scored=True,scoring_executions=1,refit_or_retuning=False,
            elapsed_seconds=time.perf_counter()-start,feature_and_six_model_inference_seconds=inference_seconds,
            cpu_threads=torch.get_num_threads(),primary=primary,
            supports_positive_auroc_difference=primary['lower']>0,
            bootstrap=plan['uncertainty'],verification='PASS: round-trip predictions, confusion totals and unchanged frozen artifacts',
            files_sha256={p.name:a.sha(p) for p in OUT.iterdir() if p.is_file()}))
        print(json.dumps(dict(status='completed',primary=primary,metrics=rows),indent=2),flush=True)
    except Exception:
        save(OUT/'failure.json',dict(status='failed',utc=now(),stage=stage,labels_may_have_been_opened=True,traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    assert args.self_test != args.execute,'Choose exactly one mode.'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    selftest() if args.self_test else execute()
