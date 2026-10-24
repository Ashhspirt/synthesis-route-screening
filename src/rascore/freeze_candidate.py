"""Freeze a concrete candidate and comparator using development data only."""
from pathlib import Path
import json
import time
import threading
import datetime
import improvement_experiments as e
import numpy as np
import pandas as pd
import torch
import psutil
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

OUT=e.a.ROOT/'results'/'frozen_candidate_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'freeze_protocol.json').exists():raise SystemExit('Frozen run exists; never overwrite.')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    validation=pd.read_csv(e.OUT/'validation_selection.csv')
    ranking=validation.groupby('condition').log_loss.mean().sort_values()
    assert ranking.index[0]=='count_standard'
    protocol={'status':'method_locked_before_refit','date':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'candidate':'count_standard','comparator':'binary_standard','selection':'Equal weight mean ensemble validation log loss over six development partitions; no internal-test ranking.',
      'mean_validation_loss':ranking.to_dict(),'ensemble_seeds':[42,43,44],
      'default_policy':'recall90','user_preference':'Prioritize fewer false negatives; accept more false positives.',
      'secondary_policy':'balanced','calibration':'None; no temperature scaling fitted.',
      'final_training_pool':'Same fixed 20000 development molecules; no expansion to full original training pool.',
      'final_split':'GroupShuffleSplit n_splits=1 test_size=0.2 random_state=404 on existing scaffold column.',
      'validation_caveat':'Refit validation drawn from previously used development pool; not an independent evaluation.',
      'refit_policy':'Keep best validation-BCE checkpoint; no refit on validation after threshold selection.',
      'training':{'hidden':[256,128,64],'dropout':.2,'weight_decay':0.,'learning_rate':.001,'batch_size':64,'max_epochs':50,'patience':5,'threads':2},
      'fingerprint':'Morgan radius2 length1024 no chirality; log1p counts; comparator binary, same 5 standardized descriptors.',
      'source_sha256':e.a.sha(Path(__file__)),
      'dependency_sha256':{n:e.a.sha(e.a.ROOT/n) for n in ['improvement_experiments.py','feature_ablation.py','pilot.py','experiment_20k.py']},
      'sample_sha256':e.a.sha(e.a.DATA/'pilot_sample.csv'),'official_test_used':False}
    e.a.pilot.save_json(OUT/'freeze_protocol.json',protocol)
    sample=pd.read_csv(e.a.DATA/'pilot_sample.csv');cache=np.load(e.a.DATA/'pilot_features.npz')
    binary,y=cache['x'].astype(np.float32),cache['y']
    assert (sample.source_split=='train').all()
    counts=np.load(e.OUT/'count_features.npz')['counts'];d=e.a.descriptors(sample)
    assert np.array_equal(counts>0,binary>0)
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=404).split(sample,y,groups=sample.scaffold))
    assert not set(sample.iloc[tr].scaffold)&set(sample.iloc[va].scaffold)
    assert len(np.unique(np.r_[tr,va]))==20000 and len(np.unique(y[tr]))==len(np.unique(y[va]))==2
    np.savez(OUT/'partition_indices.npz',train=tr,validation=va)
    assignments=sample[['source_id','canonical_smiles','scaffold']].copy();assignments['partition']='train';assignments.loc[va,'partition']='validation'
    assignments.to_csv(OUT/'partitions.csv',index=False)
    sc=StandardScaler().fit(d[tr]);scaled=sc.transform(d)
    e.a.pilot.save_json(OUT/'scaler.json',{'features':e.a.FEATURES,'mean':sc.mean_.tolist(),'scale':sc.scale_.tolist(),'n_train':len(tr)})
    np.testing.assert_allclose(sc.mean_,d[tr].mean(0),atol=1e-10)
    np.testing.assert_allclose(sc.scale_,d[tr].std(0),atol=1e-10)
    rows,resources,thresholds,ensemble_predictions=[],[],{},{}
    proc=psutil.Process();finished=threading.Event();rss=[proc.memory_info().rss]
    def monitor():
        while not finished.wait(.05):rss.append(proc.memory_info().rss)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    for condition in ['count_standard','binary_standard']:
        x=e.inputs(binary,counts,scaled,condition);models=[];probs=[]
        for seed in [42,43,44]:
            folder=OUT/condition/f'seed{seed}';folder.mkdir(parents=True)
            model,info=e.fit(x,y,tr,va,seed,condition,folder)
            p=e.a.predict(model,x[va]).astype(np.float64)
            checkpoint=torch.load(folder/'model.pt',weights_only=True,map_location='cpu')
            reloaded=e.network(condition);reloaded.load_state_dict(checkpoint['state_dict'])
            np.testing.assert_allclose(p,e.a.predict(reloaded,x[va]),atol=1e-7,rtol=1e-6)
            hist=pd.read_csv(folder/'history.csv');assert checkpoint['best_epoch']==hist.loc[hist.validation_loss.idxmin(),'epoch']
            np.save(folder/'validation_probability.npy',p)
            models.append(model);probs.append(p);rows.append({'condition':condition,'seed':seed,**info})
            print(f'{condition}/{seed} best epoch {info["best_epoch"]}',flush=True)
        ep=np.mean(probs,axis=0);ensemble_predictions[condition]=ep
        thresholds[condition]={policy:e.threshold(y[va],ep,policy) for policy in ['recall90','balanced']}
        pred=pd.DataFrame({'source_id':sample.iloc[va].source_id.to_numpy(),'label':y[va],'probability':ep})
        pred.to_csv(OUT/condition/'validation_predictions.csv',index=False)
        for policy,t in thresholds[condition].items():
            e.a.pilot.save_json(OUT/condition/f'validation_{policy}.json',e.metrics(y[va],ep,t))
        assert e.metrics(y[va],ep,thresholds[condition]['recall90'])['recall']>=.9
        for nmodels in [1,3]:
            elapsed=[]
            # Warm-up followed by five timings on the same development validation input.
            for repeat in range(6):
                start=time.perf_counter()
                acc=[]
                for m in models[:nmodels]:
                    acc.append(np.concatenate([e.a.predict(m,x[va[i:i+1024]]) for i in range(0,len(va),1024)]))
                avg=np.mean(acc,axis=0)
                if repeat:elapsed.append(time.perf_counter()-start)
            resources.append({'condition':condition,'models':nmodels,'validation_n':len(va),
                'median_prediction_seconds':float(np.median(elapsed)),'min_seconds':min(elapsed),'max_seconds':max(elapsed),
                'checkpoint_bytes':sum((OUT/condition/f'seed{s}'/'model.pt').stat().st_size for s in [42,43,44][:nmodels])})
    start=time.perf_counter();_=e.count_features(sample.iloc[:1000]);_=e.a.descriptors(sample.iloc[:1000])
    featurization_seconds=time.perf_counter()-start
    finished.set();watcher.join()
    pd.DataFrame(rows).to_csv(OUT/'training_info.csv',index=False)
    pd.DataFrame(resources).to_csv(OUT/'resource_benchmark.csv',index=False)
    cost=[]
    for condition,ep in ensemble_predictions.items():
        for policy,t in thresholds[condition].items():
            m=e.metrics(y[va],ep,t)
            for ratio in [.25,.5,1,2,5,10]:
                cost.append({'condition':condition,'policy':policy,'fn_to_fp_cost_ratio':ratio,
                    'fp':m['fp'],'fn':m['fn'],'n':m['n'],'cost_per_molecule':(m['fp']+ratio*m['fn'])/m['n']})
    pd.DataFrame(cost).to_csv(OUT/'validation_cost_sensitivity.csv',index=False)
    e.a.pilot.save_json(OUT/'resources.json',{'cpu_threads':2,'memory_total_gib':psutil.virtual_memory().total/2**30,
        'process_peak_rss_mib_sampled':max(rss)/2**20,'sampling_seconds':.05,'rss_includes_imports_and_feature_arrays':True,
        'first_1000_development_featurization_seconds':featurization_seconds,'training_seconds':sum(r['training_seconds'] for r in rows),
        'inference_excludes_features_disk_load':True,'no_gpu':True})
    files=['freeze_protocol.json','partition_indices.npz','partitions.csv','scaler.json']
    files += [str(p.relative_to(OUT)).replace('\\','/') for p in OUT.glob('*/seed*/model.pt')]
    manifest={'status':'frozen_not_tested','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'candidate':'count_standard','comparator':'binary_standard','seeds':[42,43,44],
        'default_policy':'recall90','thresholds':thresholds,'prediction_rule':'Arithmetic mean of 3 sigmoid outputs; positive if mean >= frozen threshold.',
        'data':{'development_n':len(sample),'train_n':len(tr),'validation_n':len(va),
                'train_positive_fraction':float(y[tr].mean()),'validation_positive_fraction':float(y[va].mean()),
                'train_validation_scaffold_overlap':0,'sample_sha256':protocol['sample_sha256']},
        'files_sha256':{f:e.a.sha(OUT/f) for f in files},'official_test_used':False}
    e.a.pilot.save_json(OUT/'frozen_manifest.json',manifest)
    e.a.pilot.save_json(OUT/'verification.json',{'passed':True,'models_reloaded':6,'best_epochs_checked':True,
       'training_only_scaler_checked':True,'scaffold_disjoint':True,'validation_recall_at_least_90':True,
       'thresholds_frozen':True,'official_test_scored':False})
    print(json.dumps({'train':len(tr),'validation':len(va),'thresholds':thresholds,'resources':resources}),flush=True)


if __name__=='__main__':main()
