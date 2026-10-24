"""Predeclared size x regularization experiment; fixed sample/splits, 3 training seeds."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import platform
import random
import shutil
import time

import pilot  # sets CPU thread and plotting-cache environment before imports
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, balanced_accuracy_score, accuracy_score, precision_score, recall_score)
import joblib

ROOT=Path(__file__).resolve().parent
SEEDS=[42,43,44]
CONFIGS={
    'large_standard': {'hidden':[256,128,64], 'dropout':.2, 'weight_decay':0.},
    'small_standard': {'hidden':[64,32,16], 'dropout':.2, 'weight_decay':0.},
    'large_regularized': {'hidden':[256,128,64], 'dropout':.4, 'weight_decay':1e-4},
    'small_regularized': {'hidden':[64,32,16], 'dropout':.4, 'weight_decay':1e-4},
}


def network(config):
    layers=[]
    n=1024
    for width in config['hidden']:
        layers.extend([nn.Linear(n,width),nn.ReLU(),nn.Dropout(config['dropout'])])
        n=width
    layers.append(nn.Linear(n,1))
    return nn.Sequential(*layers)


def scores(y,p,threshold):
    pred=p>=threshold
    return {'auroc':float(roc_auc_score(y,p)),
        'ap':float(average_precision_score(y,p)),
        'brier':float(brier_score_loss(y,p)),
        'log_loss':float(log_loss(y,p)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'accuracy':float(accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0))}


def fit_nn(x,y,tr,va,config,seed,folder,epochs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model=network(config)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=config['weight_decay'])
    loss_fn=nn.BCEWithLogitsLoss()
    xt=torch.from_numpy(x)
    yt=torch.from_numpy(y.astype(np.float32))
    loader=DataLoader(TensorDataset(xt[tr],yt[tr]),batch_size=64,shuffle=True,
        generator=torch.Generator().manual_seed(seed),num_workers=0)
    best_loss=float('inf')
    stale=0
    history=[]
    start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        train_loss=0.
        for xb,yb in loader:
            optimizer.zero_grad()
            loss=loss_fn(model(xb).squeeze(1),yb)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            train_loss+=float(loss.detach())*len(yb)
        model.eval()
        with torch.no_grad():
            validation_loss=float(loss_fn(model(xt[va]).squeeze(1),yt[va]))
        history.append({'epoch':epoch,'train_loss':train_loss/len(tr),'validation_loss':validation_loss})
        if validation_loss<best_loss:
            best_loss=validation_loss
            best_state=copy.deepcopy(model.state_dict())
            best_epoch=epoch
            stale=0
        else:
            stale+=1
        if stale>=5:
            break
    elapsed=time.perf_counter()-start
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vp=torch.sigmoid(model(xt[va]).squeeze(1)).numpy()
        tp=torch.sigmoid(model(xt[tr]).squeeze(1)).numpy()
    checkpoint={'state_dict':best_state,'config':config,'seed':seed,'best_epoch':best_epoch,
        'fingerprint':{'radius':2,'bits':1024,'includeChirality':False}}
    torch.save(checkpoint,folder/'model.pt')
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    return vp, {'training_seconds':elapsed,'best_epoch':best_epoch,'epochs_run':len(history),
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'train_log_loss_at_best':float(log_loss(y[tr],tp))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'experiment_20000_three_seeds')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--split-seed',type=int,default=42)
    parser.add_argument('--prepared-data',type=Path)
    args=parser.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    # Explicit refusal prevents accidental destruction of completed experiments.
    if (out/'metrics.csv').exists():
        raise SystemExit('Completed results already exist. Choose a new --output directory.')
    start=time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol={'sample_size':20000,'sample_seed':42,'split_seed':args.split_seed,'training_seeds':SEEDS,
        'configs':CONFIGS,'epochs_max':args.epochs,'early_stopping_patience':5,
        'optimizer':'Adam','learning_rate':.001,'batch_size':64,'threads':2,
        'regularization_note':'Dropout and L2 weight decay changed together; cannot isolate their individual effects.',
        'seed_scope':'Fixed sample and fixed partitions; seeds vary initialization, dropout, minibatch order and forest randomness. Not split/subsample uncertainty.',
        'validation_selection':'Lowest mean validation log_loss over 3 training seeds, separately per split, neural candidates only.',
        'test_policy':'Internal evaluation from official training pool only. Official test never scored.',
        'python':platform.python_version(),'torch':torch.__version__,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    pilot.save_json(out/'protocol.json',protocol)
    prepared=out/'data'
    prepared.mkdir(exist_ok=True)
    if args.prepared_data:
        for name in ['pilot_sample.csv','pilot_features.npz','data_audit.json']:
            source=args.prepared_data/name
            target=prepared/name
            if target.exists():
                assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
            else:
                shutil.copy2(source,target)
    if (prepared/'pilot_features.npz').exists():
        audit=json.loads((prepared/'data_audit.json').read_text(encoding='utf-8'))
        assert audit['pilot_sample_rows']==20000 and audit['seed']==42
        sample=pd.read_csv(prepared/'pilot_sample.csv')
        cache=np.load(prepared/'pilot_features.npz')
        x,y=cache['x'].astype(np.float32),cache['y']
        assert np.array_equal(sample.activity.to_numpy(),y)
    else:
        sample,x,y=pilot.audit(pilot.DEFAULT_DATA,prepared,20000,42)
    splits={}
    split_info={}
    validation_rows=[]
    for mode in ['random','scaffold']:
        tr,va,te=pilot.split_indices(sample,y,mode,args.split_seed)
        splits[mode]=(tr,va,te)
        np.savez(out/f'{mode}_indices.npz',train=tr,validation=va,internal_test=te)
        assignment=sample[['source_id','canonical_smiles','scaffold']].copy()
        assignment['partition']=''
        split_info[mode]={}
        for label,idx in zip(['train','validation','internal_test'],[tr,va,te]):
            assignment.loc[idx,'partition']=label
            split_info[mode][label]={'n':len(idx),'positive_fraction':float(y[idx].mean()),
                'scaffolds':int(sample.iloc[idx].scaffold.nunique())}
        assignment.to_csv(out/f'{mode}_partitions.csv',index=False)
        split_info[mode]['train_test_scaffold_overlap']=len(set(sample.iloc[tr].scaffold)&set(sample.iloc[te].scaffold))
        split_info[mode]['train_test_fingerprint_overlap']=len(set(map(bytes,x[tr].astype(np.uint8)))&set(map(bytes,x[te].astype(np.uint8))))
        print(f'{mode}: {len(tr)}/{len(va)}/{len(te)} train/validation/internal_test',flush=True)
        for seed in SEEDS:
            baselines={
                'majority':DummyClassifier(strategy='prior'),
                'logistic':LogisticRegression(C=1.,solver='liblinear',max_iter=1000,random_state=seed),
                'random_forest':RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=2,n_jobs=2,random_state=seed),
            }
            for name in list(baselines)+list(CONFIGS):
                folder=out/'runs'/mode/f'seed{seed}'/name
                folder.mkdir(parents=True,exist_ok=True)
                if name in baselines:
                    model=baselines[name]
                    t0=time.perf_counter()
                    model.fit(x[tr],y[tr])
                    info={'training_seconds':time.perf_counter()-t0,'best_epoch':0,'epochs_run':0,
                        'parameter_count':0,'train_log_loss_at_best':float(log_loss(y[tr],model.predict_proba(x[tr])[:,1]))}
                    vp=model.predict_proba(x[va])[:,1]
                    joblib.dump(model,folder/'model.joblib')
                else:
                    vp,info=fit_nn(x,y,tr,va,CONFIGS[name],seed,folder,args.epochs)
                thresholds=np.arange(.05,.951,.01)
                threshold=.5 if name=='majority' else float(thresholds[np.argmax([
                    balanced_accuracy_score(y[va],vp>=v) for v in thresholds])])
                row={'split':mode,'seed':seed,'model':name,'threshold':threshold,**info,**scores(y[va],vp,threshold)}
                row['generalization_gap']=row['log_loss']-row['train_log_loss_at_best']
                pilot.save_json(folder/'validation.json',row)
                pd.DataFrame({'source_id':sample.iloc[va].source_id.to_numpy(),'label':y[va],'probability':vp}).to_csv(folder/'validation_predictions.csv',index=False)
                validation_rows.append(row)
                print(f'{mode}/seed{seed}/{name}: validation BCE={row["log_loss"]:.4f}, best epoch={info["best_epoch"]}, seconds={info["training_seconds"]:.2f}',flush=True)
    pilot.save_json(out/'split_checks.json',split_info)
    val=pd.DataFrame(validation_rows)
    val.to_csv(out/'validation_metrics.csv',index=False)
    selection={}
    for mode in splits:
        means=val[(val.split==mode)&val.model.isin(CONFIGS)].groupby('model').log_loss.mean()
        selection[mode]={'selected_neural_config':means.idxmin(),'mean_validation_log_loss':means.to_dict()}
    # Freeze config choices BEFORE calculating internal test metrics.
    pilot.save_json(out/'validation_selection.json',selection)
    rows=[]
    for mode,(tr,va,te) in splits.items():
        for seed in SEEDS:
            for name in ['majority','logistic','random_forest']+list(CONFIGS):
                folder=out/'runs'/mode/f'seed{seed}'/name
                row=json.loads((folder/'validation.json').read_text(encoding='utf-8'))
                if name in CONFIGS:
                    ckpt=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
                    model=network(ckpt['config'])
                    model.load_state_dict(ckpt['state_dict'])
                    model.eval()
                    with torch.no_grad():
                        p=torch.sigmoid(model(torch.from_numpy(x[te])).squeeze(1)).numpy()
                else:
                    p=joblib.load(folder/'model.joblib').predict_proba(x[te])[:,1]
                test_metrics=scores(y[te],p,row['threshold'])
                row.update(test_metrics)
                row['internal_test_n']=len(te)
                row['positive_fraction']=float(y[te].mean())
                row['selected_neural_config']=name==selection[mode]['selected_neural_config']
                # This gap is a VALIDATION-minus-training gap; retain explicit column name.
                row['validation_generalization_gap']=row.pop('generalization_gap')
                rows.append(row)
                pd.DataFrame({'source_id':sample.iloc[te].source_id.to_numpy(),'label':y[te],
                    'probability':p,'prediction':(p>=row['threshold']).astype(int)}).to_csv(folder/'internal_test_predictions.csv',index=False)
    metrics=pd.DataFrame(rows)
    metrics.to_csv(out/'metrics.csv',index=False)
    columns=['auroc','ap','brier','log_loss','balanced_accuracy','training_seconds',
        'best_epoch','parameter_count','validation_generalization_gap']
    summary=metrics.groupby(['split','model'])[columns].agg(['mean','std'])
    summary.columns=['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(out/'summary.csv',index=False)
    contrasts=[]
    pairs=[('size_at_standard','small_standard','large_standard'),
        ('regularization_at_large','large_regularized','large_standard'),
        ('regularization_at_small','small_regularized','small_standard'),
        ('combined','small_regularized','large_standard')]
    for mode in splits:
        subset=metrics[metrics.split==mode]
        for label,new,old in pairs:
            a=subset[subset.model==new].set_index('seed')
            b=subset[subset.model==old].set_index('seed')
            for metric in ['auroc','ap','brier','log_loss','validation_generalization_gap']:
                delta=a[metric]-b[metric]
                contrasts.append({'split':mode,'contrast':label,'metric':metric,
                    'mean_paired_delta':float(delta.mean()),'sd_paired_delta':float(delta.std(ddof=1)),
                    'delta_seed42':float(delta.loc[42]),'delta_seed43':float(delta.loc[43]),'delta_seed44':float(delta.loc[44])})
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False)
    pilot.save_json(out/'completion.json',{'complete':True,'fitted_models':len(rows),'neural_training_runs':24,
        'elapsed_seconds':time.perf_counter()-start,'official_test_scored':False})
    print(summary[['auroc_mean','auroc_std','brier_mean','brier_std']].to_string(),flush=True)
    print('Completed all 42 fits; 24 neural runs. Official test not scored.',flush=True)


if __name__=='__main__':
    main()
