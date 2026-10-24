"""Count fingerprints x regularization, repeated fits and validation-only thresholds."""
from pathlib import Path
import argparse
import copy
import json
import random
import time
import feature_ablation as a
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, log_loss

OUT = a.ROOT/'results'/'improvement_round2'
CONDITIONS = {
    'binary_standard': {'representation':'binary', 'dropout':.2,'weight_decay':0.},
    'binary_regularized': {'representation':'binary','dropout':.4,'weight_decay':1e-4},
    'count_standard': {'representation':'log1p_count','dropout':.2,'weight_decay':0.},
    'count_regularized': {'representation':'log1p_count','dropout':.4,'weight_decay':1e-4},
}
POLICIES = ['balanced','recall90']


def count_features(sample):
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024,includeChirality=False)
    rows=[generator.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in sample.canonical_smiles]
    raw=np.array(rows,dtype=np.uint32)
    return raw


def network(condition):
    model=a.network(1029)
    for layer in model:
        if isinstance(layer,nn.Dropout):
            layer.p=CONDITIONS[condition]['dropout']
    return model


def inputs(binary,counts,scaled,condition):
    fp=binary if CONDITIONS[condition]['representation']=='binary' else np.log1p(counts.astype(np.float32))
    return np.ascontiguousarray(np.concatenate([fp,scaled],axis=1),dtype=np.float32)


def threshold(y,p,policy):
    if policy=='balanced':
        grid=np.arange(.05,.951,.01)
        return float(grid[np.argmax([balanced_accuracy_score(y,p>=t) for t in grid])])
    # Largest threshold retaining >=90% validation positives; ties retained by >=.
    positive=np.sort(p[y==1])[::-1]
    return float(positive[int(np.ceil(.9*len(positive)))-1])


def metrics(y,p,t):
    result=a.scores(y,p,t)
    pred=p>=t
    n0,n1=int((y==0).sum()),int((y==1).sum())
    fp,fn=int(((y==0)&pred).sum()),int(((y==1)&~pred).sum())
    result.update(n=len(y),n_negative=n0,n_positive=n1,fp=fp,fn=fn,
                  fpr=fp/n0,fnr=fn/n1,threshold=t)
    return result


def fit(x,y,tr,va,seed,condition,folder):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=network(condition)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=CONDITIONS[condition]['weight_decay'])
    loss_fn=nn.BCEWithLogitsLoss()
    xt,yt=torch.from_numpy(x),torch.from_numpy(y.astype(np.float32))
    loader=DataLoader(TensorDataset(xt[tr],yt[tr]),batch_size=64,shuffle=True,
                      generator=torch.Generator().manual_seed(seed),num_workers=0)
    best,stale,history=float('inf'),0,[]
    start=time.perf_counter()
    for epoch in range(1,51):
        model.train();train_loss=0.
        for xb,yb in loader:
            optimizer.zero_grad();loss=loss_fn(model(xb).squeeze(1),yb)
            assert torch.isfinite(loss)
            loss.backward();optimizer.step();train_loss+=float(loss.detach())*len(yb)
        model.eval()
        with torch.no_grad(): vl=float(loss_fn(model(xt[va]).squeeze(1),yt[va]))
        history.append({'epoch':epoch,'train_loss':train_loss/len(tr),'validation_loss':vl})
        if vl<best:
            best,stale,best_epoch=vl,0,epoch;state=copy.deepcopy(model.state_dict())
        else: stale+=1
        if stale>=5:break
    model.load_state_dict(state)
    torch.save({'state_dict':state,'condition':condition,'seed':seed,'best_epoch':best_epoch},folder/'model.pt')
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    return model,{'best_epoch':best_epoch,'epochs_run':len(history),'training_seconds':time.perf_counter()-start,
                  'parameter_count':sum(p.numel() for p in model.parameters()),
                  'train_log_loss':float(log_loss(y[tr],a.predict(model,x[tr])))}


def load_model(folder,condition):
    if condition=='binary_standard':
        ref=json.loads((folder/'baseline_reference.json').read_text())
        path=Path(ref['path'])
        assert a.sha(path)==ref['sha256']
    else:path=folder/'model.pt'
    cp=torch.load(path,map_location='cpu',weights_only=True)
    model=network(condition);model.load_state_dict(cp['state_dict'])
    return model,cp,path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    out=parser.parse_args().output;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise SystemExit('Existing protocol/results protected; select a fresh --output.')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol={'conditions':CONDITIONS,'split_seeds':a.SPLIT_SEEDS,'training_seeds':a.TRAIN_SEEDS,
        'n_sample':20000,'new_fits':54,'reused_fits':18,'hidden':[256,128,64],'descriptors':a.FEATURES,
        'fingerprint':'radius 2, size 1024, no chirality; binary vs log1p folded counts. Same occupied bits checked.',
        'training':'Adam lr .001, batch64, max50 epochs, patience5, validation BCE; CPU2 threads',
        'preprocessing':'Descriptor StandardScaler fit only on training partition; count log1p is fixed, not fitted.',
        'regularization':'Dropout and weight decay changed together; cannot isolate their effects.',
        'paired_initialization':'Same input dimension and initialization seed in all four conditions; same training seed schedule. Changing dropout changes optimization.',
        'selection':'Lowest ensemble validation log loss over four conditions, separately per split; freeze choices and thresholds before new internal-test scoring.',
        'ensemble':'Equal arithmetic mean of three seed probabilities; no learned weights; 3x model storage/inference.',
        'thresholds':{'balanced':'Validation balanced accuracy grid .05:.01:.95; first maximum.',
                      'recall90':'Largest observed positive validation score retaining >=90% validation positives. Exploratory tradeoff, no test guarantee.'},
        'primary_metrics':['auroc','brier','log_loss'],'error_metrics':['fpr','fnr','corrected','new_errors'],
        'error_groups':'Training-only q75 high MolWt/RingCount/HBD/HBA versus remainder, selected ensemble vs binary ensemble, matched policies. Counts and sparse denominators included.',
        'aggregation':'Average three training seeds then summarize 3 split means; ensembles are 3 partition results, not 9 runs. Repeated holdouts overlap.',
        'test_policy':'Previously viewed internal partitions are development-only. Official test untouched.',
        'label_scope':'Historical planner route found; not laboratory synthesizability.',
        'sample_sha256':a.sha(a.DATA/'pilot_sample.csv'),'fingerprint_sha256':a.sha(a.DATA/'pilot_features.npz'),
        'source_sha256':a.sha(Path(__file__)),
        'dependency_sha256':{n:a.sha(a.ROOT/n) for n in ['feature_ablation.py','experiment_20k.py','pilot.py']},
        'split_sha256':{f'{ss}/{mode}':a.sha(a.MAIN/f'split_{ss}'/f'{mode}_indices.npz') for ss in a.SPLIT_SEEDS for mode in ['random','scaffold']}}
    a.pilot.save_json(out/'protocol.json',protocol)
    sample=pd.read_csv(a.DATA/'pilot_sample.csv');cache=np.load(a.DATA/'pilot_features.npz')
    binary,y=cache['x'].astype(np.float32),cache['y']
    assert (sample.source_split=='train').all() and np.array_equal(y,sample.activity)
    d=a.descriptors(sample);counts=count_features(sample)
    assert np.array_equal(counts>0,binary>0)
    np.savez_compressed(out/'count_features.npz',counts=counts)
    pd.DataFrame(d,columns=a.FEATURES).assign(source_id=sample.source_id).to_csv(out/'descriptors.csv',index=False)
    a.pilot.save_json(out/'feature_audit.json',{'count_max':int(counts.max()),'shape':list(counts.shape),
        'count_occupied_bits_match_binary':True,'count_sha256':a.sha(out/'count_features.npz')})
    validation_rows,info_rows=[],[]
    for ss in a.SPLIT_SEEDS:
        for mode in ['random','scaffold']:
            part=out/f'split_{ss}'/mode;part.mkdir(parents=True)
            idx=np.load(a.MAIN/f'split_{ss}'/f'{mode}_indices.npz');tr,va=idx['train'],idx['validation']
            scaler=StandardScaler().fit(d[tr]);scaled=scaler.transform(d)
            a.pilot.save_json(part/'scaler.json',{'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'n_train':len(tr)})
            for condition in CONDITIONS:
                x=inputs(binary,counts,scaled,condition);probs=[]
                for seed in a.TRAIN_SEEDS:
                    folder=part/f'seed{seed}'/condition;folder.mkdir(parents=True)
                    if condition=='binary_standard':
                        old=a.OUT/f'split_{ss}'/mode/f'seed{seed}'/'fp_all'
                        a.pilot.save_json(folder/'baseline_reference.json',{'path':str(old/'model.pt'),'sha256':a.sha(old/'model.pt')})
                        model,_,_=load_model(folder,condition)
                        oldinfo=json.loads((old/'validation.json').read_text())
                        info={k:oldinfo[k] for k in ['best_epoch','epochs_run','training_seconds','parameter_count','train_log_loss']}
                        vp=a.predict(model,x[va]);saved=pd.read_csv(old/'validation_predictions.csv')
                        assert np.array_equal(saved.source_id,sample.iloc[va].source_id)
                        np.testing.assert_allclose(vp,saved.probability,atol=1e-7,rtol=1e-6)
                    else:
                        model,info=fit(x,y,tr,va,seed,condition,folder);vp=a.predict(model,x[va])
                    key={'split_seed':ss,'split':mode,'condition':condition,'seed':seed,'kind':'single'}
                    info_rows.append({**key,**info});probs.append(vp.astype(np.float64))
                    pd.DataFrame({'source_id':sample.iloc[va].source_id.to_numpy(),'label':y[va],'probability':vp.astype(np.float64)}).to_csv(folder/'validation_predictions.csv',index=False)
                    for policy in POLICIES:
                        t=threshold(y[va],vp,policy)
                        validation_rows.append({**key,'policy':policy,**metrics(y[va],vp,t)})
                    print(f'{ss}/{mode}/{condition}/{seed}: val loss {log_loss(y[va],vp):.4f}, epoch {info["best_epoch"]}',flush=True)
                ep=np.mean(probs,axis=0);ensemble=part/'ensemble'/condition;ensemble.mkdir(parents=True)
                pd.DataFrame({'source_id':sample.iloc[va].source_id.to_numpy(),'label':y[va],'probability':ep}).to_csv(ensemble/'validation_predictions.csv',index=False)
                for policy in POLICIES:
                    t=threshold(y[va],ep,policy)
                    validation_rows.append({'split_seed':ss,'split':mode,'condition':condition,'seed':-1,'kind':'ensemble','policy':policy,**metrics(y[va],ep,t)})
    validation=pd.DataFrame(validation_rows);validation.to_csv(out/'validation_metrics.csv',index=False)
    pd.DataFrame(info_rows).to_csv(out/'training_info.csv',index=False)
    choices=validation[(validation.kind=='ensemble')&(validation.policy=='balanced')][['split_seed','split','condition','log_loss']].copy()
    choices['selected']=choices.log_loss==choices.groupby(['split_seed','split']).log_loss.transform('min')
    choices.to_csv(out/'validation_selection.csv',index=False)
    # All thresholds and choices saved before any new internal test predictions.
    results=[]
    for ss in a.SPLIT_SEEDS:
        for mode in ['random','scaffold']:
            part=out/f'split_{ss}'/mode;idx=np.load(a.MAIN/f'split_{ss}'/f'{mode}_indices.npz');te=idx['internal_test']
            sc=json.loads((part/'scaler.json').read_text());scaled=(d-np.array(sc['mean']))/np.array(sc['scale'])
            for condition in CONDITIONS:
                x=inputs(binary,counts,scaled,condition);probs=[]
                for seed in a.TRAIN_SEEDS:
                    folder=part/f'seed{seed}'/condition;model,_,_=load_model(folder,condition)
                    p=a.predict(model,x[te]).astype(np.float64);probs.append(p)
                    pd.DataFrame({'source_id':sample.iloc[te].source_id.to_numpy(),'label':y[te],'probability':p}).to_csv(folder/'internal_test_predictions.csv',index=False)
                    for policy in POLICIES:
                        row=validation[(validation.split_seed==ss)&(validation.split==mode)&(validation.condition==condition)&(validation.seed==seed)&(validation.policy==policy)].iloc[0]
                        results.append({k:row[k] for k in ['split_seed','split','condition','seed','kind','policy']}|metrics(y[te],p,row.threshold))
                ep=np.mean(probs,axis=0)
                pd.DataFrame({'source_id':sample.iloc[te].source_id.to_numpy(),'label':y[te],'probability':ep}).to_csv(part/'ensemble'/condition/'internal_test_predictions.csv',index=False)
                for policy in POLICIES:
                    row=validation[(validation.split_seed==ss)&(validation.split==mode)&(validation.condition==condition)&(validation.kind=='ensemble')&(validation.policy==policy)].iloc[0]
                    results.append({k:row[k] for k in ['split_seed','split','condition','seed','kind','policy']}|metrics(y[te],ep,row.threshold))
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
    print('Completed 54 new fits, 18 reused models, 24 equal-weight ensembles and two threshold policies.',flush=True)


if __name__=='__main__':main()
