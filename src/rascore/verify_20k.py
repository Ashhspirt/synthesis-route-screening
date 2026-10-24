"""Reload all saved models and independently verify the experiment artifacts."""
from pathlib import Path
import argparse
import json
import zipfile
import io
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results'/'experiment_20000_three_seeds')
    out=parser.parse_args().output
    torch.set_num_threads(2)
    sample=pd.read_csv(out/'data'/'pilot_sample.csv')
    cache=np.load(out/'data'/'pilot_features.npz')
    x,y=cache['x'].astype(np.float32),cache['y']
    assert len(sample)==20000 and sample.source_id.is_unique and sample.canonical_smiles.is_unique
    assert (sample.source_split=='train').all() and np.array_equal(sample.activity,y)
    # Read original IDs to establish that the official test set was not reused.
    archive=Path(__file__).resolve().parents[2]/'work'/'rascore-data'/'data.zip'
    with zipfile.ZipFile(archive) as z:
        official_train=pd.read_csv(io.BytesIO(z.read('data/uspto_chembl_classification_train.csv')),index_col=0)
        official_test=pd.read_csv(io.BytesIO(z.read('data/uspto_chembl_classification_test.csv')),index_col=0)
    assert set(sample.source_id)<=set(official_train.index)
    assert not set(sample.source_id)&set(official_test.index)
    val=pd.read_csv(out/'validation_metrics.csv')
    test=pd.read_csv(out/'metrics.csv')
    selection=json.loads((out/'validation_selection.json').read_text())
    assert len(test)==len(val)==42
    assert set(test.seed)=={42,43,44}
    checks=[]
    for mode in ['random','scaffold']:
        indices=np.load(out/f'{mode}_indices.npz')
        train,valid,held=[indices[k] for k in ['train','validation','internal_test']]
        assert sorted(np.concatenate([train,valid,held]))==list(range(20000))
        for a,b in [(train,valid),(train,held),(valid,held)]:
            assert not set(sample.iloc[a].canonical_smiles)&set(sample.iloc[b].canonical_smiles)
            if mode=='scaffold':
                assert not set(sample.iloc[a].scaffold)&set(sample.iloc[b].scaffold)
        for row in test[test.split==mode].itertuples(index=False):
            folder=out/'runs'/mode/f'seed{row.seed}'/row.model
            if (folder/'model.pt').exists():
                ckpt=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
                # Reconstruct without using the experiment's model-building helper.
                modules=[]
                width=1024
                for h in ckpt['config']['hidden']:
                    modules += [nn.Linear(width,h),nn.ReLU(),nn.Dropout(ckpt['config']['dropout'])]
                    width=h
                modules.append(nn.Linear(width,1))
                model=nn.Sequential(*modules)
                model.load_state_dict(ckpt['state_dict'])
                model.eval()
                def predict(a):
                    with torch.no_grad():
                        return torch.sigmoid(model(torch.from_numpy(a)).squeeze(1)).numpy()
                history=pd.read_csv(folder/'history.csv')
                assert ckpt['best_epoch']==int(history.loc[history.validation_loss.idxmin(),'epoch'])
                assert sum(p.numel() for p in model.parameters())==row.parameter_count
                assert history.shape[0]==row.epochs_run
            else:
                model=joblib.load(folder/'model.joblib')
                def predict(a):
                    return model.predict_proba(a)[:,1]
            for part,idx,table in [('validation',valid,val),('internal_test',held,test)]:
                saved=pd.read_csv(folder/f'{part}_predictions.csv')
                assert np.array_equal(saved.source_id.to_numpy(),sample.iloc[idx].source_id.to_numpy())
                assert np.array_equal(saved.label.to_numpy(),y[idx])
                assert np.allclose(saved.probability,predict(x[idx]),atol=1e-6)
                recorded=table[(table.split==mode)&(table.seed==row.seed)&(table.model==row.model)].iloc[0]
                for metric,fn in [('auroc',roc_auc_score),('ap',average_precision_score),('brier',brier_score_loss),('log_loss',log_loss)]:
                    assert abs(fn(saved.label,saved.probability)-recorded[metric])<2e-6
            checks.append({'split':mode,'seed':row.seed,'model':row.model,'passed':True})
        neural=val[(val.split==mode)&val.model.str.startswith(('large_','small_'))]
        selected=neural.groupby('model').log_loss.mean().idxmin()
        assert selected==selection[mode]['selected_neural_config']
    summary=pd.read_csv(out/'summary.csv')
    for row in summary.itertuples(index=False):
        g=test[(test.split==row.split)&(test.model==row.model)]
        assert len(g)==3
        assert abs(g.auroc.mean()-row.auroc_mean)<1e-10
        assert abs(g.auroc.std(ddof=1)-row.auroc_std)<1e-10
    report={'passed':True,'models_checked':len(checks),'prediction_sets_checked':2*len(checks),
        'official_test_excluded':True,'checkpoint_reloads_match':True,'metrics_recalculated':True,
        'validation_only_config_selection_verified':True,'aggregate_mean_sd_verified':True,
        'checks':checks}
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))


if __name__=='__main__':
    main()
