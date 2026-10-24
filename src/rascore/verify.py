"""Independent checks of saved partitions, metrics, and model reloads."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results'/'pilot_5000_seed42')
    folder=parser.parse_args().output
    sample=pd.read_csv(folder/'pilot_sample.csv')
    data=np.load(folder/'pilot_features.npz')
    assert len(sample)==len(data['x'])
    assert set(sample.source_split)=={'train'}
    assert sample.canonical_smiles.is_unique
    metrics=pd.read_csv(folder/'metrics.csv')
    checks=[]
    torch.set_num_threads(2)
    for split in ['random','scaffold']:
        assignment=pd.read_csv(folder/f'{split}_split.csv')
        assert assignment.source_id.tolist()==sample.source_id.tolist()
        assert set(assignment.partition)=={'train','validation','pilot_test'}
        if split=='scaffold':
            assert assignment.groupby('scaffold').partition.nunique().max()==1
        held=np.flatnonzero((assignment.partition=='pilot_test').to_numpy())
        for name in ['majority','logistic','random_forest','mlp']:
            saved=pd.read_csv(folder/f'{split}_{name}_predictions.csv')
            assert set(saved.source_id)==set(sample.iloc[held].source_id)
            assert saved.source_id.is_unique and sample.source_id.is_unique
            ordered=sample.reset_index().set_index('source_id').loc[saved.source_id,'index'].to_numpy()
            assert np.array_equal(saved.activity.to_numpy(),data['y'][ordered])
            assert np.isfinite(saved.predicted_probability).all()
            assert saved.predicted_probability.between(0,1).all()
            row=metrics[(metrics.split==split)&(metrics.model==name)].iloc[0]
            assert abs(roc_auc_score(saved.activity,saved.predicted_probability)-row.auroc)<1e-7
            assert abs(average_precision_score(saved.activity,saved.predicted_probability)-row.average_precision)<1e-7
        checkpoint=torch.load(folder/f'{split}_mlp.pt',map_location='cpu',weights_only=True)
        model=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Dropout(.2),
            nn.Linear(256,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,64),
            nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        saved=pd.read_csv(folder/f'{split}_mlp_predictions.csv')
        ordered=sample.reset_index().set_index('source_id').loc[saved.source_id,'index'].to_numpy()
        with torch.no_grad():
            p=torch.sigmoid(model(torch.from_numpy(data['x'][ordered].astype(np.float32))).squeeze(1)).numpy()
        assert np.allclose(p,saved.predicted_probability.to_numpy(),atol=1e-6)
        history=pd.read_csv(folder/'learning_curves.csv')
        h=history[history.split==split]
        assert checkpoint['best_epoch']==int(h.loc[h.validation_loss.idxmin(),'epoch'])
        checks.append({'split':split,'rows':len(held),'checkpoint_prediction_match':True,'metric_recalculation_match':True,'best_epoch_is_validation_minimum':True})
    report={'passed':True,'official_test_scored':False,'checks':checks}
    (folder/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
