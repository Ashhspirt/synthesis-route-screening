"""CPU pilot: audit official data, train on a sample of original training data only."""
from pathlib import Path
import argparse
import copy
import hashlib
import io
import json
import os
import platform
import random
import time
import urllib.request
import zipfile

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split, GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parents[1] / 'work' / 'rascore-data'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT.parents[1] / 'work' / 'matplotlib-cache'))
SOURCE_COMMIT = 'cb77db503ee5cbf0e8bb8963df6e5b76b3a94f06'
URL = f'https://raw.githubusercontent.com/reymond-group/RAscore/{SOURCE_COMMIT}/data/data.zip'
EXPECTED_SHA256 = 'f73b3028592ff95feeeb2f4e0285005d956976c810a3818fc399c20480979ef3'


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def audit(data_dir, out, sample_size, seed):
    t0 = time.perf_counter()
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / 'data.zip'
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_SHA256, 'Dataset archive checksum mismatch'
    frames = []
    with zipfile.ZipFile(archive) as z:
        for split in ['train', 'test']:
            name = f'data/uspto_chembl_classification_{split}.csv'
            frame = pd.read_csv(io.BytesIO(z.read(name)), index_col=0)
            frame.index.name = 'source_id'
            frame = frame.reset_index()
            frame['source_split'] = split
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    counts = raw.groupby(['source_split', 'activity']).size()
    report = {
        'data_url': URL,
        'repository_commit': SOURCE_COMMIT,
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'archive_bytes': archive.stat().st_size,
        'columns': raw.columns.tolist(),
        'raw_rows': len(raw),
        'raw_class_counts': {f'{a}:label_{b}': int(v) for (a, b), v in counts.items()},
        'label_meaning': 'Historical AiZynthFinder solved/unsolved; not wet-lab feasibility.',
        'test_policy': 'Official test audited for quality/overlap only; excluded from pilot training and performance evaluation.',
        'canonicalization': 'RDKit canonical isomeric SMILES, retaining salts and stereochemistry; no tautomer normalization.',
    }
    rejected = []
    canonical = []
    RDLogger.DisableLog('rdApp.*')
    for i, row in enumerate(raw.itertuples(index=False)):
        reason = None
        smi = row.smi
        if row.activity not in (0, 1):
            reason = 'invalid_label'
        elif not isinstance(smi, str) or not smi.strip():
            reason = 'missing_smiles'
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumAtoms() == 0:
                reason = 'unparseable_smiles'
        if reason:
            canonical.append(None)
            rejected.append({'source_id': row.source_id, 'source_split': row.source_split, 'reason': reason, 'smi': smi})
        else:
            canonical.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        if (i + 1) % 20000 == 0:
            print(f'Audit: {i+1}/{len(raw)} structures checked', flush=True)
    raw['canonical_smiles'] = canonical
    good = raw[raw.canonical_smiles.notna()].copy()
    label_n = good.groupby('canonical_smiles').activity.nunique()
    conflict_keys = set(label_n[label_n > 1].index)
    conflicts = good[good.canonical_smiles.isin(conflict_keys)]
    conflicts.to_csv(out / 'conflicting_labels.csv', index=False)
    pd.DataFrame(rejected, columns=['source_id', 'source_split', 'reason', 'smi']).to_csv(out / 'rejected_rows.csv', index=False)
    good = good[~good.canonical_smiles.isin(conflict_keys)]
    train = good[good.source_split == 'train'].copy()
    test = good[good.source_split == 'test'].copy()
    overlap = set(train.canonical_smiles) & set(test.canonical_smiles)
    train[train.canonical_smiles.isin(overlap)].to_csv(out / 'official_split_overlap.csv', index=False)
    report.update({
        'rejected_rows': len(rejected), 'conflicting_structures': len(conflict_keys),
        'conflicting_rows': len(conflicts),
        'train_duplicate_rows_after_conflict_removal': int(train.canonical_smiles.duplicated().sum()),
        'test_duplicate_rows_after_conflict_removal': int(test.canonical_smiles.duplicated().sum()),
        'official_split_overlapping_structures': len(overlap),
    })
    train = train.drop_duplicates('canonical_smiles')
    train = train[~train.canonical_smiles.isin(overlap)].reset_index(drop=True)
    report['clean_training_pool_rows'] = len(train)
    report['clean_official_test_unique_rows'] = test.canonical_smiles.nunique()
    if sample_size >= len(train):
        sample = train.copy()
    else:
        sample, _ = train_test_split(train, train_size=sample_size, stratify=train.activity, random_state=seed)
    sample = sample.reset_index(drop=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=False)
    x = np.empty((len(sample), 1024), dtype=np.uint8)
    scaffolds = []
    feature_start = time.perf_counter()
    for i, smi in enumerate(sample.canonical_smiles):
        mol = Chem.MolFromSmiles(smi)
        x[i] = generator.GetFingerprintAsNumPy(mol)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.append(scaffold or '__ACYCLIC__')
    sample['scaffold'] = scaffolds
    sample.to_csv(out / 'pilot_sample.csv', index=False)
    np.savez_compressed(out / 'pilot_features.npz', x=x, y=sample.activity.to_numpy(dtype=np.int64))
    report.update({
        'pilot_sample_rows': len(sample), 'pilot_positive_fraction': float(sample.activity.mean()),
        'pilot_unique_scaffolds': int(sample.scaffold.nunique()),
        'pilot_acyclic_rows': int((sample.scaffold == '__ACYCLIC__').sum()),
        'empty_scaffold_policy': 'All acyclic molecules form one group; never spread across scaffold splits.',
        'fingerprint': {'type': 'Morgan binary', 'radius': 2, 'bits': 1024, 'includeChirality': False},
        'fingerprint_note': 'Labels retain stereochemistry; chirality-blind features are an explicit pilot limitation.',
        'fingerprint_and_scaffold_seconds': time.perf_counter()-feature_start,
        'audit_and_preparation_seconds': time.perf_counter()-t0,
        'seed': seed, 'rdkit_version': rdBase.rdkitVersion,
    })
    save_json(out / 'data_audit.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return sample, x.astype(np.float32), sample.activity.to_numpy(dtype=np.int64)


def split_indices(sample, y, mode, seed):
    idx = np.arange(len(y))
    if mode == 'random':
        tr, rest = train_test_split(idx, test_size=.30, stratify=y, random_state=seed)
        va, te = train_test_split(rest, test_size=.5, stratify=y[rest], random_state=seed)
    else:
        groups = sample.scaffold.to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, train_size=.70, random_state=seed)
        tr, rest = next(splitter.split(idx, y, groups))
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed).split(rest, y[rest], groups[rest]))
        va, te = rest[a], rest[b]
    sets = [tr, va, te]
    assert len(set(np.concatenate(sets))) == len(y)
    for i, a in enumerate(sets):
        assert len(np.unique(y[a])) == 2, 'Each partition must contain both labels.'
        assert (sample.iloc[a].source_split == 'train').all()
        for b in sets[i+1:]:
            assert not set(sample.iloc[a].canonical_smiles) & set(sample.iloc[b].canonical_smiles)
            if mode == 'scaffold':
                assert not set(sample.iloc[a].scaffold) & set(sample.iloc[b].scaffold)
    return tr, va, te


def train(sample, x, y, out, seed, epochs):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
        balanced_accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, log_loss)
    from sklearn.calibration import calibration_curve
    import sklearn
    import psutil
    import matplotlib.pyplot as plt
    import joblib

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    save_json(out / 'environment.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'processor': platform.processor(), 'cpu_logical_threads': os.cpu_count(),
        'ram_gib': psutil.virtual_memory().total/2**30,
        'available_ram_gib_at_start': psutil.virtual_memory().available/2**30,
        'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
        'sklearn': sklearn.__version__, 'rdkit': rdBase.rdkitVersion,
        'device': 'cpu', 'training_threads': 2,
    })
    metrics = []
    histories = []
    checks = {}
    plot_data = {}
    for mode in ['random', 'scaffold']:
        tr, va, te = split_indices(sample, y, mode, seed)
        assignment = sample[['source_id', 'canonical_smiles', 'scaffold']].copy()
        assignment['partition'] = ''
        for label, inds in zip(['train','validation','pilot_test'], [tr,va,te]):
            assignment.loc[inds, 'partition'] = label
        assignment.to_csv(out / f'{mode}_split.csv', index=False)
        checks[mode] = {label: {'rows':len(inds), 'positive_fraction':float(y[inds].mean()),
            'scaffolds':int(sample.iloc[inds].scaffold.nunique())}
            for label, inds in zip(['train','validation','pilot_test'], [tr,va,te])}
        checks[mode]['canonical_overlap'] = 0
        checks[mode]['scaffold_overlap_train_test'] = len(set(sample.iloc[tr].scaffold) & set(sample.iloc[te].scaffold))
        checks[mode]['feature_overlap_train_test'] = len(set(map(bytes,x[tr].astype(np.uint8))) & set(map(bytes,x[te].astype(np.uint8))))
        print(f'{mode}: train={len(tr)}, validation={len(va)}, pilot_test={len(te)}', flush=True)
        model_dict = {
            'majority': DummyClassifier(strategy='prior'),
            'logistic': LogisticRegression(C=1.0, max_iter=1000, solver='liblinear', random_state=seed),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2, n_jobs=2, random_state=seed),
        }
        predictions = {}
        for name, model in model_dict.items():
            t0 = time.perf_counter()
            model.fit(x[tr], y[tr])
            elapsed = time.perf_counter()-t0
            start = time.perf_counter()
            vprob, prob = model.predict_proba(x[va])[:,1], model.predict_proba(x[te])[:,1]
            predictions[name] = (vprob,prob,elapsed,time.perf_counter()-start)
            joblib.dump(model, out / f'{mode}_{name}.joblib')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(1024,256), nn.ReLU(), nn.Dropout(.2),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128,64),
            nn.ReLU(), nn.Dropout(.2), nn.Linear(64,1))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        loss_fn = nn.BCEWithLogitsLoss()
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y.astype(np.float32))
        loader = DataLoader(TensorDataset(xt[tr],yt[tr]),batch_size=64,shuffle=True,
            generator=torch.Generator().manual_seed(seed),num_workers=0)
        best_loss, stale = float('inf'), 0
        best_epoch, best_state = None, None
        t0 = time.perf_counter()
        for epoch in range(1, epochs+1):
            model.train()
            total = 0.
            for xb,yb in loader:
                optimizer.zero_grad()
                loss=loss_fn(model(xb).squeeze(1),yb)
                assert torch.isfinite(loss), 'Non-finite loss'
                loss.backward()
                optimizer.step()
                total += float(loss.detach())*len(yb)
            model.eval()
            with torch.no_grad():
                vl = float(loss_fn(model(xt[va]).squeeze(1),yt[va]))
            histories.append({'split':mode,'epoch':epoch,'train_loss':total/len(tr),'validation_loss':vl})
            if vl < best_loss-1e-6:
                best_loss,best_epoch,best_state,stale=vl,epoch,copy.deepcopy(model.state_dict()),0
            else:
                stale += 1
            print(f'{mode} epoch {epoch}: train={total/len(tr):.4f} val={vl:.4f}',flush=True)
            if stale >= 5:
                break
        elapsed=time.perf_counter()-t0
        model.load_state_dict(best_state)
        model.eval()
        start=time.perf_counter()
        with torch.no_grad():
            vprob=torch.sigmoid(model(xt[va]).squeeze(1)).numpy()
            prob=torch.sigmoid(model(xt[te]).squeeze(1)).numpy()
        predictions['mlp']=(vprob,prob,elapsed,time.perf_counter()-start)
        torch.save({'state_dict':best_state,'layers':[1024,256,128,64,1],'dropout':.2,
            'best_epoch':best_epoch,'seed':seed,'fingerprint_radius':2,'fingerprint_bits':1024,
            'fingerprint_include_chirality':False},out/f'{mode}_mlp.pt')
        checks[mode]['mlp_best_epoch']=best_epoch
        checks[mode]['mlp_parameter_count']=sum(p.numel() for p in model.parameters())
        plot_data[mode]={}
        for name,(vp,p,seconds,infer_seconds) in predictions.items():
            thresholds=np.arange(.05,.951,.01)
            threshold=.5 if name=='majority' else float(thresholds[np.argmax([
                balanced_accuracy_score(y[va],vp>=v) for v in thresholds])])
            pred=(p>=threshold).astype(int)
            result={'split':mode,'model':name,'seed':seed,'test_rows':len(te),
                'positive_fraction':float(y[te].mean()), 'auroc':float(roc_auc_score(y[te],p)),
                'average_precision':float(average_precision_score(y[te],p)),
                'accuracy':float(accuracy_score(y[te],pred)),
                'balanced_accuracy':float(balanced_accuracy_score(y[te],pred)),
                'precision':float(precision_score(y[te],pred,zero_division=0)),
                'recall':float(recall_score(y[te],pred,zero_division=0)),
                'f1':float(f1_score(y[te],pred,zero_division=0)),
                'brier':float(brier_score_loss(y[te],p)),
                'log_loss':float(log_loss(y[te],p)),
                'validation_selected_threshold':threshold,'training_seconds':seconds,
                'validation_plus_test_prediction_seconds':infer_seconds}
            metrics.append(result)
            output=sample.iloc[te][['source_id','canonical_smiles','activity','scaffold']].copy()
            output['predicted_probability']=p
            output['prediction']=pred
            output.to_csv(out/f'{mode}_{name}_predictions.csv',index=False)
            plot_data[mode][name]=(y[te],p)
            print(f'{mode}/{name}: AUROC={result["auroc"]:.4f}, AP={result["average_precision"]:.4f}, train={seconds:.1f}s',flush=True)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(histories).to_csv(out/'learning_curves.csv',index=False)
    save_json(out/'split_checks.json',checks)
    save_json(out/'metrics.json',metrics)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for i,mode in enumerate(['random','scaffold']):
        h=pd.DataFrame(histories).query('split == @mode')
        axes[0,i].plot(h.epoch,h.train_loss,label='Train')
        axes[0,i].plot(h.epoch,h.validation_loss,label='Validation')
        axes[0,i].set(title=f'{mode.title()} split: MLP learning curve',xlabel='Epoch',ylabel='Binary cross-entropy')
        axes[0,i].legend()
        axes[1,i].plot([0,1],[0,1],'--',color='gray')
        for name,(target,p) in plot_data[mode].items():
            if name=='majority': continue
            observed,predicted=calibration_curve(target,p,n_bins=8,strategy='quantile')
            axes[1,i].plot(predicted,observed,'o-',label=name)
        axes[1,i].set(title=f'{mode.title()} split: pilot calibration',xlabel='Mean predicted probability',ylabel='Observed positive fraction',xlim=(0,1),ylim=(0,1))
        axes[1,i].legend()
    fig.tight_layout()
    fig.savefig(out/'training_and_calibration.png',dpi=150)
    plt.close(fig)
    print('Pilot complete. Official test set was NOT scored.',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'pilot_5000_seed42')
    parser.add_argument('--sample-size',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--reuse-audit',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.reuse_audit:
        report=json.loads((args.output/'data_audit.json').read_text(encoding='utf-8'))
        assert report['seed']==args.seed and report['pilot_sample_rows']==args.sample_size
        sample=pd.read_csv(args.output/'pilot_sample.csv')
        features=np.load(args.output/'pilot_features.npz')
        x,y=features['x'].astype(np.float32),features['y']
        assert len(sample)==len(x)==len(y) and np.array_equal(sample.activity.to_numpy(),y)
    else:
        sample,x,y=audit(args.data_dir,args.output,args.sample_size,args.seed)
    if not args.audit_only:
        train(sample,x,y,args.output,args.seed,args.epochs)


if __name__=='__main__':
    main()
