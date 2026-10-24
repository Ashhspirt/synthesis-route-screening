"""Fixed-protocol descriptor ablations on existing development partitions."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import random
import time
import pilot
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, log_loss
from experiment_20k import scores, CONFIGS, network as old_network

ROOT = Path(__file__).resolve().parent
MAIN = ROOT/'results'/'main_models_and_splits'
DATA = ROOT/'results'/'experiment_20000_three_seeds'/'data'
OUT = ROOT/'results'/'error_analysis_and_feature_ablation'
FEATURES = ['MolWt', 'RingCount', 'RotatableBonds', 'HBD', 'HBA']
CONDITIONS = {
    'fp_only': {'fp': True, 'descriptors': []},
    'descriptors_only': {'fp': False, 'descriptors': FEATURES},
    'fp_all': {'fp': True, 'descriptors': FEATURES},
    'fp_no_mass': {'fp': True, 'descriptors': FEATURES[1:]},
    'fp_no_topology': {'fp': True, 'descriptors': ['MolWt', 'HBD', 'HBA']},
    'fp_no_hbond': {'fp': True, 'descriptors': FEATURES[:3]},
}
SPLIT_SEEDS = [101, 202, 303]
TRAIN_SEEDS = [42, 43, 44]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptors(sample):
    rows = []
    for smi in sample.canonical_smiles:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        rows.append([Descriptors.MolWt(mol), Lipinski.RingCount(mol),
                     Lipinski.NumRotatableBonds(mol), Lipinski.NumHDonors(mol),
                     Lipinski.NumHAcceptors(mol)])
    a = np.array(rows, dtype=np.float64)
    assert np.isfinite(a).all()
    return a


def network(n_inputs):
    layers = []
    for width in [256, 128, 64]:
        layers.extend([nn.Linear(n_inputs, width), nn.ReLU(), nn.Dropout(.2)])
        n_inputs = width
    layers.append(nn.Linear(n_inputs, 1))
    return nn.Sequential(*layers)


def inputs(fp, scaled, condition):
    config = CONDITIONS[condition]
    parts = [fp] if config['fp'] else []
    if config['descriptors']:
        parts.append(scaled[:, [FEATURES.index(d) for d in config['descriptors']]])
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x)).squeeze(1)).numpy()


def fit(x, y, tr, va, seed, folder, condition):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = network(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    loss_fn = nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    loader = DataLoader(TensorDataset(xt[tr], yt[tr]), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    history, best, stale = [], float('inf'), 0
    start = time.perf_counter()
    for epoch in range(1, 51):
        model.train()
        train_loss = 0.
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(yb)
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(xt[va]).squeeze(1), yt[va]))
        history.append({'epoch': epoch, 'train_loss': train_loss/len(tr), 'validation_loss': vl})
        if vl < best:
            best, stale, best_epoch = vl, 0, epoch
            state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(state)
    torch.save({'state_dict': state, 'input_dim': x.shape[1], 'seed': seed,
                'condition': condition, 'best_epoch': best_epoch}, folder/'model.pt')
    pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
    vp = predict(model, x[va])
    thresholds = np.arange(.05, .951, .01)
    threshold = float(thresholds[np.argmax([balanced_accuracy_score(y[va], vp >= v) for v in thresholds])])
    info = {'threshold': threshold, 'best_epoch': best_epoch, 'epochs_run': len(history),
            'training_seconds': time.perf_counter()-start,
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'train_log_loss': float(log_loss(y[tr], predict(model, x[tr])))}
    return model, vp, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise SystemExit('Results/protocol already exist; use a fresh --output directory.')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = {
        'conditions': CONDITIONS, 'split_seeds': SPLIT_SEEDS, 'training_seeds': TRAIN_SEEDS,
        'sample_sha256': sha(DATA/'pilot_sample.csv'), 'fingerprint_sha256': sha(DATA/'pilot_features.npz'),
        'architecture': [256,128,64], 'dropout': .2, 'weight_decay': 0., 'learning_rate': .001,
        'optimizer': 'Adam', 'batch_size': 64, 'epochs_max': 50, 'patience': 5, 'threads': 2,
        'scaling': 'StandardScaler mean and population std fitted only on each partition training set; saved for all seeds/conditions.',
        'baseline': 'Reuse original large_standard fits, not validation-selected architectures; frozen source hashes checked.',
        'selection': 'Epoch by validation BCE; threshold by validation balanced accuracy on 0.05:0.01:0.95. Feature candidates ranked by mean validation log loss per split before new internal test evaluation.',
        'primary_metrics': ['auroc','brier','log_loss'],
        'aggregation': 'Mean 3 training seeds within each partition, then mean and sample SD over 3 split seeds. Paired condition differences; repeated holdouts overlap, no independence or significance claims.',
        'error_analysis': 'Original fp_only; train-only quartile descriptor bins with duplicate edges collapsed; report FPR and FNR, count each true class separately; sparse class denominators under 30 flagged. Unique-molecule examples ranked by out-of-sample mean wrong-class probability, descriptive not causal.',
        'label_scope': 'Historical planner route-found labels, not experimental synthesizability.',
        'test_policy': 'Reused internal partitions are development results. Official test never scored or loaded here.',
        'initialization_note': 'Identical seed schedules, but different input sizes change random initialization draws. Paired seeds do not imply identical initial weights.',
        'descriptor_definitions': {'MolWt':'RDKit Descriptors.MolWt', 'RingCount':'Lipinski.RingCount (SSSR count)',
            'RotatableBonds':'Lipinski.NumRotatableBonds default strict', 'HBD':'Lipinski.NumHDonors', 'HBA':'Lipinski.NumHAcceptors'},
        'rdkit': rdBase.rdkitVersion, 'torch': torch.__version__,
        'source_sha256': sha(Path(__file__)),
        'baseline_sources': {n: sha(ROOT/n) for n in ['pilot.py','experiment_20k.py']},
        'split_files': {f'{s}/{m}':sha(MAIN/f'split_{s}'/f'{m}_indices.npz') for s in SPLIT_SEEDS for m in ['random','scaffold']}}
    pilot.save_json(out/'protocol.json', protocol)
    sample = pd.read_csv(DATA/'pilot_sample.csv')
    cache = np.load(DATA/'pilot_features.npz')
    fp, y = cache['x'].astype(np.float32), cache['y']
    assert np.array_equal(sample.activity, y) and sample.source_id.is_unique
    d = descriptors(sample)
    df = pd.DataFrame(d, columns=FEATURES)
    df.insert(0,'source_id',sample.source_id)
    df.to_csv(out/'descriptors.csv', index=False)
    pilot.save_json(out/'descriptor_audit.json', {'rows':len(d), 'finite':True,
                    'summary':pd.DataFrame(d,columns=FEATURES).describe().to_dict()})
    val_rows, jobs, baseline_hashes = [], [], {}
    for split_seed in SPLIT_SEEDS:
        for mode in ['random','scaffold']:
            source = MAIN/f'split_{split_seed}'
            idx = np.load(source/f'{mode}_indices.npz')
            tr, va, te = [idx[k] for k in ['train','validation','internal_test']]
            part = out/f'split_{split_seed}'/mode
            part.mkdir(parents=True)
            scaler = StandardScaler().fit(d[tr])
            scaled = scaler.transform(d)
            pilot.save_json(part/'scaler.json', {'features':FEATURES,'mean':scaler.mean_.tolist(),
                'scale':scaler.scale_.tolist(), 'n_samples_seen':int(scaler.n_samples_seen_),
                'train_indices_sha256':hashlib.sha256(tr.tobytes()).hexdigest()})
            for seed in TRAIN_SEEDS:
                for condition in CONDITIONS:
                    folder = part/f'seed{seed}'/condition
                    folder.mkdir(parents=True)
                    key = {'split_seed':split_seed,'split':mode,'seed':seed,'condition':condition}
                    if condition == 'fp_only':
                        old = source/'runs'/mode/f'seed{seed}'/'large_standard'
                        row = json.loads((old/'validation.json').read_text())
                        cp = torch.load(old/'model.pt', map_location='cpu', weights_only=True)
                        model = old_network(cp['config'])
                        model.load_state_dict(cp['state_dict'])
                        vp = predict(model, fp[va])
                        saved = pd.read_csv(old/'validation_predictions.csv')
                        assert np.array_equal(saved.source_id, sample.iloc[va].source_id)
                        np.testing.assert_allclose(vp, saved.probability, atol=1e-7, rtol=1e-6)
                        baseline_hashes[str(old.relative_to(ROOT))] = sha(old/'model.pt')
                        info = {k:row[k] for k in ['threshold','best_epoch','epochs_run','training_seconds','parameter_count']}
                        info['train_log_loss'] = row['train_log_loss_at_best']
                        pilot.save_json(folder/'baseline_reference.json',{'path':str(old),'sha256':sha(old/'model.pt')})
                    else:
                        x = inputs(fp, scaled, condition)
                        model, vp, info = fit(x, y, tr, va, seed, folder, condition)
                    row = {**key, **info, **scores(y[va], vp, info['threshold'])}
                    val_rows.append(row)
                    pilot.save_json(folder/'validation.json', row)
                    pd.DataFrame({'source_id':sample.iloc[va].source_id.to_numpy(),'label':y[va],
                                  'probability':vp}).to_csv(folder/'validation_predictions.csv', index=False)
                    jobs.append((key, folder))
                    print(f'{split_seed}/{mode}/{seed}/{condition}: val loss {row["log_loss"]:.4f}, epoch {info["best_epoch"]}', flush=True)
    val = pd.DataFrame(val_rows)
    val.to_csv(out/'validation_metrics.csv', index=False)
    selection = val.groupby(['split_seed','split','condition']).log_loss.mean().reset_index()
    selection['selected'] = selection.log_loss == selection.groupby(['split_seed','split']).log_loss.transform('min')
    selection.to_csv(out/'validation_selection.csv', index=False)
    pilot.save_json(out/'baseline_hashes.json', baseline_hashes)
    rows = []
    for key, folder in jobs:
        idx = np.load(MAIN/f'split_{key["split_seed"]}'/f'{key["split"]}_indices.npz')
        te = idx['internal_test']
        if key['condition'] == 'fp_only':
            ref = json.loads((folder/'baseline_reference.json').read_text())
            old = Path(ref['path'])
            cp = torch.load(old/'model.pt',map_location='cpu',weights_only=True)
            model = old_network(cp['config'])
            x = fp
        else:
            cp = torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            model = network(cp['input_dim'])
            sc = json.loads((folder.parent.parent/'scaler.json').read_text())
            x = inputs(fp, (d-np.array(sc['mean']))/np.array(sc['scale']), key['condition'])
        model.load_state_dict(cp['state_dict'])
        p = predict(model, x[te])
        info = json.loads((folder/'validation.json').read_text())
        row = {**info, **scores(y[te],p,info['threshold']), 'internal_test_n':len(te)}
        rows.append(row)
        pd.DataFrame({'source_id':sample.iloc[te].source_id.to_numpy(),'label':y[te],
                      'probability':p,'prediction':(p>=info['threshold']).astype(int)}).to_csv(folder/'internal_test_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    print(f'Completed {len(rows)} runs ({len(rows)-18} new fits + 18 reused baseline fits).',flush=True)


if __name__ == '__main__':
    main()
