"""Real frozen-model inference on three deterministically selected validation molecules."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import Descriptors,Lipinski,rdFingerprintGenerator
ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'results/frozen_candidate_v1'
torch.set_num_threads(2)

def _csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def load_demo():
    manifest=json.loads((FROZEN/'frozen_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files_sha256'].items():
        assert hashlib.sha256((FROZEN/name).read_bytes()).hexdigest()==expected,name
    sample=_csv(ROOT/'results/experiment_20000_three_seeds/data/pilot_sample.csv')
    predictions=_csv(FROZEN/'count_standard/validation_predictions.csv')
    # Use the first three saved validation rows, without choosing by probability or correctness.
    chosen=predictions[:3]
    byid={row['source_id']:row for row in sample}
    rows=[byid[row['source_id']] for row in chosen]
    scaler=json.loads((FROZEN/'scaler.json').read_text(encoding='utf-8'))
    models=[]
    for seed in [42,43,44]:
        layers=[];dim=1029
        for width in [256,128,64]:
            layers += [nn.Linear(dim,width),nn.ReLU(),nn.Dropout(.2)];dim=width
        layers.append(nn.Linear(dim,1))
        model=nn.Sequential(*layers)
        saved=torch.load(FROZEN/f'count_standard/seed{seed}/model.pt',weights_only=True,map_location='cpu')
        model.load_state_dict(saved['state_dict']);model.eval();models.append(model)
    return {'rows':rows,'saved':chosen,'scaler':scaler,'models':models,'threshold':manifest['thresholds']['count_standard']['recall90']}

def molecule_state(demo):
    return [{'source_id':int(r['source_id']),'historical_label':int(r['activity']),'SMILES':r['canonical_smiles']} for r in demo['rows']]

def features(demo):
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024,includeChirality=False)
    molecules=[Chem.MolFromSmiles(r['canonical_smiles']) for r in demo['rows']]
    counts=np.array([generator.GetCountFingerprintAsNumPy(m) for m in molecules],dtype=np.uint32)
    desc=np.array([[Descriptors.MolWt(m),Lipinski.RingCount(m),Lipinski.NumRotatableBonds(m),Lipinski.NumHDonors(m),Lipinski.NumHAcceptors(m)] for m in molecules])
    scaled=(desc-np.array(demo['scaler']['mean']))/np.array(demo['scaler']['scale'])
    x=np.ascontiguousarray(np.concatenate([np.log1p(counts.astype(np.float32)),scaled],axis=1),dtype=np.float32)
    # Show a real repeated count from the first molecule, not a hypothetical example.
    index=int(np.argmax(counts[0]))
    state={'input_shape':list(x.shape),'fingerprint_width':1024,'first_molecule_raw_descriptors':dict(zip(demo['scaler']['features'],desc[0].round(3).tolist())),'first_molecule_scaled_descriptors':dict(zip(demo['scaler']['features'],scaled[0].round(3).tolist())),'example_hashed_bin':index,'bin_count':int(counts[0,index]),'binary_value':int(counts[0,index]>0),'log1p_value':round(float(x[0,index]),4),'scaler_training_n':demo['scaler']['n_train']}
    return x,state

def forward(demo,x):
    probabilities=[];layers=[]
    with torch.no_grad():
        for i,model in enumerate(demo['models']):
            h=torch.from_numpy(x)
            for layer in model:
                h=layer(h)
                if i==0 and isinstance(layer,nn.ReLU):
                    layers.append({'width':int(h.shape[1]),'first_molecule_first_6_activations':[round(float(v),4) for v in h[0,:6]],'nonzero_units':int(torch.count_nonzero(h[0]))})
            probabilities.append(torch.sigmoid(h.squeeze(1)).numpy().astype(np.float64))
    probs=np.stack(probabilities,axis=1)
    state={'seed42_hidden_layers':layers,'three_seed_probabilities':probs.round(6).tolist(),'dropout_in_evaluation':False,'training_performed':False}
    return probs,state

def layer_summary(h):
    """Small, inspectable summary of one hidden layer for the presentation."""
    return {'shape':list(h.shape),'active_units_per_molecule':[int(v) for v in (h>0).sum(dim=1)],'first_molecule_first_6':[round(float(v),4) for v in h[0,:6]]}

def decision(demo,probs):
    means=probs.mean(axis=1)
    np.testing.assert_allclose(means,[float(r['probability']) for r in demo['saved']],rtol=1e-6,atol=1e-7)
    result=[]
    for row,p in zip(demo['rows'],means):
        result.append({'source_id':int(row['source_id']),'mean_probability':round(float(p),6),'frozen_threshold':round(demo['threshold'],6),'predicted_label':int(p>=demo['threshold']),'historical_label':int(row['activity'])})
    return result

def saved_results():
    rows=_csv(ROOT/'results/official_holdout_v1/metrics.csv')
    count=next(r for r in rows if r['model']=='count_standard' and r['policy']=='recall90')
    binary=next(r for r in rows if r['model']=='binary_standard' and r['policy']=='recall90')
    return {'official_n':int(count['n']),'count_AUROC':round(float(count['auroc']),4),'binary_AUROC':round(float(binary['auroc']),4),'count_recall_percent':round(100*float(count['recall']),2),'false_negatives':int(count['fn']),'false_positives':int(count['fp']),'fewer_misses':int(binary['fn'])-int(count['fn']),'extra_false_positives':int(count['fp'])-int(binary['fp']),'deferred':int(count['tn'])+int(count['fn'])}

def audit_demo():
    demo=load_demo();one=molecule_state(demo);x,two=features(demo);probs,three=forward(demo,x);four=decision(demo,probs)
    return {'selection':'first three rows of saved final validation predictions, before examining scores','states':[one,two,three,four],'saved_official_summary':saved_results(),'new_official_test_inference':False,'validation_predictions_match':True}
