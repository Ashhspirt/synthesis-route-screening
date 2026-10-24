"""Independent checks on stored outputs only; no model inference or test archive access."""
import evaluate_holdout as h
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

r=h.OUT
run=h.read(r/'execution.json')
for name,digest in run['files_sha256'].items():assert h.a.sha(r/name)==digest,name
pred=pd.read_csv(r/'predictions.csv',float_precision='round_trip')
m=pd.read_csv(r/'metrics.csv',float_precision='round_trip')
y=pred.activity.to_numpy()
plan=h.read(h.PLAN)
for model in h.MODELS:
    individual=np.array([pred[f'{model}_seed{s}'] for s in [42,43,44]])
    np.testing.assert_array_equal(individual.mean(axis=0),pred[model])
    for policy,t in plan['policies'][model].items():
        row=m[(m.model==model)&(m.policy==policy)].iloc[0]
        decision=pred[model].to_numpy()>=t
        np.testing.assert_array_equal(decision,pred[f'{model}_{policy}'])
        tn,fp,fn,tp=confusion_matrix(y,decision,labels=[0,1]).ravel()
        np.testing.assert_array_equal([tn,fp,fn,tp],[row.tn,row.fp,row.fn,row.tp])
        assert roc_auc_score(y,pred[model])==row.auroc
        assert average_precision_score(y,pred[model])==row.average_precision
boot=pd.read_csv(r/'bootstrap_replicates.csv',float_precision='round_trip')
ci=pd.read_csv(r/'confidence_intervals.csv',float_precision='round_trip')
assert boot.shape==(2000,25) and len(ci)==25
np.testing.assert_allclose(boot.paired_auroc_difference,boot['count_standard/auroc']-boot['binary_standard/auroc'],atol=0,rtol=0)
for row in ci.itertuples():
    np.testing.assert_allclose([row.lower,row.upper],boot[row.metric].quantile([.025,.975]).to_numpy(),atol=1e-15,rtol=0)
first=h.bootstrap(y,{name:pred[name].to_numpy() for name in h.MODELS},plan['policies'],1,20261004)
np.testing.assert_allclose(first.iloc[0].to_numpy(),boot.iloc[0].to_numpy(),atol=0,rtol=0)
rel=pd.read_csv(r/'reliability.csv')
assert (rel.groupby('model').n.sum()==len(y)).all()
subs=pd.read_csv(r/'structure_subgroups.csv')
for _,g in subs.groupby(['model','policy','feature']):
    assert len(g)==2 and g.n.sum()==len(y) and g.n_positive.sum()==y.sum()
for name,t in plan['policies'].items():
    saved=h.read(h.FROZEN/'frozen_manifest.json')['thresholds'][name]
    assert t==saved
h.check_frozen_candidate.main()
h.save(r/'independent_verification.json',dict(status='PASS',utc=h.now(),checks=[
    'all recorded artifact hashes','serialized equal-weight seed averages','sklearn confusion matrices and probability rankings',
    'all 25 percentile intervals from 2000 saved resamples','paired AUROC differences',
    'first bootstrap replicate reproduced with fixed seed','reliability and subgroup count conservation','unchanged frozen plan and model hashes'],
    used_saved_predictions_only=True,new_model_inference=False))
print('PASS: heldout outputs independently verified without reopening source test or refitting.')
