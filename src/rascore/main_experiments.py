"""Execute three new repeated-holdout partitions, with three training seeds each."""
from pathlib import Path
import json
import hashlib
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'main_models_and_splits'
DATA=ROOT/'results'/'experiment_20000_three_seeds'/'data'
SPLIT_SEEDS=[101,202,303]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'sample_size':20000,'sample_seed':42,'split_seeds':SPLIT_SEEDS,
        'training_seeds':[42,43,44],'split_types':['random','scaffold'],
        'models':['majority','logistic','random_forest','large_standard','small_standard','large_regularized','small_regularized'],
        'fits_planned':126,'neural_fits_planned':72,
        'data_hashes':{n:digest(DATA/n) for n in ['pilot_sample.csv','pilot_features.npz']},
        'code_hashes':{n:digest(ROOT/n) for n in ['main_experiments.py','experiment_20k.py','pilot.py','verify_20k.py']},
        'analysis':'Average three training runs within each split seed, then summarize three split-level means; do not treat nine runs as independent datasets.',
        'split_caveat':'Repeated holdouts drawn from same sample. Test sets can overlap across repetitions; not independent blind external validation.',
        'model_selection':'Neural configuration selected per split type and split seed on mean validation log_loss only, before internal test scoring.',
        'official_test_policy':'No training, tuning, or performance scoring on official test.',
        'date':'2026-10-04'}
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text())==protocol,'Protocol changed; use a new experiment directory.'
    else:
        protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    start=time.perf_counter()
    for seed in SPLIT_SEEDS:
        folder=OUT/f'split_{seed}'
        if not (folder/'completion.json').exists():
            command=[sys.executable,'-u',str(ROOT/'experiment_20k.py'),'--output',str(folder),
                '--prepared-data',str(DATA),'--split-seed',str(seed)]
            print(f'Starting partition seed {seed}',flush=True)
            with (OUT/f'split_{seed}_training.log').open('w',encoding='utf-8') as log:
                process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if 'validation BCE' in line or 'Completed all' in line:
                        print(line.rstrip(),flush=True)
                code=process.wait()
            if code:
                raise RuntimeError(f'Partition {seed} failed; inspect saved log.')
        print(f'Verifying partition seed {seed}',flush=True)
        subprocess.run([sys.executable,str(ROOT/'verify_20k.py'),'--output',str(folder)],check=True)
    (OUT/'completion.json').write_text(json.dumps({'complete':True,'elapsed_this_invocation_seconds':time.perf_counter()-start,
        'fits_completed':126,'neural_fits_completed':72,'official_test_scored':False},indent=2),encoding='utf-8')
    print('All model/partition runs completed and checked.',flush=True)


if __name__=='__main__':
    main()
