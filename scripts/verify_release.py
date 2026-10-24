"""Read-only checks of frozen provenance and stored evaluation; standard library only."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'src/rascore'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    frozen = ROOT/'results/frozen_candidate_v1'
    manifest = read(frozen/'frozen_manifest.json')
    for name, expected in manifest['files_sha256'].items():
        require(sha(frozen/name) == expected, f'Frozen file changed: {name}')
    protocol = read(frozen/'freeze_protocol.json')
    require(sha(ROOT/'freeze_candidate.py') == protocol['source_sha256'], 'Freeze source changed')
    for name, expected in protocol['dependency_sha256'].items():
        require(sha(ROOT/name) == expected, f'Computational source changed: {name}')
    sample = ROOT/'results/experiment_20000_three_seeds/data/pilot_sample.csv'
    require(sha(sample) == manifest['data']['sample_sha256'], 'Development sample changed')
    lock = read(ROOT/'report/test_plan_lock.json')
    require(sha(ROOT/'report/heldout_test_plan.json') == lock['sha256'], 'Test plan changed')
    require(sha(frozen/'frozen_manifest.json') == lock['frozen_manifest_sha256'], 'Manifest changed')
    official = ROOT/'results/official_holdout_v1'
    execution = read(official/'execution.json')
    for name, expected in execution['files_sha256'].items():
        require(sha(official/name) == expected, f'Evaluation artifact changed: {name}')
    with (official/'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    require(len(predictions) == 19935, 'Unexpected official evaluation size')
    with (official/'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        metrics = list(csv.DictReader(stream))
    for metric in metrics:
        model, policy = metric['model'], metric['policy']
        threshold = float(metric['threshold'])
        counts = dict(tp=0, tn=0, fp=0, fn=0)
        for row in predictions:
            score = float(row[model])
            mean = sum(float(row[f'{model}_seed{seed}']) for seed in [42, 43, 44])/3
            require(math.isclose(score, mean, abs_tol=1e-12), 'Seed average disagrees')
            prediction = int(score >= threshold)
            require(prediction == int(row[f'{model}_{policy}']), 'Saved decision disagrees')
            counts[{(1, 1): 'tp', (1, 0): 'fn', (0, 1): 'fp', (0, 0): 'tn'}[
                int(row['activity']), prediction]] += 1
        require(all(counts[key] == int(metric[key]) for key in counts), 'Confusion matrix disagrees')
    print(json.dumps({'status': 'PASS', 'frozen_files_checked': len(manifest['files_sha256']),
                      'prediction_rows': len(predictions), 'operating_policies': len(metrics),
                      'model_inference': False, 'writes': False}, indent=2))


if __name__ == '__main__':
    main()
