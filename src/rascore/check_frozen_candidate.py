"""Read-only preflight: do not open or score official test records."""
import json
from pathlib import Path
import feature_ablation as a

def main():
    root=a.ROOT/'results'/'frozen_candidate_v1'
    m=json.loads((root/'frozen_manifest.json').read_text())
    assert m['status']=='frozen_not_tested' and m['default_policy']=='recall90'
    for name,h in m['files_sha256'].items():assert a.sha(root/name)==h,name
    p=json.loads((root/'freeze_protocol.json').read_text())
    assert a.sha(a.ROOT/'freeze_candidate.py')==p['source_sha256']
    for name,h in p['dependency_sha256'].items():assert a.sha(a.ROOT/name)==h,name
    assert a.sha(a.DATA/'pilot_sample.csv')==m['data']['sample_sha256']
    plan=a.ROOT/'report'/'heldout_test_plan.json'
    if plan.exists():
        plan_data=json.loads(plan.read_text(encoding='utf-8'))
        assert plan_data['frozen_manifest_sha256']==a.sha(root/'frozen_manifest.json')
        lock=a.ROOT/'report'/'test_plan_lock.json'
        if lock.exists():assert json.loads(lock.read_text())['sha256']==a.sha(plan)
    print('PASS: frozen models, preprocessing, thresholds, source and test-plan hashes match. Official test was not opened.')

if __name__=='__main__':main()
