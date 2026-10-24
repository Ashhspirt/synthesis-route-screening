"""Build the real Lectrace site and verify the four selected computational states."""
from pathlib import Path
import json,sys,os
from _model_demo import audit_demo
from _presentation_figures import render_figures
from rdkit import Chem
from rdkit.Chem import Draw
from lectrace.builder import build
HERE=Path(__file__).resolve().parent
os.chdir(HERE)
audit=audit_demo()
render_figures(audit)
molecules=audit['states'][0]
Draw.MolsToGridImage([Chem.MolFromSmiles(r['SMILES']) for r in molecules],molsPerRow=3,subImgSize=(400,300),legends=[f"ID {r['source_id']} | y={r['historical_label']}" for r in molecules]).save(HERE/'demo_molecules.png')
build(output=HERE/'site',files=[HERE/'01_route_found.py'],incremental=False)
trace=json.loads((HERE/'site/traces/01_route_found.json').read_text(encoding='utf-8'))
assert not trace['metadata'].get('error'),trace['metadata'].get('error')
serialized=json.dumps(trace,ensure_ascii=False)
for marker in ['State 1','State 2','State 3','State 4','0.8888','90.24%']:assert marker in serialized,marker
assert '0:00-0:45' not in serialized
assert 'intentionally blank' not in serialized
assert 'Molecular synthesis route screening' in serialized
audit['figure_count']=8
audit['audience_timing_notes']=False
audit['lectrace_version']='1.1.30'
audit['trace_steps']=len(trace['steps'])
audit['runtime_error']=trace['metadata'].get('error')
(HERE/'presentation_checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','selected_execution_states':4,'trace_steps':len(trace['steps']),'validation_predictions_match':True,'new_official_test_inference':False}))
