"""Copy an already-built LecTrace site into the static deployment directory."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'src/rascore/presentation/site'
if not (source/'index.html').exists():
    raise SystemExit('Run src/rascore/presentation/build_presentation.py first.')
for file in source.rglob('*'):
    if file.is_file() and not any(part.startswith('.') for part in file.relative_to(source).parts):
        target = ROOT/'docs/lectrace'/file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
print('Public viewer copied to docs/lectrace.')
