"""Create an empty experiment workspace without copying reference checkpoints."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT/'work/training-run')
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        parser.error('Destination already exists; choose a new workspace.')
    source = ROOT/'src/rascore'
    target = destination/'src/rascore'
    target.mkdir(parents=True)
    for file in source.iterdir():
        if file.is_file() and file.suffix in {'.py', '.ps1', '.txt'}:
            shutil.copyfile(file, target/file.name)
    print(f'Created {destination}\nRun the development experiments from this workspace.')


if __name__ == '__main__':
    main()
