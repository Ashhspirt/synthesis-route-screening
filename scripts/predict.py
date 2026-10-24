"""Score supplied SMILES with the frozen three-seed count ensemble."""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/rascore/presentation'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--smiles', action='append', help='One SMILES; repeat for a batch.')
    inputs.add_argument('--input', type=Path, help='CSV containing a smiles column.')
    parser.add_argument('--column', default='smiles')
    parser.add_argument('--output', type=Path, help='CSV destination; default is stdout.')
    args = parser.parse_args()

    if args.input:
        with args.input.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if args.column not in (reader.fieldnames or []):
                parser.error(f'Input CSV needs a {args.column!r} column.')
            smiles = [row[args.column] for row in reader]
    else:
        smiles = args.smiles
    if not smiles:
        parser.error('No molecules supplied.')

    from rdkit import Chem
    from _model_demo import load_demo, features, forward
    invalid = [index + 1 for index, value in enumerate(smiles)
               if not value or (mol := Chem.MolFromSmiles(value)) is None or mol.GetNumAtoms() == 0]
    if invalid:
        parser.error(f'Invalid or empty SMILES at input rows {invalid}.')
    demo = load_demo()
    demo['rows'] = [{'canonical_smiles': value} for value in smiles]
    x, _ = features(demo)
    probabilities, _ = forward(demo, x)
    means = probabilities.mean(axis=1)
    threshold = demo['threshold']
    output = args.output.open('w', encoding='utf-8', newline='') if args.output else sys.stdout
    try:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(['smiles', 'route_found_score', 'threshold', 'prioritize'])
        for value, probability in zip(smiles, means):
            writer.writerow([value, float(probability), threshold, int(probability >= threshold)])
    finally:
        if args.output:
            output.close()


if __name__ == '__main__':
    main()
