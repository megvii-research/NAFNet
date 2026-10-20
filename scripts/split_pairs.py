"""Seeded image-level manifest splitting. Prefer official scene/sequence splits."""
import argparse
import csv
import os
import random
from pathlib import Path
from workflow.data import discover_pairs


def split_pairs(input_dir, target_dir, output, ratios=(0.8, 0.1, 0.1), seed=42):
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios)-1) > 1e-8:
        raise ValueError('three positive ratios must sum to one')
    pairs = discover_pairs(Path.cwd(), {'input': str(Path(input_dir).resolve()), 'target': str(Path(target_dir).resolve())})
    random.Random(seed).shuffle(pairs)
    counts = [int(len(pairs)*ratios[0]), int(len(pairs)*ratios[1])]
    counts.append(len(pairs)-sum(counts))
    if min(counts) < 1:
        raise ValueError('not enough pairs for three nonempty splits')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    offset = 0
    for name, count in zip(('train', 'val', 'test'), counts):
        with (output/f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['id', 'input', 'target'])
            writer.writeheader()
            for pair in pairs[offset:offset+count]:
                writer.writerow({'id': pair.name, 'input': os.path.relpath(pair.input, output),
                                 'target': os.path.relpath(pair.target, output)})
        offset += count
    return dict(zip(('train', 'val', 'test'), counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'target', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--ratios', type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(split_pairs(args.input, args.target, args.output, args.ratios, args.seed))


if __name__ == '__main__':
    main()
