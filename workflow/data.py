"""Strict RGB pairs, CSV mappings and independent dataset splits."""
import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


@dataclass(frozen=True)
class Pair:
    name: str
    input: Path
    target: Path


def read_rgb(path):
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError(f'expected 8-bit RGB image, got {image.mode}: {path}')
        return np.asarray(image).copy()


def image_index(folder):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f'missing directory: {folder}')
    found = {}
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.relative_to(folder).with_suffix('').as_posix()
            if key in found:
                raise ValueError(f'ambiguous duplicate stem: {key}')
            found[key] = path.resolve()
    if not found:
        raise ValueError(f'no images: {folder}')
    return found


def discover_pairs(root, spec):
    root = Path(root)
    if 'manifest' in spec:
        pairs = []
        with (root/spec['manifest']).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'input', 'target'} <= set(reader.fieldnames or []):
                raise ValueError('manifest requires input,target columns; optional id')
            for row in reader:
                if not row['input'] or not row['target']:
                    raise ValueError('empty input/target')
                pairs.append(Pair(row.get('id') or Path(row['input']).with_suffix('').as_posix(),
                                  (root/row['input']).resolve(), (root/row['target']).resolve()))
    else:
        inputs, targets = image_index(root/spec['input']), image_index(root/spec['target'])
        if inputs.keys() != targets.keys():
            raise ValueError(f'unpaired images; missing target={sorted(inputs.keys()-targets.keys())[:10]}, '
                             f'missing input={sorted(targets.keys()-inputs.keys())[:10]}')
        pairs = [Pair(key, inputs[key], targets[key]) for key in sorted(inputs)]
    if not pairs:
        raise ValueError('empty paired split')
    names, paths = set(), set()
    for pair in pairs:
        if pair.name in names or pair.input in paths or pair.target in paths or pair.input == pair.target:
            raise ValueError(f'duplicate ID/file or identical input/target: {pair.name}')
        names.add(pair.name)
        paths.update((pair.input, pair.target))
        a, b = read_rgb(pair.input), read_rgb(pair.target)
        if a.shape != b.shape:
            raise ValueError(f'paired shape mismatch: {pair.name}: {a.shape}, {b.shape}')
    return pairs


def validate_splits(data, config_dir):
    root = (Path(config_dir)/data['root']).resolve()
    splits = {name: discover_pairs(root, spec) for name, spec in data['splits'].items()}
    paths, hashes = {}, {}
    for name, pairs in splits.items():
        for pair in pairs:
            for path in (pair.input, pair.target):
                if path in paths and paths[path] != name:
                    raise ValueError(f'cross-split file leakage: {path}')
                paths[path] = name
                if data.get('check_content_leakage', True):
                    pixels = read_rgb(path)
                    digest = hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
                    if digest in hashes and hashes[digest] != name:
                        raise ValueError(f'cross-split content leakage: {path}')
                    hashes[digest] = name
    return splits


def to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()/255


class PairedDataset(Dataset):
    def __init__(self, pairs, crop_size=None, augment=False):
        self.pairs, self.crop_size, self.augment = pairs, crop_size, augment
        if crop_size is not None:
            if crop_size < 1:
                raise ValueError('crop_size must be positive')
            for pair in pairs:
                if min(read_rgb(pair.input).shape[:2]) < crop_size:
                    raise ValueError(f'image smaller than crop_size: {pair.input}')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        a, b = read_rgb(pair.input), read_rgb(pair.target)
        if self.crop_size:
            h, w = a.shape[:2]
            s = self.crop_size
            top, left = random.randint(0, h-s), random.randint(0, w-s)
            a, b = a[top:top+s, left:left+s], b[top:top+s, left:left+s]
        if self.augment:
            if random.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
            if random.random() < 0.5:
                a, b = a[::-1], b[::-1]
            k = random.randrange(4)
            a, b = np.rot90(a, k), np.rot90(b, k)
        return {'input': to_tensor(a), 'target': to_tensor(b), 'id': pair.name}
