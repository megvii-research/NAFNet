"""Create RGB PNG fixtures, not a public dataset or real shale experiment."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter


def create(root, seed=123):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    for split, count in [('train', 2), ('val', 1), ('test', 1), ('unpaired', 1)]:
        for i in range(count):
            pixels = rng.integers(20, 236, (192, 192, 3), dtype=np.uint8)
            target = Image.fromarray(pixels)
            blurred = np.asarray(target.filter(ImageFilter.GaussianBlur(0.7))).astype(np.float32)
            degraded = np.clip(blurred+rng.normal(0, 5, pixels.shape), 0, 255).round().astype(np.uint8)
            source = root/split/'input'
            source.mkdir(parents=True, exist_ok=True)
            Image.fromarray(degraded).save(source/f'{i:03d}.png')
            if split != 'unpaired':
                dest = root/split/'target'
                dest.mkdir(parents=True, exist_ok=True)
                target.save(dest/f'{i:03d}.png')
    (root/'README.txt').write_text('Synthetic pipeline fixtures only; not real shale experiments.\n')
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='datasets/workflow_smoke')
    print(create(parser.parse_args().output))


if __name__ == '__main__':
    main()
