import copy
import csv
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from models.build import VARIANTS
from scripts.create_smoke_data import create
from scripts.split_pairs import split_pairs
from workflow.data import discover_pairs, read_rgb, validate_splits
from workflow.engine import train, evaluate, load_checkpoint, restore_model
from workflow.metrics import score_image


class TestWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        create(cls.root/'data')
        cls.config = dict(seed=42, cpu_threads=2, deterministic=True, device='cpu', output='runs',
            backbone=dict(width=4, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1], middle_blk_num=1),
            data=dict(root='data', splits={s: dict(input=f'{s}/input', target=f'{s}/target') for s in ('train', 'val', 'test')}),
            train=dict(epochs=1, batch_size=1, num_workers=0, crop_size=16, augment=True,
                       lr=0.001, milestones=[1], gamma=0.5))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pair_manifest_and_leakage(self):
        splits = validate_splits(self.config['data'], self.root)
        self.assertEqual({k: len(v) for k, v in splits.items()}, dict(train=2, val=1, test=1))
        (self.root/'data/pairs.csv').write_text('id,input,target\none,train/input/000.png,train/target/000.png\n')
        self.assertEqual(discover_pairs(self.root/'data', {'manifest': 'pairs.csv'})[0].name, 'one')
        with self.assertRaisesRegex(ValueError, 'unpaired'):
            discover_pairs(self.root/'data', dict(input='train/input', target='val/target'))
        bad = copy.deepcopy(self.config['data'])
        bad['splits']['test'] = bad['splits']['val']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_splits(bad, self.root)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for d in ('input', 'target'):
                (folder/d).mkdir()
            Image.new('RGB', (12, 12)).save(folder/'input/a.png')
            Image.new('RGB', (13, 12)).save(folder/'target/a.png')
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                discover_pairs(folder, dict(input='input', target='target'))
            Image.new('L', (12, 12)).save(folder/'gray.png')
            with self.assertRaisesRegex(ValueError, 'RGB'):
                read_rgb(folder/'gray.png')

    def test_seeded_split_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            pool = Path(folder)
            for d in ('input', 'target'):
                (pool/d).mkdir()
            pairs = [p for values in validate_splits(self.config['data'], self.root).values() for p in values]
            for i, pair in enumerate(pairs):
                shutil.copy(pair.input, pool/f'input/{i}.png')
                shutil.copy(pair.target, pool/f'target/{i}.png')
            for name in ('m1', 'm2'):
                self.assertEqual(split_pairs(pool/'input', pool/'target', pool/name, (0.5, 0.25, 0.25)),
                                 dict(train=2, val=1, test=1))
            for split in ('train', 'val', 'test'):
                self.assertEqual((pool/f'm1/{split}.csv').read_text(), (pool/f'm2/{split}.csv').read_text())
            data = dict(root=str(pool/'m1'), splits={s: dict(manifest=f'{s}.csv') for s in ('train', 'val', 'test')})
            self.assertEqual(sum(map(len, validate_splits(data, self.root).values())), 4)

    def test_duplicate_pixels_across_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'data'
            shutil.copytree(self.root/'data', root)
            shutil.copy(root/'val/input/000.png', root/'test/input/000.png')
            data = copy.deepcopy(self.config['data'])
            data['root'] = str(root)
            with self.assertRaisesRegex(ValueError, 'content leakage'):
                validate_splits(data, self.root)

    def test_metrics_and_unavailable_status(self):
        pixels = read_rgb(self.root/'data/test/target/000.png')
        scores = score_image(pixels, pixels)
        self.assertEqual(scores['psnr'], float('inf'))
        self.assertAlmostEqual(scores['ssim'], 1.0)
        self.assertEqual(scores['niqe_status'], 'ok')
        self.assertTrue(np.isfinite(scores['niqe']))
        small = score_image(pixels[:8, :8], pixels[:8, :8])
        self.assertIsNone(small['ssim'])
        self.assertIsNone(small['niqe'])
        self.assertEqual(set(score_image(pixels)), {'niqe', 'niqe_status'})
        constant = score_image(np.zeros((192, 192, 3), np.uint8))
        self.assertIsNone(constant['niqe'])
        self.assertTrue(constant['niqe_status'].startswith('unavailable'))

    def test_four_one_epoch_runs_and_exports(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                run = train(self.config, self.root, variant, output=self.root/'four'/variant)
                saved = load_checkpoint(run/'last.pt')
                self.assertEqual(saved['epoch'], 1)
                self.assertEqual(saved['global_step'], 2)
                for name in ('best.pt', 'history.csv', 'val_epoch_0001/per_image.csv'):
                    self.assertTrue((run/name).is_file())
                model = restore_model(saved, torch.device('cpu'))
                pixels = self.root/'data/unpaired/input/000.png'
                report = evaluate(model, [pixels], run/'unpaired', torch.device('cpu'), paired=False)
                self.assertEqual(set(report['metrics']), {'niqe'})
                with (run/'unpaired/per_image.csv').open(encoding='utf-8-sig') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertNotIn('psnr', rows[0])
                self.assertNotIn('ssim', rows[0])
                enhanced = read_rgb(rows[0]['output'])
                self.assertEqual(enhanced.shape, read_rgb(pixels).shape)
                self.assertEqual(report['metrics']['niqe']['valid'], 1)
                self.assertEqual(float(rows[0]['niqe']), score_image(enhanced)['niqe'])

    def test_resume_equals_uninterrupted(self):
        resumed = train(self.config, self.root, 'fdmnet', output=self.root/'resumed')
        train(self.config, self.root, 'fdmnet', output=resumed, resume=resumed/'last.pt', epochs=2)
        continuous = train(self.config, self.root, 'fdmnet', output=self.root/'continuous', epochs=2)
        a, b = load_checkpoint(resumed/'last.pt'), load_checkpoint(continuous/'last.pt')
        self.assertEqual(a['global_step'], b['global_step'])
        self.assertEqual(a['scheduler'], b['scheduler'])
        for name in a['model']:
            torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
        bad = copy.deepcopy(self.config)
        bad['train']['lr'] = 0.05
        with self.assertRaisesRegex(ValueError, 'differs'):
            train(bad, self.root, 'fdmnet', output=self.root/'bad_resume', resume=resumed/'last.pt', epochs=3)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_training(self):
        config = copy.deepcopy(self.config)
        config.update(device='cuda', deterministic=False)
        run = train(config, self.root, 'fdmnet', output=self.root/'cuda')
        self.assertEqual(load_checkpoint(run/'last.pt')['epoch'], 1)
