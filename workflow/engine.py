"""Shared training, safe checkpoints and full-image evaluation."""
import csv
import hashlib
import json
import logging
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from models import build_model
from models.build import DEFAULT_BACKBONE
from .data import PairedDataset, read_rgb, to_tensor, validate_splits
from .metrics import score_image, summarize


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as stream:
        return yaml.safe_load(stream), path.parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def select_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('supported devices: auto, cpu, cuda, cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
    return device


def close_logger(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def logger_for(folder):
    logger = logging.getLogger(str(Path(folder).resolve()))
    close_logger(logger)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (logging.StreamHandler(), logging.FileHandler(Path(folder)/'run.log', encoding='utf-8')):
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    return logger


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return '+inf' if value == float('inf') else '-inf' if value == -float('inf') else None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get('format_version') != 1:
        raise ValueError('expected workflow format_version=1 checkpoint')
    return state


def restore_model(state, device):
    model = build_model(state['variant'], state['backbone'], **state['module_options']).to(device)
    model.load_state_dict(state['model'], strict=True)
    return model


def rng_state(generator):
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'].cpu())
    generator.set_state(state['loader'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def dataset_signature(splits):
    manifest = {}
    for split, pairs in splits.items():
        manifest[split] = []
        for pair in pairs:
            item = {'id': pair.name}
            for kind in ('input', 'target'):
                path = getattr(pair, kind)
                item[kind] = str(path)
                with path.open('rb') as stream:
                    item[kind+'_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            manifest[split].append(item)
    return manifest


def evaluate(model, items, output, device, *, paired, crop_border=0, logger=None, metadata=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    images = output/'images'
    images.mkdir()
    own_logger = logger is None
    logger = logger or logger_for(output)
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.inference_mode():
            for index, item in enumerate(items):
                path = item.input if paired else item
                name = item.name if paired else str(item)
                source = read_rgb(path)
                x = to_tensor(source).unsqueeze(0).to(device)
                prediction = model(x)
                if prediction.shape != x.shape or not torch.isfinite(prediction).all():
                    raise RuntimeError(f'invalid output: {name}')
                pixels = prediction[0].clamp(0, 1).mul(255).round().byte().cpu().permute(1, 2, 0).numpy()
                destination = (images/f'{index:06d}_{Path(path).stem}.png').resolve()
                Image.fromarray(pixels).save(destination)
                scores = score_image(pixels, read_rgb(item.target) if paired else None, crop_border)
                row = {'id': name, 'input': str(path), 'output': str(destination), **scores}
                if paired:
                    row['target'] = str(item.target)
                rows.append(row)
                logger.info('image=%s metrics=%s', name, json_safe(scores))
    finally:
        model.train(was_training)
        if own_logger:
            close_logger(logger)
    if not rows:
        raise ValueError('evaluation has no images')
    summary = summarize(rows, paired)
    protocol = {'pixels': 'saved RGB uint8 PNG, clipped to [0,1] and rounded to [0,255]',
                'crop_border': crop_border,
                'niqe': 'BasicSR pristine model; Y channel; 96x96 blocks; OpenCV linear second scale; no resizing'}
    if paired:
        protocol.update(psnr='RGB MSE, range=255',
                        ssim='RGB channel mean, Gaussian 11x11 sigma=1.5, population covariance')
    summary.update(paired=paired, metadata=metadata or {}, protocol=protocol)
    with (output/'per_image.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output/'per_image.json', rows)
    write_json(output/'summary.json', summary)
    with (output/'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['metric', 'mean', 'valid', 'unavailable', 'images'])
        writer.writeheader()
        for name, values in summary['metrics'].items():
            writer.writerow(dict(metric=name, images=len(rows), **values))
    return summary


def _train(config, config_dir, variant, *, output=None, resume=None, epochs=None, device_override=None):
    cfg = dict(config['train'])
    if epochs is not None:
        cfg['epochs'] = epochs
    if cfg['epochs'] < 1 or cfg['batch_size'] < 1 or cfg.get('num_workers', 0) < 0:
        raise ValueError('invalid epochs, batch_size or workers')
    if cfg['batch_size'] > 1 and not cfg.get('crop_size'):
        raise ValueError('batch_size > 1 requires crop_size')
    splits = validate_splits(config['data'], config_dir)
    if not {'train', 'val'} <= splits.keys():
        raise ValueError('independent train/val required')
    run = Path(output).resolve() if output else (Path(config_dir)/config['output']/variant).resolve()
    if run.exists() and any(run.iterdir()) and not resume:
        raise ValueError('output is not empty; use a new directory or resume')
    run.mkdir(parents=True, exist_ok=True)
    logger = logger_for(run)
    device = select_device(device_override or config.get('device', 'auto'))
    seed = config.get('seed', 42)
    seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(config.get('deterministic', True))
    if 'cpu_threads' in config:
        torch.set_num_threads(config['cpu_threads'])
    generator = torch.Generator().manual_seed(seed)
    backbone = {**DEFAULT_BACKBONE, **config.get('backbone', {})}
    module_options = config.get('module_options', {})
    model = build_model(variant, backbone, **module_options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0.0))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.get('milestones', []), gamma=cfg.get('gamma', 0.5))
    amp = bool(cfg.get('amp', False) and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loader = DataLoader(PairedDataset(splits['train'], cfg.get('crop_size'), cfg.get('augment', True)),
                        batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg.get('num_workers', 0),
                        worker_init_fn=seed_worker, generator=generator, pin_memory=device.type == 'cuda', drop_last=False)
    signature = dict(data=dataset_signature(splits), variant=variant, backbone=backbone, module_options=module_options,
                     seed=seed, train={k: v for k, v in cfg.items() if k != 'epochs'}, metrics=config.get('metrics', {}),
                     amp_enabled=amp, deterministic=config.get('deterministic', True))
    start, best, step = 0, -float('inf'), 0
    if resume:
        saved = load_checkpoint(resume)
        if saved['signature'] != signature:
            raise ValueError('resume configuration/data differs; only total epochs and output may change')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        start, best, step = saved['epoch'], saved['best_psnr'], saved['global_step']
        restore_rng(saved['rng'], generator)
    if cfg['epochs'] <= start:
        raise ValueError('epochs is total target and must exceed checkpoint epoch')
    write_json(run/'config.json', dict(config, train=cfg, variant=variant, resolved_device=str(device)))
    write_json(run/'data_manifest.json', signature['data'])
    logger.info('variant=%s device=%s amp=%s parameters=%d train=%d val=%d start=%d',
                variant, device, amp, sum(p.numel() for p in model.parameters()), len(splits['train']), len(splits['val']), start)
    for epoch in range(start+1, cfg['epochs']+1):
        model.train()
        loss_sum, examples, batches = 0.0, 0, 0
        used_lr = optimizer.param_groups[0]['lr']
        for batch in loader:
            x, target = batch['input'].to(device), batch['target'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = nn.functional.l1_loss(model(x), target)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.get('grad_clip', 1.0), error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()*x.shape[0]
            examples += x.shape[0]
            batches += 1
            step += 1
            logger.info('epoch=%d step=%d loss=%.8f grad_norm=%.6f', epoch, step, loss.item(), norm.item())
        if not batches:
            raise RuntimeError('zero training batches')
        report = evaluate(model, splits['val'], run/f'val_epoch_{epoch:04d}', device, paired=True,
                          crop_border=config.get('metrics', {}).get('crop_border', 0), logger=logger,
                          metadata={'variant': variant, 'epoch': epoch, 'split': 'val'})
        psnr = report['metrics']['psnr']['mean']
        improved = psnr > best
        best = max(best, psnr)
        scheduler.step()
        saved = dict(format_version=1, variant=variant, backbone=backbone, module_options=module_options,
                     model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler={**scheduler.state_dict(), 'milestones': dict(scheduler.milestones)},
                     scaler=scaler.state_dict(), epoch=epoch, global_step=step, best_psnr=best,
                     signature=signature, rng=rng_state(generator))
        save_checkpoint(run/'last.pt', saved)
        if improved:
            save_checkpoint(run/'best.pt', saved)
        interval = cfg.get('save_every', 1)
        if interval > 0 and epoch % interval == 0:
            save_checkpoint(run/f'epoch_{epoch:04d}.pt', saved)
        record = dict(epoch=epoch, steps=batches, global_step=step, loss_l1=loss_sum/examples, lr=used_lr,
                      **{k: v['mean'] for k, v in report['metrics'].items()})
        history = run/'history.csv'
        exists = history.exists()
        with history.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            if not exists:
                writer.writeheader()
            writer.writerow(record)
        logger.info('epoch_summary=%s', json_safe(record))
    return run


def train(config, config_dir, variant, *, output=None, resume=None, epochs=None, device_override=None):
    run = Path(output).resolve() if output else (Path(config_dir)/config['output']/variant).resolve()
    try:
        return _train(config, config_dir, variant, output=output, resume=resume, epochs=epochs, device_override=device_override)
    finally:
        close_logger(logging.getLogger(str(run)))
