"""PyCharm-friendly entry point; working directory is the repository root."""
import argparse
import hashlib
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from models.build import VARIANTS
from workflow.data import image_index, validate_splits
from workflow.engine import evaluate, load_checkpoint, load_config, restore_model, select_device, train


def main(argv=None):
    parser = argparse.ArgumentParser(description='FDM-Net train/validate/test/unpaired infer')
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check-data')
    check.add_argument('--config', required=True)
    fit = commands.add_parser('train')
    fit.add_argument('--config', required=True)
    fit.add_argument('--variant', choices=VARIANTS, default='fdmnet')
    fit.add_argument('--resume')
    fit.add_argument('--epochs', type=int, help='total target epochs, not additional epochs')
    fit.add_argument('--output')
    fit.add_argument('--device')
    for name in ('validate', 'test', 'infer'):
        command = commands.add_parser(name)
        command.add_argument('--checkpoint', required=True)
        command.add_argument('--output', required=True, help='new output directory')
        command.add_argument('--device', default='auto')
        command.add_argument('--crop-border', type=int)
        command.add_argument('--threads', type=int, default=2)
        command.add_argument('--input' if name == 'infer' else '--config', required=True)
    args = parser.parse_args(argv)
    if args.command in ('check-data', 'train'):
        config, directory = load_config(args.config)
        if args.command == 'check-data':
            print({k: len(v) for k, v in validate_splits(config['data'], directory).items()})
        else:
            train(config, directory, args.variant, output=args.output, resume=args.resume,
                  epochs=args.epochs, device_override=args.device)
        return
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    saved = load_checkpoint(args.checkpoint)
    device = select_device(args.device)
    model = restore_model(saved, device)
    paired = args.command != 'infer'
    if paired:
        config, directory = load_config(args.config)
        splits = validate_splits(config['data'], directory)
        split = 'val' if args.command == 'validate' else 'test'
        if split not in splits:
            parser.error(f'missing {split} split')
        items = splits[split]
    else:
        path = Path(args.input).resolve()
        items = [path] if path.is_file() else list(image_index(path).values())
    border = args.crop_border if args.crop_border is not None else (config.get('metrics', {}).get('crop_border', 0) if paired else 0)
    with Path(args.checkpoint).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    evaluate(model, items, args.output, device, paired=paired, crop_border=border,
             metadata={'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': digest,
                       'variant': saved['variant'], 'epoch': saved['epoch'], 'split': args.command})
    print((Path(args.output)/'summary.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
