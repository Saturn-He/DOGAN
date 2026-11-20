#!/usr/bin/env python3
"""Run DOGAN with a JSON configuration."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments(settings):
    result = []
    for key, value in settings.items():
        if value is True:
            result.append('--' + key)
        elif value is not False and value is not None:
            result.extend(['--' + key, str(value)])
    return result



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.json')
    args, extra = parser.parse_known_args()
    config = json.loads(args.config.read_text())
    settings = dict(config['common'])
    settings.update(config['test' if args.mode == 'test' else 'train'])
    for key in ['dataroot', 'dino_checkpoint', 'checkpoints_dir', 'results_dir']:
        if key in settings:
            path = Path(settings[key]).expanduser()
            settings[key] = str(path if path.is_absolute() else ROOT / path)
    os.chdir(ROOT)
    os.environ.setdefault('TORCH_HOME', str(ROOT / 'pretrained/torch'))
    script = ROOT / ('train.py' if args.mode == 'train' else 'test.py')
    sys.argv = [str(script)] + arguments(settings) + extra
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
