#!/usr/bin/env python3
"""Start the local demo, reset its database, or prefetch the existing classifier."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=['start', 'reset', 'prepare'], default='start')
    parser.add_argument('--skip-build', action='store_true', help='Use an already built frontend')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == 'reset':
        from lonn.demo.store import Store
        store = Store()
        store.reset()
        store.engine.dispose()
        print('Demo reset to 64 seeded requests and default settings. Customer key rotated. Provider credential retained.')
        return
    if args.command == 'prepare':
        from lonn.task_classifier import TaskClassifier
        TaskClassifier().classify('Explain binary search.')
        print('Classifier cached. Automatic routing is ready for offline local analysis.')
        return
    if not args.skip_build:
        manager = shutil.which('pnpm') or shutil.which('npm')
        if not manager:
            sys.exit('Install Node.js 20+ and npm or pnpm; then retry. Use --skip-build only after a frontend build.')
        lock = 'pnpm-lock.yaml' if Path(manager).name == 'pnpm' else 'package-lock.json'
        install = ['install', '--frozen-lockfile'] if Path(manager).name == 'pnpm' else (['ci'] if (ROOT/'frontend'/lock).exists() else ['install'])
        subprocess.run([manager, *install], cwd=ROOT/'frontend', check=True)
        subprocess.run([manager, 'run', 'build'], cwd=ROOT/'frontend', check=True)
    if not (ROOT/'frontend/dist/index.html').exists():
        sys.exit('Frontend build is missing; run without --skip-build.')
    print('Lönn workspace: http://127.0.0.1:8000\nCustomer app: http://127.0.0.1:8000/customer', flush=True)
    import uvicorn
    uvicorn.run('lonn.demo.api:app', host='127.0.0.1', port=8000, access_log=False)


if __name__ == '__main__':
    main()
