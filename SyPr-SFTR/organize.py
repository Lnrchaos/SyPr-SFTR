import os
import shutil
from pathlib import Path

def organize_project():
    base_dir = Path(__file__).parent
    src_dir = base_dir / 'src'
    package_dir = src_dir / 'syprsftr'
    
    # Create necessary directories
    (package_dir / 'federated').mkdir(exist_ok=True)
    (package_dir / 'symbolic').mkdir(exist_ok=True)
    (package_dir / 'probabilistic').mkdir(exist_ok=True)
    (package_dir / 'self_supervised').mkdir(exist_ok=True)
    (package_dir / 'models').mkdir(exist_ok=True)
    
    # Move files to their respective directories
    files_to_move = [
        (src_dir / 'federated.py', package_dir / 'federated' / '__init__.py'),
        (src_dir / 'symbolic.py', package_dir / 'symbolic' / '__init__.py'),
        (src_dir / 'probabilistic.py', package_dir / 'probabilistic' / '__init__.py'),
        (src_dir / 'self_supervised.py', package_dir / 'self_supervised' / '__init__.py'),
        (src_dir / 'transformer.py', package_dir / 'models' / 'transformer.py'),
    ]
    
    for src, dst in files_to_move:
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"Moved {src} to {dst}")
    
    # Create __init__.py files
    for dirpath, dirnames, _ in os.walk(package_dir):
        if '__pycache__' in dirnames:
            dirnames.remove('__pycache__')
        if not (Path(dirpath) / '__init__.py').exists():
            (Path(dirpath) / '__init__.py').touch()
    
    print("Project structure organized successfully!")

if __name__ == "__main__":
    organize_project()
