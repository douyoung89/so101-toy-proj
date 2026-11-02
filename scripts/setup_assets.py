"""Fetch the pinned SO-101 model and explicitly initialize EasyOCR's local cache."""
import argparse
from pathlib import Path
import subprocess
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'a758567c3978dfeefe282ede0500085a48fe8f78'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ocr', action='store_true', help='also download EasyOCR English weights (~100 MB)')
    args = parser.parse_args()
    with ZipFile(ROOT/'data/labels/images.zip') as archive:
        archive.extractall(ROOT/'data/labels')
    target = ROOT/'external/SO-ARM100'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',
                        'https://github.com/TheRobotStudio/SO-ARM100.git',str(target)], check=True)
        subprocess.run(['git','-C',str(target),'sparse-checkout','set','Simulation/SO101'],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',COMMIT],check=True)
    actual = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'], text=True).strip()
    if actual != COMMIT:
        raise ValueError(f'Existing external checkout is {actual}; expected {COMMIT}; preserved without changes')
    assets = target/'Simulation/SO101/assets'
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    print(f'SO-101 model ready: {actual}')
    if args.ocr:
        import easyocr
        easyocr.Reader(['en'],gpu=False,download_enabled=True,
                       model_storage_directory=str(ROOT/'data/easyocr_models'),
                       user_network_directory=str(ROOT/'data/easyocr_models/user_network'))
        print('EasyOCR cache ready (ignored by Git).')

if __name__ == '__main__':
    main()
