"""Unpack bundled simulation assets and download English OCR weights if missing."""
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'simulation/assets'


def main():
    for name in ('meshes', 'labels'):
        destination = ASSETS / name
        if not destination.is_dir():
            destination.mkdir(parents=True)
            with ZipFile(ASSETS / f'{name}.zip') as archive:
                archive.extractall(destination)
            print(f'Prepared {name}', flush=True)
    images = ROOT / 'data/images'
    for path in sorted(images.glob('images_*.zip')):
        with ZipFile(path) as archive:
            if any(not (images / name).is_file() for name in archive.namelist()):
                archive.extractall(images)
                print(f'Prepared {path.name}', flush=True)
    cache = ASSETS / 'ocr_models'
    if not all((cache / name).is_file() for name in ('craft_mlt_25k.pth', 'english_g2.pth')):
        import easyocr
        print('Downloading English OCR weights (first run only)...', flush=True)
        easyocr.Reader(['en'], gpu=False, download_enabled=True,
                       model_storage_directory=str(cache),
                       user_network_directory=str(cache / 'user_network'))
    print('Simulation assets ready', flush=True)


if __name__ == '__main__':
    main()
