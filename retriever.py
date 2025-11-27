"""
Simple URL downloader for dataset retrieval (wget-like).

Usage:
  python -m data.retriever --url https://example.com/dataset.zip --dest data/mydataset

This will download the archive and automatically unzip it into the destination directory.
"""

import argparse
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse
import requests
import zipfile
import tarfile


def _download_url(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[retriever] Downloading {url} to {out_path} ...")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"[retriever] Download complete: {out_path}")


def _is_zip_file(path: Path) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False


def _is_tar_file(path: Path) -> bool:
    try:
        return tarfile.is_tarfile(path)
    except Exception:
        return False


def fetch_from_url(url: str, dest: str, unzip: bool = True):
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(url)
    filename = Path(parsed.path).name
    if not filename:
        raise ValueError('Unable to extract filename from URL')

    # save in a temporary location under the dest dir
    out_file = dest_path / filename

    _download_url(url, out_file)

    if unzip:
        # If archive, extract to dest, then remove archive
        if _is_zip_file(out_file):
            print('[retriever] Extracting zip archive...')
            with zipfile.ZipFile(out_file, 'r') as zf:
                zf.extractall(dest_path)
            out_file.unlink()
            print('[retriever] Extraction complete')
        elif _is_tar_file(out_file):
            print('[retriever] Extracting tar archive...')
            with tarfile.open(out_file, 'r:*') as tf:
                tf.extractall(dest_path)
            out_file.unlink()
            print('[retriever] Extraction complete')
        else:
            # Not an archive; leave as-is
            print('[retriever] File does not appear to be a zip or tar archive; left as-is')

    return dest_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download dataset archive from a URL and unpack it into a destination folder')
    parser.add_argument('--url', type=str, required=True, help='URL to the dataset (zip or tar)')
    parser.add_argument('--dest', type=str, default='data', help='Destination directory to store the dataset')
    parser.add_argument('--no-unzip', dest='unzip', action='store_false', help='If set, do not attempt to unzip the downloaded file')

    args = parser.parse_args()
    try:
        path = fetch_from_url(args.url, args.dest, unzip=args.unzip)
        print(f"[retriever] Dataset available in: {path}")
    except Exception as e:
        print(f"[retriever] Error fetching dataset: {e}")
