import os
import tarfile
import requests
from pathlib import Path
import spatialdata as sd
ALT_ZENODO_URL = (
    "https://zenodo.org/record/15226841/files/nsclc4301_truncated.zarr.tar?"
    "download=1"
)
GITHUB_URL = (
    "https://raw.githubusercontent.com/clinicalomx/napari-prism/main/datasets/"
    "nsclc4301_truncated.zarr.tar"
)

ZARR_NAME = "nsclc4301_truncated.zarr"
TAR_NAME = f"{ZARR_NAME}.tar"

def nsclc4301_truncated():
    cwd = Path.cwd()
    tar_path = cwd / TAR_NAME
    zarr_path = cwd / ZARR_NAME

    if not zarr_path.exists():
        print(f"Downloading {TAR_NAME} from Github...")
        with requests.get(GITHUB_URL, stream=True) as r:
            r.raise_for_status()
            with open(tar_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Extracting {TAR_NAME}...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=cwd)

        tar_path.unlink()  # remove the tar file
    
    # Then load zarr with spatialdata
    sdata = sd.read_zarr(zarr_path)
    return sdata