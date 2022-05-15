# Designed and implemented jointly with Noureddine Gueddach, Anne Marx, Mateusz Nowak (ETH Zurich)

import datetime
import os
import shutil
import urllib3
import warnings
import zipfile


DATASET_ZIP_URLS = {
    "backgrounds": "https://polybox.ethz.ch/index.php/s/7hE6WIct12CZi66/download",
    "ptb_xl": "https://polybox.ethz.ch/index.php/s/6cYYSheXDP6ZiC5/download"
}


def download_dataset(dataset_name):
    """
    Download the dataset with the given name and unpack it into a newly created subdirectory of the "datasets"
    directory; the URL of the ZIP to be downloaded can be specified in utils.py
    :param dataset_name: name of the dataset to download
    :return: True on success; False on failure
    """
    destination_path = os.path.join(*["datasets", dataset_name.lower()])
    ts_path = os.path.join(destination_path, "download_timestamp.txt")
    zip_path = f"{destination_path}.zip"

    url = next((v for k, v in DATASET_ZIP_URLS.items() if dataset_name.lower() == k.lower()), None)
    if url is None:
        warnings.warn(f"Dataset '{dataset_name}' unknown... error in download_dataset()")
        return False

    # check if data already downloaded; use timestamp file written *after* successful download for the check
    if os.path.exists(ts_path):
        return True
    else:
        os.makedirs(destination_path, exist_ok=True)

    # data doesn't exist yet
    print("Downloading Dataset...")
    pool = urllib3.PoolManager()
    try:
        with pool.request("GET", url, preload_content=False) as response, open(zip_path, "wb") as file:
            shutil.copyfileobj(response, file)
    except Exception as e:
        warnings.warn(f"Error encountered while downloading dataset '{dataset_name}': {str(e)}")
        return False
    print("...Done!")

    print("Extracting files...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(destination_path)
        print("...Done!")
    print("Removing zip file...")
    os.unlink(zip_path)
    print("...Done!")

    with open(ts_path, "w") as file:
        file.write(str(datetime.datetime.now()))

    return True
