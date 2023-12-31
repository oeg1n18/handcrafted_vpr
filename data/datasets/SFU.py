import os
import urllib
import zipfile
import urllib.request
from os.path import isdir

import numpy as np
import glob
import config

NAME = 'SFU'
def get_query_paths():
    if isdir(config.root_dir + '/data/raw_data/SFU'):
        return sorted(glob.glob(config.root_dir + '/data/raw_data/SFU/jan/*.jpg'))
    else:
        print("must download dataset first into rootdir + /data/raw_data/ directory")


def get_map_paths():
    if isdir(config.root_dir + '/data/raw_data/SFU'):
        return sorted(glob.glob(config.root_dir + '/data/raw_data/SFU/dry/*.jpg'))
    else:
        print("must download dataset first into rootdir + /data/raw_data/ directory")

def get_gtmatrix(gt_type='soft'):
    gt_data = np.load(config.root_dir + '/data/raw_data/SFU/GT.npz')
    if gt_type=='hard':
        GT = gt_data['GThard'].astype('bool')
    else:
        GT = gt_data['GTsoft'].astype('bool')
    return GT.astype(np.uint8)

def download(rootdir=None):
    destination = config.root_dir + '/data/raw_data/SFU'
    print('===== SFU dataset does not exist. Download to ' + destination + '...')
    fn = 'SFU.zip'
    url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn

    # create folders
    path = os.path.expanduser(destination)
    os.makedirs(path, exist_ok=True)

    # download
    urllib.request.urlretrieve(url, path + fn)

    # unzip
    with zipfile.ZipFile(path + fn, 'r') as zip_ref:
        zip_ref.extractall(destination)

    # remove zipfile
    os.remove(destination + fn)