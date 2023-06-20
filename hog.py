
import cv2
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

NAME = 'HOG'

def compute_query_desc(Q, dataset_name=None, disable_pbar=False):

    ref_map = [cv2.imread(pth, 0) for pth in Q]

    winSize = (512, 512)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (16, 16)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    ref_desc_list = []
    for ref_image in tqdm(ref_map, desc='Computing Query Descriptors', disable=disable_pbar):
        if ref_image is not None:
            hog_desc = hog.compute(cv2.resize(ref_image, winSize))
        ref_desc_list.append(hog_desc)
    q_desc = np.array(ref_desc_list).astype(np.float32)
    return q_desc

def compute_map_features(M, dataset_name=None, disable_pbar=False):
    ref_map = [cv2.imread(pth, 0) for pth in M]

    winSize = (512, 512)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (16, 16)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    ref_desc_list = []
    for ref_image in tqdm(ref_map, desc='Computing Map Descriptors', disable=disable_pbar):
        if ref_image is not None:
            hog_desc = hog.compute(cv2.resize(ref_image, winSize))
        ref_desc_list.append(hog_desc)
    m_desc = np.array(ref_desc_list).astype(np.float32)
    return m_desc


def matching_method(q_desc, m_desc):
    return cosine_similarity(q_desc, m_desc)