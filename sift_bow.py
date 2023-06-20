from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from glob import glob
import random
import config
import os

# please fill in db_path with your directory to the training images of choice
db_path = os.path.join(os.getcwd() + '/data/raw_data/SFU/dry')
n_visual_words = 200

NAME = "SIFT-BOW"

if db_path == None:
    raise NotImplementedError


def train(n_clusters=200, training_size=1):
    """
    input: n_clusters the size of the visual word vocabulary to create
    output: a trained KMeans object containing the bag of visual words
    """
    assert os.path.exists(db_path)
    db_paths = glob(db_path + '/*')
    random.shuffle(db_paths)
    # sample just a subset so clustering doesnt take too long
    db_paths = db_paths[:training_size]


    if len(db_paths) == 0:
        raise Exception("Path to database of images was incorrect")
    # Instantiate the opencv SIFT object
    sift = cv2.SIFT_create()

    # Read in the images as numpy arrays
    imgs = [np.array(Image.open(img_pth), dtype=np.uint8) for img_pth in db_paths]

    # Convert the images to grayscale
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    # Extract the sift features for the images to create a dataset
    # There are multiple features per image so stacking into a single matrix
    features = np.vstack([sift.detectAndCompute(img, None)[1] for img in imgs]).astype(np.float32)

    # Clustering the features into a bag of words
    print("Clustering the database")
    kmeans = KMeans(n_clusters=n_clusters).fit(features)
    return kmeans


# Cluster the dataset to get bagovw
kmeans = train(n_clusters=n_visual_words)


def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    """
    inputs: Q a list aboslute image paths
    outputs: a 2d numpy matrix image features
    """
    sift = cv2.SIFT_create()
    all_representations = []
    for img_pth in Q:
        # Read in the image
        img = np.array(Image.open(img_pth))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Compute the sift features
        img_features = sift.detectAndCompute(img, None)[1]
        # Quantize the sift features with BOW representation
        word_indexes = kmeans.predict(img_features)
        img_representation = np.histogram(word_indexes, bins=np.arange(n_visual_words+1))[0]
        # Normalize the histogram to zero mean and unit standard deviation
        img_representation = (img_representation - img_representation.mean()) / img_representation.std()
        all_representations.append(img_representation)
    return np.array(all_representations).astype(np.float32)


def compute_map_features(M, dataset_name=None, disable_pbar=False):
    """
    inputs: Q a list aboslute image paths
    outputs: a 2d numpy matrix image features
    """
    sift = cv2.SIFT_create()
    all_representations = []
    for img_pth in M:
        # Read in the image
        img = np.array(Image.open(img_pth))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Compute the sift features
        img_features = sift.detectAndCompute(img, None)[1]
        # Quantize the sift features with BOW representation
        word_indexes = kmeans.predict(img_features)
        img_representation = np.histogram(word_indexes, bins=np.arange(n_visual_words+1))[0]
        # Normalize the histogram to zero mean and unit standard deviation
        img_representation = (img_representation - img_representation.mean()) / img_representation.std()
        all_representations.append(img_representation)
    return np.array(all_representations).astype(np.float32)



def matching_method(q_desc, m_desc):
    return cosine_similarity(q_desc, m_desc)