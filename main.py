from data.datasets import SFU, GardensPointWalking, StLucia

# To download the datasets
StLucia.download()
GardensPointWalking.download()
SFU.download()


# To load the Datasets
Q = SFU.get_query_paths() # Loads a list of paths to the query images
M = SFU.get_map_paths() # Loads a list of paths to the reference images
GT = SFU.get_gtmatrix() # Loads the ground truth matrix



""" 
The Similarity matrix S[i,j] gives the similarity score between Q[i] and M[j]
The Ground Truth Matrix GT gives the grount truth such that G[i,j] == 1 if Q[i] and M[j] depict the same place
"""

# ========== HOG - (Histogram of Gradients VPR technqiues ===============
import hog
q_desc = hog.compute_query_desc(Q) # Compute the query descriptors
m_desc = hog.compute_map_features(M) # compute the map descriptors
S = hog.matching_method(q_desc, m_desc) # compute the similarity matrix

# ============= SIFT-BOW (sift feature extraction with bag of visual words feature aggregation) ================
import sift_bow
q_desc = sift_bow.compute_query_desc(Q) # compute the query descriptors
m_desc = sift_bow.compute_map_features(M) # compute the map descriptors
S = sift_bow.matching_method(q_desc, m_desc) # compute the similarity matrix

# =============================== CoHog ===========================
import cohog
q_desc = cohog.compute_query_desc(Q[:10]) # compute the query descriptors
m_desc = cohog.compute_map_features(M[:10]) # compute the map descriptors
S = cohog.matching_method(q_desc, m_desc) # compute the similarity matrix



