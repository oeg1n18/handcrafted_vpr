


"""
This is the main file for our Python implementation of CoHOG published in IEEE RAL 2020. The HOG_feature folder contains files that have been
modified by us, however, the original rights for all employed libraries lie with the corresponding contributors and we thanks them for
open-sourcing their implementations. For the rest of the libraries (skimage, scipy, matplotlib, cv2, numpy etc), please use standard methods of
installation for your IDE, if not already installed.

For any queries or problems, send me an email at mubarizzaffar@gmail.com. If you find this useful, please cite our paper titled
"CoHOG: A Light-weight, Compute-efficient and Training-free Visual Place Recognition Technique for Changing Environments".

Created on Thu Jan  3 11:18:57 2019

@author: mubariz
"""

import cv2
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from tqdm import tqdm
import numpy as np
import math



####################### PARAMETERS #########################
magic_width = 512
magic_height = 512
cell_size = 16  # HOG cell-size
bin_size = 8  # HOG bin size
image_frames = 1  # 1 for grayscale, 3 for RGB
descriptor_depth = bin_size * 4 * image_frames  # x4 is here for block normalization due to nature of HOG
ET = 0.4  # Entropy threshold, vary between 0-1.

total_no_of_regions = int((magic_width / cell_size - 1) * (magic_width / cell_size - 1))

#############################################################

#################### GLOBAL VARIABLES ######################

d1d2dot_matrix = np.zeros([total_no_of_regions, total_no_of_regions], dtype=np.float64)
d1d2matches_maxpooled = np.zeros([total_no_of_regions], dtype=np.float64)
d1d2matches_regionallyweighted = np.zeros([total_no_of_regions], dtype=np.float64)

matched_local_pairs = []
ref_desc = []


############################################################





class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / float(np.max(img)))
        self.img = self.img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert 360 % self.bin_size == 0, "bin_size should be divisible by 360"

    #    @jit(nopython=True)
    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        # hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image




def largest_indices_thresholded(ary):
    good_list = np.where(ary >= ET)
    #    no_of_good_regions=len(good_list[0])
    print(len(good_list))
    print(len(good_list[0]))

    return good_list


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""

    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


# @jit(nopython=False)
def conv_match_dotproduct(d1, d2, regional_gd, total_no_of_regions):  # Assumed aspect 1:1 here

    global d1d2dot_matrix
    global d1d2matches_maxpooled
    global d1d2matches_regionallyweighted
    global matched_local_pairs

    np.dot(d1, d2, out=d1d2dot_matrix)

    np.max(d1d2dot_matrix, axis=1, out=d1d2matches_maxpooled)  # Select best matched ref region for every query region

    np.multiply(d1d2matches_maxpooled, regional_gd,
                out=d1d2matches_regionallyweighted)  # Weighting regional matches with regional goodness

    score = np.sum(d1d2matches_regionallyweighted) / np.sum(regional_gd)  # compute final match score

    return score


##############################################################################################

NAME = 'CoHog'
def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    all_out_lists = []
    for query in tqdm(Q, desc='Computing Query Descriptors', disable=disable_pbar):
        outlist = []
        query = cv2.imread(query)
        img_2 = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
        img_2rgb = query
        if (img_2 is not None):
            img_2 = cv2.resize(img_2, (magic_height, magic_width))
            img_2rgb = cv2.resize(img_2rgb, (magic_height, magic_width))

            hog_desc = Hog_descriptor(img_2, cell_size=cell_size, bin_size=bin_size)
            #height, width, angle_unit = initialize(img_2, cell_size, bin_size)
            vector_2 = hog_desc.extract()
            vector_2 = np.asfortranarray(vector_2, dtype=np.float32)

            ################# Entropy Map ###############################
            #        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_as_ubyte(img_2), (100, 100))
            #        ent_time=time.time()
            entropy_image = cv2.resize(entropy(img_gray, disk(5)), (magic_width, magic_height))
            #        print('Entropy Time:',time.time()-ent_time)

            ################# Finding Regions #####################
            local_goodness = np.zeros([int(magic_height / cell_size - 1), int(magic_width / cell_size - 1)], dtype=np.float32)
            for a in range(int(magic_height / cell_size - 1)):
                for b in range(int(magic_width / cell_size - 1)):
                    #                local_staticity=1 #Disabling staticity here, can be accommodated in future by employing YOLO etc.
                    local_entropy = np.sum(entropy_image[a * cell_size:a * cell_size + 2 * cell_size,
                                           b * cell_size:b * cell_size + 2 * cell_size]) / (8 * (cell_size * 4 * cell_size))

                    if (local_entropy >= ET):
                        local_goodness[a, b] = 1
                    else:
                        # local_goodness[a,b]=local_entropy
                        local_goodness[a, b] = 0

            regional_goodness = local_goodness.flatten()
            outlist.append(vector_2)
            outlist.append(regional_goodness)
            all_out_lists.append(outlist)
    return all_out_lists


def compute_map_features(M, dataset_name=None, disable_pbar=False):  # ref_map is a 1D list of images in this case.
    "INPUT: reference list of images to be matched."
    "OUTPUT: Feature descriptors of all reference images to be matched."
    ref_map = M
    ref_desc = []
    for ref in tqdm(range(len(ref_map)), desc='Computing Map Features', disable=disable_pbar):
        img_1 = cv2.imread(ref_map[ref])
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

        if (img_1 is not None):
            img_1 = cv2.resize(img_1, (magic_height, magic_width))

            #        startencodetimer=time.time()

            hog_desc = Hog_descriptor(img_1, cell_size=cell_size, bin_size=bin_size)
            #height, width, angle_unit = initialize(img_1, cell_size, bin_size)
            vector_1 = hog_desc.extract()
            #        vector_1=np.asarray(vector_1.reshape(total_no_of_regions,len(vector_1[0][0])))
            vector_1 = np.asfortranarray(np.array(vector_1).transpose(), dtype=np.float32)
            ref_desc.append(vector_1)
    return ref_desc


def perform_vpr(q_path, m_desc):  # ref_map_features is a 1D list of feature descriptors of reference images in this case.
    "INPUT: Query desc and reference list of images' features to be matched."
    "OUTPUT: Matching Score and Best Matched Image."

    ref_map_features = m_desc
    query_info = compute_query_desc([q_path])[0]
    vector_2 = query_info[0]
    regional_goodness = query_info[1]

    confusion_vector = np.zeros(len(ref_map_features), dtype=np.float32)
    ref_desc = ref_map_features

    for ref in range(len(ref_map_features)):
        score = conv_match_dotproduct(vector_2.astype('float64'), ref_desc[ref].astype('float64'), regional_goodness,
                                      total_no_of_regions)
        # print(score, ref)
        confusion_vector[ref] = score

    return int(np.argmax(confusion_vector)), float(np.amax(confusion_vector))


def matching_method(q_desc, m_desc, disable_pbar=False):
    S = []
    for desc in tqdm(q_desc, desc='Computing Similarities', disable=disable_pbar):
        ref_map_features = m_desc
        query_info = desc
        ref_map_features = list(ref_map_features)
        vector_2 = query_info[0]
        regional_goodness = query_info[1]

        confusion_vector = np.zeros(len(ref_map_features), dtype=np.float32)
        ref_desc = ref_map_features

        for ref in range(len(ref_map_features)):
            score = conv_match_dotproduct(vector_2.astype('float64'), ref_desc[ref].astype('float64'),
                                          regional_goodness,
                                          total_no_of_regions)
            # print(score, ref)
            confusion_vector[ref] = score
        S.append(confusion_vector)
    return np.array(S)
