#这个extract有问题
from skimage import feature as ft
import cv2
# from .HOG import HOG

def extract_hog_feature(img, cell_size):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features = ft.hog(img,  # input image
                  orientations=9,  # number of bins
                  pixels_per_cell=(4,4), # pixel per cell
                  cells_per_block=(4,4), # cells per blcok
                  block_norm = 'L1', #  block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                  transform_sqrt = True, # power law compression (also known as gamma correction)
                  feature_vector=True, # flatten the final vectors
                  visualize=False) # return HOG map

    return features

def extract_cn_feature():
    raise NotImplementedError
