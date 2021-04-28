import scipy.ndimage as ndimage
import numpy as np

from skimage import measure
from matplotlib import image
from PIL import Image


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """RGB to graysc of ndarray using PIL."""
    vol_max = rgb.max()
    np_normalized = (rgb / vol_max * 255).astype(np.uint8)
    r = Image.fromarray(np_normalized).convert('L')
    return np.array(r)


def contour_matrix(r: np.ndarray, find_thres=220) -> np.ndarray:
    """2D contours of grayscale image r."""
    contours = measure.find_contours(r, find_thres)
    contour_matrix = np.zeros(r.shape)
    if not len(contours):
        return contour_matrix
    for x, y in contours[0]:
        contour_matrix[round(x), round(y)] = 1
    return contour_matrix
    
    
def contour_matrix_mask(contour_matrix: np.ndarray) -> np.ndarray:
    """Binary mask on brain from 2D contour matrix."""
    return ndimage.binary_fill_holes(contour_matrix) # ~ to invert


def contour_tensor_mask(contour_tensor: np.ndarray) -> np.ndarray:
    """Binary mask on brain from 3D contour tensor."""
    return np.apply_along_axis(contour_matrix_mask, 1, contour_tensor)


def brain_hull(volume: np.ndarray) -> np.ndarray:
    """3D contours of volume represented as a sparse matrix of volume shape."""
    contour_tensor = []
    for idx in range(volume.shape[0]): # idx x 240 x 240 x 4
        img = volume[idx, :, :, :-1]
        r = rgb_to_grayscale(img)
        contour_tensor.append(contour_matrix(r))
    return np.array(contour_tensor)


def brain_hull_idxs(hull_tensor: np.ndarray):
    """x, y, z of 3D contours of volume for 3d mesh gen."""
    return np.where(hull_tensor == 1)