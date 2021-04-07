import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from tqdm import tqdm
from scipy.signal import resample_poly
from visualization.visualization import VisualizationApp
from visualization.contours import brain_hull, contour_tensor_mask, brain_hull_idxs
from model.gradcam import load_vgg16, init_gcpp
from model.preprocessing import UnprocessedDataset, transform_datapoint


# Init logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s] - [%(levelname)s] - %(message)s'
)

# Parse args
ap = argparse.ArgumentParser()
ap.add_argument(
    '-f', '--folder', 
    type=str, 
    default='data/sample_volume', 
    help='Brain volume folder path.'
)
ap = ap.parse_args()

# Params for entire training set
mean = np.load('data/mean.npy')
std = np.load('data/std.npy')

# Load volume
training_data = UnprocessedDataset(ap.folder)
sample_volume = training_data[0]
volume_name = training_data.files[0]
output_path = f'data/gradcam_output/{volume_name}.npy'


if __name__ == '__main__':    
    logging.debug('Starting brain slice transformations.')
    transformed_imgs = []
    for brain_slice in tqdm(sample_volume):
        img_transformed = transform_datapoint(brain_slice, mean, std)
        transformed_imgs.append(img_transformed)
        
    # Permute and convert images to ndarray
    transformed_imgs_np = np.array([
        np.array(img[0].permute(1, 2, 0)) for img in transformed_imgs
    ])
    hull = brain_hull(transformed_imgs_np)
    tensor_mask = contour_tensor_mask(hull)
        
    # Load or compute GradCAM++ output for brain
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.debug(f'GradCAM++ output for volume "{volume_name}" found. Loading...')
        mask_3d = np.load(output_path)
    else:
        logging.debug(f'GradCAM++ output for volume "{volume_name}" not found.')
        vgg = load_vgg16(path='data/trained_model.pt')
        Gcpp = init_gcpp(vgg)
        logging.debug('VGG16 model loaded.')

        # Transform and GradCAM++ images
        logging.debug('Starting GradCAM++ on brain volume slices.')
        slice_masks = []
        for img_transformed in tqdm(transformed_imgs):
            mask, _ = Gcpp(img_transformed, class_idx=0)
            np_mask = mask[0, 0, :].numpy()
            slice_masks.append(np_mask)
        mask_3d = np.stack(slice_masks)
        np.save(output_path, mask_3d)
        logging.debug(f'GradCAM++ output saved at "{output_path}".')
    
    # Downsample volume hull
    hull_downsampled = np.nan_to_num(hull.copy())
    # Downsample brain volume mask, inside brain activation only
    mask_downsampled_inside = np.nan_to_num(mask_3d.copy())
    # Downsample brain volume activations
    mask_downsampled = np.nan_to_num(mask_3d.copy())
    mask_downsampled_inside[~tensor_mask] = 0
    for k in range(3):
        hull_downsampled = resample_poly(hull_downsampled, 1, 8, axis=k)
        mask_downsampled = resample_poly(mask_downsampled, 1, 8, axis=k)
        mask_downsampled_inside = resample_poly(mask_downsampled_inside, 1, 8, axis=k)
    hull_downsampled[hull_downsampled > 0.01] = 1 # > Fix hardcoded thres 0.01
    logging.debug('Downsampling complete.')
    
    # Normalize
    inside_squared = mask_downsampled_inside**2
    inside_squared /= inside_squared.flatten().max()
    
    logging.debug('Starting 3d visualization Dash app.')
    mesh_xyz = brain_hull_idxs(hull_downsampled)
    vis_data = {
        'all-activation': mask_downsampled,
        'inside-activation': mask_downsampled_inside,
        'inside-activation-squared': inside_squared
    }
    Vis = VisualizationApp(vis_data, mesh_xyz)
    Vis.run()