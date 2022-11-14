# Copyright Parker Institute for Cancer Immunotherapy, 2022
# 
# Licensing :
# This script is released under the Apache 2.0 License
# Note however that the programs it calls may be subject to different licenses.
# Users are responsible for checking that they are authorized to run all programs
# before running this script.

import sys
import json
import numpy as np
import pandas as pd
import tifffile as tf
from skimage import io
from sklearn.decomposition import PCA

def get_marker(page):
    """Extract channel marker label from MIBI Tiff Image Page"""
    desc = json.loads(page.tags['ImageDescription'].value)
    return desc['channel.target']

def extract_markers(tif_file):
    """Extract all MIBI channel information, returning a list where index of 
    marker name for channel matches the index of that channel in the image
    cube."""
    with tf.TiffFile(tif_file) as tif:
        return [get_marker(p) for p in tif.pages]


# input and output parameters
mibi_file = sys.argv[1] #'2047f620-b638-40de-82df-27b9417d9639.tiff'
out_tiff = sys.argv[2] #'pca_image_cube.tiff'
out_contributions_csv = sys.argv[3] #'pca_marker_contributions.csv'


# read MIBI image, calculate MIBI image info and views
mibi = io.imread(mibi_file)
chan, x, y = mibi.shape
flat_mibi = np.reshape(mibi, (chan, x*y))
pixel_stack = flat_mibi.T

# Run PCA and get back result (as chan x pixel matrix)
pca = PCA()
result = pca.fit_transform(pixel_stack)

# Reshape back to image and save as output_tiff_file
pca_img = np.reshape(result.T, (chan, x, y))
io.imsave(out_tiff, pca_img)

# Align markers with contributions to each principal component
markers = extract_markers(mibi_file)
pca_breakdown = pd.DataFrame(pca.components_.T)
pca_breakdown['marker_contribution'] = markers
pca_breakdown.to_csv(out_contributions_csv)
