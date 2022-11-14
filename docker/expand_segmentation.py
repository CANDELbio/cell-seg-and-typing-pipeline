# Overview: Expand the original cell segmentation using the pixel type assignments from the bootstrap_mibi_pixels.py
#           script. This algorithm is particularly useful for irregularly shaped cell types (ex. fibroblasts,
#           macrophages, etc).
#
# Inputs:
#   'mask_file_name':       String containing the path/name of the segmentation mask.
#   'segment_types_name':   String containing the path/name of the csv output from the type_segments.py
#                           script that contains the cell type assigned to each segment.
#   'pixel_types_name':     String containing the path/name of the csv output from the bootstrap_mibi_pixels.py
#                           script that contains the cell type assigned to each pixel.
#
# Outputs:
#   expanded_segment_types.csv:   Csv file containing the cell type assigned to each expanded/added segment.
#   expanded_full_mask.tif:       Tiff file containing the expanded segments. All pixels belonging to a given segment
#                                 are equal to the segment id.
#
#   These outputs have fixed names and will be written to the working directory.
#
# Copyright Parker Institute for Cancer Immunotherapy, 2022
# 
# Licensing :
# This script is released under the Apache 2.0 License
# Note however that the programs it calls may be subject to different licenses.
# Users are responsible for checking that they are authorized to run all programs
# before running this script.

import numpy as np
import pandas as pd
import math
import argparse
from skimage import io
from skimage.segmentation import flood_fill
from skimage import measure

parser = argparse.ArgumentParser(
        description='''Expand current segmentations to include unassigned, touching pixels of the same type.'''
)

parser.add_argument('mask_file_name', metavar='M', type=str,
                    help='whole cell segmentation mask.')
parser.add_argument('segment_types_name', metavar='S', type=str,
                    help='name of csv with segment types.')
parser.add_argument('pixel_types_name', metavar='P', type=str,
                    help='name of csv with pixel types.')


args = parser.parse_args()

segmentation_name = args.mask_file_name
pixel_type_name = args.pixel_types_name
segment_type_name = args.segment_types_name
output_name = "expanded_segment_types.csv"
output_mask_name = "expanded_full_mask.tif"

args = parser.parse_args()


def alloc_matrix2d(W, H):
    """ Pre-allocate a 2D matrix of empty lists. """
    return [ [ [] for i in range(W) ] for j in range(H) ]


def find_dist(overlap_seg_idx, centroids, pix_idx):
    dist_1 = []
    for i in range(len(overlap_seg_idx)):
        dist_1.append(math.sqrt((pix_idx[0] - centroids[overlap_seg_idx[i]-1][0]) ** 2 +  ### overlap_seg_idx holds segment idx that starts from 1
                                (pix_idx[1] - centroids[overlap_seg_idx[i]-1][1]) ** 2))  ### need index to start from 0
    return dist_1


segmentation_im = np.squeeze(io.imread(segmentation_name))
tiff_dim = segmentation_im.shape
num_segments = np.max(segmentation_im)

with open(pixel_type_name) as csv_file:
    pixel_types = pd.read_csv(csv_file, delimiter=",")

with open(segment_type_name) as csv_file:
    segment_types = pd.read_table(csv_file, delimiter=",")

cell_types = pd.unique(segment_types['type'])
cell_types = np.delete(cell_types, np.where(cell_types == "Unknown"))

final_seg_output = segmentation_im

for cell in cell_types:

    ### Initialize variables
    pixel_mask = np.zeros((tiff_dim[0], tiff_dim[1]))
    display_mask = np.zeros((tiff_dim[0], tiff_dim[1]))
    segment_mask = np.zeros((tiff_dim[0], tiff_dim[1]))
    new_segmentation_im = np.zeros((tiff_dim[0], tiff_dim[1]))

    ### Create mask with current cell type pixels only
    pixel_rows = pixel_types.index[pixel_types['type'] == cell]
    pixel_idx = pixel_types.values[pixel_rows, 0]
    pixel_coords = np.unravel_index(pixel_idx.astype(int), (tiff_dim[0], tiff_dim[1]), order='C')
    pixel_mask[pixel_coords[0], pixel_coords[1]] = 1
    display_mask[pixel_coords[0], pixel_coords[1]] = 1

    ### Create mask with current cell type segments only
    segment_idx = segment_types.index[segment_types["type"] == cell].tolist()
    if len(segment_idx) > 0:
        count = 0
        for i in segment_idx:
            seg_pix = np.where(segmentation_im == i+1)
            count = count+1
            new_segmentation_im[seg_pix] = count
            segment_mask[seg_pix] = 1
            display_mask[seg_pix] = 0.5

        ### Remove pixels already belonging to other segments (from original segmentation)
        existing_segments = np.where(segmentation_im != 0)
        pixel_mask[existing_segments] = 0

        ### For each segment, expand to include all touching, typed pixels

        centroids = np.zeros((count, 2))
        obj_pixels = list()
        for i in range(count):
            seg_pix = np.where(new_segmentation_im == i+1)
            row_val = np.floor(np.mean(seg_pix[0])).astype(int)
            col_val = np.floor(np.mean(seg_pix[1])).astype(int)
            centroids[i, 0] = row_val
            centroids[i, 1] = col_val
            seg_im = np.zeros((tiff_dim[0], tiff_dim[1]))
            pixel_mask[seg_pix] = 1
            pixel_mask[row_val, col_val] = 1
            ff_im = flood_fill(pixel_mask, (row_val.astype(int), col_val.astype(int)), 10)
            new_pix = np.where(ff_im == 10)
            seg_im[new_pix] = 1
            seg_im[seg_pix] = 1
            unique_pixels = np.unique(np.concatenate([np.ravel_multi_index(new_pix, (tiff_dim[0], tiff_dim[1])),
                                                      np.ravel_multi_index(seg_pix, (tiff_dim[0], tiff_dim[1]))]))
            obj_pixels.append(np.unravel_index(unique_pixels.astype(int), (tiff_dim[0], tiff_dim[1])))
            pixel_mask[seg_pix] = 0
            pixel_mask[row_val, col_val] = 0
            del(ff_im, seg_im, seg_pix)

        ### Loop through obj_pixels and append obj id to 2048x2048 matrix of lists
        ### Have list of objects each pixel belongs to
        list_matrix_obj = alloc_matrix2d(tiff_dim[0], tiff_dim[1])
        overlap_seg = np.zeros((tiff_dim[0], tiff_dim[1]))
        for i in range(len(obj_pixels)):
            l1 = obj_pixels[i]
            for j in range(len(l1[0])):
                list_matrix_obj[l1[0][j]][l1[1][j]].append(i+1)
                overlap_seg[l1[0][j]][l1[1][j]] = len(list_matrix_obj[l1[0][j]][l1[1][j]])

        multi_pix_idx = np.where(overlap_seg > 1)
        single_pix_idx = np.where(overlap_seg <= 1)

        final_segmentation = np.zeros((tiff_dim[0], tiff_dim[1]))

        for i in range(tiff_dim[0]):
            for j in range(tiff_dim[1]):
                if len(list_matrix_obj[i][j]) > 1:
                    final_segmentation[i][j] = 0
                else:
                    if len(list_matrix_obj[i][j]) == 1:
                        final_segmentation[i][j] = np.array(list_matrix_obj[i][j])

        min_dist = []
        if len(multi_pix_idx[0]) != 0:
            for i in range(len(multi_pix_idx[0])):
                pix_idx = [multi_pix_idx[0][i], multi_pix_idx[1][i]]
                overlap_seg_idx = list_matrix_obj[pix_idx[0]][pix_idx[1]]
                dist_vec = find_dist(overlap_seg_idx, centroids, pix_idx)
                min_dist.append(min(dist_vec))
                min_seg_idx = overlap_seg_idx[dist_vec.index(min(dist_vec))]
                final_segmentation[pix_idx[0]][pix_idx[1]] = min_seg_idx

        ### Create master segmentation with expanded boundaries for all cell types (and original segment ids)
        current_idx = range(1, final_segmentation.max().astype(int)+1)
        for s_idx in range(final_segmentation.max().astype(int)):
            final_seg_output[np.where(final_segmentation == current_idx[s_idx])] = segment_idx[s_idx]+1

        ### Add pixels not included in a current segment to their own segments
        temp_seg = np.zeros((tiff_dim[0], tiff_dim[1]))
        temp_seg[final_segmentation != 0] = 1
        diff_pix = pixel_mask - temp_seg
        diff_pix[diff_pix == -1] = 0
        new_obj = measure.label(diff_pix)
        unique_new_obj, new_obj_size = np.unique(new_obj[:, :], return_counts=True)
        unique_new_obj = np.delete(unique_new_obj, 0)  ### Remove 0 (background) from array
        new_obj_size = np.delete(new_obj_size, 0)

        ### Compare objs to mean size of already established objects
        unique_curr_obj, curr_obj_size = np.unique(final_segmentation[:, :], return_counts=True)
        unique_curr_obj = np.delete(unique_curr_obj, 0) ### Remove 0 (background) from array
        curr_obj_size = np.delete(curr_obj_size, 0)
        curr_obj_mean = curr_obj_size.mean()
        curr_obj_std = curr_obj_size.std()
        min_limit = max(20, curr_obj_mean - curr_obj_std)
        new_obj_idx = np.where((new_obj_size > min_limit) & (new_obj_size < curr_obj_mean + curr_obj_std))
        if np.size(new_obj_idx) != 0:
            for i in range(len(new_obj_idx[0])):
                new_idx = num_segments + 1
                num_segments = new_idx
                new_segment_idx = unique_new_obj[new_obj_idx[0][i]]
                final_seg_output[np.where(new_obj == new_segment_idx)] = new_idx

                segment_types.loc[new_idx-1] = [new_idx, cell]


### Ensure that original segmentations are not changed
original_segment_idx = np.where(segmentation_im != 0)
final_seg_output[original_segment_idx] = segmentation_im[original_segment_idx]

final_seg_dwn = final_seg_output.astype(np.uint16)
io.imsave(output_mask_name, final_seg_dwn)

segment_types.to_csv(output_name, index=False)
