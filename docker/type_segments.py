# Overview: Assign each segment (resulting from a segmentation algorithm) a cell type using the typed pixels output by
#           the bootstrap_mibi_pixels.py script
#
# Inputs:
#   'mask_file_name':           String containing the path/name of the segmentation mask.
#   'pixel_types_names':        String containing the path/name of the csv output from the bootstrap_mibi_pixels.py
#                               script that contains the cell type assigned to each pixel.
#   'hierarchy':                String containing the path/name of the yaml hierarchy file that was previously used in
#                               the bootstrap_mibi_pixels.py script.
#
# Outputs:
#   'segment_types.csv':        Csv file containing the cell type assigned to each segment.
#   'conflicted_segment_types.csv':     Csv file containing the potential cell types for each segment labeled as
#                                       conflicted (these are segments with similar likelihoods for more than one
#                                       cell type).
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
from skimage import io
import argparse
import yaml

parser = argparse.ArgumentParser(
        description='''Assigns cell types to segments based on majority of enclosed pixel types.'''
)

parser.add_argument('mask_file_name', metavar='M', type=str,
                    help='whole cell segmentation mask.')
parser.add_argument('pixel_types_name', metavar='P', type=str,
                    help='name of csv with pixel types.')
parser.add_argument('hierarchy', metavar='H', type=str,
                    help='the hierarchy file that groups markers by cell type.')
args = parser.parse_args()


def read_hierarchy_file(fname):
    with open(fname) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


segmentation_path = args.mask_file_name
pixel_type_path = args.pixel_types_name
hierarchy_yaml = args.hierarchy
output_csv_path = "segment_types.csv"
output_csv_path_c = "conflicted_segment_types.csv"

hierarchy_data = read_hierarchy_file(hierarchy_yaml)

name_list = []
priority_list = []
for k, v in hierarchy_data['hierarchy'].items():
    for n in v:
        name_list.append(n)
        priority_list.append(k)

cell_types = name_list
cell_hierarchy = np.array(priority_list)
cell_types_df = pd.DataFrame({"Name": name_list, "Priority": priority_list})

cell_type2index = {cell_type: cell_types.index(cell_type) for cell_type in cell_types}

with open(pixel_type_path) as csv_file:
    pixel_types = pd.read_csv(csv_file, delimiter=',')

pixel_types = pixel_types.sort_values(by='pixel_id', axis=0)

segmentation_im = np.squeeze(io.imread(segmentation_path))
tiff_dim = segmentation_im.shape
num_segments = np.max(segmentation_im)

for f in range(len(cell_types)):
    cell_type = cell_types[f]
    pixel_types = pixel_types.replace(to_replace=cell_type, value=f + 1)

pixel_types = pixel_types.replace(to_replace='Unknown', value=0)
tumor_num = cell_types.index('Tumor')+1

### Reshape each row
typed_im = np.reshape(pixel_types['type'].values, (tiff_dim[0], tiff_dim[1]), order='C') # row major

### Assign types to segments
num_segments = segmentation_im.max()
segment_type = []
conflicted_segment_num = []
conflicted_segment_types = []
for f in range(num_segments):
    seg_pix = np.where(segmentation_im == f+1)
    seg_area = len(seg_pix[0])
    pix_types = typed_im[seg_pix]
    pix_obj, pix_counts = np.unique(pix_types, return_counts=True)

    ### Remove any categories that constitute less than 5% of the segment area
    area_fraction = pix_counts / seg_area
    low_area_idx = np.where(pix_counts < 6)
    pix_counts = np.delete(pix_counts, np.unique(low_area_idx))
    pix_obj = np.delete(pix_obj, np.unique(low_area_idx))

    ### Prioritize tumor, add max norm and then look at if max is > 60% higher than other markers. If not, look at close categories and apply hierarchy
    if sum(pix_obj == tumor_num) != 0:
        segment_type.append("Tumor")

    elif len(pix_counts) > 1:
        if sum(pix_obj == 0) > 0:
            zero_idx = np.where(pix_obj == 0)
            pix_counts = np.delete(pix_counts, zero_idx)
            pix_obj = np.delete(pix_obj, zero_idx)

        pix_perc = pix_counts / pix_counts.max()
        high_idx = np.where(pix_perc >= 0.4)
        high_perc = pix_perc[high_idx]
        high_vals = pix_obj[high_idx]
        if len(high_perc) > 1:
            cell_type_idx = high_vals
            cell_type_for_indexing = cell_type_idx.astype(int)-1
            hierarchy_max = cell_hierarchy[cell_type_for_indexing].max()
            hierarchy_obj, hierarchy_counts = np.unique(cell_hierarchy[cell_type_for_indexing], return_counts=True)
            num_elements = hierarchy_counts[np.where(hierarchy_obj == hierarchy_max)]
            if num_elements > 1: ### Allocate ties to 'Conflicted'
                segment_type.append("Conflicted")
                conflicted_segment_num.append(f+1)
                conflicted_segment_types.append(np.array(cell_types)[cell_type_for_indexing.astype(int)])
            else:
                hierarchy_max_idx = cell_hierarchy[cell_type_for_indexing].argmax()
                final_type = cell_types[cell_type_for_indexing[hierarchy_max_idx].astype(int)]
                segment_type.append(final_type)

        else:
            cell_type_idx = high_vals
            cell_type_for_indexing = cell_type_idx.astype(int)-1
            final_type = cell_types[int(cell_type_for_indexing)]
            segment_type.append(final_type)

    else:
        if len(pix_counts) == 0:
            segment_type.append("Unknown")
        elif pix_obj[np.where(pix_counts == pix_counts.max())] == 0:
            segment_type.append("Unknown")  # No classified pixels in the segment
        else:
            segment_type.append(cell_types[int(pix_obj)-1])

type_df = pd.DataFrame({"segment": np.arange(1, num_segments+1, 1),
                        "type": segment_type})

conflicted_type_df = pd.DataFrame({"segment": conflicted_segment_num,
                                   "type": conflicted_segment_types})

### Save cell segments types
type_df.to_csv(output_csv_path, index=False)
conflicted_type_df.to_csv(output_csv_path_c, index=False)
