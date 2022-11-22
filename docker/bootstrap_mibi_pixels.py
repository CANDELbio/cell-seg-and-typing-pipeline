# Overview: Assign cell types to pixels based on distributions of markers across a single slide
#
# Inputs:
#   'mibifile':     String containing path/name of the image file to be analyzed.
#                   Ex: '~/image_folder/image_file.tiff'
#   'classfile':    String containing the path/name of the output classfile tiff (see Outputs)
#   'hierarchy':    String containing the path/name of the yaml file that contains three user defined lists.
#                   'cell_types' contains the individual markers that are expected to be expressed in each cell type.
#                   'hierarchy' contains a numerically defined hierarchy that indicates which cell type should be
#                   prioritized in instances of a scoring tie between more than one cell type (higher hierarchy value
#                   has a higher weight).
#                   'location' indicates the markers that are expressed in the nucleus.
#                   Ex:
#                       cell_types:
#                           CD4 T cell:
#                               - CD45
#                               - CD3
#                               - CD4
#                           CD8 T cell:
#                               - CD45
#                               - CD3
#                               - CD8
#                           Tumor:
#                               - SOX10
#                           ...
#
#                       hierarchy:
#                           1:
#                               - Endothelial
#                               - Fibroblast
#                               - Immune
#                           2:
#                               - Plasma cell
#                               - Myeloid
#                               - B cell
#                               - T cell
#                               - Monocyte
#                               - Dendritic
#                               - NK cell
#                               - Macrophage
#                           3:
#                               - CD4 T cell
#                               - CD8 T cell
#                               - Neutrophil
#                               - NKT cell
#                           4:
#                               - Treg
#                           5:
#                               - Tumor
#                       location:
#                           nucleus:
#                               - SOX10
#                               - FOXP3
#   'output-classes.csv':       String containing the path/name of the output output_classes csv file (see Outputs).
#   'output-pixel-labels.csv':  String containing the path/name of the output output_pixel_labels csv file
#                               (see Outputs).
#   'marker-threshold':         Value between 0 and 1 that indicates the cut-off value of the marker distributions. Any
#                               individual pixel marker values that fall below this threshold of the marker's
#                               distribution are considered background.
#                               Ex: 'marker-threshold' = 0.4 indicates that markers values in the bottom 40% of the
#                               distribution are considered background noise.
#
# Outputs:
#   classfile.tiff:             Output tiff image in which each pixel has been set to it's designated cell type (each
#                               cell type is assigned to a unique integer)
#   output-classes.csv:         Csv file containing the lookup between cell type and integer value used in the
#                               classfile.tiff.
#   output-pixel-labels.csv:    Csv file that contains the cell type assigned to each pixel in the image.
#
#   The names and save paths for these outputs are set by the input variables of the same name.
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
from numpy.random import default_rng
import json
import yaml
import skimage
import tifffile as tf
from skimage import io
from functools import reduce
import argparse
import itertools

seed = 123456
rng = default_rng(seed)

parser = argparse.ArgumentParser(
        description='''Performs cell type classification on an image using
        the bootstrap method.'''
)
parser.add_argument('mibifile', metavar='M', type=str,
                    help='the mibi file to be classified.')
parser.add_argument('classfile', metavar='M', type=str,
                    help='the classified image (integer code per class)')
parser.add_argument('--hierarchy', metavar='H', type=str, 
                    help='the hierarchy file that groups markers by cell type.',
                    default='markers_hierarchy.yml')
parser.add_argument('--output-classes', metavar='S', type=str,
                    help='the csv file to be output for class labels')
parser.add_argument('--output-pixel-labels', metavar='P', type=str,
                     help='the csv file to be output for individual pixel classes')
parser.add_argument('--marker-threshold', metavar='T', type=float,
                    help='percentile threshold of distribution used as cutoff' +
                         ' for pixels to be classified as positive for marker.',
                    default=0.4)
args = parser.parse_args()

mibi_file = args.mibifile
hierarchy_yaml = args.hierarchy

args = parser.parse_args()


def read_hierarchy_file(fname):
    with open(fname) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def hierarchy2matrix(hierarchy_data, lineage_markers):
    """Generates a matrix of cell types by marker using data provided
    in the hierarchy.yml file."""
    by_cell_type = pd.DataFrame({'marker': lineage_markers})
    ctypes = list(hierarchy_data['cell_types'].keys())
    for ct in ctypes:
        by_cell_type[ct] = False
    by_cell_type.set_index('marker', inplace=True)
    for cell_type, markers in hierarchy_data['cell_types'].items():
        for marker in markers:
            by_cell_type.loc[marker][cell_type] = True
    return by_cell_type


def random_indices(percent, length):
    """Given flat length of an array and percent of indices to sample, returns
    a random sample of indices.""" 
    sample_size = np.int64(np.floor(percent * length))
    return np.unique(rng.integers(0, length-1, sample_size))


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


def enrich_panel(panel_df):
    """Adds name, full name, and priority fields to the panel data frame."""
    # Full Name: cell type, population, subpopulation, state, filtering out NAs
    # Name is the last entry of Full Name
    # Priority is the length of Full Name
    panel_df['Full Name'] = panel_df.apply(
            lambda row: list(filter(lambda item: not pd.isna(item),
                             [row['Cell Type'], row['Population'],
                              row['Subpopulation'], row['State']])),
            axis=1)
    panel_df['Name'] = panel_df['Full Name'].apply(lambda item: item[-1])
    panel_df['Priority'] = panel_df['Full Name'].apply(lambda item: len(item))
    return panel_df


def generate_null_distributions(pixel_stack, N=10, percent=0.10):
    """Generates null distributions per marker by calculating sample means for
    10% of each channel image, computed N times."""
    distributions = []
    for i in range(0, N):
        distribution_seed_inds = random_indices(percent, pixel_stack.shape[0])
        seed_array = pixel_stack[distribution_seed_inds, :]
        dist_means = np.mean(seed_array, axis=0)
        distributions.append(dist_means)
    all_distributions = np.vstack(distributions)
    return all_distributions


def calculate_percentile_weights(pixel_stack, thresh=0.5):
    """For each marker channel, calculates the percentile breakdown
    of non-zero values then returns an output image weighted by the
    one percentile placement of intensity in the distribution of values
    for that channel. I.e. if a value is higher than 99% of other values
    in the channel, it will be given the value 0.01, or 87% 0.13 and
    so on."""
    perc_thresh = int(100 * thresh)
    output = np.zeros(pixel_stack.shape, dtype=np.float32)
    for i in range(0, pixel_stack.shape[1]):
        this_chan = pixel_stack[:, i]
        this_chan_nz = this_chan[np.where(this_chan > 0.01)]
        breaks = np.arange(perc_thresh, 100, 1)
        perc = np.percentile(this_chan_nz, breaks)
        break_lookup = np.searchsorted(perc, this_chan)
        output[:, i] = 1.0 - (((break_lookup) - 1) * 0.01) + thresh)
    return output


# read the hierarchy data file and construct matrix
hierarchy_data = read_hierarchy_file(hierarchy_yaml)
ctypes = list(hierarchy_data['cell_types'].keys())
num_cell_types = len(ctypes)
marker_list = list(hierarchy_data['cell_types'].values())
marker_list_unique = list(itertools.chain.from_iterable(marker_list))
lineage_markers = pd.Series(list(set(marker_list_unique)))

mxct_matrix = hierarchy2matrix(hierarchy_data, lineage_markers)

# this fixes e.g. DC.SIGN -> DC-SIGN (-, ' ', etc. replaced with '.' from excel
lineage_markers = lineage_markers.str.replace('\.', '-', regex=True)

# read in all MIBI data and channel -> marker information for all markers present
# in MIBI file.
mibi = io.imread(mibi_file)
markers = extract_markers(mibi_file)
marker_lookup = {val:markers.index(val) for val in markers}
lineage_markers_i = [marker_lookup[m] for m in lineage_markers]

# calculate MIBI image info and views
chan, x, y = mibi.shape
flat_mibi = np.reshape(mibi, (chan, x*y))

# -- for reference, how to get back to original mibi array shape --
# rest_mibi = np.reshape(flat_mibi, (chan, x, y))
pixel_stack = flat_mibi.T

# -- If nuclear markers are not co-expressed with dsDNA, set marker to 0

nucleus_markers = []
for k, v in hierarchy_data['location'].items():
    for n in v:
        if k == "nucleus":
            nucleus_markers.append(n)

### Could generalize to main nuclear marker (ex DAPI, dsDNA, etc)
dsDNA_idx = markers.index("dsDNA")
for marker in nucleus_markers:
    marker_idx = markers.index(marker)
    for i in range(0, np.shape(pixel_stack)[0]):
        if pixel_stack[i, dsDNA_idx] == 0 and pixel_stack[i, marker_idx] > 0:
            pixel_stack[i, marker_idx] = 0

# -- BOOTSTRAP calculations start --
weights = calculate_percentile_weights(pixel_stack, thresh=args.marker_threshold)
positives = (1 - weights) * (weights <= (1 - args.marker_threshold))
### Add penalty for missing markers
positives[np.where(positives == 0)] = -0.1
###
lib_matrix = np.array(mxct_matrix)
cell_type_weights = positives[:, lineage_markers_i] @ lib_matrix
### Bring negative weights back to 0
cell_type_weights[np.where(cell_type_weights < 0)] = 0
###
maxes = np.max(cell_type_weights, axis=1)
max_full_dims = np.vstack([maxes]*num_cell_types)
argmaxes = (cell_type_weights == max_full_dims.T) & (cell_type_weights != 0.0)

# ordered cell types the same as the `lib_matrix` cell type dimension
cell_types = list(mxct_matrix.columns)

name_list = []
priority_list = []
for k, v in hierarchy_data['hierarchy'].items():
    for n in v:
        name_list.append(n)
        priority_list.append(k)

priority_df = pd.DataFrame({"Name": name_list, "Priority": priority_list})

# explicitly set Tumor class priority to 10
tumor_entry = priority_df[priority_df['Name'] == 'Tumor']
priority_df.loc[tumor_entry.index[0], 'Priority'] = 10

# get priority values as appropriately ordered vector
priority_df.set_index('Name', inplace=True)
priority_ordered = priority_df.loc[cell_types]
priority_vec = np.array(priority_ordered['Priority'])

# assign priority value for each possible class, the argmax of this is
# the best match.
typematch_w_priority = argmaxes * priority_vec
best_match = np.argmax(typematch_w_priority, axis=1)

# find pixels where every class match is 0.0 and set these to the Unknown
# class
all_zeros = np.sum(typematch_w_priority == np.repeat(0.0, num_cell_types), axis=1) == num_cell_types
best_match[all_zeros] = -1

# output class image as coded image w/0 as unknown (by incrementing each class #)
class_image = np.reshape(best_match, (x, y)) + 1
io.imsave(args.classfile, class_image)

# output csv of class labels when arg passed
if args.output_classes:
    classes = ['Unknown'] + cell_types
    class_df = pd.DataFrame({'class_code': range(0, len(classes)),
                             'class_label': classes})
    class_df.to_csv(args.output_classes, index=False)

if args.output_pixel_labels:
    classes = ['Unknown'] + cell_types
    marker_lookup = {val: classes.index(val) for val in classes}
    output_df = pd.DataFrame(columns=['pixel_id', 'type'])
    for i in marker_lookup.values():
        idx = np.ravel_multi_index(np.where(class_image == i), dims=class_image.shape)
        input_data = np.transpose(np.vstack([idx, [classes[i]] * len(idx)]))
        output_df = pd.concat([output_df, pd.DataFrame(input_data, columns=['pixel_id', 'type'])], axis=0)

    output_df.to_csv(args.output_pixel_labels, index=False)
