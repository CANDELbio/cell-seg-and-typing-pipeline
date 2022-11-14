# Sum the specified layers into a new single layer:
# Args: inputfilename, outputfilename, indexes (json list of integers)
# i.e.: python collapse.py "inputfile.tiff" "outputfile.tiff" "[4, 13, 20, 23]"
#
# Copyright Parker Institute for Cancer Immunotherapy, 2022
# 
# Licensing :
# This script is released under the Apache 2.0 License
# Note however that the programs it calls may be subject to different licenses.
# Users are responsible for checking that they are authorized to run all programs
# before running this script.
 
import skimage.io as skio
import numpy as np
import json, sys

infile = sys.argv[1]
outfile = sys.argv[2]
indexes = json.loads(sys.argv[3])

imstack = skio.imread(infile, plugin="tifffile")
out_img = np.zeros_like(imstack[1,:,:])
out_img = np.sum(imstack[indexes,:,:], axis=0)
skio.imsave(outfile, out_img, plugin="tifffile")
