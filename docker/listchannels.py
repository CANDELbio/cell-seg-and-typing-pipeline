# This will output the index and page name of all the channels:
#
# Copyright Parker Institute for Cancer Immunotherapy, 2022
# 
# Licensing :
# This script is released under the Apache 2.0 License
# Note however that the programs it calls may be subject to different licenses.
# Users are responsible for checking that they are authorized to run all programs
# before running this script.

import tifffile, sys
from ome_types import from_tiff


infile = sys.argv[1]
outfile = sys.argv[2]
outstr = ""

def tryOutputAndQuit(outfile, outstr):
	if outstr:
		f = open(outfile, 'wt', encoding='utf-8')
		f.write(outstr)
		f.close()
		quit()

# Parse OME-TIFF metadata
try:
	ome = from_tiff(infile)
	for index, channel in enumerate(ome.images[0].pixels.channels):
		outstr += f"{index} {channel.name}\n"
except:
	pass

tryOutputAndQuit(outfile, outstr)

# Parse MIBItiff metadata
try:
	with tifffile.TiffFile(infile) as tif:
		for index, page in enumerate(tif.pages):
			outstr += str(index) + " " + page.tags['PageName'].value + "\n"
except:
	pass

tryOutputAndQuit(outfile, outstr)

raise Exception("Unable to parse TIFF metadata.")