# This will split an input tiff into tiles of the chosen size.
# Edge tiles will be smaller than the chosen size.
# Usage:
# python3 ./tile.py <inputFilePath> <outputFolder> <tilwWidth> <tileLength>

# Copyright Parker Institute for Cancer Immunotherapy, 2022
# 
# Licensing :
# This script is released under the Apache 2.0 License
# Note however that the programs it calls may be subject to different licenses.
# Users are responsible for checking that they are authorized to run all programs
# before running this script.

import pyvips, sys, os

infilePath = sys.argv[1]
outfolder = sys.argv[2]
tileWidth = int(sys.argv[3])
tileHeight = int(sys.argv[4])

# Load the image and get the relevant stats.
# pyvips loads all of the tiff pages/images as one image with all of the pages vertically joined.
# so page height is the height of one of the images, and n pages is the number of pages/images in
# the tiff.
image = pyvips.Image.new_from_file(infilePath, n=-1)
imageWidth = image.width
pageHeight = image.get('page-height')
nPages = image.get('n-pages')

print('Image stats:')
print(f'Width: {imageWidth}')
print(f'Height: {pageHeight}')
print(f'Total Height: {image.height}')
print(f'Number of Pages: {nPages}')

infileBasename = os.path.basename(infilePath)
splitBasename = infileBasename.split('.')
infileName = splitBasename[0]
outfileSuffix = '.'.join(splitBasename[1:])

for x in range(0, imageWidth, tileWidth):
  for y in range(0, pageHeight, tileHeight):
    # Calculate tile width and tile length in case we are on the last tile and it would go over the image boundaries.
    curTileWidth = min(tileWidth, imageWidth - x)
    curTileHeight = min(tileHeight, pageHeight - y)
    # Generate filename and filepath
    outfileName = f"{infileName}_{x}_{y}.{outfileSuffix}"
    outfilePath = os.path.join(outfolder, outfileName)
    # Notify and crop
    print(f"Cropping region {x}, {y}")
    images = []
    for z in range(0, nPages):
      images.append(image.crop(x, y + (pageHeight * z), curTileWidth, curTileHeight))
    # Join the images in the array together
    joined = pyvips.Image.arrayjoin(images, across=1)
    # Set the page height to the tile height so the pages/images in the tiff are properly separated/recognized.
    joined.set_type(pyvips.GValue.gint_type, 'page-height', curTileHeight)
    joined.write_to_file(outfilePath)