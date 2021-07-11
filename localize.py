# -*- coding: utf8 -*-

"""
Localization of all connex components (area >= area_min)
and extraction of all subsequent patches as individual 
images : a brute version on one hand, and an enhanced 
version in other hand. 

"""

__author__ =  'Thierry BROUARD'
__version__=  '0.1'

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from skimage import io 
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.color import label2rgb, gray2rgb
from skimage.morphology import binary_closing
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--area", type=int, default=10,
    help="min area for a connex component")
ap.add_argument("-eb", "--extbrute", type=int, default=0,
    help="if 1, save brute pictures into brutepath directory")
ap.add_argument("-bp", "--brutepath", type=str, default="./brute/",
    help="path to output brute pictures")
ap.add_argument("-ee", "--extenhanced", type=int, default=0,
    help="if 1, save enhanced pictures into enhpath directory")
ap.add_argument("-ep", "--enhpath", type=str, default="./enhanced/",
    help="path to output enhanced pictures")
ap.add_argument("-i", "--image", required=True, type=str,
    help="image filename to process")
args = vars(ap.parse_args())

# parameters
area_min = args["area"]
extract_brute = args["extbrute"]==1
extract_enhanced = args["extenhanced"]==1
brute_path = args["brutepath"]
enhanced_path = args["enhpath"]
imageName = args["image"]



print( "[INFO] reading ", imageName, "..." )
# load image
imgOrig = io.imread( imageName )


print( "[INFO] processing image ..." )
# preprocessing
thresh = threshold_otsu(imgOrig)
imgBin = imgOrig <= thresh


print( "[INFO] labeling ..." )
# labeling 
label_image = label(imgBin)


print( "[INFO] getting area of all connex components ..." )
# filtering bounding boxes
regions = []
for region in regionprops(label_image):

    minr, minc, maxr, maxc = region.bbox
    area = (maxr-minr) * (maxc-minc)
    if area >= area_min:
        regions.append([minr, minc, maxr, maxc])
    regions.append([minr, minc, maxr, maxc])


print( "[INFO] extraction of ", len(regions)," connex components ..." )
# extraction
num_extracted = 1
for region in regions :

    minr, minc, maxr, maxc = region
    imgExtracted = ( imgOrig[minr:maxr, minc:maxc] )
  
    # brute
    if extract_brute:
        img_name_ext = "%08d.png" % ( num_extracted , )
        io.imsave(brute_path+img_name_ext, imgExtracted, check_contrast=False)

    if extract_enhanced:
        # enhanced : only one CC by BB + morpho
        thresh = threshold_otsu(imgOrig)
        imgBin = ( imgExtracted <= thresh )
        imgBin = binary_closing(imgBin, selem=np.ones((2, 2)))
        label_image = label(imgBin)
    
        # if more than one cc in bbox
        liste_cc = regionprops( label_image )

        if len( liste_cc ) > 1 :
            max_area = liste_cc[0].area
            max_label = liste_cc[0].label
            for cc in liste_cc :
                if cc.area > max_area :
                    max_area = cc.area
                    max_label = cc.label


            imgBin = ( label_image == max_label )

        # mask brute by enhanced
        imgMasked = 255 - ((255 - imgExtracted) * imgBin)

        img_name_ext = "%08d.png" % ( num_extracted , )
        io.imsave(enhanced_path+img_name_ext, imgMasked, check_contrast=False)

    num_extracted = num_extracted + 1


print( "[INFO] building result image ..." )
# draw bounding boxes on a colored image
from skimage.draw import line, set_color
from skimage.draw import rectangle_perimeter

imgColor = gray2rgb(imgOrig)

for region in regions :

    minr, minc, maxr, maxc = region
    area = (maxr-minr) * (maxc-minc)
    if area >= area_min :
        start = (minr, minc)
        end = (maxr, maxc)
        rr, cc = rectangle_perimeter(start, end=end)
        set_color(imgColor, (rr, cc), [255, 0, 0])

io.imsave('./labeled-'+imageName, imgColor, check_contrast=False)
print( "[INFO] ", './labeled-'+imageName, " saved." )

# displaying
image_label_overlay = label2rgb(label_image, image=imgOrig, bg_label=0)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

for region in regions :

    minr, minc, maxr, maxc = region

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()
