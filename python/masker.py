from pyalicevision import matchingImageCollection as avmic   
from pyalicevision import matching as avmatch  
from pyalicevision import feature as avfeat
from pyalicevision import system as avsys

from common import *

import math
import os
import logging

from pathlib import Path


def apply_masks(inputSfMData, imagePairsList, warpFolder, certaintyFolder, masksFolder, masksExtension, outputCertaintyFolder, rangeIteration, rangeBlocksCount):
    
    # Parse sfm
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    # Retrieve list of images pairs to process
    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, 0, -1, False):
        raise RuntimeError("Error in image pairs list loading")

    
    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(plist))
    if not valid:
        logging.error("Error computing range.")
        raise RuntimeError("Error computing range.")
    

    # loop over pairs of images
    for id in range(rangeStart, rangeEnd):

        item = plist[id]
        
        #id of views
        referenceId = item[0]
        otherId = item[1]

        #retrieve ImageInfos
        reference_iinfo = iinfos[referenceId]
        other_iinfo = iinfos[otherId]

        #Build paths
        pair_string = str(referenceId) + "_" + str(otherId)
        path_warp = os.path.join(warpFolder, pair_string + "_warp.exr")
        path_certainty = os.path.join(warpFolder, pair_string + "_certainty.exr")

        logging.info(f"Processing pair {pair_string}")
        
        #load images
        warp_A_B = open_image_as_numpy(path_warp)
        certainty_A_B = open_image_as_numpy(path_certainty, True)

        #get properties
        warpHeight = certainty_A_B.shape[0]
        warpWidth = certainty_A_B.shape[1]

        #upgrade coordinates
        warp_A_B[:, :, 0] *= warp_A_B.shape[1]
        warp_A_B[:, :, 1] *= warp_A_B.shape[0]

        #Build resized masks if possible
        maskReference = None
        maskOther = None
        if len(masksFolder) > 0 :
            # Replace the extension with the mask extension
            stem = os.path.splitext(os.path.basename(reference_iinfo.path))[0]
            mask_filename = f"{stem}.{masksExtension}"

            # Build the path to the correct mask
            path_mask = os.path.join(masksFolder, mask_filename)

            #Build mask for reference
            if os.path.exists(path_mask):
                maskLarge = open_image(path_mask, isBW=True, isFloat=False)
                maskSmall = avimage.Image_uchar()
                avimage.resampleImage(warpWidth, warpHeight, maskLarge, maskSmall, False);
                maskReference = np.squeeze(maskSmall.getNumpyArray())

            # Replace the extension with the mask extension
            stem = os.path.splitext(os.path.basename(other_iinfo.path))[0]
            mask_filename = f"{stem}.{masksExtension}"

            # Build the path to the correct mask
            path_mask = os.path.join(masksFolder, mask_filename)

            if os.path.exists(path_mask):
                maskLarge = open_image(path_mask, isBW=True, isFloat=False)
                maskSmall = avimage.Image_uchar()
                avimage.resampleImage(warpWidth, warpHeight, maskLarge, maskSmall, False);
                maskOther = np.squeeze(maskSmall.getNumpyArray())

        #make sure masks exist
        if maskReference is None or maskOther is None:
            continue
        
        #masking using reference mask is straightforward
        certainty_A_B[maskReference == 0] = 0

        # #masking by checking warped coordinates
        x = warp_A_B[:, :, 0].astype(int)
        y = warp_A_B[:, :, 1].astype(int)
        
        # # Create mask for valid coordinates (within bounds)
        outOfBounds = (x < 0) | (x >= maskOther.shape[1]) | (y < 0) | (y >= maskOther.shape[0])
        x[outOfBounds] = 0
        y[outOfBounds] = 0
        
        invalid = outOfBounds | ~maskOther[y, x]
        
        # # Set certainty to 0 where invalid
        certainty_A_B[invalid] = 0
        
        
        #write output
        path_certainty_output = os.path.join(outputCertaintyFolder, pair_string + "_certainty.exr")
        save_image(path_certainty_output, certainty_A_B, True)

if __name__ == '__main__':
    import argparse

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    # create the parser for the "warp" sub-command
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--warpFolder', type=str, help='')
    parser.add_argument('--certaintyFolder', type=str, help='')
    parser.add_argument('--masksFolder', type=str, help='')
    parser.add_argument('--masksExtension', type=str, help='')
    parser.add_argument('--outputCertaintyFolder', type=str, help='')
    parser.add_argument('--rangeIteration', type=int, help='')
    parser.add_argument('--rangeBlocksCount', type=int, help='')
    parser.set_defaults(func=apply_masks)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                    imagePairsList=args.imagePairsList,
                    warpFolder=args.warpFolder,
                    certaintyFolder=args.certaintyFolder,
                    masksFolder=args.masksFolder,
                    masksExtension=args.masksExtension,
                    outputCertaintyFolder=args.outputCertaintyFolder,
                    rangeIteration=args.rangeIteration,
                    rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
