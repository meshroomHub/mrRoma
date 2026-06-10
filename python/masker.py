from pyalicevision import matchingImageCollection as avmic   
from pyalicevision import matching as avmatch  
from pyalicevision import feature as avfeat
from pyalicevision import system as avsys

from common import *

import math
import os
import logging

from pathlib import Path


def parse_masks_folders(masksFolders):
    if masksFolders is None:
        return []

    if isinstance(masksFolders, (list, tuple)):
        folders = []
        for item in masksFolders:
            folders.extend(parse_masks_folders(item))
        deduped = []
        seen = set()
        for folder in folders:
            if folder not in seen:
                deduped.append(folder)
                seen.add(folder)
        return deduped

    if not isinstance(masksFolders, str):
        return []

    raw = masksFolders.strip()
    if len(raw) == 0:
        return []

    return [raw]


def find_mask_path(masksFolders, mask_filename):
    for folder in masksFolders:
        candidate = os.path.join(folder, mask_filename)
        if os.path.exists(candidate):
            return candidate
    return None


def apply_masks(inputSfMData, imagePairsList, warpFolder, confidenceFolder, masksFolders, masksExtension, outputConfidenceFolder, rangeIteration, rangeBlocksCount):
    
    # Parse sfm
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    # Retrieve list of images pairs to process
    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, False):
        raise RuntimeError("Error in image pairs list loading")

    
    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(plist))
    if not valid:
        logging.error("Error computing range.")
        raise RuntimeError("Error computing range.")

    masksFolders = parse_masks_folders(masksFolders)
    if len(masksFolders) == 0:
        logging.error("At least one masks folder is required.")
        raise RuntimeError("At least one masks folder is required.")
    

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
        path_confidence = os.path.join(confidenceFolder, pair_string + "_confidence.exr")

        logging.info(f"Processing pair {pair_string}")
        
        #load images
        warp_A_B = open_image_as_numpy(path_warp)
        confidence_A_B = open_image_as_numpy(path_confidence, True)

        #get properties
        warpHeight = confidence_A_B.shape[0]
        warpWidth = confidence_A_B.shape[1]

        #upgrade coordinates
        warp_A_B[:, :, 0] *= warp_A_B.shape[1]
        warp_A_B[:, :, 1] *= warp_A_B.shape[0]

        #Build resized masks
        maskReference = None
        maskOther = None
        # Replace the extension with the mask extension
        stem = os.path.splitext(os.path.basename(reference_iinfo.path))[0]
        mask_filename = f"{stem}.{masksExtension}"

        # Build the path to the correct mask
        path_mask = find_mask_path(masksFolders, mask_filename)

        #Build mask for reference
        if path_mask is not None:
            logging.info(f"Found reference mask at {path_mask}")
            maskLarge = open_image(path_mask, isBW=True, isFloat=False)
            maskSmall = avimage.Image_uchar()
            avimage.resampleImage(warpWidth, warpHeight, maskLarge, maskSmall, False);
            maskReference = np.squeeze(maskSmall.getNumpyArray())

        # Replace the extension with the mask extension
        stem = os.path.splitext(os.path.basename(other_iinfo.path))[0]
        mask_filename = f"{stem}.{masksExtension}"

        # Build the path to the correct mask
        path_mask = find_mask_path(masksFolders, mask_filename)

        if path_mask is not None:
            logging.info(f"Found comparison mask at {path_mask}")
            maskLarge = open_image(path_mask, isBW=True, isFloat=False)
            maskSmall = avimage.Image_uchar()
            avimage.resampleImage(warpWidth, warpHeight, maskLarge, maskSmall, False);
            maskOther = np.squeeze(maskSmall.getNumpyArray())

        #make sure masks exist
        if maskReference is None:
            continue
        
        #masking using reference mask is straightforward
        confidence_A_B[maskReference == 0] = 0

        if maskOther is not None:
            x = warp_A_B[:, :, 0].astype(np.int64)
            y = warp_A_B[:, :, 1].astype(np.int64)

            valid = (x >= 0) & (y >= 0) & (x < maskOther.shape[1]) & (y < maskOther.shape[0])

            masked_other = np.zeros_like(valid, dtype=bool)
            masked_other[valid] = maskOther[y[valid], x[valid]] == 0
            confidence_A_B[masked_other] = 0
         
        
        #write output
        path_confidence_output = os.path.join(outputConfidenceFolder, pair_string + "_confidence.exr")
        save_image(path_confidence_output, confidence_A_B, True)

if __name__ == '__main__':
    import argparse

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    # create the parser for the "warp" sub-command
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--warpFolder', type=str, help='')
    parser.add_argument('--confidenceFolder', type=str, help='')
    parser.add_argument('--masksFolders', nargs='*', default=None, help='')
    parser.add_argument('--masksExtension', type=str, help='')
    parser.add_argument('--outputConfidenceFolder', type=str, help='')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    parser.set_defaults(func=apply_masks)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                    imagePairsList=args.imagePairsList,
                    warpFolder=args.warpFolder,
                    confidenceFolder=args.confidenceFolder,
                    masksFolders=args.masksFolders,
                    masksExtension=args.masksExtension,
                    outputConfidenceFolder=args.outputConfidenceFolder,
                    rangeIteration=args.rangeIteration,
                    rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
