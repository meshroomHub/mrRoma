from romav2.device import device
from romav2 import RoMaV2
from romav2.io import tensor_to_pil

from common import *

import os
import torch
import logging


def prepare_warp(w):
    
    """ Transform the warp tensor from roma to a RGB image with B value being always 1 """
    w = ((w + 1.0) / 2.0).detach().cpu().numpy().copy()
    w = np.concatenate([w, np.zeros([w.shape[0], w.shape[1], 1], dtype=np.float32)], axis=-1)
    

    return w

def prepare_confidence(c):
    """ Transform the confidence tensor from roma to a 3 dimensional array 
    (Last dimension being of size 1)
    """
    c = c.detach().cpu().numpy().copy()
    
    return c


def checkUncertaintyLoops(warp_A_B, warp_B_A, certainty_A_B, certainty_B_A, upsampleResolution):
    """ Take the minimum of certainty between the original certainty, and the certainty of the warped pixel.
        Will update certainty_A_B

    Parameters:
        warp_A_B the warp image between A and B
        warp_B_A the warp image between B and A
        certainty_A_B certainty of warp_A_B
        certainty_B_A certainty of warp_B_A
        upsampleResolution tuple of roma resolution
    """
    H = upsampleResolution[0]
    W = upsampleResolution[1]

    coords = warp_A_B[:, :, :2].copy().reshape(-1, 2)
    
    coords_Xm = (coords[:, 0] * W).astype(int)
    coords_Xp = coords_Xm + 1
    coords_Ym = (coords[:, 1] * H).astype(int)
    coords_Yp = coords_Ym + 1

    coords_Xm = np.clip(coords_Xm, 0, W - 1)
    coords_Xp = np.clip(coords_Xp, 0, W - 1)
    coords_Ym = np.clip(coords_Ym, 0, H - 1)
    coords_Yp = np.clip(coords_Yp, 0, H - 1)

    c11 = certainty_B_A[coords_Ym, coords_Xm]
    c12 = certainty_B_A[coords_Ym, coords_Xp]
    c21 = certainty_B_A[coords_Yp, coords_Xm]
    c22 = certainty_B_A[coords_Yp, coords_Xp]
    
    maxuncertainty = np.maximum(np.maximum(np.maximum(c11, c12), c21), c22)
    c = maxuncertainty.reshape(H, W, 1)    
    np.minimum(certainty_A_B, c, out=certainty_A_B)

def compute_densematches(inputSfMData, imagePairsList, outputWarpFolder, outputCertaintyFolder, checkLoops, rangeIteration, rangeBlocksCount):
    """ This high level function is computing the warp between pairs of images

    Parameters:
        inputSfmData : the sfmData containing the descriptions of the images to match
        imagePairsList : a list of pair of images uids which list the warp to compute
        outputWarpFolder : a destination folder for the warp images
        outputCertaintyFolder : a destination folder for the certainty images
    """
    

    #Parse sfmdata, create compatible images
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, 0, -1, False):
        raise RuntimeError("Error in image pairs list loading")
    pairsToProcess = list(plist)

    blockSize = int(len(pairsToProcess) / rangeBlocksCount)
    rangeStart = rangeIteration * blockSize
    rangeEnd = rangeStart + blockSize
    if rangeIteration + 1 == rangeBlocksCount:
        rangeEnd = len(pairsToProcess)

    
    pairsToProcess = pairsToProcess[rangeStart:rangeEnd]
    logging.info(f"Processing elements {rangeStart} to {rangeEnd}")

    logging.info("Loading model ....")
    model = RoMaV2()
    model.apply_setting("precise")
    upsampleResolution = (model.H_lr, model.W_lr) if (model.H_hr is None or model.W_hr is None) else (model.H_hr, model.W_hr) 

    
    for item in pairsToProcess:
        referenceId = item[0]
        otherId = item[1]
        referenceInfo = iinfos[referenceId]
        otherInfo = iinfos[otherId]   

        # Effectively do the matching
        # Output is (batch_size, 2, H, W)
        logging.info(f"Matching {referenceId} with {otherId}")

        imA = open_image_to_pil(referenceInfo.path)
        imB = open_image_to_pil(otherInfo.path)

        preds = model.match(imA, imB)
        warp_A_B = prepare_warp(preds["warp_AB"][0])
        warp_B_A = prepare_warp(preds["warp_BA"][0])
        certainty_A_B, certainty_B_A = (
            prepare_confidence(preds["overlap_AB"][0]),
            prepare_confidence(preds["overlap_BA"][0]),
        )
      
        if checkLoops:
            checkUncertaintyLoops(warp_A_B, warp_B_A, certainty_A_B, certainty_B_A, upsampleResolution)

        logging.info("saving matches")
        pair_string = str(referenceId) + "_" + str(otherId)
        path_warp = os.path.join(outputWarpFolder, pair_string + "_warp.exr")
        path_certainty = os.path.join(outputCertaintyFolder, pair_string + "_certainty.exr")
        save_image(path_warp, warp_A_B, False)
        save_image(path_certainty, certainty_A_B, True)

if __name__ == '__main__':
    import argparse
    
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaMatcher')

    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--outputWarpFolder', type=str, help='')
    parser.add_argument('--outputCertaintyFolder', type=str, help='')
    parser.add_argument('--checkLoops', type=str_to_bool, help='', default=False)
    parser.add_argument('--rangeIteration', type=int, help='')
    parser.add_argument('--rangeBlocksCount', type=int, help='')
    parser.set_defaults(func=compute_densematches)

    args = parser.parse_args()

    if hasattr(args, 'func'): 
        args.func(inputSfMData=args.inputSfMData,
                    imagePairsList=args.imagePairsList,
                    outputWarpFolder=args.outputWarpFolder,
                    outputCertaintyFolder=args.outputCertaintyFolder,
                    checkLoops=args.checkLoops,
                    rangeIteration=args.rangeIteration,
                    rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()