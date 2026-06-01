from romav2.device import device
from romav2 import RoMaV2
from romav2.io import tensor_to_pil
from romav2.features import Descriptor

from common import *

import os
import torch
import logging


def prepare_warp(w):
    
    """ 
    Transform the warp tensor from roma to a RGB image with B value being always 0.
    First step : w contains values between -1 and 1.0. Updates it to be between 0 and 1.
    Second step : converts a 2 channel image to a 3 channel image, filling the last channel with 0 
    """
    w = ((w + 1.0) / 2.0).detach().cpu().numpy().copy()
    w = np.concatenate([w, np.zeros([w.shape[0], w.shape[1], 1], dtype=np.float32)], axis=-1)

    return w

def prepare_confidence(c):
    
    """ 
    Transform the confidence tensor from roma to a 3 dimensional array 
    (Last dimension being of size 1)
    """
    c = c.detach().cpu().numpy().copy()
    
    return c

def check_loop(warp_A_B, warp_B_A):
    """Check the loop consistency of a warp pair.

    For each pixel in A, follow warp_A_B to B, then warp_B_A back to A.
    Returns the round-trip distance (in pixels) from the original position.

    Parameters:
        warp_A_B: warp image from A to B, values in [0, 1] (shape H x W x 2)
        warp_B_A: warp image from B to A, values in [0, 1] (shape H x W x 2)

    Returns:
        Distance map of shape (H, W, 1), or None if inputs are invalid.
        Pixels whose forward coordinate falls outside the image are set to inf.
    """
    if warp_A_B is None or warp_B_A is None:
        return None

    if warp_A_B.shape != warp_B_A.shape:
        return None

    height, width = warp_A_B.shape[0], warp_A_B.shape[1]

    # Pixel coordinates of where each A pixel maps to in B
    x_A_B = warp_A_B[:, :, 0] * width
    y_A_B = warp_A_B[:, :, 1] * height

    # Integer neighbors for sampling warp_B_A
    ixm = np.floor(x_A_B).astype(int)
    iym = np.floor(y_A_B).astype(int)
    ixp = ixm + 1
    iyp = iym + 1

    # Valid mask (before clipping): forward coordinate must lie inside the image
    valid = (ixm >= 0) & (iym >= 0) & (ixp < width) & (iyp < height)

    # Clip to valid range to avoid out-of-bounds access
    ixm = np.clip(ixm, 0, width - 1)
    iym = np.clip(iym, 0, height - 1)
    ixp = np.clip(ixp, 0, width - 1)
    iyp = np.clip(iyp, 0, height - 1)

    # Original pixel coordinates (column = x, row = y)
    y_orig, x_orig = np.mgrid[0:height, 0:width].astype(np.float32)

    # Sample warp_B_A at each of the 4 neighbors and measure round-trip error
    def dist_at(iy, ix):
        back_x = warp_B_A[iy, ix, 0] * width
        back_y = warp_B_A[iy, ix, 1] * height
        return np.sqrt((back_x - x_orig) ** 2 + (back_y - y_orig) ** 2)

    dist = np.minimum(
        np.minimum(dist_at(iym, ixm), dist_at(iym, ixp)),
        np.minimum(dist_at(iyp, ixm), dist_at(iyp, ixp)),
    )
    dist = np.where(valid, dist, np.inf)

    return dist[..., np.newaxis]

def updateUncertaintyWithLoops(warp_A_B, warp_B_A, confidence_A_B, confidence_B_A, threshold):
    """Zero out confidence where the A→B→A loop error exceeds the threshold.

    For each pixel in A the round-trip error is computed: the pixel is mapped
    to B via warp_A_B, then mapped back to A via warp_B_A.  If the Euclidean
    distance between the recovered position and the original pixel exceeds
    *threshold* (in pixels), both confidence_A_B and confidence_B_A are set to
    zero at that location.

    Parameters:
        warp_A_B      warp image from A to B, values in [0, 1]
        warp_B_A      warp image from B to A, values in [0, 1]
        confidence_A_B  confidence map for warp_A_B (modified in-place)
        confidence_B_A  confidence map for warp_B_A (modified in-place)
        threshold     maximum accepted round-trip distance in pixels
    """
    dist = check_loop(warp_A_B, warp_B_A)
    if dist is None:
        return

    invalid = (dist > threshold)[..., 0]  # shape (H, W)
    confidence_A_B[invalid] = 0.0
    confidence_B_A[invalid] = 0.0

def compute_densematches(inputSfMData, imagePairsList, minConfidence, outputWarpFolder, outputConfidenceFolder, outputCovarianceFolder, checkLoops, loopThreshold, outputCovarianceFlag, rangeIteration, rangeBlocksCount):
 
    from pyalicevision import system as avsys

    # Parse sfmdata, create compatible images
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    # Load the image pairs to process (referenceView, matchingView)
    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, False):
        raise RuntimeError("Error in image pairs list loading")
    pairsToProcess = list(plist)

    #Compute parallelization parameters using the number of pairs to process
    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(pairsToProcess))
    if not valid:
        logging.error("Range is out of bounds.")
        return

    
    pairsToProcess = pairsToProcess[rangeStart:rangeEnd]
    
    logging.info(f"Processing elements {rangeStart} to {rangeEnd}")
    logging.info("Loading model")

    # Try to load model from disk if found.
    # Otherwise, the model will be downloaded on the net
    roma_weights = None
    dinov3_path = None
    if "ROMATCH_MODELS_PATH" in os.environ:
        modelPath = os.environ["ROMATCH_MODELS_PATH"]
        romaModelPath = os.path.join(modelPath, "romav2.pt")
        roma_weights = torch.load(romaModelPath, weights_only=True)
        dinov3_path = os.path.join(modelPath, "dinov3")

    # Create roma objects
    descCfg = Descriptor.Cfg(module_path=dinov3_path)
    romaCfg = RoMaV2.Cfg(descriptor=descCfg, weights=roma_weights, compile=False)
    model = RoMaV2(cfg=romaCfg)
    model.apply_setting("precise")
    upsampleResolution = (model.H_lr, model.W_lr) if (model.H_hr is None or model.W_hr is None) else (model.H_hr, model.W_hr) 

    
    # Loop over all pairs of images received as parameter.
    # Each pair of images will be processed independently
    for item in pairsToProcess:
        
        # SfmData was previously parsed,
        # Retrieve the loaded information for the pair
        referenceId = item[0]
        otherId = item[1]
        referenceInfo = iinfos[referenceId]
        otherInfo = iinfos[otherId]   

        # Effectively do the matching
        # Output is (batch_size, 2, H, W)
        logging.info(f"Matching {referenceId} with {otherId}")

        # Load images from disk
        imA = open_image_to_pil(referenceInfo.path)
        imB = open_image_to_pil(otherInfo.path)
        
        # Effectively call roma processing
        preds = model.match(imA, imB)

        # Convert output to required format
        warp_A_B = prepare_warp(preds["warp_AB"][0])
        warp_B_A = prepare_warp(preds["warp_BA"][0])
        confidence_A_B, confidence_B_A = (
            prepare_confidence(preds["overlap_AB"][0]),
            prepare_confidence(preds["overlap_BA"][0]),
        )

        if checkLoops:
            # Update uncertainty by analyzing the loop
            updateUncertaintyWithLoops(warp_A_B, warp_B_A, confidence_A_B, confidence_B_A, loopThreshold)

        low_confidence = confidence_A_B[..., 0] < minConfidence
        warp_A_B[low_confidence] = 0
        confidence_A_B[low_confidence] = 0

        # Saving warp image and confidence image
        logging.info("saving matches")
        pair_string = str(referenceId) + "_" + str(otherId)
        path_warp = os.path.join(outputWarpFolder, pair_string + "_warp.exr")
        path_confidence = os.path.join(outputConfidenceFolder, pair_string + "_confidence.exr")
        path_covariance = os.path.join(outputCovarianceFolder, pair_string + "_covariance.exr")
        save_image(path_warp, warp_A_B, False)
        save_image(path_confidence, confidence_A_B, True)

        if outputCovarianceFlag:
            precision_AB = preds["precision_AB"][0]
            shape = preds["precision_AB"].shape
            std_AB = prepare_confidence(torch.linalg.det(precision_AB) ** (-1 / 4))
            save_image(path_covariance, std_AB[..., np.newaxis], True)

if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)
    
    # create the top-level parserS
    parser = argparse.ArgumentParser(prog='romaMatcher')

    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--minConfidence', type=float, help='', default=0.0)
    parser.add_argument('--outputWarpFolder', type=str, help='')
    parser.add_argument('--outputConfidenceFolder', type=str, help='')
    parser.add_argument('--outputCovarianceFolder', type=str, help='')
    parser.add_argument('--checkLoops', type=str_to_bool, help='', default=False)
    parser.add_argument('--loopThreshold', type=float, help='Maximum accepted round-trip distance in pixels', default=3.0)
    parser.add_argument('--outputCovarianceFlag', type=str_to_bool, help='', default=False)
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    parser.set_defaults(func=compute_densematches)

    args = parser.parse_args()

    if hasattr(args, 'func'): 
        args.func(inputSfMData=args.inputSfMData,
                    imagePairsList=args.imagePairsList,
                    minConfidence=args.minConfidence,
                    outputWarpFolder=args.outputWarpFolder,
                    outputConfidenceFolder=args.outputConfidenceFolder,
                    outputCovarianceFolder=args.outputCovarianceFolder,
                    checkLoops=args.checkLoops,
                    loopThreshold=args.loopThreshold,
                    outputCovarianceFlag=args.outputCovarianceFlag,
                    rangeIteration=args.rangeIteration,
                    rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()