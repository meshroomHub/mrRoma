from romav2.device import device
from romav2 import RoMaV2
from romav2.io import tensor_to_pil
from romav2.features import Descriptor

from common import *

import os
import time
import torch
import logging
import h5py, hdf5plugin


def prepare_warp(w):
    """
    Convert a RoMa warp tensor to a NumPy array with normalized [0, 1] coordinates.

    The input tensor uses [-1, 1] coordinates (RoMa convention). This function
    rescales them to [0, 1] and appends a zero channel to produce a 3-channel
    array suitable for saving as an EXR image.

    Args:
        w (torch.Tensor): Warp tensor of shape (H, W, 2) with values in [-1, 1].

    Returns:
        numpy.ndarray: Float32 array of shape (H, W, 3) with xy coordinates in
            [0, 1] and a zero third channel.
    """
    w = ((w + 1.0) / 2.0).detach().cpu().numpy().copy()
    w = np.concatenate([w, np.zeros([w.shape[0], w.shape[1], 1], dtype=np.float32)], axis=-1)

    return w

def prepare_confidence(c):
    """
    Convert a RoMa confidence tensor to a NumPy array.

    Args:
        c (torch.Tensor): Confidence tensor of arbitrary shape.

    Returns:
        numpy.ndarray: Detached, CPU-side copy of the tensor as a NumPy array.
    """
    c = c.detach().cpu().numpy().copy()
    
    return c

def check_loop(warp_A_B, warp_B_A):
    """
    Compute the round-trip (loop-closure) error for a pair of dense warps.

    For every pixel in image A, the forward warp maps it to a location in B.
    The function then samples the backward warp at that location and measures
    how far the result deviates from the original pixel position. Bilinear
    sampling is approximated by taking the minimum distance among the four
    nearest-neighbor corners.

    Args:
        warp_A_B (numpy.ndarray | None): Dense warp from A to B, shape (H, W, 3),
            with xy coordinates normalised to [0, 1] in the first two channels.
        warp_B_A (numpy.ndarray | None): Dense warp from B to A, same layout.

    Returns:
        numpy.ndarray | None: Per-pixel round-trip error in pixels, shape
            (H, W, 1), or None if either warp is None or the shapes differ.
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
    """
    Zero out confidence values where the round-trip warp error exceeds a threshold.

    Calls :func:`check_loop` to obtain per-pixel loop errors and sets both
    ``confidence_A_B`` and ``confidence_B_A`` to 0 at all pixels whose error
    is greater than ``threshold``. Modifies the confidence arrays in-place.

    Args:
        warp_A_B (numpy.ndarray): Dense warp from A to B, shape (H, W, 3).
        warp_B_A (numpy.ndarray): Dense warp from B to A, shape (H, W, 3).
        confidence_A_B (numpy.ndarray): Confidence map for A→B, shape (H, W, 1).
            Modified in-place.
        confidence_B_A (numpy.ndarray): Confidence map for B→A, shape (H, W, 1).
            Modified in-place.
        threshold (float): Maximum acceptable round-trip error in pixels.
    """
    dist = check_loop(warp_A_B, warp_B_A)
    if dist is None:
        return

    invalid = (dist > threshold)[..., 0]  # shape (H, W)
    confidence_A_B[invalid] = 0.0
    confidence_B_A[invalid] = 0.0

def compute_densematches(inputSfMData, imagePairsList, minConfidence, outputWarpArchive, outputConfidenceArchive, outputCovarianceArchive, checkLoops, loopThreshold, outputCovarianceFlag, rangeIteration, rangeBlocksCount):
    """
    Run RoMa dense matching on a set of image pairs and save the results to disk.

    For each pair in the assigned processing range the function:
    - loads both images,
    - runs the RoMa model to obtain bidirectional dense warps and overlap maps,
    - optionally filters matches using loop-closure consistency,
    - filters matches below ``minConfidence``,
    - writes the warp, confidence, and (optionally) covariance EXR files.

    Output filenames follow the pattern ``<referenceId>_<otherId>_{warp,confidence,covariance}.exr``.

    Args:
        inputSfMData (str): Path to the input SfM data file.
        imagePairsList (str): Path to the file listing image pairs to process.
        minConfidence (float): Minimum confidence threshold; matches below this
            value are discarded (warp and confidence set to 0).
        outputWarpArchive (str): Archive path for warp.
        outputConfidenceArchive (str): Archive path for confidence.
        outputCovarianceArchive (str): Archive path for covariance.
        checkLoops (bool): If True, apply loop-closure filtering via
            :func:`updateUncertaintyWithLoops`.
        loopThreshold (float): Maximum round-trip error in pixels used for
            loop-closure filtering.
        outputCovarianceFlag (bool): If True, compute and save per-pixel
            uncertainty (std-dev derived from precision matrix determinant).
        rangeIteration (int): Index of the current processing block (for parallelization).
        rangeBlocksCount (int): Total number of processing blocks (for parallelization).
    """
    from pyalicevision import system as avsys

    # Parse sfmdata, create compatible images
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    # Load the image pairs to process (referenceView, matchingView)
    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, False):
        raise RuntimeError("Error in image pairs list loading")
    logging.info(f"Loaded {len(plist)} pairs from list")

    # Filter out the pairs with views not in current sfmData
    pairsToProcessTmp = list(plist)
    pairsToProcess = list()
    for pair in pairsToProcessTmp:
        if not pair[0] in iinfos or not pair[1] in iinfos:
            continue
        pairsToProcess.append(pair)
    logging.info(f"Kept {len(plist)} filtered pairs from list")

    #Compute parallelization parameters using the number of pairs to process
    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(pairsToProcess))
    if not valid:
        logging.error("Range is out of bounds.")
        return
    
    pairsToProcess = pairsToProcess[rangeStart:rangeEnd]

    from collections import defaultdict
    pairsByReference = defaultdict(list)
    for pair in pairsToProcess:
        pairsByReference[pair[0]].append(pair[1])

    logging.info(f"Processing elements {rangeStart} to {rangeEnd}")
    logging.info("Loading model")

    # Try to load model from disk if found.
    # Otherwise, the model will be downloaded on the net
    roma_weights = None
    dinov3_path = None
    if "ROMATCH_MODELS_PATH" in os.environ:
        modelPath = os.environ["ROMATCH_MODELS_PATH"]
        romaModelPath = os.path.join(modelPath, "romav2.0.1.pt")
        roma_weights = torch.load(romaModelPath, weights_only=True)
        dinov3_path = os.path.join(modelPath, "dinov3")

    # Create roma objects
    descCfg = Descriptor.Cfg(module_path=dinov3_path)
    romaCfg = RoMaV2.Cfg(descriptor=descCfg, weights=roma_weights, compile=False)
    model = RoMaV2(cfg=romaCfg)
    model.apply_setting("precise")
    upsampleResolution = (model.H_lr, model.W_lr) if (model.H_hr is None or model.W_hr is None) else (model.H_hr, model.W_hr) 

    # Setup filenames
    lock_path = f"{outputWarpArchive}.lock"

    # Loop over all pairs of images received as parameter.
    # Each pair of images will be processed independently
    for referenceId, listOthers in pairsByReference.items():
        
        referenceInfo = iinfos[referenceId]
        imA = open_image_to_pil(referenceInfo.path)

        for otherId in listOthers:

                # SfmData was previously parsed,
                # Retrieve the loaded information for the pair
                otherInfo = iinfos[otherId]   

                # Effectively do the matching
                # Output is (batch_size, 2, H, W)
                logging.info(f"Matching {referenceId} with {otherId}")

                # Load images from disk
                imB = open_image_to_pil(otherInfo.path)
                
                # Effectively call roma processing
                logging.info(f"Start processing")
                preds = model.match(imA, imB)
                logging.info(f"end processing")

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

                # Atomic exclusive-create lock: O_CREAT|O_EXCL is guaranteed
                # atomic on NFS/Lustre, unlike flock which is not supported.
                while True:
                    try:
                        _lfd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(_lfd)
                        break
                    except FileExistsError:
                        time.sleep(0.1)
                try:
                    with h5py.File(outputWarpArchive, "a") as f_warp, \
                         h5py.File(outputConfidenceArchive, "a") as f_conf:
                        if pair_string in f_warp:
                            del f_warp[pair_string]
                        f_warp.create_dataset(pair_string, data=warp_A_B, dtype=np.float16, **hdf5plugin.LZ4(), chunks=True)
                        if pair_string in f_conf:
                            del f_conf[pair_string]
                        f_conf.create_dataset(pair_string, data=(confidence_A_B * 255.0).astype(np.uint8), dtype=np.uint8, **hdf5plugin.LZ4(), chunks=True)
                    if outputCovarianceFlag:
                        precision_AB = preds["precision_AB"][0]
                        shape = preds["precision_AB"].shape
                        std_AB = prepare_confidence(torch.linalg.det(precision_AB) ** (-1 / 4))
                        with h5py.File(outputCovarianceArchive, "a") as f_cv:
                            if pair_string in f_cv:
                                del f_cv[pair_string]
                            f_cv.create_dataset(pair_string, data=std_AB, dtype=np.float16, **hdf5plugin.LZ4(), chunks=True)                    
                except:
                    logging.error("Error writing output files")
                    raise RuntimeError()
                finally:
                    os.unlink(lock_path)
                    

if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)
    
    # create the top-level parserS
    parser = argparse.ArgumentParser(prog='romaMatcher')

    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--minConfidence', type=float, help='', default=0.0)
    parser.add_argument('--outputWarpArchive', type=str, help='')
    parser.add_argument('--outputConfidenceArchive', type=str, help='')
    parser.add_argument('--outputCovarianceArchive', type=str, help='')
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
                    outputWarpArchive=args.outputWarpArchive,
                    outputConfidenceArchive=args.outputConfidenceArchive,
                    outputCovarianceArchive=args.outputCovarianceArchive,
                    checkLoops=args.checkLoops,
                    loopThreshold=args.loopThreshold,
                    outputCovarianceFlag=args.outputCovarianceFlag,
                    rangeIteration=args.rangeIteration,
                    rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()