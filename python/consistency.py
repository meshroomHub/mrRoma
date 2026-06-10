from common import *
import logging
from pathlib import Path
from itertools import pairwise
import os
import math
import numpy as np

def parse_folders(inputFolders):
    """
    Normalise the folder argument into a deduplicated list of path strings.

    Accepts ``None``, a single path string, or a (possibly nested) list/tuple of
    path strings. Empty strings and whitespace-only entries are silently ignored.
    Duplicate paths are removed while preserving order.

    Args:
        inputFolders: Path string, list/tuple of path strings, or ``None``.

    Returns:
        list[str]: Ordered, deduplicated list of non-empty folder paths.
    """
    if inputFolders is None:
        return []

    if isinstance(inputFolders, (list, tuple)):
        folders = []
        for item in inputFolders:
            folders.extend(parse_folders(item))
        deduped = []
        seen = set()
        for folder in folders:
            if folder not in seen:
                deduped.append(folder)
                seen.add(folder)
        return deduped

    if not isinstance(inputFolders, str):
        return []

    raw = inputFolders.strip()
    if len(raw) == 0:
        return []

    return [raw]


def find_path(folders, filename):
    """
    Search a list of folders for a file and return the first match.

    Args:
        folders (list[str]): Ordered list of directories to search.
        filename (str): Filename to look for (e.g. ``"123_456_warp.exr"``).

    Returns:
        str | None: Full path to the first existing file, or ``None`` if the
            file is not found in any of the provided folders.
    """
    for folder in folders:
        candidate = os.path.join(folder, filename)
        if os.path.exists(candidate):
            return candidate
    return None

def check_consistency(warp_A_B, warp_A_C, warp_B_C):
    """
    Compute the triplet-consistency error for three dense warps.

    Given warps A→B, A→C, and B→C, the function checks whether the composition
    B→C ∘ A→B agrees with A→C at every pixel. For each pixel in A it follows
    A→B to a location in B, samples B→C at that location (using the four
    nearest-neighbour corners), and measures the distance between the result
    and the direct A→C prediction.

    Args:
        warp_A_B (numpy.ndarray | None): Dense warp from A to B, shape (H, W, 3)
            with xy coordinates normalised to [0, 1] in the first two channels.
        warp_A_C (numpy.ndarray | None): Dense warp from A to C, same layout.
        warp_B_C (numpy.ndarray | None): Dense warp from B to C, same layout.

    Returns:
        numpy.ndarray | None: Per-pixel consistency error in pixels, shape
            (H, W, 1), or ``None`` if any warp is ``None`` or the shapes differ.
    """
    if warp_A_B is None or warp_A_C is None or warp_B_C is None:
        return None

    if warp_A_B.shape != warp_A_C.shape or warp_A_B.shape != warp_B_C.shape:
        return None

    # Vectorized computation
    height, width = warp_A_B.shape[0], warp_A_B.shape[1]
        
    # Compute normalized coordinates
    x_A_B = warp_A_B[:, :, 0] * width
    y_A_B = warp_A_B[:, :, 1] * height
    x_A_C = warp_A_C[:, :, 0] * width
    y_A_C = warp_A_C[:, :, 1] * height
    
    # Get integer and fractional parts
    ixm = np.floor(x_A_B).astype(int)
    iym = np.floor(y_A_B).astype(int)
    ixp = ixm + 1
    iyp = iym + 1
    
    # Create mask for valid coordinates (before clipping)
    valid = (ixm >= 0) & (iym >= 0) & (ixp < width) & (iyp < height)
    
    # Clip indices to valid range to prevent out-of-bounds access
    ixm = np.clip(ixm, 0, width - 1)
    iym = np.clip(iym, 0, height - 1)
    ixp = np.clip(ixp, 0, width - 1)
    iyp = np.clip(iyp, 0, height - 1)
    
    # Sample 4 neighbors
    dx = warp_B_C[iym, ixm, 0] * width - x_A_C
    dy = warp_B_C[iym, ixm, 1] * height - y_A_C
    dist_1 = np.sqrt(dx**2 + dy**2)
    
    dx = warp_B_C[iym, ixp, 0] * width - x_A_C
    dy = warp_B_C[iym, ixp, 1] * height - y_A_C
    dist_2 = np.sqrt(dx**2 + dy**2)
    
    dx = warp_B_C[iyp, ixm, 0] * width - x_A_C
    dy = warp_B_C[iyp, ixm, 1] * height - y_A_C
    dist_3 = np.sqrt(dx**2 + dy**2)
    
    dx = warp_B_C[iyp, ixp, 0] * width - x_A_C
    dy = warp_B_C[iyp, ixp, 1] * height - y_A_C
    dist_4 = np.sqrt(dx**2 + dy**2)
    
    # Take minimum
    dist = np.minimum(np.minimum(dist_1, dist_2), np.minimum(dist_3, dist_4))
    dist = np.where(valid, dist, np.inf)

    return dist[..., np.newaxis]



def compute_consistency(referenceSfMData, framesSfMData, warpFolders, confidenceFolders, outputConfidenceFolder, maxDistance, rangeIteration, rangeBlocksCount):
    """
    Filter confidence maps using triplet warp consistency across a frame sequence.

    For each reference image in the assigned processing range and for each
    consecutive frame pair (A, B) in the frame sequence, the function evaluates
    the consistency of the composition warp(ref→A) ∘ warp(A→B) against
    warp(ref→B) via :func:`check_consistency`. The per-pixel consistency errors
    are accumulated by taking the element-wise minimum across all frame pairs,
    and the confidence map for each ref→frame pair is zeroed out wherever the
    accumulated error exceeds ``maxDistance``.

    Filtered confidence maps are written to ``outputConfidenceFolder``.

    Args:
        referenceSfMData (str): Path to the SfM data file for the reference views.
        framesSfMData (str): Path to the SfM data file for the frame sequence.
        warpFolders (list[str] | str | None): One or more directories to search
            for warp EXR files. Parsed by :func:`parse_folders`.
        confidenceFolders (list[str] | str | None): One or more directories to
            search for input confidence EXR files.
        outputConfidenceFolder (str): Directory where filtered confidence EXR
            files are written.
        maxDistance (float): Maximum acceptable consistency error in pixels;
            confidence is zeroed above this threshold.
        rangeIteration (int): Index of the current processing block (for parallelization).
        rangeBlocksCount (int): Total number of processing blocks (for parallelization).
    """
    from pyalicevision import system as avsys

    # Parse sfm
    refinfos = get_imageinfos_from_sfmdata(referenceSfMData)
    framesinfos = get_imageinfos_from_sfmdata(framesSfMData)

    frame_items = list(framesinfos.items())
    ref_items = list(refinfos.items())

    # Computeing parallelization parameters
    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(ref_items))
    if not valid:
        logging.error("Range is out of bounds.")
        return
        
    ref_items = ref_items[rangeStart:rangeEnd]

    for idxRef, (keyref, _) in enumerate(ref_items):
        
        previousDistanceMap = None

        for idxFrame, (k1, _) in enumerate(frame_items):

            k2 = k1
            if idxFrame + 1 < len(frame_items):
                k2, _ = frame_items[idxFrame + 1]
            
            # ref -> A
            pair_string = str(keyref) + "_" + str(k1)
            path_warp = find_path(warpFolders, pair_string + "_warp.exr")        
            warp_ref_A = open_image_as_numpy(path_warp)

            # ref -> B
            pair_string = str(keyref) + "_" + str(k2)
            path_warp = find_path(warpFolders, pair_string + "_warp.exr")        
            warp_ref_B = open_image_as_numpy(path_warp)

            # A -> B
            pair_string = str(k1) + "_" + str(k2)
            path_warp = find_path(warpFolders, pair_string + "_warp.exr")          
            warp_A_B = open_image_as_numpy(path_warp)

            distanceMap = check_consistency(warp_ref_A, warp_ref_B, warp_A_B)

            if previousDistanceMap is not None:
                if distanceMap is None:
                    distanceMap = previousDistanceMap
                else:
                    distanceMap = np.minimum(distanceMap, previousDistanceMap)

            if distanceMap is None:
                continue

            previousDistanceMap = distanceMap

            #Read source confidence
            pair_string = str(keyref) + "_" + str(k1)
            path_confidence = find_path(confidenceFolders, pair_string + "_confidence.exr")
            confidence_keyref_A = open_image_as_numpy(path_confidence, True)
            confidence_keyref_A[distanceMap > maxDistance] = 0.0

            #write output
            path_confidence_output = os.path.join(outputConfidenceFolder, pair_string + "_confidence.exr")
            save_image(path_confidence_output, confidence_keyref_A, True)

if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    # create the parser for the "warp" sub-command
    parser.add_argument('--referenceSfMData', type=str, help='')
    parser.add_argument('--framesSfMData', type=str, help='')
    parser.add_argument('--warpFolders', nargs='*', default=None, help='')
    parser.add_argument('--confidenceFolders', nargs='*', default=None, help='')
    parser.add_argument('--outputConfidenceFolder', type=str, help='')
    parser.add_argument('--maxDistance', type=float, help='')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    parser.set_defaults(func=compute_consistency)


    args = parser.parse_args()

    print(args.rangeBlocksCount, flush=True)

    if hasattr(args, 'func'):
        args.func(referenceSfMData=args.referenceSfMData,
                framesSfMData=args.framesSfMData,
                warpFolders=args.warpFolders,
                confidenceFolders=args.confidenceFolders,
                outputConfidenceFolder=args.outputConfidenceFolder,
                maxDistance=args.maxDistance,
                rangeIteration=args.rangeIteration,
                rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
