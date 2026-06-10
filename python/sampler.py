from common import *

import scipy
import os
import math
import re
from pathlib import Path
import json
import logging
import torch
import torch.nn.functional as F

def max_pool_confidence(confidence, radiusMP):
    """
    Apply non-maximum suppression to a confidence map using a square max-pool window.

    Only pixels that are the strict local maximum within their pooling window are
    retained; all others are set to 0. A small per-pixel tiebreaker is added before
    pooling so that ties are broken deterministically.

    Args:
        confidence (numpy.ndarray): 2-D or single-channel (H, W, 1) confidence map.
        radiusMP (int): Half-size of the pooling window. The full kernel is
            ``(2*radiusMP+1) x (2*radiusMP+1)``. Values <= 0 return the input unchanged.

    Returns:
        numpy.ndarray: Confidence map with non-maxima zeroed out, same shape as input.
    """
    if radiusMP <= 0:
        return confidence

    confidence2d = confidence.squeeze()
    if confidence2d.ndim != 2:
        raise ValueError("confidence must be a 2d array or a single-channel image")

    kernelSize = radiusMP * 2 + 1

    # Add a unique per-pixel tiebreaker so that within each window only one pixel
    # can match the pooled maximum, even when multiple pixels share the same value.
    tiebreaker = np.arange(confidence2d.size, dtype=np.float32).reshape(confidence2d.shape)
    tiebreaker *= 1e-10 / (confidence2d.max() + 1e-12)
    confidence2d_tb = confidence2d + tiebreaker

    confidenceTensor = torch.from_numpy(np.ascontiguousarray(confidence2d_tb)).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(confidenceTensor, kernel_size=kernelSize, stride=1, padding=radiusMP)
    pooled = pooled.squeeze(0).squeeze(0).numpy()

    confidence2d = np.where(confidence2d_tb == pooled, confidence2d, 0)
    if confidence.ndim == 3:
        confidence2d = confidence2d[:, :, np.newaxis]

    return confidence2d

def kde(x, std = 0.1):
    """
    Estimate the local density of a set of 2-D points using a Gaussian KDE.

    For each point the density is computed as the sum of Gaussian kernel values
    evaluated at its nearest neighbours (up to 200, within a radius derived from
    the standard deviation). Points outside the effective radius contribute zero.

    Args:
        x (numpy.ndarray): Array of shape (N, 2) containing the point coordinates.
        std (float): Standard deviation of the Gaussian kernel. Defaults to 0.1.

    Returns:
        numpy.ndarray: 1-D array of shape (N,) with the density estimate for each point.
    """

    #Using inverse of gaussian to compute upper bound for this stddev
    #std::log(1e-8) ~ -18.42
    limit = np.sqrt(-2.0 * std * std * -18.42)

    tree = scipy.spatial.KDTree(x)
    dd, _ = tree.query(x, 200, distance_upper_bound = limit)
    
    #because of limits, we may have less than k neighboors,
    #and the matrix may be filled with infs
    dd[dd > limit] = 0

    scores = np.exp(-(dd**2)/(2*(std**2)))
    density = scores.sum(axis=-1)

    return density

def load_filters(filtersFolder):
    """
    Load match-filter descriptors from JSON files in a folder.

    Scans ``filtersFolder`` for files matching ``matches_<N>.json``, parses each
    one, and concatenates all filter entries into a single list. If
    ``filtersFolder`` is empty the function returns an empty list immediately.

    Args:
        filtersFolder (str): Path to the folder containing filter JSON files.
            Pass an empty string to skip loading.

    Returns:
        list: All filter entries found across the JSON files. Each entry format
            matches the structure stored in the files (pair identifier + model
            parameters).
    """
    filters = []

    #If filter folder is not empty
    if len(filtersFolder) > 0:

        #May contains multiple files because of chunks
        pattern = re.compile(r"^matches_[0-9]+.json")
        files = [f for f in Path(filtersFolder).iterdir() if f.is_file() and pattern.match(f.name)]

        #Parse json
        for path in files:
            filtersFile = str(path)
            with open(filtersFile, "r") as f:
                content = json.load(f)
                for filter in content:
                    filters.append(filter)
    
    return filters

def create_coordinates(width, height):
    """
    Build a normalised (u, v) coordinate grid for an image of the given size.

    Each pixel (col, row) maps to ``(col / width, row / height)`` so that
    coordinates span ``[0, 1)`` in both dimensions.

    Args:
        width (int): Number of columns.
        height (int): Number of rows.

    Returns:
        numpy.ndarray: Float64 array of shape (H, W, 2) where the last dimension
            holds ``(u, v)`` normalised coordinates.
    """
    # one array for the x coordinates, one array for the y coordinates
    xs = 1.0 / width
    ys = 1.0 / height
    x = np.linspace(0.0, 1 - xs, width)
    y = np.linspace(0.0, 1 - ys, height)
    X, Y = np.meshgrid(x, y, indexing='xy')  

    # each 2d coordinates contains 2 elements, one for x, one for y
    return np.stack([X, Y], axis = 2)

def updateUncertainty(grid, warp, confidence, model, threshold, reference_iinfo, other_iinfo):
    """
    Zero out confidence values whose epipolar residual exceeds a threshold.

    For each pixel the function computes the Sampson-like distance between the
    reference coordinate and the epipolar line induced by ``model`` (a fundamental
    or essential matrix), then sets the confidence to 0 wherever that distance
    exceeds ``threshold``. Modifies ``confidence`` in-place.

    Args:
        grid (numpy.ndarray): Normalised coordinate grid for the reference image,
            shape (H, W, 2), as returned by :func:`create_coordinates`.
        warp (numpy.ndarray): Dense warp to the other image, shape (H, W, 3),
            with xy coordinates in [0, 1] in the first two channels.
        confidence (numpy.ndarray): Confidence map, shape (H, W, 1). Modified in-place.
        model (numpy.ndarray): 3x3 fundamental/essential matrix.
        threshold (float): Maximum acceptable epipolar distance (in pixels).
        reference_iinfo: Image info for the reference view (must expose ``.width``
            and ``.height``).
        other_iinfo: Image info for the other view (must expose ``.width`` and
            ``.height``).
    """
    width = confidence.shape[1]
    height = confidence.shape[0]

    coords = grid.copy().reshape(-1, 2)
    coords = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis = 1)

    matches = warp[:, :, :2].copy().reshape(-1, 2)
    matches = np.concatenate([matches, np.ones((matches.shape[0], 1))], axis = 1)

    coords *= np.array([reference_iinfo.width, reference_iinfo.height, 1])
    matches *= np.array([other_iinfo.width, other_iinfo.height, 1])
    
    #Compute x = F * coords
    x = model @ coords.transpose()

    #compute ||x[1:2]||
    norm = np.linalg.norm(x[:2, :], axis=0)
    y = np.sum(matches.transpose() * x, axis=0)
    y = np.abs(y) / norm
    
    #Confidence to 0 for pixels which do not pass geometric check
    y = y.reshape((height, width))
    confidence[y > (threshold)] = 0

def build_uncertainties(iinfos, warpFolder, confidenceFolder, imagePairsList, filters, minConfidence):
    """
    Load and optionally filter confidence maps for a list of image pairs.

    For each pair the function:
    - loads the pre-computed confidence EXR,
    - discards values below ``minConfidence``,
    - if a matching filter exists, applies epipolar-geometry filtering via
      :func:`updateUncertainty`,
    - skips pairs whose confidence file is missing or that have no filter entry
      when filters are provided.

    Args:
        iinfos (dict): Mapping from view ID to image-info objects.
        warpFolder (str): Directory containing warp EXR files.
        confidenceFolder (str): Directory containing confidence EXR files.
        imagePairsList (list): List of (referenceId, otherId) tuples to process.
        filters (list): Filter entries as returned by :func:`load_filters`.
            Pass an empty list to skip geometric filtering.
        minConfidence (float): Confidence values below this are set to 0.

    Returns:
        dict: Mapping from ``(referenceId, otherId)`` tuples to their filtered
            confidence arrays (shape H x W x 1).
    """
    uncertaintiesByPair = dict()

    # loop over pairs of images
    for item in imagePairsList:

        referenceId = item[0]
        otherId = item[1]

        reference_iinfo = iinfos[referenceId]
        other_iinfo = iinfos[otherId]

        # Find the associated filter
        hasFilter = False
        for filter in filters:
            if filter[0][0] == referenceId and filter[0][1] == otherId:
                values = filter[1]
                v = values["model"]
                model = np.array([[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]])
                threshold = values["threshold"]
                hasFilter = True

        if len(filters) > 0 and hasFilter is False:
            #If a filter is not found : no matches
            logging.debug(f"filtered {referenceId} {otherId}")
            continue
        
        pair_string = str(referenceId) + "_" + str(otherId)
        path_warp = os.path.join(warpFolder, pair_string + "_warp.exr")
        path_confidence = os.path.join(confidenceFolder, pair_string + "_confidence.exr")

        if not Path(path_confidence).is_file():
            continue

        #load images
        warp_A_B = open_image_as_numpy(path_warp)
        confidence_A_B = open_image_as_numpy(path_confidence, True)
        confidence_A_B[confidence_A_B < minConfidence] = 0.0

        #Filter images
        if hasFilter:
            warpHeight = confidence_A_B.shape[0]
            warpWidth = confidence_A_B.shape[1]
            grid = create_coordinates(warpWidth, warpHeight)
            updateUncertainty(grid, warp_A_B, confidence_A_B, model, threshold, reference_iinfo, other_iinfo)
        
        uncertaintiesByPair[item] = confidence_A_B
    
    return uncertaintiesByPair


def get_samples(confidence, minConfidence, maxMatches, radiusMP):
    """
    Sample a spatially balanced set of high-confidence pixel coordinates.

    The function performs two-stage sampling:
    1. Non-maximum suppression via :func:`max_pool_confidence` followed by
       confidence-weighted random sampling to obtain an initial candidate set.
    2. Density-penalised re-sampling via :func:`kde` to promote spatial spread,
       discarding isolated points with very low local density.

    Args:
        confidence (numpy.ndarray): Confidence map, shape (H, W) or (H, W, 1).
        minConfidence (float): Minimum confidence threshold; pixels below are excluded.
        maxMatches (int): Maximum number of samples to return.
        radiusMP (int): Radius passed to :func:`max_pool_confidence` for NMS.

    Returns:
        numpy.ndarray: Array of shape (N, 2) with normalised (u, v) coordinates
            of the selected samples, where N <= ``maxMatches``.
    """
    sample_thresh = minConfidence 
    pooledConfidence = max_pool_confidence(confidence, radiusMP)

    #Create 2d grids
    coords2d = create_coordinates(pooledConfidence.shape[1], pooledConfidence.shape[0])
    
    #reshape to vector
    pooledConfidence = pooledConfidence.squeeze()
    coords = coords2d.reshape(-1, 2)
    pooledConfidence = pooledConfidence.reshape(-1)

    #remove bad elements
    coords = coords[pooledConfidence > sample_thresh]
    pooledConfidence = pooledConfidence[pooledConfidence > sample_thresh]

    if confidence.shape[0] == 0:
        return np.array(())

    max_samples = min(maxMatches * 4, len(pooledConfidence))
    probabilities = pooledConfidence / pooledConfidence.sum()
    
    samples = np.random.choice(len(probabilities), size = max_samples, p = probabilities, replace=False)

    good_coords = coords[samples]

    # #penalize high density
    density = kde(good_coords, std=0.05)
    p = 1 / (density+1)

    #remove isolated points
    p[density < 10] = 1e-7

    max_samples = min(maxMatches, len(samples))

    probabilities =  p /  p.sum()
    balanced_samples = np.random.choice(len(probabilities), size = max_samples, p = probabilities, replace=False)

    final_coords = good_coords[balanced_samples]
    
    return final_coords


def compute_warp_scale_windowed(warp, window_size=17, eps=1e-12):
    """
    Estimate the local scale change induced by a dense warp using a sliding window.

    For each pixel, a local affine transform is fitted to the warp within a square
    neighbourhood via windowed covariance statistics. The returned scale is the
    square root of the absolute determinant of the 2x2 affine matrix, which
    approximates the local area-stretch factor.

    Args:
        warp (numpy.ndarray): Dense warp array of shape (H, W, >=2) with normalised
            [0, 1] coordinates in the first two channels.
        window_size (int): Side length of the averaging window. Must be >= 3;
            even values are incremented by 1. Defaults to 17.
        eps (float): Small value added to the covariance determinant to avoid
            division by zero. Defaults to 1e-12.

    Returns:
        numpy.ndarray: Float64 array of shape (H, W) with the per-pixel scale estimate.
    """
    if window_size < 3:
        raise ValueError("window_size must be >= 3")

    if window_size % 2 == 0:
        window_size += 1

    height = warp.shape[0]
    width = warp.shape[1]

    source = create_coordinates(width, height)
    u = source[:, :, 0].astype(np.float64)
    v = source[:, :, 1].astype(np.float64)
    x = warp[:, :, 0].astype(np.float64)
    y = warp[:, :, 1].astype(np.float64)

    filt = scipy.ndimage.uniform_filter
    size = window_size

    mean_u = filt(u, size=size, mode='nearest')
    mean_v = filt(v, size=size, mode='nearest')
    mean_x = filt(x, size=size, mode='nearest')
    mean_y = filt(y, size=size, mode='nearest')

    mean_uu = filt(u * u, size=size, mode='nearest')
    mean_uv = filt(u * v, size=size, mode='nearest')
    mean_vv = filt(v * v, size=size, mode='nearest')

    mean_xu = filt(x * u, size=size, mode='nearest')
    mean_xv = filt(x * v, size=size, mode='nearest')
    mean_yu = filt(y * u, size=size, mode='nearest')
    mean_yv = filt(y * v, size=size, mode='nearest')

    cov_uu = mean_uu - mean_u * mean_u
    cov_uv = mean_uv - mean_u * mean_v
    cov_vv = mean_vv - mean_v * mean_v

    cov_xu = mean_xu - mean_x * mean_u
    cov_xv = mean_xv - mean_x * mean_v
    cov_yu = mean_yu - mean_y * mean_u
    cov_yv = mean_yv - mean_y * mean_v

    det_cov = cov_uu * cov_vv - cov_uv * cov_uv
    det_cov = np.maximum(det_cov, eps)

    a11 = (cov_xu * cov_vv - cov_xv * cov_uv) / det_cov
    a12 = (cov_xv * cov_uu - cov_xu * cov_uv) / det_cov
    a21 = (cov_yu * cov_vv - cov_yv * cov_uv) / det_cov
    a22 = (cov_yv * cov_uu - cov_yu * cov_uv) / det_cov

    det_a = a11 * a22 - a12 * a21

    return np.sqrt(np.abs(det_a))


def get_matches(coords, warp, confidence):
    """
    Look up warp destinations and confidence values for a set of source coordinates.

    For each normalised (u, v) coordinate the function reads the corresponding
    pixel from ``warp`` and ``confidence`` by nearest-neighbour sampling.

    Args:
        coords (numpy.ndarray): Array of shape (N, 2) with normalised [0, 1]
            source coordinates (u along width, v along height).
        warp (numpy.ndarray): Dense warp array of shape (H, W, >=2).
        confidence (numpy.ndarray): Confidence map of shape (H, W, 1).

    Returns:
        numpy.ndarray: Array of shape (N, 3) where columns are
            ``[warp_u, warp_v, confidence]`` for each input coordinate.
    """
    height = warp.shape[0]
    width = warp.shape[1]

    ret = np.zeros((coords.shape[0], 3))
    countSamples = coords.shape[0]
    
    for i in range(0, countSamples):
        x = coords[i, 0]
        y = coords[i, 1]

        ix = int(x * float(width))
        iy = int(y * float(height))

        ret[i, 0] = warp[iy, ix, 0]
        ret[i, 1] = warp[iy, ix, 1]
        ret[i, 2] = confidence[iy, ix, 0]
    
    return ret

def compute_samples(inputSfMData, imagePairsList, warpFolder, confidenceFolder, samplesFolder, filtersFolder, minConfidence, maxMatches, radiusMP, rangeIteration, rangeBlocksCount):
    """
    Extract and save feature samples and their matches from dense warp fields.

    For each reference image in the assigned processing range the function:
    - aggregates confidence maps across all its associated pairs,
    - draws a spatially balanced sample of up to ``maxMatches`` feature locations,
    - scales the normalised coordinates back to the original image resolution,
    - looks up warp destinations in each pair and scales them similarly,
    - saves a ``.npy`` file per reference image (keypoint positions + scale) and
      per pair (match positions + confidence + scale).

    Output files are named ``<referenceId>.npy`` and
    ``<referenceId>_<otherId>.npy`` inside ``samplesFolder``.

    Args:
        inputSfMData (str): Path to the input SfM data file.
        imagePairsList (str): Path to the file listing image pairs to process.
        warpFolder (str): Directory containing warp EXR files.
        confidenceFolder (str): Directory containing confidence EXR files.
        samplesFolder (str): Directory where output ``.npy`` sample files are written.
        filtersFolder (str): Directory with optional geometric filter JSON files
            (pass an empty string to disable filtering).
        minConfidence (float): Minimum confidence threshold for sample selection.
        maxMatches (int): Maximum number of feature matches to extract per reference.
        radiusMP (int): Non-maximum suppression radius used during sampling.
        rangeIteration (int): Index of the current processing block (for parallelization).
        rangeBlocksCount (int): Total number of processing blocks (for parallelization).
    """
    from pyalicevision import system as avsys
    
    # First of all, load the optional filters
    filters = load_filters(filtersFolder)

    # Parse sfm
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    # Retrieve list of images pairs to process
    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, False):
        raise RuntimeError("Error in image pairs list loading")
    
    # build a list of image pairs indexed by their reference images
    plistByRef = dict()
    for item in plist:
        ref = item[0]
        if ref in plistByRef:
            plistByRef[ref].append(item)
        else:
            plistByRef[ref] = [item]
    refsToProcess = list(plistByRef)
    
    # Parallelization is done by splitting pairs based on their reference image
    # We want to have access to all the pairs from the same reference

    # Computeing parallelization parameters
    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(refsToProcess))
    if not valid:
        logging.error("Range is out of bounds.")
        return
        
    refsToProcess = refsToProcess[rangeStart:rangeEnd]

     #Loop over all reference images
    for referenceId in refsToProcess:

        # Retrieve all pairs for this reference image
        pairs = plistByRef[referenceId]
        
        logging.info(f"Processing reference #{referenceId}")

        # Load uncertainties and store them using pair as key
        uncertaintiesByPair = build_uncertainties(iinfos, warpFolder, confidenceFolder, pairs, filters, minConfidence)
        if len(uncertaintiesByPair) == 0:
            logging.info(f"No uncertainties for reference #{referenceId}")
            continue

        #we sum the certainties together for the same reference image
        #We also sample once for all pairs with the same reference image
        grouped = None
        for item in uncertaintiesByPair:
            if grouped is None:
                grouped = uncertaintiesByPair[item].copy()
            else:
                #mask = (grouped == 0) | (uncertaintiesByPair[item] == 0)
                grouped += uncertaintiesByPair[item]
                #grouped[mask] = 0
        
        reference_iinfo = iinfos[referenceId]
        samples_A_B = get_samples(grouped, 0.0, maxMatches, radiusMP)
        if len(samples_A_B.shape) == 1:
            logging.info(f"No valid samples for reference #{referenceId}")
            continue

        #Compute scale of the features
        wscale = math.log(float(reference_iinfo.width) / float(grouped.shape[1]), 2)
        hscale = math.log(float(reference_iinfo.height) / float(grouped.shape[0]), 2)
        scale = max(wscale, hscale)

        #scale to original size
        scaledSamples = np.zeros((samples_A_B.shape[0], 4))
        scaledSamples[:, 0] = samples_A_B[:, 0] * reference_iinfo.width 
        scaledSamples[:, 1] = samples_A_B[:, 1] * reference_iinfo.height
        scaledSamples[:, 2] = 1.0
        scaledSamples[:, 3] = scale

        path_output = os.path.join(samplesFolder, str(referenceId))
        np.save(path_output, scaledSamples)

        # loop over pairs of images
        for item in uncertaintiesByPair :

            referenceId = item[0]
            otherId = item[1]

            reference_iinfo = iinfos[referenceId]
            other_iinfo = iinfos[otherId]

            pair_string = str(referenceId) + "_" + str(otherId)
            path_warp = os.path.join(warpFolder, pair_string + "_warp.exr")

            #load images
            warp_A_B = open_image_as_numpy(path_warp)
            confidence_A_B = uncertaintiesByPair[item]
            match_A_B = get_matches(samples_A_B, warp_A_B, confidence_A_B)

            #scale to original size
            scaledSamples = np.zeros((match_A_B.shape[0], 4))
            scaledSamples[:, 0] = match_A_B[:, 0] * other_iinfo.width
            scaledSamples[:, 1] = match_A_B[:, 1] * other_iinfo.height
            scaledSamples[:, 2] = match_A_B[:, 2]
            scaledSamples[:, 3] = scale

            #scale to original size
            path_output = os.path.join(samplesFolder, str(referenceId) + "_" + str(otherId))
            np.save(path_output, scaledSamples)

if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    # create the parser for the "warp" sub-command
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--warpFolder', type=str, help='')
    parser.add_argument('--confidenceFolder', type=str, help='')
    parser.add_argument('--samplesFolder', type=str, help='')
    parser.add_argument('--filtersFolder', type=str, help='')
    parser.add_argument('--maxMatches', type=int, help='')
    parser.add_argument('--radiusMP', type=int, help='')
    parser.add_argument('--minConfidence', type=float, help='')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    parser.set_defaults(func=compute_samples)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                    imagePairsList=args.imagePairsList,
                    warpFolder=args.warpFolder,
                    confidenceFolder=args.confidenceFolder,
                    samplesFolder=args.samplesFolder,
                    filtersFolder=args.filtersFolder,
                    minConfidence=args.minConfidence,
                    maxMatches=args.maxMatches,
                    radiusMP=args.radiusMP,
                    rangeIteration=args.rangeIteration,
                    rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
