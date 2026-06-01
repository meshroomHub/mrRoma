from common import *

import scipy
import os
import math
import re
from pathlib import Path
import json
import logging

def kde(x, std = 0.1):
    """
    A reimplementation in numpy of Kde
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
    Load json files containing filters

    Parameters:
        filtersFolder is the path of the directory containing json files
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
     
    # one array for the x coordinates, one array for the y coordinates
    xs = 1.0 / width
    ys = 1.0 / height
    x = np.linspace(0.0, 1 - xs, width)
    y = np.linspace(0.0, 1 - ys, height)
    X, Y = np.meshgrid(x, y, indexing='xy')  

    # each 2d coordinates contains 2 elements, one for x, one for y
    return np.stack([X, Y], axis = 2)

def updateUncertainty(grid, warp, confidence, model, threshold, reference_iinfo, other_iinfo):
    """ Update confidence array using geometric filter. 
    Assumes the filter has been computed externally.

    Parameters:
        grid : the coordinates grid (W,H,2)
        warp : the warped coordinates grid (W, H, 2)
        confidence : per pixel confidence grid (W, H, 2)
        model : the 3x3 geometric matrix containing the fundamental matrix
        threshold : maximal distance allowed
        reference_iinfo : info about the first view
        other_iinfo : info about the second view
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

    """ Build up and filter uncertainties 

    Parameters:
        iinfos : the image infos containing the descriptions of the images to match
        warpFolder : folder containing the warp images
        confidenceFolder : folder containing the confidence images
        imagePairsList : a list of pair of images uids which list the warp to compute
        filters : filters used to geometrically filter the samples
        minConfidence : minimal Confidence value

    Return:
        dict of uncertainties indexed by pairs
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


def get_samples(confidence, minConfidence, maxMatches):
    """ Using uncertainty array, extract a list of samples

    Parameters:
        confidence a 2d array containing the warp uncertainty
        minConfidence minimal confidence allowed
        maxMatches maximal number of matches
    """

    sample_thresh = minConfidence 
    
    #Create 2d grids
    coords2d = create_coordinates(confidence.shape[1], confidence.shape[0])
    
    #reshape to vector
    confidence = confidence.squeeze()
    coords = coords2d.reshape(-1, 2)
    confidence = confidence.reshape(-1)

    #remove bad elements
    coords = coords[confidence > sample_thresh]
    confidence = confidence[confidence > sample_thresh]

    if confidence.shape[0] == 0:
        return np.array(())

    max_samples = min(maxMatches * 4, len(confidence))
    probabilities = confidence / confidence.sum()
    
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
    Compute the local warp scale using an affine fit over a sliding window.

    This is more robust than differentiating the warp over one pixel because the
    warp is low resolution and can be aliased. For each pixel, an affine map is
    fit from source coordinates to warped coordinates over a local window, and
    the scale is derived from the determinant of its linear part.

    Parameters:
        warp : (H, W, 3) coordinate map, channels 0/1 are x/y in [0, 1]
        window_size : odd side length of the sliding window in pixels
        eps : numerical stability term

    Return:
        (H, W) array containing sqrt(|det(A)|), where A is the locally fitted
        affine transform between source and warped coordinates.
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
    Using a list of coordinates, extract the associated coordinates using the warp image
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

def compute_samples(inputSfMData, imagePairsList, warpFolder, confidenceFolder, samplesFolder, filtersFolder, minConfidence, maxMatches, rangeIteration, rangeBlocksCount):

    """ This high level function is extracting samples form the warp images

    Parameters:
        inputSfmData : the sfmData containing the descriptions of the images to match
        imagePairsList : a list of pair of images uids which list the warp to compute
        warpFolder : folder containing the warp images
        confidenceFolder : folder containing the confidence images
        samplesFolder : output folder for the samples files
        filtersFolder : folder containing the filters used to geometrically filter the samples
        masksFolder : folder containing the masks for input images
        minConfidence: threshold for confidence validity
        maxMatches: Maximal amount of matches per pair
        rangeIteration: if of chunk to compute between [0; rangeBlocksCount[
        rangeBlocksCount: count of chunks blocks
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
        samples_A_B = get_samples(grouped, 0.0, maxMatches)
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
                    rangeIteration=args.rangeIteration,
                    rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
