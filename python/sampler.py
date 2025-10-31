from common import *

import scipy
import os
import math
import re
from pathlib import Path
import json

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

def updateUncertainty(grid, warp, certainty, model, threshold, reference_iinfo, other_iinfo):
    """ Update certainty array using geometric filter. 
    Assumes the filter has been computed externally.

    Parameters:
        grid : the coordinates grid (W,H,2)
        warp : the warped coordinates grid (W, H, 2)
        certainty : per pixel confidence grid (W, H, 2)
        model : the 3x3 geometric matrix containing the fundamental matrix
        threshold : maximal distance allowed
        reference_iinfo : info about the first view
        other_iinfo : info about the second view
    """
    width = certainty.shape[1]
    height = certainty.shape[0]

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
    
    #Certainty to 0 for pixels which do not pass geometric check
    y = y.reshape((height, width))
    certainty[y > (threshold)] = 0

def build_uncertainties(iinfos, warpFolder, certaintyFolder, imagePairsList, filters, minCertainty):

    """ Build up and filter uncertainties 

    Parameters:
        iinfos : the image infos containing the descriptions of the images to match
        warpFolder : folder containing the warp images
        certaintyFolder : folder containing the certainty images
        imagePairsList : a list of pair of images uids which list the warp to compute
        filters : filters used to geometrically filter the samples
        minCertainty : minimal Certainty value

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
            print(f"filtered {referenceId} {otherId}")
            continue
        
        pair_string = str(referenceId) + "_" + str(otherId)
        path_warp = os.path.join(warpFolder, pair_string + "_warp.exr")
        path_certainty = os.path.join(certaintyFolder, pair_string + "_certainty.exr")

        #load images
        warp_A_B = open_image_as_numpy(path_warp)
        certainty_A_B = open_image_as_numpy(path_certainty, True)
        certainty_A_B[certainty_A_B < minCertainty] = 0.0
        warpHeight = certainty_A_B.shape[0]
        warpWidth = certainty_A_B.shape[1]
        grid = create_coordinates(warpWidth, warpHeight)

        #Filter images
        if hasFilter:
            updateUncertainty(grid, warp_A_B, certainty_A_B, model, threshold, reference_iinfo, other_iinfo)
        
        uncertaintiesByPair[item] = certainty_A_B
    
    return uncertaintiesByPair

def get_samples(certainty, minCertainty, maxMatches):
    """ Using uncertainty array, extract a list of samples

    Parameters:
        certainty a 2d array containing the warp uncertainty
        minCertainty minimal certainty allowed
        maxMatches maximal number of matches
    """

    sample_thresh = minCertainty 
    
    #Create 2d grids
    coords2d = create_coordinates(certainty.shape[1], certainty.shape[0])
    
    #reshape to vector
    certainty = certainty.squeeze()
    coords = coords2d.reshape(-1, 2)
    certainty = certainty.reshape(-1)

    #remove bad elements
    coords = coords[certainty > sample_thresh]
    certainty = certainty[certainty > sample_thresh]

    if certainty.shape[0] == 0:
        return np.array(())

    max_samples = min(maxMatches * 4, len(certainty))
    probabilities = certainty / certainty.sum()
    
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

def get_matches(coords, warp, certainty):
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
        ret[i, 2] = certainty[iy, ix, 0]
    
    return ret

def compute_samples(inputSfMData, imagePairsList, warpFolder, certaintyFolder, samplesFolder, filtersFolder, groupUncertainties, minCertainty, maxMatches, rangeIteration, rangeBlocksCount):

    """ This high level function is extracting samples form the warp images

    Parameters:
        inputSfmData : the sfmData containing the descriptions of the images to match
        imagePairsList : a list of pair of images uids which list the warp to compute
        warpFolder : folder containing the warp images
        certaintyFolder : folder containing the certainty images
        samplesFolder : output folder for the samples files
        filtersFolder : folder containing the filters used to geometrically filter the samples
        masksFolder : folder containing the masks for input images
        minCertainty: threshold for certainty validity
        maxMatches: Maximal amount of matches per pair
        rangeIteration: if of chunk to compute between [0; rangeBlocksCount[
        rangeBlocksCount: count of chunks blocks
    """

    # First of all, load the optional filters
    filters = load_filters(filtersFolder)

    # Parse sfm
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    # Retrieve list of images pairs to process
    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, 0, -1, False):
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
    
    #Compute parallelization
    blockSize = int(len(refsToProcess) / rangeBlocksCount)
    rangeStart = rangeIteration * blockSize
    rangeEnd = rangeStart + blockSize
    if rangeIteration + 1 == rangeBlocksCount:
        rangeEnd = len(refsToProcess)
    refsToProcess = refsToProcess[rangeStart:rangeEnd]

     #Loop over all reference images
    for referenceId in refsToProcess:
        
        pairs = plistByRef[referenceId]
        
        print(f"Processing reference #{referenceId}", flush=True)

        # Load uncertainties
        uncertaintiesByPair = build_uncertainties(iinfos, warpFolder, certaintyFolder, pairs, filters, minCertainty)
        if len(uncertaintiesByPair) == 0:
            continue

        #If groupUncertainties, we sum the certainties together for the same reference image
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
            certainty_A_B = uncertaintiesByPair[item]
            match_A_B = get_matches(samples_A_B, warp_A_B, certainty_A_B)

            #scale to original size
            scaledSamples = np.zeros((match_A_B.shape[0], 4))
            scaledSamples[:, 0] = match_A_B[:, 0] * reference_iinfo.width
            scaledSamples[:, 1] = match_A_B[:, 1] * reference_iinfo.height
            scaledSamples[:, 2] = match_A_B[:, 2]
            scaledSamples[:, 3] = scale

            #scale to original size
            path_output = os.path.join(samplesFolder, str(referenceId) + "_" + str(otherId))
            np.save(path_output, scaledSamples)

if __name__ == '__main__':
    import argparse

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    # create the parser for the "warp" sub-command
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--warpFolder', type=str, help='')
    parser.add_argument('--certaintyFolder', type=str, help='')
    parser.add_argument('--samplesFolder', type=str, help='')
    parser.add_argument('--filtersFolder', type=str, help='')
    parser.add_argument('--groupUncertainties', type=bool, help='')
    parser.add_argument('--maxMatches', type=int, help='')
    parser.add_argument('--minCertainty', type=float, help='')
    parser.add_argument('--rangeIteration', type=int, help='')
    parser.add_argument('--rangeBlocksCount', type=int, help='')
    parser.set_defaults(func=compute_samples)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                    imagePairsList=args.imagePairsList,
                    warpFolder=args.warpFolder,
                    certaintyFolder=args.certaintyFolder,
                    samplesFolder=args.samplesFolder,
                    filtersFolder=args.filtersFolder,
                    groupUncertainties=args.groupUncertainties,
                    minCertainty=args.minCertainty,
                    maxMatches=args.maxMatches,
                    rangeIteration=args.rangeIteration,
                    rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
