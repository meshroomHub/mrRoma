from pyalicevision import matchingImageCollection as avmic   
from pyalicevision import matching as avmatch  
from pyalicevision import feature as avfeat

from common import *

import os
import re
import scipy
from pathlib import Path
import json

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
            print(path)
            filtersFile = str(path)
            with open(filtersFile, "r") as f:
                content = json.load(f)
                for filter in content:
                    filters.append(filter)
    
    return filters
 

    
def kde(x, std = 0.1):
    """
    A reimplementation in numpy of Kde
    """

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


def export_features(regionsMap, idView, coords, width, height):

    if coords.shape[0] == 0:
        return 0

    if idView in regionsMap:
        regionsRef = regionsMap[idView]
        start = regionsRef.RegionCount()
    else:
        regionsRef = avfeat.SiftRegions()
        start = 0
   
    for coord in coords:
        regionsRef.Descriptors().append(avfeat.SiftDescriptor())
        regionsRef.Features().append(avfeat.PointFeature(coord[0] * width, coord[1] * height, 1.0, 0.0))

    regionsMap[idView] = regionsRef
    return start
    
def get_matches(coords, warp):
    """
    Using a list of coordinates, extract the associated coordinates using the warp image
    """
    height = warp.shape[0]
    width = warp.shape[1]

    ret = coords.copy()
    countSamples = coords.shape[0]
    
    for i in range(0, countSamples):
        x = coords[i, 0]
        y = coords[i, 1]

        ix = int(x * float(width))
        iy = int(y * float(height))

        ret[i, 0] = warp[iy, ix, 0]
        ret[i, 1] = warp[iy, ix, 1]
    
    return ret

def store_matches(global_matches, refId, otherId, refOffset, otherOffset, count):
    
    matches = avmatch.IndMatches() 
    for i in range(0, count):
        matches.append(avmatch.IndMatch(refOffset + i, otherOffset + i))

    perdesc = avmatch.MatchesPerDescType()
    perdesc[avmatch.EImageDescriberType_SIFT] = matches

    pair = avmatch.Pair(refId, otherId)
    global_matches[pair] = perdesc
        
def saveFeatures(regionsMap, outputFolder):
    
    for (key, region) in regionsMap.items():
        
        ffeat = f"{outputFolder}/{key}.sift.feat"
        fdesc = f"{outputFolder}/{key}.sift.desc"
        
        region.Save(ffeat, fdesc)

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

def build_uncertainties(iinfos, warpFolder, masksFolder, masksExtension, imagePairsList, filters):

    """ Build up and filter uncertainties 

    Parameters:
        iinfos : the image infos containing the descriptions of the images to match
        warpFolder : folder containing the warp images
        masksFolder : folder containing the masks for input images
        imagePairsList : a list of pair of images uids which list the warp to compute
        filters : filters used to geometrically filter the samples

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
        path_certainty = os.path.join(warpFolder, pair_string + "_certainty.exr")

        #load images
        warp_A_B = open_image_as_numpy(path_warp)
        certainty_A_B = open_image_as_numpy(path_certainty, True)
        warpHeight = certainty_A_B.shape[0]
        warpWidth = certainty_A_B.shape[1]
        grid = create_coordinates(warpWidth, warpHeight)

        mask = None
        if len(masksFolder) > 0 :
            # Replace the extension with the mask extension
            stem = os.path.splitext(os.path.basename(reference_iinfo.path))[0]
            mask_filename = f"{stem}.{masksExtension}"

            # Build the path to the correct mask
            path_mask = os.path.join(masksFolder, mask_filename)

            if os.path.exists(path_mask):
                maskLarge = open_image(path_mask, isBW=True, isFloat=False)
                maskSmall = avimage.Image_uchar()
                avimage.resampleImage(warpWidth, warpHeight, maskLarge, maskSmall, False);
                mask = maskSmall.getNumpyArray()

        #Apply mask if exists
        if mask is not None:
            if mask.shape[0] == warp_A_B.shape[0] and mask.shape[1] == warp_A_B.shape[1]:
                certainty_A_B[mask == 0] = 0

        #Filter images
        if hasFilter:
            updateUncertainty(grid, warp_A_B, certainty_A_B, model, threshold, reference_iinfo, other_iinfo)

        uncertaintiesByPair[item] = certainty_A_B
    
    return uncertaintiesByPair

def compute_samples(inputSfMData, imagePairsList, warpFolder, featuresFolder, matchesFolder, filtersFolder, masksFolder, masksExtension, groupUncertainties, minCertainty, maxMatches):

    """ This high level function is extracting samples form the warp images

    Parameters:
        inputSfmData : the sfmData containing the descriptions of the images to match
        imagePairsList : a list of pair of images uids which list the warp to compute
        warpFolder : folder containing the warp images
        featuresFolder : output folder for the features files
        matchesFolder : output folder for the features files
        filtersFolder : folder containing the filters used to geometrically filter the samples
        masksFolder : folder containing the masks for input images
        groupUncertainties : group all uncertainties starting from the same view
        minCertainty: threshold for certainty validity
        maxMatches: Maximal amount of matches per pair
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
    
    regionsMap = dict()
    global_matches = avmatch.PairwiseMatches()

    #Loop over all reference images
    for referenceId, pairs in plistByRef.items():

        print(f"Ref {referenceId}")

        # Load uncertainties
        uncertaintiesByPair = build_uncertainties(iinfos, warpFolder, masksFolder, masksExtension, pairs, filters)
        if groupUncertainties:

            #If groupUncertainties, we sum the certainties together for the same reference image
            #We also sample once for all pairs with the same reference image
            grouped = None
            for item in uncertaintiesByPair:
                if grouped is None:
                    grouped = uncertaintiesByPair[item]
                else:
                    grouped += uncertaintiesByPair[item]
            
            reference_iinfo = iinfos[referenceId]
            samples_A_B = get_samples(grouped, minCertainty, maxMatches)
            offset1 = export_features(regionsMap, referenceId, samples_A_B, reference_iinfo.width, reference_iinfo.height)

        # loop over pairs of images
        for item in uncertaintiesByPair :

            print(f"processing pair {item}")
            referenceId = item[0]
            otherId = item[1]

            reference_iinfo = iinfos[referenceId]
            other_iinfo = iinfos[otherId]

            pair_string = str(referenceId) + "_" + str(otherId)
            path_warp = os.path.join(warpFolder, pair_string + "_warp.exr")

            #load images
            warp_A_B = open_image_as_numpy(path_warp)
            if not groupUncertainties:
                certainty_A_B = uncertaintiesByPair[item]

            #get samples coordinates
            if not groupUncertainties:
                samples_A_B = get_samples(certainty_A_B, minCertainty, maxMatches)
                offset1 = export_features(regionsMap, referenceId, samples_A_B, reference_iinfo.width, reference_iinfo.height)

            #Get matches based on warp images
            match_A_B = get_matches(samples_A_B, warp_A_B)
            offset2 = export_features(regionsMap, otherId, match_A_B, other_iinfo.width, other_iinfo.height)

            #export matches
            store_matches(global_matches, referenceId, otherId, offset1, offset2, len(match_A_B))

    #Save all features and matches
    avmatch.Save(global_matches, matchesFolder, "txt", False, "")
    saveFeatures(regionsMap, featuresFolder)

if __name__ == '__main__':
    import argparse
    
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    # create the parser for the "warp" sub-command
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--warpFolder', type=str, help='')
    parser.add_argument('--featuresFolder', type=str, help='')
    parser.add_argument('--matchesFolder', type=str, help='')
    parser.add_argument('--filtersFolder', type=str, help='')
    parser.add_argument('--masksFolder', type=str, help='')
    parser.add_argument('--masksExtension', type=str, help='')
    parser.add_argument('--groupUncertainties', type=bool, help='')
    parser.add_argument('--maxMatches', type=int, help='')
    parser.add_argument('--minCertainty', type=float, help='')
    parser.set_defaults(func=compute_samples)

    args = parser.parse_args()

    if hasattr(args, 'func'): 
        args.func(inputSfMData=args.inputSfMData,
                    imagePairsList=args.imagePairsList,
                    warpFolder=args.warpFolder,
                    featuresFolder=args.featuresFolder,
                    matchesFolder=args.matchesFolder,
                    filtersFolder=args.filtersFolder,
                    masksFolder=args.masksFolder,
                    masksExtension=args.masksExtension,
                    groupUncertainties=args.groupUncertainties,
                    minCertainty=args.minCertainty,
                    maxMatches=args.maxMatches)
    else:
        parser.print_help()