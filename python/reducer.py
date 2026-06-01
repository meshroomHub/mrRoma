from pyalicevision import matchingImageCollection as avmic   
from pyalicevision import matching as avmatch  
from pyalicevision import feature as avfeat

from common import *

import logging
import os

def export_features(regionsMap, idView, coords):

    regionsRef = regionsMap[idView]
    start = regionsRef.RegionCount()
   
    count = 0
    for coord in coords:
        if coord[2] > 1e-6:
            regionsRef.Descriptors().append(avfeat.RomaDescriptor())
            regionsRef.Features().append(avfeat.PointFeature(coord[0], coord[1], coord[3], 0.0))
            count = count + 1
           
        
    regionsMap[idView] = regionsRef

    return start

def saveFeatures(regionsMap, outputFolder):
    
    for (key, region) in regionsMap.items():
        
        ffeat = f"{outputFolder}/{key}.roma.feat"
        fdesc = f"{outputFolder}/{key}.roma.desc"
        
        region.Save(ffeat, fdesc)

def reduce_samples(inputSfMData, imagePairsList, samplesFolder, featuresFolder, matchesFolder):

    """ This high level function is extracting samples form the warp images

    Parameters:
        inputSfmData : the sfmData containing the descriptions of the images to match
        imagePairsList : a list of pair of images uids which list the warp to compute
        samplesFolder : input folder for the samples files
        featuresFolder : output folder for the features
        matchesFolder : output folder for the matches
    """

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
    
    #Start features objects
    regionsMap = dict()
    for key, item in iinfos.items():
        regionsMap[key] = avfeat.RomaRegions()

    global_matches = avmatch.PairwiseMatches()

    for referenceId, pairs in plistByRef.items():
        logging.info(f"Processing reference #{referenceId}")

        # Output features file for the reference image
        path_coords = os.path.join(samplesFolder, str(referenceId) + ".npy")
        
        # Load file with coordinates for a given reference view
        # This is an array of size [n, 3], where n is the number of coordinates used in this reference view
        # first column is the x coordinate
        # second column is the y coordinate
        # third column is the 1
        # fourth column is the scale
        try:
            coords_A_B = np.load(path_coords)
        except:
            coords_A_B = np.array(())

        # Transform this array to a set of feature points for the reference view
        # referenceId may already exists in regionMap, because the reference view may have been used as
        # a matching view. We therefore return the offset in the list of features to be able to compute
        # indices for matches
        refOffset = export_features(regionsMap, referenceId, coords_A_B)

        # For all pairs with the current reference view
        for item in pairs:

            otherId = item[1]

            # Load file with precomputed matches for a given reference view
            # This is an array of size [n, 3], where n is the number of matches
            # first column is the x coordinates in the other image
            # second column is the y coordinates in the other image
            # third column is the confidence score
            # fourth column is the scale
            path_samples = os.path.join(samplesFolder, str(referenceId) + "_" + str(otherId) + ".npy")

            try:
                match_A_B = np.load(path_samples)
            except:
                continue

            # Transform this array to a set of feature points for the other view
            # otherId may already exists in regionMap, because the view may have been used as
            # another matching view. We therefore return the offset in the list of features to be able to compute
            # indices for matches
            otherOffset = export_features(regionsMap, otherId, match_A_B)
            
            # Build a list of matches using offset and indices
            pos = 0
            matches = avmatch.IndMatches() 
            for rowId in range(0, match_A_B.shape[0]):
                if match_A_B[rowId, 2] > 1e-6:
                    matches.append(avmatch.IndMatch(refOffset + rowId, otherOffset + pos))
                    pos = pos + 1
            
            # Save matches for pair per desc.
            # Obviously, here we only have on descriptor type
            perdesc = avmatch.MatchesPerDescType()
            perdesc[avmatch.EImageDescriberType_ROMA] = matches

            # Save matches for pair per desc in a global container
            pair = avmatch.Pair(referenceId, otherId)
            global_matches[pair] = perdesc

    #Save all features and matches
    avmatch.Save(global_matches, matchesFolder, "txt", False, "")
    saveFeatures(regionsMap, featuresFolder)
            

if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    # create the parser for the "warp" sub-command
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--samplesFolder', type=str, help='')
    parser.add_argument('--featuresFolder', type=str, help='')
    parser.add_argument('--matchesFolder', type=str, help='')
    
    parser.set_defaults(func=reduce_samples)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                imagePairsList=args.imagePairsList,
                samplesFolder=args.samplesFolder,
                featuresFolder=args.featuresFolder,
                matchesFolder=args.matchesFolder)
    else:
        parser.print_help()
