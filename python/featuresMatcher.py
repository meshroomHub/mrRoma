from pyalicevision import matching as avmatch  
from pyalicevision import feature as avfeat

import scipy.ndimage
import logging

from common import *

import os
import math

def regionToNumpy(region):

    size = len(region.Features())
    array = np.empty(shape=(size, 2))

    vec = region.Features()
    for idx in range(0, size):

        f = vec[idx]
        array[idx, 0] = f.x()
        array[idx, 1] = f.y()

    return array

def compute_featuresMatcher(inputSfMData, imagePairsList, warpFolder, featuresFolders, matchesFolder, masksFolder, masksExtension, minCertainty, rangeIteration, rangeBlocksCount):
    
     # Parse sfm
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
        
    global_matches = avmatch.PairwiseMatches()

    for item in pairsToProcess:
        referenceId = item[0]
        otherId = item[1]
        referenceInfo = iinfos[referenceId]
        otherInfo = iinfos[otherId]   
        
        #load features
        regionsRef = avfeat.SiftRegions()
        regionsOther = avfeat.SiftRegions()
        for folder in featuresFolders:
            ref_feat = f"{folder}/{referenceId}.dspsift.feat"
            ref_desc = f"{folder}/{referenceId}.dspsift.desc"
            if os.path.exists(ref_feat) and os.path.exists(ref_desc):
                regionsRef.Load(ref_feat, ref_desc)
                break
        for folder in featuresFolders:
            other_feat = f"{folder}/{otherId}.dspsift.feat"
            other_desc = f"{folder}/{otherId}.dspsift.desc"
            if os.path.exists(other_feat) and os.path.exists(other_desc):
                regionsOther.Load(other_feat, other_desc)
                break

        #load warp
        pair_string = str(referenceId) + "_" + str(otherId)
        path_warp = os.path.join(warpFolder, pair_string + "_warp.exr")
        path_certainty = os.path.join(warpFolder, pair_string + "_certainty.exr")
        warp_A_B = open_image_as_numpy(path_warp)
        certainty_A_B = open_image_as_numpy(path_certainty, True)

        W = certainty_A_B.shape[1]
        H = certainty_A_B.shape[0]

        #Load mask
        mask = None
        if len(masksFolder) > 0 :
            # Replace the extension with the mask extension
            stem = os.path.splitext(os.path.basename(referenceInfo.path))[0]
            mask_filename = f"{stem}.{masksExtension}"

            # Build the path to the correct mask
            path_mask = os.path.join(masksFolder, mask_filename)

            if os.path.exists(path_mask):
                maskLarge = open_image(path_mask, isBW=True, isFloat=False)
                maskSmall = avimage.Image_uchar()
                avimage.resampleImage(W, H, maskLarge, maskSmall, False);
                mask = maskSmall.getNumpyArray()

        #Apply mask if exists
        if mask is not None:
            if mask.shape[0] == warp_A_B.shape[0] and mask.shape[1] == warp_A_B.shape[1]:
                certainty_A_B[mask == 0] = 0

        #retrieve a list of coordinates for reference features
        refCoords = regionToNumpy(regionsRef)
        refCoords[:, 0] *= float(W) / float(referenceInfo.width)
        refCoords[:, 1] *= float(H) / float(referenceInfo.height)
        refCoordsInt = refCoords.astype(int)

        #retrieve a list of coordinates for other features
        otherCoords = regionToNumpy(regionsOther)
        otherCoords[:, 0] *= float(W) / float(otherInfo.width)
        otherCoords[:, 1] *= float(H) / float(otherInfo.height)
        otherCoordsInt = otherCoords.astype(int)

        grid = [[[] for i in range(H)] for j in range(W)]
        for row in range(otherCoords.shape[0]):
            ix = otherCoordsInt[row, 0]
            iy = otherCoordsInt[row, 1]
            grid[iy][ix].append(row)

        filtered_certainty = scipy.ndimage.minimum_filter(certainty_A_B, size=3, mode='constant', cval=np.inf)
        minimum_warp = scipy.ndimage.minimum_filter(warp_A_B, size=3, mode='constant', cval=np.inf)
        maximum_warp = scipy.ndimage.maximum_filter(warp_A_B, size=3, mode='constant', cval=-np.inf)

        matches = avmatch.IndMatches() 

        for row in range(refCoords.shape[0]):
            ix = refCoordsInt[row, 0]
            iy = refCoordsInt[row, 1]

            if ix < 1 or iy < 1:
                continue
            
            if ix >= (W - 1)  or iy >= (H - 1):
                continue

            if filtered_certainty[iy, ix] < minCertainty:
                continue

            #get region of search
            minx = minimum_warp[iy, ix, 0] * W
            miny = minimum_warp[iy, ix, 1] * H
            maxx = maximum_warp[iy, ix, 0] * W
            maxy = maximum_warp[iy, ix, 1] * H
            iminx = int(np.floor(minx))
            iminy = int(np.floor(miny))
            imaxx = int(np.ceil(maxx))
            imaxy = int(np.ceil(maxy))
            
            #extract features in region of search
            subgrid = grid[iminy:imaxy][iminx:imaxx]
            
            firstId = -1
            firstDist = 0
            secondId = -1
            secondDist = 0

            subgrid = [x for l in subgrid for x in l]
            subgrid = [x for l in subgrid for x in l]

            for otherid in subgrid:
                dist = regionsRef.SquaredDescriptorDistance(row, regionsOther, otherid)
                if dist < firstDist or firstId < 0:
                    firstDist = dist
                    firstId = otherid
                elif dist < secondDist or secondId < 0:
                    secondDist = dist
                    secondId = otherid
            
            if firstId < 0 or secondId < 0:
                continue
            
            if secondDist * 0.8 < firstDist:
                continue

            matches.append(avmatch.IndMatch(row, firstId))

        perdesc = avmatch.MatchesPerDescType()
        perdesc[avmatch.EImageDescriberType_DSPSIFT] = matches

        pair = avmatch.Pair(referenceId, otherId)
        global_matches[pair] = perdesc
    
    #Save all features and matches
    avmatch.Save(global_matches, matchesFolder, "txt", False, f"{rangeIteration}_")

if __name__ == '__main__':
    import argparse
    
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--warpFolder', type=str, help='')
    parser.add_argument('--featuresFolders', type=str, nargs='+', help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--masksFolder', type=str, help='')
    parser.add_argument('--masksExtension', type=str, help='')
    parser.add_argument('--minCertainty', type=float, help='')
    parser.add_argument('--rangeIteration', type=int, help='')
    parser.add_argument('--rangeBlocksCount', type=int, help='')
    parser.set_defaults(func=compute_featuresMatcher)

    args = parser.parse_args()

    if hasattr(args, 'func'): 
        args.func(inputSfMData=args.inputSfMData,
                imagePairsList=args.imagePairsList,
                warpFolder=args.warpFolder,
                featuresFolders=args.featuresFolders,
                matchesFolder=args.output,
                masksFolder=args.masksFolder,
                masksExtension=args.masksExtension,
                minCertainty=args.minCertainty,
                rangeIteration=args.rangeIteration,
                rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()