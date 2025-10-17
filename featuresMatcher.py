

def regionToNumpy(region):

    size = len(region.Features())
    array = np.empty(shape=(size, 2))

    vec = region.Features()
    for idx in range(0, size):

        f = vec[idx]
        array[idx, 0] = f.x()
        array[idx, 1] = f.y()

    return array

def interp(warp_A_B, certainty_A_B, coords, minCertainty):

    warp_A_B = warp_A_B[:, :, :2]

    index = np.arange(coords.shape[0])
    index = index[:, np.newaxis]

    #keep feature index after filtering
    coords = np.hstack((coords, index))
    coords = coords[coords[:, 0] >= 0.0]
    coords = coords[coords[:, 1] >= 0.0]
    coords = coords[coords[:, 0] < W - 2]
    coords = coords[coords[:, 1] < H - 2]

    X = coords[:, 0]
    Y = coords[:, 1]

    Xm = X.astype(int)
    Xp = Xm + 1
    Ym = Y.astype(int)
    Yp = Ym + 1

    w11 = warp_A_B[Ym, Xm]
    w12 = warp_A_B[Ym, Xp]
    w21 = warp_A_B[Yp, Xm]
    w22 = warp_A_B[Yp, Xp]

    certainty_A_B = certainty_A_B.squeeze()
    c11 = certainty_A_B[Ym, Xm]
    c12 = certainty_A_B[Ym, Xp]
    c21 = certainty_A_B[Yp, Xm]
    c22 = certainty_A_B[Yp, Xp]

    cx = X - Xm
    cy = Y - Ym
    cx = cx[:, np.newaxis]
    cy = cy[:, np.newaxis]

    warp = cy * (cx * w11 + (1-cx) * w12) + (1.0 - cy) * (cx * w21 + (1-cx) * w22)
    certainty = np.minimum(c11, np.minimum(c12, np.minimum(c21, c22)))

    coords = coords[certainty < minCertainty]
    warp = warp[certainty < minCertainty]
    certainty = certainty[certainty < minCertainty]

    coords[:, :2] = warp[:, :2]

    return coords


def compute_featuresMatcher(inputSfMData, imagePairsList, warpFolder, featuresFolder, matchesFolder, masksFolder, masksExtension, minCertainty, rangeIteration, rangeBlocksCount):
    
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
    print(f"Processing elements {rangeStart} to {rangeEnd}")
        
    global_matches = avmatch.PairwiseMatches()

    for item in pairsToProcess:
        referenceId = item[0]
        otherId = item[1]
        referenceInfo = iinfos[referenceId]
        otherInfo = iinfos[otherId]   
        
        #load features
        regionsRef = avfeat.SiftRegions()
        regionsOther = avfeat.SiftRegions()
        regionsRef.LoadFeatures(f"{featuresFolder}/{referenceId}.dspsift.feat")
        regionsOther.LoadFeatures(f"{featuresFolder}/{otherId}.dspsift.feat")

        #load warp
        pair_string = str(referenceId) + "_" + str(otherId)
        path_warp = os.path.join(warpFolder, pair_string + "_warp.exr")
        path_certainty = os.path.join(warpFolder, pair_string + "_certainty.exr")
        warp_A_B = open_image_as_numpy(path_warp)
        certainty_A_B = open_image_as_numpy(path_certainty, True)
        
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

        interpolated = interp(warp_A_B, certainty_A_B, refCoords, minCertainty)
        interpolated[:, 0] *= float(referenceInfo.width)
        interpolated[:, 1] *= float(referenceInfo.height)

        otherCoords = regionToNumpy(regionsOther)
        tree = scipy.spatial.KDTree(otherCoords)
        (dd, ii) = tree.query(interpolated[:, :2], distance_upper_bound=20.0)
        
        interpolated = interpolated[ii < otherCoords.shape[0]]
        ii = ii[ii < otherCoords.shape[0]]

        print(ii)

        matches = avmatch.IndMatches() 
        for idx in range(0, ii.shape[0]):
            i = int(interpolated[idx, 2])
            j = int(ii[idx])
            
            matches.append(avmatch.IndMatch(i, j))

        perdesc = avmatch.MatchesPerDescType()
        perdesc[avmatch.EImageDescriberType_DSPSIFT] = matches

        pair = avmatch.Pair(referenceId, otherId)
        global_matches[pair] = perdesc
    
    avmatch.Save(global_matches, matchesFolder, "txt", False, f"{rangeIteration}_")

if __name__ == '__main__':
    import argparse
    
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--warpFolder', type=str, help='')
    parser.add_argument('--featuresFolder', type=str, help='')
    parser.add_argument('--matchesFolder', type=str, help='')
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
                featuresFolder=args.featuresFolder,
                matchesFolder=args.matchesFolder,
                masksFolder=args.masksFolder,
                masksExtension=args.masksExtension,
                minCertainty=args.minCertainty,
                rangeIteration=args.rangeIteration,
                rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()