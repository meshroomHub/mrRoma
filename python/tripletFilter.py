from common import *

from itertools import permutations
import logging
import h5py, hdf5plugin
from pyalicevision import matching as avmatch  
from pyalicevision import feature as avfeat

def filter_matches(inputSfMData, imagePairsList, warpArchive, confidenceArchive, featuresFolder, matchesFolder, output):

    # Todo : I need to find a way to guess it automatically.
    romaWidth = 1280
    romaHeight = 1280
    
    # Parse sfm
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    # Retrieve list of images pairs to process
    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, False):
        raise RuntimeError("Error in image pairs list loading")
    
    pairsToProcess = list()
    for pair in plist:
        if not pair[0] in iinfos or not pair[1] in iinfos:
            continue
        pairsToProcess.append(pair)
    
    # build a list of image pairs indexed by their reference images
    plistByRef = dict()
    for item in pairsToProcess:
        ref = item[0]
        plistByRef.setdefault(ref, []).append(item)
    
    refsToProcess = list(plistByRef)

    # Load existing matches
    input_matches = avmatch.PairwiseMatches()
    types = avmatch.EImageDescriberTypeVector()
    matches = avmatch.Load(input_matches, iinfos.keys(), [matchesFolder], types)
    

    # Tuple is (viewId, featureId, x, y)
    # dictTracks is a dictionnary whose keys are Tuple and values are vectors of Tuple.
    # For each referencePoints, list the associated otherPoints matched in other nodes.
    dictTracks = dict()

    for (pairViews, matchesPerDescs) in input_matches.items():

        referenceId = pairViews[0]
        otherId = pairViews[1]

        if not pairViews in pairsToProcess:
            continue
        
        perdesc = avmatch.MatchesPerDescType()

        # Compute the scale to apply to convert from image to roma scale
        refScaleY = romaHeight / iinfos[referenceId].height
        refScaleX = romaWidth / iinfos[referenceId].width
        otherScaleY = romaHeight / iinfos[otherId].height
        otherScaleX = romaWidth / iinfos[otherId].width

        for (desc, matchesPerDesc) in matchesPerDescs.items():
            
            #load features for both images
            regionsRef = avfeat.SiftRegions()
            regionsOther = avfeat.SiftRegions()
            regionsRef.Load(f"{featuresFolder}/{referenceId}.{avfeat.EImageDescriberType_enumToString(desc)}.feat", f"{featuresFolder}/{referenceId}.{avfeat.EImageDescriberType_enumToString(desc)}.desc")
            regionsOther.Load(f"{featuresFolder}/{otherId}.{avfeat.EImageDescriberType_enumToString(desc)}.feat", f"{featuresFolder}/{otherId}.{avfeat.EImageDescriberType_enumToString(desc)}.desc")
            featuresRef = regionsRef.Features()
            featuresOther = regionsOther.Features()

            for match in matchesPerDesc:
                
                # Retrieve coordinates
                pRef = featuresRef[match._i]
                pOther = featuresOther[match._j]

                # Scale points to ROMA scale
                ox = pOther.x() * otherScaleX
                oy = pOther.y() * otherScaleY
                rx = pRef.x() * refScaleX
                ry = pRef.y() * refScaleY

                # Store the otherTuple to the list of Tuples attached to the reference point
                refTuple = (referenceId, match._i, rx, ry)
                otherTuple = (otherId, match._j, ox, oy)
                dictTracks.setdefault(refTuple, []).append(otherTuple)
    
    # Create a list of pairs of points to check per pair of images
    # Indexing per pair of images enable to factorize loading of warp images
    pairsPerReference = dict()
    for referenceTuple, vecOtherTuples in dictTracks.items():
        for t1, t2 in permutations(vecOtherTuples, 2):
            viewPair = (t1[0], t2[0])
            featureIds = (t1, t2)
            pairsPerReference.setdefault(viewPair, []).append(featureIds)

    # Triplet distance threshold (Roma pixels)
    tau = 4.0

    # Directed leaf-leaf pairs that passed the triangle constraint.
    # Each entry is ((viewId1, featIdx1), (viewId2, featIdx2)).
    consistent_pairs = set()

    # Read content from warp
    with h5py.File(warpArchive, "r") as f_warp_h5, \
         h5py.File(confidenceArchive, "r") as f_conf_h5:

        for pair, features in pairsPerReference.items():

            pair_string = str(pair[0]) + "_" + str(pair[1])
            if pair_string not in f_warp_h5 or pair_string not in f_conf_h5:
                continue

            warp = f_warp_h5[pair_string][()].astype(np.float32)
            conf = f_conf_h5[pair_string][()].astype(np.float32) / 255.0

            H, W = warp.shape[0], warp.shape[1]

            # Source (t1) and target (t2) coordinates already in Roma space
            coords1 = np.array([[f[0][2], f[0][3]] for f in features], dtype=np.float32)  # (N, 2) x,y
            coords2 = np.array([[f[1][2], f[1][3]] for f in features], dtype=np.float32)  # (N, 2) x,y

            ixm = np.floor(coords1[:, 0]).astype(int)
            iym = np.floor(coords1[:, 1]).astype(int)
            ixp = ixm + 1
            iyp = iym + 1

            in_bounds = (ixm >= 0) & (iym >= 0) & (ixp < W) & (iyp < H)

            ixm_c = np.clip(ixm, 0, W - 1)
            iym_c = np.clip(iym, 0, H - 1)
            ixp_c = np.clip(ixp, 0, W - 1)
            iyp_c = np.clip(iyp, 0, H - 1)

            # Stack all 4 corners: shape (4, N)
            corner_iy = np.stack([iym_c, iym_c, iyp_c, iyp_c])
            corner_ix = np.stack([ixm_c, ixp_c, ixm_c, ixp_c])

            px = warp[corner_iy, corner_ix, 0] * W   # (4, N)
            py = warp[corner_iy, corner_ix, 1] * H   # (4, N)
            dx = px - coords2[np.newaxis, :, 0]       # (4, N)
            dy = py - coords2[np.newaxis, :, 1]       # (4, N)
            dists = np.sqrt(dx * dx + dy * dy)        # (4, N)

            best_idx = np.argmin(dists, axis=0)       # (N,)
            n_idx = np.arange(len(features))
            best_dist = dists[best_idx, n_idx]
            consistent = in_bounds & (best_dist < tau)

            for i, (t1, t2) in enumerate(features):
                if consistent[i]:
                    consistent_pairs.add(((t1[0], t1[1]), (t2[0], t2[1])))

    # Per-star pruning: for each star (reference point + its matched leaves),
    # iteratively remove the leaf with the lowest triangle-support ratio until
    # all remaining leaves satisfy the threshold, then discard stars that are
    # too small to be trustworthy.
    #
    # minRatioSupport – fraction of other leaves in the star that must form a
    #                   consistent triangle (either warp direction) with a leaf.
    # minStarSize     – minimum number of surviving leaves to keep any match.
    minRatioSupport = 0.4
    minStarSize = 2

    # Set of (ref_key, leaf_key) edges that survived per-star pruning.
    #   ref_key  = (refViewId,   refFeatIdx)
    #   leaf_key = (otherViewId, otherFeatIdx)
    surviving_matches = set()

    for referenceTuple, vecOtherTuples in dictTracks.items():
        ref_key = (referenceTuple[0], referenceTuple[1])
        leaves = list(vecOtherTuples)

        while True:
            n = len(leaves)
            if n < 2:
                # Cannot form any triangle; discard all leaves in this star.
                leaves = []
                break

            # Count how many other leaves form a consistent triangle with each
            # leaf, accepting either warp direction as evidence.
            support = []
            for i, t_i in enumerate(leaves):
                key_i = (t_i[0], t_i[1])
                s = sum(
                    1 for j, t_j in enumerate(leaves) if i != j and (
                        (key_i, (t_j[0], t_j[1])) in consistent_pairs or
                        ((t_j[0], t_j[1]), key_i) in consistent_pairs
                    )
                )
                support.append(s)

            possible = n - 1
            min_support = min(support)

            if min_support / possible >= minRatioSupport:
                break  # All leaves meet the threshold; star is clean.

            # Remove the single weakest leaf and iterate.
            leaves.pop(support.index(min_support))

        if len(leaves) >= minStarSize:
            for t in leaves:
                surviving_matches.add((ref_key, (t[0], t[1])))

    # Keep only pairwise matches where the specific ref->leaf edge survived pruning.
    output_matches = avmatch.PairwiseMatches()

    for (pairViews, matchesPerDescs) in input_matches.items():
        if pairViews not in pairsToProcess:
            continue

        referenceId = pairViews[0]
        otherId = pairViews[1]
        perdesc = avmatch.MatchesPerDescType()

        for (desc, matchesPerDesc) in matchesPerDescs.items():
            filtered = avmatch.IndMatches()
            for match in matchesPerDesc:
                ref_key  = (referenceId, match._i)
                leaf_key = (otherId,     match._j)
                if (ref_key, leaf_key) in surviving_matches:
                    filtered.append(match)
            if len(filtered) > 0:
                perdesc[desc] = filtered

        if len(perdesc) > 0:
            output_matches[pairViews] = perdesc

    avmatch.Save(output_matches, output, "txt", False, "")


if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    # create the parser for the "warp" sub-command
    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--warpArchive', type=str, help='')
    parser.add_argument('--confidenceArchive', type=str, help='')
    parser.add_argument('--featuresFolder', type=str, help='')
    parser.add_argument('--matchesFolder', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.set_defaults(func=filter_matches)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                    imagePairsList=args.imagePairsList,
                    warpArchive=args.warpArchive,
                    confidenceArchive=args.confidenceArchive,
                    featuresFolder=args.featuresFolder,
                    matchesFolder=args.matchesFolder,
                    output=args.output)
    else:
        parser.print_help()
