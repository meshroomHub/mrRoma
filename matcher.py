from romatch import roma_outdoor

from common import *

import os



def prepare_warp(w):
    """ Transform the warp tensor from roma to a RGB image with B value being always 1 """
    w = ((w + 1.0) / 2.0).detach().cpu().numpy()
    w = np.concatenate([w, np.zeros([w.shape[0], w.shape[1], 1], dtype=np.float32)], axis=-1)
    return w

def prepare_confidence(c):
    """ Transform the confidence tensor from roma to a 3 dimensional array 
    (Last dimension being of size 1)
    """
    c = c.detach().cpu().numpy()
    c = np.expand_dims(c, axis=-1)
    return c

def prepare_roma_outputs(w, c, upsampleResolution):
    """ Transform output of roma to usable data
    """
    H = upsampleResolution[0]
    W = upsampleResolution[1]

    w_a_b = w[:, :W, 2:4]
    c_a_b = c[:, :W]
    w_b_a = w[:, W:, 0:2]
    c_b_a = c[:, W:]
    
    return prepare_warp(w_a_b), prepare_warp(w_b_a), prepare_confidence(c_a_b), prepare_confidence(c_b_a)

def checkUncertaintyLoops(warp_A_B, warp_B_A, certainty_A_B, certainty_B_A, upsampleResolution):
    """ Take the minimum of certainty between the original certainty, and the certainty of the warped pixel.
        Will update certainty_A_B

    Parameters:
        warp_A_B the warp image between A and B
        warp_B_A the warp image between B and A
        certainty_A_B certainty of warp_A_B
        certainty_B_A certainty of warp_B_A
        upsampleResolution tuple of roma resolution
    """
    H = upsampleResolution[0]
    W = upsampleResolution[1]

    coords = warp_A_B[:, :, :2].copy().reshape(-1, 2)
    
    coords_Xm = (coords[:, 0] * W).astype(int)
    coords_Xp = coords_Xm + 1
    coords_Ym = (coords[:, 1] * H).astype(int)
    coords_Yp = coords_Ym + 1

    coords_Xm = np.clip(coords_Xm, 0, W - 1)
    coords_Xp = np.clip(coords_Xp, 0, W - 1)
    coords_Ym = np.clip(coords_Ym, 0, H - 1)
    coords_Yp = np.clip(coords_Yp, 0, H - 1)

    c11 = certainty_B_A[coords_Ym, coords_Xm]
    c12 = certainty_B_A[coords_Ym, coords_Xp]
    c21 = certainty_B_A[coords_Yp, coords_Xm]
    c22 = certainty_B_A[coords_Yp, coords_Xp]
    
    maxuncertainty = np.maximum(np.maximum(np.maximum(c11, c12), c21), c22)
    c = maxuncertainty.reshape(H, W, 1)    
    np.minimum(certainty_A_B, c, out=certainty_A_B)

def compute_densematches(inputSfMData, imagePairsList, outputFolder, checkLoops, rangeIteration, rangeBlocksCount):
    """ This high level function is computing the warp between pairs of images

    Parameters:
        inputSfmData : the sfmData containing the descriptions of the images to match
        imagePairsList : a list of pair of images uids which list the warp to compute
        outputFolder : a destination folder for the warp images
    """
    
    upsampleResolution = (864, 864)

    #Parse sfmdata, create compatible images
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

    print("Loading model ....")
    matcher = roma_outdoor(device="cuda", upsample_res=upsampleResolution)
    
    for item in pairsToProcess:
        referenceId = item[0]
        otherId = item[1]
        referenceInfo = iinfos[referenceId]
        otherInfo = iinfos[otherId]   

        # Effectively do the matching
        # Output is (batch_size, 2, H, W)
        print(f"Matching {referenceId} with {otherId}")

        imA = open_image_to_pil(referenceInfo.path)
        imB = open_image_to_pil(otherInfo.path)
        torch_warp, torch_certainty = matcher.match(imA, imB, device="cuda")
        
        # prepare and stack data
        warp_A_B, warp_B_A, certainty_A_B, certainty_B_A = prepare_roma_outputs(torch_warp, torch_certainty, upsampleResolution) 

        if checkLoops:
            checkUncertaintyLoops(warp_A_B, warp_B_A, certainty_A_B, certainty_B_A, upsampleResolution)

        print("saving matches")
        pair_string = str(referenceId) + "_" + str(otherId)
        path_warp = os.path.join(outputFolder, pair_string + "_warp.exr")
        path_certainty = os.path.join(outputFolder, pair_string + "_certainty.exr")
        save_image(path_warp, warp_A_B)
        save_image(path_certainty, certainty_A_B, True)

if __name__ == '__main__':
    import argparse
    
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaMatcher')

    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--outputFolder', type=str, help='')
    parser.add_argument('--checkLoops', type=bool, help='')
    parser.add_argument('--rangeIteration', type=int, help='')
    parser.add_argument('--rangeBlocksCount', type=int, help='')
    parser.set_defaults(func=compute_densematches)

    args = parser.parse_args()

    if hasattr(args, 'func'): 
        args.func(inputSfMData=args.inputSfMData,
                    imagePairsList=args.imagePairsList,
                    outputFolder=args.outputFolder,
                    checkLoops=args.checkLoops,
                    rangeIteration=args.rangeIteration,
                    rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()