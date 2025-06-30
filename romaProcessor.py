import os
import numpy as np

H, W = 864, 864

def open_image(path, isBW = False):
    from pyalicevision import image as avimg

    image = []
    if isBW:
        image = avimg.Image_float()
    else:
        image = avimg.Image_RGBfColor()

    optRead = avimg.ImageReadOptions(avimg.EImageColorSpace_NO_CONVERSION)
    avimg.readImage(path, image, optRead)

    return image.getNumpyArray()

def save_image(path, array, isBW = False):
    from pyalicevision import image as avimg
    
    image = []
    if isBW:
        image = avimg.Image_float()
    else: 
        image = avimg.Image_RGBfColor()
    
    image.fromNumpyArray(array)

    optWrite = avimg.ImageWriteOptions()
    optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
    avimg.writeImage(path, image, optWrite)

def get_images_info_from_sfmdata(inputSfMData, outputFolder):

    from pyalicevision import sfmData as avsfmData
    from pyalicevision import sfmDataIO as avsfmDataIO
    from pyalicevision import image as avimg
    
    data = avsfmData.SfMData()
    ret = avsfmDataIO.load(data, inputSfMData, avsfmDataIO.VIEWS)
    if not ret:
        raise RuntimeError("Error with sfm data")
        
    views = data.getViews()
    
    nb_images = len(views)
    
    #sort by frameid
    views = dict(sorted(views.items(), key=lambda v: v[1].getFrameId()))
    
    # utils variables

    infos = dict()
    for (key, view) in views.items():
        item = dict()
        item["uid"] = key
        item["width"] = int(view.getImage().getWidth())
        item["height"] = int(view.getImage().getHeight())

        #load image
        path = view.getImage().getImagePath()
        img = open_image(path)

        #save to new format
        newpath = f"{outputFolder}/{key}.png"
        save_image(newpath, img)

        #store new path
        item["path"] = newpath
        infos[key] = item

    return infos
    
def prepare_warp(w):
    w = ((w + 1.0) / 2.0).detach().cpu().numpy()
    w = np.concatenate([w, np.zeros([w.shape[0], w.shape[1], 1], dtype=np.float32)], axis=-1)
    return w

def prepare_confidence(c):
    c = c.detach().cpu().numpy()

    c = np.expand_dims(c, axis=-1)
    return c

def prepare_roma_outputs(w, c):

    w_a_b = w[:, :W, 2:4]
    c_a_b = c[:, :W]
    w_b_a = w[:, W:, 0:2]
    c_b_a = c[:, W:]
    return prepare_warp(w_a_b), prepare_warp(w_b_a), prepare_confidence(c_a_b), prepare_confidence(c_b_a)
    
def compute_densematches(inputSfMData, imagePairsList, outputFolder):

    import torch
    from romatch import roma_outdoor
    from pyalicevision import matchingImageCollection as avmic
    from pyalicevision import image as avimg
    
    iinfos = get_images_info_from_sfmdata(inputSfMData, outputFolder)

    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist):
        raise RuntimeError("Error in image pairs list loading")

    print("Loading model ....")
    matcher = roma_outdoor(device="cuda")
    
    for item in plist:
        referenceId = item[0]
        otherId = item[1]
        referenceInfo = iinfos[referenceId]
        otherInfo = iinfos[otherId]   

        print(f"Matching {referenceId} with {otherId}")
        torch_warp, torch_certainty = matcher.match(referenceInfo["path"], otherInfo["path"], device="cuda")
        
        #output is (batch_size, 2, H, W)

        # prepare and stack data
        warp_A_B, warp_B_A, certainty_A_B, certainty_B_A = prepare_roma_outputs(torch_warp, torch_certainty) 

        print("save A->B")
        pair_string = str(referenceId) + "_" + str(otherId)
        path_warp = os.path.join(outputFolder, pair_string + "_warp.exr")
        path_certainty = os.path.join(outputFolder, pair_string + "_certainty.exr")
        save_image(path_warp, warp_A_B)
        save_image(path_certainty, certainty_A_B, True)
    
        print("save B->A")
        pair_string = str(otherId) + "_" + str(referenceId)
        path_warp = os.path.join(outputFolder, pair_string + "_warp.exr")
        path_certainty = os.path.join(outputFolder, pair_string + "_certainty.exr")
        save_image(path_warp, warp_B_A)
        save_image(path_certainty, certainty_B_A, True)

def get_samples(certainty):

    import torch
    from romatch.utils.kde import kde

    sample_thresh = 0.05
    certainty = torch.from_numpy(certainty)
    certainty = certainty.squeeze()
    certainty[certainty > sample_thresh] = 1
    if not certainty.sum(): 
        certainty = certainty + 1e-8
    
    x = torch.arange(W)
    y = torch.arange(H) 

    # Create meshgrid
    X, Y = torch.meshgrid(x, y, indexing='ij')  
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    coords = torch.stack([X, Y], dim = 1)
    
    #get certainty size
    certainty = certainty.reshape(-1)

    
    max_samples = min(40000, len(certainty))
    samples = torch.multinomial(certainty, num_samples = max_samples, replacement=False)

    good_coords = coords[samples]

    #penalize high density
    density = kde(good_coords, std=0.1)
    p = 1 / (density+1)

    #remove isolated points
    p[density < 10] = 1e-7

    max_samples = min(10000, len(samples))
    balanced_samples = torch.multinomial(p, num_samples = max_samples, replacement=False)

    final_coords = good_coords[balanced_samples]
    
    return final_coords.detach().cpu().numpy()



def compute_samples(inputSfMData, imagePairsList, warpFolder, outputFolder):

    from romatch import roma_outdoor
    from pyalicevision import matchingImageCollection as avmic
    from pyalicevision import image as avimg
    
    
    iinfos = get_images_info_from_sfmdata(inputSfMData, outputFolder)

    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist):
        raise RuntimeError("Error in image pairs list loading")

    for item in plist:
        referenceId = item[0]
        otherId = item[1]
        
        pair_string = str(referenceId) + "_" + str(otherId)
        path_warp = os.path.join(warpFolder, pair_string + "_warp.exr")
        path_certainty = os.path.join(warpFolder, pair_string + "_certainty.exr")

        #load images
        warp_A_B = open_image(path_warp)
        certainty_A_B = open_image(path_certainty, True)
        samples_A_B = get_samples(certainty_A_B)
        
        print(samples_A_B)

if __name__ == '__main__':
    import argparse
    
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='romaProcessor')

    # create sub-parser
    sub_parsers = parser.add_subparsers(help='sub-command help', dest='subcommand')
    sub_parsers.required = True 

    # create the parser for the "warp" sub-command
    parser_warp = sub_parsers.add_parser('match', help='Computes and save warp maps')
    parser_warp.add_argument('--inputSfMData', type=str, help='')
    parser_warp.add_argument('--imagePairsList', type=str, help='')
    parser_warp.add_argument('--outputFolder', type=str, help='')
    parser_warp.set_defaults(func=compute_densematches)

    # create the parser for the "warp" sub-command
    parser_sample = sub_parsers.add_parser('sample', help='Computes and save samples')
    parser_sample.add_argument('--inputSfMData', type=str, help='')
    parser_sample.add_argument('--imagePairsList', type=str, help='')
    parser_sample.add_argument('--warpFolder', type=str, help='')
    parser_sample.add_argument('--outputFolder', type=str, help='')
    parser_sample.set_defaults(func=compute_samples)

    args = parser.parse_args()

    if hasattr(args, 'func'): 
        if args.subcommand == 'match': 
            args.func(inputSfMData=args.inputSfMData,
                      imagePairsList=args.imagePairsList,
                      outputFolder=args.outputFolder)
        elif args.subcommand == 'sample': 
            args.func(inputSfMData=args.inputSfMData,
                      imagePairsList=args.imagePairsList,
                      warpFolder=args.warpFolder,
                      outputFolder=args.outputFolder)
        else:
            parser.print_help()
    else:
        parser.print_help()