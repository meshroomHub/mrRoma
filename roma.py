import json
import os

import numpy as np
from PIL import Image
import cv2

# FIXME: hardcoded
MASK_EXTENTION = "exr"
# roma working size
H, W = 864, 864
# device to run roma on
DEVICE = "cuda"

#utils functions
def save_image(path, image):
    import OpenEXR
    channels = { "RGB" : image }
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
                "type" : OpenEXR.scanlineimage }
    with OpenEXR.File(header, channels) as outfile:
        outfile.write(path)

def open_image(path):
    from pyalicevision import image as img
    # import image as img
    image = img.Image_RGBfColor()
    optRead = img.ImageReadOptions(img.EImageColorSpace_NO_CONVERSION)
    img.readImage(path, image, optRead)
    return image.getNumpyArray()

def write_feature_file(keypoints, keypoint_file):
    with open(keypoint_file, "w") as kpf:
        for kp_x, kp_y in keypoints[:,0:2]:
            kpf.write("%f %f 0 0\n"%(kp_x, kp_y))

def write_descriptor_file(descriptors, desk_filename):
    from struct import pack
    with open(desk_filename, "wb") as df:
        #nb of desc, as size_t (should be 1 byte)
        nb_desv_encoded = pack('N', int(descriptors.shape[0]))
        df.write(nb_desv_encoded)
        for descriptor in descriptors:#write descriptor as floats (4 bytes)
            for d in descriptor:
                d=pack('f', d)
                df.write(d)

# prepare warp and conf
def prepare_roma_outputs(w, c):
    w = w[:, :W, :]
    c = c[:, :W]
    w = ((w[:, :, 2:4] * W + (W - 1)) / 2).detach().cpu().numpy()
    c = c.detach().cpu().numpy()
    return w, c

def open_sfm(inputSfMData):
    from pyalicevision import sfmData, sfmDataIO
    print("Loading sfm data")
    data = sfmData.SfMData()
    ret = sfmDataIO.load(data, inputSfMData, sfmDataIO.ALL)
    if not ret:
        raise RuntimeError("Error with sfm data")
    views = data.getViews()
    nb_image = len(views)
    #sort by frameid
    views = dict(sorted(views.items(), key=lambda v: v[1].getFrameId()))
    print(views)
    # utils variables
    images_uids = [v for v in views.keys()]
    print("First image path: " + views[images_uids[0]].getImage().getImagePath())
    images_paths = [v.getImage().getImagePath() for v in views.values()]
    firstImg = views[images_uids[0]].getImage()
    oW, oH = int(firstImg.getWidth()), int(firstImg.getHeight())
    return images_uids, images_paths, (oW, oH)

def get_keyframe_indices(images_uids, keyframeSteps=0, keyFrameSfMData=""):
    # if any, will load keyframe indices from another sfm
    if keyFrameSfMData != "":
        print("Loading keyframes indices from sfm")
        with open(keyFrameSfMData, "r") as json_file:
            sfm_data_kf = json.load(json_file)
        images_uids_kf = [v["viewId"] for v in sfm_data_kf["views"]]
        keyframes_indices = np.where(np.isin(images_uids, images_uids_kf))[0]
        print(keyframes_indices)
    # will just uniform  keyframe sampling otherwise
    elif keyframeSteps!=0:
        keyframes_indices = range(0, len(images_uids), keyframeSteps)
        images_uids_kf=[images_uids[i] for i in keyframes_indices]
    else:
        raise RuntimeError("Wrong argument")
    return keyframes_indices, images_uids_kf

def get_matched_frame_indices(i, keyframes_indices, nb_images):
    """
    For keyframe i, returns the indinces of the frames to be matched
    """
    start_frame = 0 if i == 0 else keyframes_indices[i-1] #+1 if not unique to avoid matching in between kfs
    end_frame = nb_images-1 if i == len(keyframes_indices)-1 else keyframes_indices[i+1]+1 #no +1 if not unique
    matched_non_keyframes = list(range(start_frame, end_frame))
    return matched_non_keyframes      

def compute_warps(inputSfMData,
                  keyFrameSfMData, keyframeSteps,
                  samplingStep, keypointsPerKeyFrame,
                  outputFolder):
    #lazy import to avoid wait time
    import torch
    from romatch import roma_outdoor

    # load sfmdata
    images_uids,images_paths, _ = open_sfm(inputSfMData)

    # loads model, this is lazy
    print("Loading model")
    matcher = roma_outdoor(device=DEVICE)

    #  get keyframe indices from sfm or step
    keyframes_indices, _ = get_keyframe_indices(images_uids, keyframeSteps, keyFrameSfMData)

    print("Matching")
    
    # for each keyframe
    for i, k in enumerate(keyframes_indices):
        # save the  certainties for sampling
        certainties = {}
        print("  %d/%d" % (i, len(keyframes_indices)))
        # compute the non keyframes range to be matched with
        matched_non_keyframes = get_matched_frame_indices(i, keyframes_indices, len(images_uids))

        print("  Matching kf %d with :"%k)
        print(matched_non_keyframes)
        # for each frame in between the keyframes run roma
        for f in matched_non_keyframes:
            # if not keyframe
            if k == f:
                continue
            # match
            print("  Matching kf %d with %d" % (k, f))
            torch_warp, torch_certainty = matcher.match(images_paths[k], images_paths[f], device=DEVICE)
            print("  Done")
            # prepare and stack data
            warp, certainty = prepare_roma_outputs(torch_warp, torch_certainty) 
            #keep certainties for sampling
            certainties[f] = certainty
            #save 
            warp=np.concatenate([warp, np.ones([warp.shape[0], warp.shape[1],1], dtype=np.float32)], axis=-1)#make rgb
            certainty=np.concatenate([np.expand_dims(certainty, axis=-1), np.ones([certainty.shape[0], certainty.shape[1],2], dtype=np.float32)], axis=-1)#make rgb
            save_image(os.path.join(outputFolder, str(images_uids[k])+"_"+str(images_uids[f])+"_warp.exr"),warp)
            save_image(os.path.join(outputFolder, str(images_uids[k])+"_"+str(images_uids[f])+"_certainty.exr"),certainty)
        
        #run sampling on keyframe, once all match between keyframe and frames have been computed
        was_sampled = np.zeros([H,W,3], dtype=bool)
        #if 0 run roma sampling
        if samplingStep == 0:
            #take median of certainties to try and be robust to false positive
            cummulated_certainties = np.median(np.stack([c for c in certainties.values()], axis=-1), axis=-1)
            print("  Running roma sampling")
            #run roma sampling (torch_warp is just used because it contains the x,y meshgrid coordinates )
            roma_matches, _ = matcher.sample(torch_warp, torch.from_numpy(cummulated_certainties).to(device=DEVICE), num=keypointsPerKeyFrame)
            # select the meshgrid and back in image coord
            roma_matches = (
                ((roma_matches[:, 0:2] * W + (W - 1)) / 2)
                .detach()
                .cpu()
                .numpy()
            )
            roma_matches = np.clip(np.round(roma_matches).astype(np.int32),0,H-1)
            #flag the pixels as 'sampled'
            was_sampled[roma_matches[:, 1], roma_matches[:, 0],:] = True
            print("  Done")
        else:#uniform sampling
            was_sampled[::samplingStep, ::samplingStep, :]=1
        
        #save sampling
        save_image(os.path.join(outputFolder, str(images_uids[k])+"_sampling.exr"),was_sampled.astype(np.float32))
        
        print("Done matching")

def compute_tracks(inputSfMData, keyFrameSfMData, keyframeSteps,
                   wrapFolder, maskFolder, 
                   confThreshold, fakeScale, minTrackLength,
                   outputTracks):

    # load sfmdata
    images_uids, _, (oW, oH)=open_sfm(inputSfMData)

    #  get keyframe indices from sfm or step
    keyframes_indices, images_uids_kf = get_keyframe_indices(images_uids, keyframeSteps, keyFrameSfMData)

    # init track for reference keyframe pixels (one track per pixel, modulo grid sampling if any)
    tracks = []
    #coordinates of pixels in downsampled images
    pys, pxs = np.meshgrid(
        np.arange(0, H),
        np.arange(0, W),
        indexing="ij",
    )
    pys = pys.flatten()
    pxs = pxs.flatten()
    #init track file format
    for i, k in enumerate(keyframes_indices):
        for j, (x, y) in enumerate(zip(pxs, pys)):
            feat = [
                int(images_uids[k]),
                {
                    #give index in order
                    "featureId": int(j),
                    # coords in original image
                    "coords": [float(oW * x / W), float(oH * y / H)],
                    #for now scale fixed
                    "scale": fakeScale,
                },
            ]
            tracks.append(
                [
                    #track index in order
                    pxs.shape[0] * i + j,
                    #fake type for viz
                    {"descType": "sift", "featPerView": [feat]},
                ]
            )


    # keeps track of how many features we have per view
    nb_feat_per_view = [0 for _ in range(len(images_uids))]
    # for each keyframe
    for i, k in enumerate(keyframes_indices):
        print("  %d/%d" % (i, len(keyframes_indices)))
        #load sampled keypoints in keyframe
        print(len(images_uids_kf))
        print(k)
        was_sampled = open_image(os.path.join(wrapFolder, str(images_uids_kf[i])+"_sampling.exr"))[:,:,0]==1
        # open mask keyframe mask if any
        mask=None
        if maskFolder != "":
            print("  Opening mask")
            mask_file=os.path.join(maskFolder, str(images_uids_kf[i]))+"."+MASK_EXTENTION
            mask = open_image(mask_file)
            mask = cv2.resize(mask, (W, H))
            if len(mask.shape) > 2:
                # If RGB, convert to single channel (keep the first one)
                mask = mask[:,:,0]
            print("  Done")
        # save the warps and certainties
        warps = {}
        certainties = {}
        # compute the non keyframes range to be matched with
        matched_non_keyframes = get_matched_frame_indices(i, keyframes_indices, len(images_uids))

        # for each frame in between the keyframes load roma results
        print("Loading matches")
        for f in matched_non_keyframes:
            # if not keyframe
            if k == f:
                continue
            warp = open_image(os.path.join(wrapFolder, str(images_uids_kf[i])+"_"+str(images_uids[f])+"_warp.exr"))[:,:,0:2]
            certainty = open_image(os.path.join(wrapFolder, str(images_uids_kf[i])+"_"+str(images_uids[f])+"_certainty.exr"))[:,:,0]
            # coords in original image size
            warp_orig = np.stack([oW * warp[:, :, 0] / W, oH * warp[:, :, 1] / H], axis=-1)
            #save for later
            warps[f] = warp_orig
            certainties[f] = certainty
        print("Done loading matches")
        
        for f in matched_non_keyframes:
            # if not keyframe 
            if k == f:
                continue
            certainty = certainties[f]
            warp_orig = warps[f]
            print("  Selecting keypoints from kf %d with %d" % (k, f))
            certain_keypoints = certainty[pys, pxs] > confThreshold
            print("  %d rejected by confidence " % (np.count_nonzero(~certain_keypoints)))
            unmasked_keypoints = np.ones_like(certain_keypoints)
            if mask is not None:
                unmasked_keypoints = mask[pys, pxs] != 0
                print("  %d rejected by masks " % (np.count_nonzero(~unmasked_keypoints)))
            print(was_sampled.dtype, certain_keypoints.dtype, unmasked_keypoints.dtype)
            print(was_sampled.shape, certain_keypoints.shape, unmasked_keypoints.shape)
            selected_keypoints = (was_sampled[pys, pxs] & certain_keypoints & unmasked_keypoints)
            print(
                "  %d/%d rejected by confidence, sampling or masks "
                % (np.count_nonzero(~selected_keypoints), H * W)
            )

            print(
                "  %d/%d point kept"
                % (np.count_nonzero(selected_keypoints), pxs.shape[0])
            )

            print("  Formating")
            for j, (x, y) in enumerate(zip(pxs, pys)):
                if not selected_keypoints[j]:
                    continue

                proj = list([float(warp_orig[y, x][0]), float(warp_orig[y, x][1])])
                corresp_track_index = pxs.shape[0] * i + j
                #give unique feature id
                featureId = nb_feat_per_view[f]
                nb_feat_per_view[f] += 1
                feat = [
                    int(images_uids[f]),
                    {
                        "featureId": featureId,
                        "coords": proj,
                        "scale": fakeScale,
                    },
                ]
                tracks[corresp_track_index][1]["featPerView"].append(feat)

            print("  Done")
    print("Done")

    filtered_tracks = []
    print("%d tracks" % (len(tracks)))
    print("Post-filtering tracks")
    for t in tracks:
        if len(t[1]["featPerView"]) >= minTrackLength:
            filtered_tracks.append(t)

    print("Done")
    print("%d tracks" % (len(tracks)))
    print("Saving")
    with open(outputTracks, "w") as tf:
        json.dump(filtered_tracks, tf, indent=2)
    print("Done")

def export_matches(inputSfMData, keyFrameSfMData, keyframeSteps,
                   wrapFolder, maskFolder, 
                   confThreshold,
                   outputFolder):
    # load sfmdata
    images_uids, _, (oW, oH)=open_sfm(inputSfMData)

    #  get keyframe indices from sfm or step
    keyframes_indices, images_uids_kf = get_keyframe_indices(images_uids, keyframeSteps, keyFrameSfMData)

    #coordinates of pixels in downsampled images
    pys, pxs = np.meshgrid(
        np.arange(0, H),
        np.arange(0, W),
        indexing="ij",
    )
    pys = pys.flatten()
    pxs = pxs.flatten()
    
    # keeps track of  features we have per view, in order
    feat_per_view = [[] for _ in range(len(images_uids))]
    matches={uid0:{uid1:[] for uid1 in images_uids} for uid0 in images_uids}

    # for each keyframe
    for i, k in enumerate(keyframes_indices):
        print("  %d/%d" % (i, len(keyframes_indices)))
        #load sampled keypoints in keyframe
        was_sampled = open_image(os.path.join(wrapFolder, str(images_uids_kf[i])+"_sampling.exr"))[:,:,0]==1
        # open mask keyframe mask if any
        mask=None
        if maskFolder != "":
            print("  Opening mask")
            mask_file=os.path.join(maskFolder, str(images_uids_kf[i]))+"."+MASK_EXTENTION
            mask = open_image(mask_file)
            mask = cv2.resize(mask, (W, H))
            if len(mask.shape)>3:
                mask = mask[:,:,0]
            print("  Done")           
        # load the warps and certainties  
        warps = {}
        certainties = {}
        # compute the non keyframes range to be matched with
        matched_non_keyframes = get_matched_frame_indices(i, keyframes_indices, len(images_uids))
        
        # for each frame in between the keyframes load roma results
        for f in matched_non_keyframes:
            # if not keyframe
            if k == f:
                continue
            warp = open_image(os.path.join(wrapFolder, str(images_uids_kf[i])+"_"+str(images_uids[f])+"_warp.exr"))[:,:,0:2]
            certainty = open_image(os.path.join(wrapFolder, str(images_uids_kf[i])+"_"+str(images_uids[f])+"_certainty.exr"))[:,:,0]
            # coords in original image size
            warp_orig = np.stack([oW * warp[:, :, 0] / W, oH * warp[:, :, 1] / H], axis=-1)
            #save for later
            warps[f] = warp_orig
            certainties[f] = certainty
        print("Done loading matches")
        
        unmasked_keypoints = np.ones_like(was_sampled)
        if mask is not None:
            unmasked_keypoints = mask[pys, pxs] != 0
            print("  %d rejected by masks " % (np.count_nonzero(~unmasked_keypoints)))

        #saving features on keyframe, if sampled and nonmask
        selected_keypoints = was_sampled & unmasked_keypoints
        selected_keypoints=selected_keypoints.flatten()
       
        selected_keypoints_index = [] 
        for x,y in zip(pxs[selected_keypoints], pys[selected_keypoints]):
            selected_keypoints_index.append(len(feat_per_view[k]))
            feat_per_view[k].append((x,y))
            
        #for each matched frame
        for f in matched_non_keyframes:
            # if not keyframe 
            if k == f:
                continue
            certainty = certainties[f]
            warp_orig = warps[f]
            for j,(x,y)in enumerate(zip(pxs[selected_keypoints], pys[selected_keypoints])):
                if  certainty[y, x] > confThreshold :
                    #save features
                    feat_idx = len(feat_per_view[f])
                    feat_per_view[f].append((x,y))
                    #save match
                    matches[images_uids[k]][images_uids[f]].append((selected_keypoints_index[j], feat_idx))

    print("Saving files")
    desc_type="dspsift"
    dummydescriptor=True
    for  uid, features in zip(images_uids, feat_per_view):
        with open(os.path.join(outputFolder,str(uid)+"."+desc_type+".feat"), "w") as kpf:
            for kp_x, kp_y in features:
                kpf.write("%f %f 0 0\n"%(kp_y, kp_x))
        if dummydescriptor:
            write_descriptor_file(np.zeros([len(features), 128]),
                                    os.path.join(outputFolder,str(uid)+"."+desc_type+".desc"))       
    print("Done")

import argparse

if __name__ == '__main__':
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='PROG')
    #parser.add_argument('--foo', action='store_true', help='foo is great option')

    # create sub-parser
    sub_parsers = parser.add_subparsers(help='sub-command help', dest='subcommand')
    sub_parsers.required = True 

    # create the parser for the "warp" sub-command
    parser_warp = sub_parsers.add_parser('warp', help='Computes and save warp maps')
    parser_warp.add_argument('--inputSfMData', type=str, help='')
    parser_warp.add_argument('--keyFrameSfMData', type=str, help='')
    parser_warp.add_argument('--keyframeSteps', type=int, help='')
    parser_warp.add_argument('--samplingStep', type=int, help='')
    parser_warp.add_argument('--keypointsPerKeyFrame', type=int, help='')
    parser_warp.add_argument('--outputFolder', type=str, help='')
    parser_warp.set_defaults(func=compute_warps)
 
    # create the parser for the "track" 
    parser_tracks = sub_parsers.add_parser('tracks', help='Will generate tracks from computed warps')
    parser_tracks.add_argument('--inputSfMData', type=str, help='')
    parser_tracks.add_argument('--wrapFolder', type=str, help='')
    parser_tracks.add_argument('--keyFrameSfMData', default="", type=str, help='')
    parser_tracks.add_argument('--keyframeSteps', type=int, help='')
    parser_tracks.add_argument('--maskFolder', type=str, default="", help='')
    parser_tracks.add_argument('--confThreshold', type=float, default=0.0, help='')
    parser_tracks.add_argument('--minTrackLength', type=int, default=2, help='')
    parser_tracks.add_argument('--fakeScale', type=float, default=1.0, help='')
    parser_tracks.add_argument('--outputTracks', type=str, help='')
    parser_tracks.set_defaults(func=compute_tracks)

    parser_export = sub_parsers.add_parser('export', help='Will export feature, desc and feat from computed warps')
    parser_export.add_argument('--inputSfMData', type=str, help='')
    parser_export.add_argument('--wrapFolder', type=str, help='')
    parser_export.add_argument('--keyFrameSfMData', default="", type=str, help='')
    parser_export.add_argument('--keyframeSteps', type=int, help='')
    parser_export.add_argument('--maskFolder', type=str, default="", help='')
    parser_export.add_argument('--confThreshold', type=float, default=0.0, help='')
    parser_export.add_argument('--outputFolder', type=str, help='')
    parser_export.set_defaults(func=export_matches)
 
    args = parser.parse_args()

    if hasattr(args, 'func'): 
        if args.subcommand == 'warp': 
            args.func(inputSfMData=args.inputSfMData,
                      keyFrameSfMData=args.keyFrameSfMData,
                      keyframeSteps=args.keyframeSteps,
                      samplingStep=args.samplingStep, 
                      keypointsPerKeyFrame=args.keypointsPerKeyFrame,
                      outputFolder=args.outputFolder) #need to unpack manually because argparse is crap
        elif args.subcommand == 'tracks':
            args.func(inputSfMData=args.inputSfMData,
                      keyFrameSfMData=args.keyFrameSfMData,
                      keyframeSteps=args.keyframeSteps,
                      wrapFolder=args.wrapFolder,
                      maskFolder=args.maskFolder,
                      confThreshold=args.confThreshold,
                      minTrackLength=args.minTrackLength,
                      fakeScale=args.fakeScale,
                      outputTracks=args.outputTracks)
        elif args.subcommand == 'export':
            args.func(inputSfMData=args.inputSfMData,
                      keyFrameSfMData=args.keyFrameSfMData,
                      keyframeSteps=args.keyframeSteps,
                      wrapFolder=args.wrapFolder,
                      maskFolder=args.maskFolder,
                      confThreshold=args.confThreshold,
                      outputFolder=args.outputFolder)
        else:
            parser.print_help()
    else:
        parser.print_help()

