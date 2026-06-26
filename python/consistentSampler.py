from common import *

from pathlib import Path
import logging
import h5py, hdf5plugin
import scipy
import os
import math

def kde(x, std = 0.1):
    """
    Estimate the local density of a set of 2-D points using a Gaussian KDE.

    For each point the density is computed as the sum of Gaussian kernel values
    evaluated at its nearest neighbours (up to 200, within a radius derived from
    the standard deviation). Points outside the effective radius contribute zero.

    Args:
        x (numpy.ndarray): Array of shape (N, 2) containing the point coordinates.
        std (float): Standard deviation of the Gaussian kernel. Defaults to 0.1.

    Returns:
        numpy.ndarray: 1-D array of shape (N,) with the density estimate for each point.
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
    Load match-filter descriptors from JSON files in a folder.

    Scans ``filtersFolder`` for files matching ``matches_<N>.json``, parses each
    one, and concatenates all filter entries into a single list. If
    ``filtersFolder`` is empty the function returns an empty list immediately.

    Args:
        filtersFolder (str): Path to the folder containing filter JSON files.
            Pass an empty string to skip loading.

    Returns:
        list: All filter entries found across the JSON files. Each entry format
            matches the structure stored in the files (pair identifier + model
            parameters).
    """
    import json
    import re
    
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
    """
    Build a normalised (u, v) coordinate grid for an image of the given size.

    Each pixel (col, row) maps to ``(col / width, row / height)`` so that
    coordinates span ``[0, 1)`` in both dimensions.

    Args:
        width (int): Number of columns.
        height (int): Number of rows.

    Returns:
        numpy.ndarray: Float64 array of shape (H, W, 2) where the last dimension
            holds ``(u, v)`` normalised coordinates.
    """
    # one array for the x coordinates, one array for the y coordinates
    xs = 1.0 / width
    ys = 1.0 / height
    x = np.linspace(0.0, 1 - xs, width)
    y = np.linspace(0.0, 1 - ys, height)
    X, Y = np.meshgrid(x, y, indexing='xy')  

    # each 2d coordinates contains 2 elements, one for x, one for y
    return np.stack([X, Y], axis = 2)

def updateUncertainty(grid, warp, confidence, model, threshold, reference_iinfo, other_iinfo):
    """
    Zero out confidence values whose epipolar residual exceeds a threshold.

    For each pixel the function computes the Sampson-like distance between the
    reference coordinate and the epipolar line induced by ``model`` (a fundamental
    or essential matrix), then sets the confidence to 0 wherever that distance
    exceeds ``threshold``. Modifies ``confidence`` in-place.

    Args:
        grid (numpy.ndarray): Normalised coordinate grid for the reference image,
            shape (H, W, 2), as returned by :func:`create_coordinates`.
        warp (numpy.ndarray): Dense warp to the other image, shape (H, W, 3),
            with xy coordinates in [0, 1] in the first two channels.
        confidence (numpy.ndarray): Confidence map, shape (H, W, 1). Modified in-place.
        model (numpy.ndarray): 3x3 fundamental/essential matrix.
        threshold (float): Maximum acceptable epipolar distance (in pixels).
        reference_iinfo: Image info for the reference view (must expose ``.width``
            and ``.height``).
        other_iinfo: Image info for the other view (must expose ``.width`` and
            ``.height``).
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

def build_uncertainties(iinfos, warpArchive, confidenceArchive, imagePairsList, filtersByPair, minConfidence):
    """
    Load and optionally filter confidence maps for a list of image pairs.

    For each pair the function:
    - loads the pre-computed confidence EXR,
    - discards values below ``minConfidence``,
    - if a matching filter exists, applies epipolar-geometry filtering via
      :func:`updateUncertainty`,
    - skips pairs whose confidence file is missing or that have no filter entry
      when filters are provided.

    Args:
        iinfos (dict): Mapping from view ID to image-info objects.
        warpArchive: Open HDF5 archive containing warp arrays.
        confidenceArchive: Open HDF5 archive containing confidence arrays.
        imagePairsList (list): List of (referenceId, otherId) tuples to process.
        filtersByPair (dict): Optional geometric filters indexed by image pair.
            Pass an empty dict to skip geometric filtering.
        minConfidence (float): Confidence values below this are set to 0.

    Returns:
        dict: Mapping from ``(referenceId, otherId)`` tuples to their filtered
            confidence arrays (shape H x W x 1).
    """
    uncertaintiesByPair = dict()

    # loop over pairs of images
    for item in imagePairsList:

        referenceId = item[0]
        otherId = item[1]

        reference_iinfo = iinfos[referenceId]
        other_iinfo = iinfos[otherId]

        filterValues = filtersByPair.get((referenceId, otherId))
        hasFilter = filterValues is not None

        if len(filtersByPair) > 0 and hasFilter is False:
            #If a filter is not found : no matches
            logging.debug(f"filtered {referenceId} {otherId}")
            continue
        
        pair_string = str(referenceId) + "_" + str(otherId)

        if pair_string not in confidenceArchive or pair_string not in warpArchive:
            continue
        confidence_A_B = confidenceArchive[pair_string][()].astype(np.float32) / 255.0
        confidence_A_B[confidence_A_B < minConfidence] = 0.0

        #Filter images
        if hasFilter:
            warp_A_B = warpArchive[pair_string][()].astype(np.float32)
            v = filterValues["model"]
            model = np.array([[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]])
            threshold = filterValues["threshold"]
            warpHeight = confidence_A_B.shape[0]
            warpWidth = confidence_A_B.shape[1]
            grid = create_coordinates(warpWidth, warpHeight)
            updateUncertainty(grid, warp_A_B, confidence_A_B, model, threshold, reference_iinfo, other_iinfo)
        
        uncertaintiesByPair[item] = confidence_A_B
    
    return uncertaintiesByPair

def get_matches(coords, warp, confidence):
    """
    Look up warp destinations and confidence values for a set of source coordinates.

    For each normalised (u, v) coordinate the function reads the corresponding
    pixel from ``warp`` and ``confidence`` by nearest-neighbour sampling.

    Args:
        coords (numpy.ndarray): Array of shape (N, 2) with normalised [0, 1]
            source coordinates (u along width, v along height).
        warp (numpy.ndarray): Dense warp array of shape (H, W, >=2).
        confidence (numpy.ndarray): Confidence map of shape (H, W, 1).

    Returns:
        numpy.ndarray: Array of shape (N, 3) where columns are
            ``[warp_u, warp_v, confidence]`` for each input coordinate.
    """
    height = warp.shape[0]
    width = warp.shape[1]

    ret = np.empty((coords.shape[0], 3), dtype=np.float32)

    ix = np.clip((coords[:, 0] * float(width)).astype(np.int64), 0, width - 1)
    iy = np.clip((coords[:, 1] * float(height)).astype(np.int64), 0, height - 1)

    ret[:, 0] = warp[iy, ix, 0]
    ret[:, 1] = warp[iy, ix, 1]
    ret[:, 2] = confidence[iy, ix, 0]
    
    return ret

def get_samples(confidence, minConfidence, maxMatches, radiusMP):
    """
    Sample high-confidence pixel coordinates with weighted hard-radius exclusion.

    Each selected pixel is drawn with probability proportional to its remaining
    confidence value. After a pixel is selected, all pixels within ``radiusMP``
    pixels are removed from the sampling distribution, so returned coordinates
    are not closer than ``radiusMP`` pixels from each other.

    Args:
        confidence (numpy.ndarray): Confidence map, shape (H, W) or (H, W, 1).
        minConfidence (float): Minimum confidence threshold; pixels below are excluded.
        maxMatches (int): Maximum number of samples to return.
        radiusMP (int): Minimum exclusion radius in pixels.

    Returns:
        numpy.ndarray: Array of shape (N, 2) with normalised (u, v) coordinates
            of the selected samples, where N <= ``maxMatches``.
    """
    if maxMatches <= 0:
        return np.empty((0, 2), dtype=np.float32)
    
    np.random.seed(0)

    confidence2d = confidence.squeeze()
    if confidence2d.ndim != 2:
        raise ValueError("confidence must be a 2d array or a single-channel image")

    height = confidence2d.shape[0]
    width = confidence2d.shape[1]
    weights = confidence2d.astype(np.float64, copy=True)
    weights[weights < minConfidence] = 0.0
    weights[~np.isfinite(weights)] = 0.0

    flat_weights = weights.reshape(-1)
    if flat_weights.sum() <= 0.0:
        return np.empty((0, 2), dtype=np.float32)

    tree = flat_weights.copy()
    for i in range(1, tree.size + 1):
        parent = i + (i & -i)
        if parent <= tree.size:
            tree[parent - 1] += tree[i - 1]

    def add_weight(index, delta):
        index += 1
        while index <= tree.size:
            tree[index - 1] += delta
            index += index & -index

    def total_weight():
        total = 0.0
        index = tree.size
        while index > 0:
            total += tree[index - 1]
            index -= index & -index
        return total

    def sample_weighted_index(target):
        index = 0
        bit = 1 << (tree.size.bit_length() - 1)
        while bit:
            next_index = index + bit
            if next_index <= tree.size and tree[next_index - 1] < target:
                target -= tree[next_index - 1]
                index = next_index
            bit >>= 1
        return index

    disk_offsets = []
    if radiusMP > 0:
        radius2 = radiusMP * radiusMP
        for dy in range(-radiusMP, radiusMP + 1):
            for dx in range(-radiusMP, radiusMP + 1):
                if dy * dy + dx * dx <= radius2:
                    disk_offsets.append((dy, dx))
    else:
        disk_offsets.append((0, 0))

    selected = []
    for _ in range(maxMatches):
        total = total_weight()
        if total <= 0.0:
            break
        
        target = np.random.random() * total
        if target <= 0.0:
            target = np.nextafter(0.0, total)

        flat_index = sample_weighted_index(target)
        y, x = divmod(flat_index, width)
        selected.append((x / float(width), y / float(height)))

        for dy, dx in disk_offsets:
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= height or xx < 0 or xx >= width:
                continue

            index = yy * width + xx
            old_weight = flat_weights[index]
            if old_weight <= 0.0:
                continue

            flat_weights[index] = 0.0
            add_weight(index, -old_weight)

    if len(selected) == 0:
        return np.empty((0, 2), dtype=np.float32)

    return np.array(selected, dtype=np.float32)

def compute_samples(inputSfMData, imagePairsList, warpArchive, confidenceArchive, samplesFolder, filtersFolder, minConfidence, maxMatches, radiusMP):
    
    
    # First of all, load the optional filters
    filtersByPair = dict()
    for filterEntry in load_filters(filtersFolder):
        filtersByPair[(filterEntry[0][0], filterEntry[0][1])] = filterEntry[1]

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
        if ref in plistByRef:
            plistByRef[ref].append(item)
        else:
            plistByRef[ref] = [item]
    refsToProcess = list(plistByRef)


    groupedPerReference = {}
    samplesPerReference = {}

    with h5py.File(warpArchive, "r") as f_warp_h5, \
         h5py.File(confidenceArchive, "r") as f_conf_h5:

        logging.info("Building grouped uncertainties.")
        #Loop over all reference images
        for referenceId in refsToProcess:

            # Retrieve all pairs for this reference image
            pairs = plistByRef[referenceId]
            
            logging.info(f"Processing reference #{referenceId}")

            # Load uncertainties and store them using pair as key
            uncertaintiesByPair = build_uncertainties(iinfos, f_warp_h5, f_conf_h5, pairs, filtersByPair, minConfidence)
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
                    grouped += uncertaintiesByPair[item]

            groupedPerReference[referenceId] = grouped

        

        for iter in range(0, 10):
            logging.info(f"Iteration #{iter}")
            logging.info("Reproject samples.")

            reuseCandidatesPerImage = {}
            
            #Loop over all reference images
            for referenceId in refsToProcess:
                logging.info(f"Processing reference #{referenceId}")
                # Retrieve all pairs for this reference image
                pairs = plistByRef[referenceId]

                if referenceId not in samplesPerReference:
                    continue

                for (_, otherId) in pairs:

                    pair_string = str(referenceId) + "_" + str(otherId)

                    if pair_string not in f_warp_h5:
                        continue

                    warp_A_B = f_warp_h5[pair_string][()].astype(np.float32)
                    coords = samplesPerReference[referenceId]  # (N, 2) normalised [0,1]

                    warpH = warp_A_B.shape[0]
                    warpW = warp_A_B.shape[1]
                    ix = np.clip((coords[:, 0] * float(warpW)).astype(np.int64), 0, warpW - 1)
                    iy = np.clip((coords[:, 1] * float(warpH)).astype(np.int64), 0, warpH - 1)

                    projected = np.stack([warp_A_B[iy, ix, 0], warp_A_B[iy, ix, 1]], axis=1)

                    # Discard points whose warp value is zero (unmatched / out-of-bounds)
                    valid = (projected[:, 0] > 0) | (projected[:, 1] > 0)
                    projected = projected[valid]

                    if len(projected) == 0:
                        continue

                    if otherId not in reuseCandidatesPerImage:
                        reuseCandidatesPerImage[otherId] = projected
                    else:
                        reuseCandidatesPerImage[otherId] = np.concatenate([reuseCandidatesPerImage[otherId], projected], axis=0)

            logging.info("Build samples.")
            #Loop over all reference images
            for referenceId in refsToProcess:
                logging.info(f"Processing reference #{referenceId}")
                if referenceId not in groupedPerReference:
                    continue

                grouped = groupedPerReference[referenceId]
                warpH = grouped.shape[0]
                warpW = grouped.shape[1]
                conf2d = (grouped[:, :, 0] if grouped.ndim == 3 else grouped).copy()

                selected = []

                # --- seed with reuse candidates first ---
                if referenceId in reuseCandidatesPerImage:

                    # Shuffle so that when multiple candidates from different source images
                    # land in the same occupancy cell, the winner is chosen uniformly at
                    # random rather than being determined by the arbitrary accumulation order.
                    reuseCandidates = reuseCandidatesPerImage[referenceId].copy()
                    #np.random.shuffle(reuseCandidates)

                    selected = []
                    for coord in reuseCandidates:
                        if len(selected) > maxMatches:
                            break

                        ix = int(coord[0] * warpW)
                        iy = int(coord[1] * warpH)

                        if ix < 0 or ix >= warpW or iy < 0 or iy >= warpH :
                            continue
                        
                        if conf2d[iy, ix] < minConfidence:
                            continue

                        selected.append((coord[0], coord[1]))

                        y_min = max(0, iy - radiusMP)
                        y_max = min(warpH, iy + radiusMP + 1)
                        x_min = max(0, ix - radiusMP)
                        x_max = min(warpW, ix + radiusMP + 1)
                        YY, XX = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
                        circle_mask = (YY - iy) ** 2 + (XX - ix) ** 2 <= radiusMP ** 2
                        conf2d[y_min:y_max, x_min:x_max][circle_mask] = 0

                    # path_confidence_output = os.path.join("/datas/servantf/", f"{referenceId}_confidence.exr")
                    # save_image(path_confidence_output, conf2d[:, :, np.newaxis], True)
                    # raise RuntimeError()

                npSelected = np.array(selected) if len(selected) > 0 else np.empty((0, 2))
                logging.info(f"Previously {npSelected.shape}")
                if len(selected) < maxMatches:
                    
                    # Compute new coords
                    new_coords = get_samples(conf2d, minConfidence, maxMatches - len(selected), radiusMP)
                    if len(new_coords) > 0:
                        npSelected = np.concatenate([npSelected, new_coords], axis=0)
                        logging.info(f"Updated to {npSelected.shape}")
 
                samplesPerReference[referenceId] = npSelected

        logging.info("Exporting.")
        #Loop over all reference images
        for referenceId in refsToProcess:
            if referenceId not in samplesPerReference:
                continue

            logging.info(f"Processing reference #{referenceId}")
            
            reference_iinfo = iinfos[referenceId]
            samples_A_B = samplesPerReference[referenceId]
            grouped = groupedPerReference[referenceId]
        
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
            pairs = plistByRef[referenceId]

            # Load uncertainties and store them using pair as key
            uncertaintiesByPair = build_uncertainties(iinfos, f_warp_h5, f_conf_h5, pairs, filtersByPair, minConfidence)
            if len(uncertaintiesByPair) == 0:
                logging.info(f"No uncertainties for reference #{referenceId}")
                continue

            if referenceId not in samplesPerReference:
                continue

            for item in uncertaintiesByPair:
                
                otherId = item[1]

                ref_iinfo = iinfos[referenceId]
                other_iinfo = iinfos[otherId]

                pair_string = str(referenceId) + "_" + str(otherId)

                #load images
                if pair_string not in f_warp_h5:
                    logging.warning(f"Warp not found for pair {pair_string}, skipping.")
                    continue
                warp_A_B = f_warp_h5[pair_string][()].astype(np.float32)
                

                confidence_A_B = uncertaintiesByPair[item]
                match_A_B = get_matches(samples_A_B, warp_A_B, confidence_A_B)

                #scale to original size
                scaledSamples = np.zeros((match_A_B.shape[0], 4))
                scaledSamples[:, 0] = match_A_B[:, 0] * other_iinfo.width
                scaledSamples[:, 1] = match_A_B[:, 1] * other_iinfo.height
                scaledSamples[:, 2] = match_A_B[:, 2]
                scaledSamples[:, 3] = scale

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
    parser.add_argument('--warpArchive', type=str, help='')
    parser.add_argument('--confidenceArchive', type=str, help='')
    parser.add_argument('--samplesFolder', type=str, help='')
    parser.add_argument('--filtersFolder', type=str, help='')
    parser.add_argument('--maxMatches', type=int, help='')
    parser.add_argument('--radiusMP', type=int, help='')
    parser.add_argument('--minConfidence', type=float, help='')
    parser.set_defaults(func=compute_samples)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                    imagePairsList=args.imagePairsList,
                    warpArchive=args.warpArchive,
                    confidenceArchive=args.confidenceArchive,
                    samplesFolder=args.samplesFolder,
                    filtersFolder=args.filtersFolder,
                    minConfidence=args.minConfidence,
                    maxMatches=args.maxMatches,
                    radiusMP=args.radiusMP)
    else:
        parser.print_help()
