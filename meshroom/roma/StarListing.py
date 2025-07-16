__version__ = "1.0"

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
import os

class StarListing(desc.Node):

    category = "ROMA"
    documentation = """"""
    size = desc.DynamicNodeSize("inputSfMData")

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData.",
            value="",
        ),
        desc.File(
            name="keySfMData",
            label="Keyframes SfMData",
            description="Input Keyframes SfMData.",
            value="",
        ),
        desc.IntParam(
            name="radiusKeyFrames",
            label="Keyframes Radius",
            description="Maximal distance between the reference keyframe and the other keyframe",
            value=1,
            range=(0, 100, 1)
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        )
    ]

    outputs = [
        desc.File(
            name="imagePairsList",
            label="Image Pairs",
            description="Path to a file which contains the list of image pairs to match.",
            value="{nodeCacheFolder}/imagesPairLists.txt"
        )
    ]

    def processChunk(self, chunk):

        from pyalicevision import sfmData as avsfmData
        from pyalicevision import sfmDataIO as avsfmDataIO
        from pyalicevision import matchingImageCollection as avmic

        chunk.logManager.start(chunk.node.verboseLevel.value)

        framesData = avsfmData.SfMData()
        ret = avsfmDataIO.load(framesData, chunk.node.inputSfMData.value, avsfmDataIO.VIEWS)
        if not ret:
            raise RuntimeError("Error with inputSfMData loading")

        keyframesData = avsfmData.SfMData()
        ret = avsfmDataIO.load(keyframesData, chunk.node.keySfMData.value, avsfmDataIO.VIEWS)
        if not ret:
            raise RuntimeError("Error with keySfMData data loading")
        
        
        # Get sorted list of keyframe frame ids
        keyframes = []
        views = keyframesData.getViews()
        for (key, view) in views.items():
            keyframes.append(view.getFrameId())
        keyframes.sort()

        if len(keyframes) == 0:
            raise RuntimeError("No keyframes found")
        
        # for all frames, build a dict frameid -> viewid
        frames = dict()
        views = framesData.getViews()
        for (key, view) in views.items():
            frames[view.getFrameId()] = key
        frames = dict(sorted(frames.items()))
        
        print(frames)


        dist = chunk.node.radiusKeyFrames.value
        framekeys = list(frames.keys())

        print(framekeys)

        plist = avmic.PairSet()
        for curKeyFrameId in range(0, len(keyframes)):
            firstKeyFrameId = max(curKeyFrameId - dist, -1)
            lastKeyFrameId = min(curKeyFrameId + dist, len(keyframes))

            curFrameId = keyframes[curKeyFrameId]

            print(firstKeyFrameId)
            print(lastKeyFrameId)
            
            firstFrameId = framekeys[0]
            if firstKeyFrameId >= 0:
                firstFrameId = keyframes[firstKeyFrameId]

            lastFrameId = framekeys[-1]
            if lastKeyFrameId < len(keyframes):
                lastFrameId = keyframes[lastKeyFrameId]

            print(firstFrameId)
            print(lastFrameId)

            referenceViewId = frames[curFrameId]
            for otherFrameId in range(firstFrameId, lastFrameId + 1):
                if otherFrameId not in frames:
                    continue
                    
                otherViewId = frames[otherFrameId]
                if referenceViewId == otherViewId:
                    continue
                plist.append((referenceViewId, otherViewId))

    
        if not avmic.savePairsToFile(chunk.node.imagePairsList.value, plist):
            raise RuntimeError("Error in image pairs list loading")

        chunk.logManager.end()
